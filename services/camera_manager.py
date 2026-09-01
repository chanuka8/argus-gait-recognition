import threading
from pathlib import Path

import yaml

from core.logger import setup_logger
from security_layer.credentials import resolve_camera_config, sanitize_rtsp_url
from services.camera_worker import CameraWorker


class CameraManager:
    def __init__(
        self,
        config_path: str = "configs/cameras.yaml",
        inference_pipeline=None,
        detection_processor=None,
    ) -> None:
        self.config_path = Path(config_path)
        self._logger = setup_logger("camera_manager")

        self._inference_pipeline = inference_pipeline
        self._detection_processor = detection_processor

        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()
        self._health_check_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._load_config()

    def _load_config(self) -> None:
        self.cameras_config = {}
        self.defaults = {
            "width": 640,
            "height": 480,
            "target_fps": 15,
            "reconnect_interval": 5,
            "max_reconnect_attempts": 3,
            "max_queue_size": 10,
        }
        self.multi_camera_config = {}

        if not self.config_path.exists():
            self._logger.error(f"Config file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            raw_cameras = config.get("cameras", {})
            resolved_cameras = {}
            if isinstance(raw_cameras, dict):
                for cid, ccfg in raw_cameras.items():
                    c_dict = dict(ccfg) if isinstance(ccfg, dict) else {}
                    c_dict.setdefault("id", cid)
                    try:
                        resolved_cameras[cid] = resolve_camera_config(c_dict)
                    except (ValueError, KeyError, TypeError, OSError):
                        resolved_cameras[cid] = c_dict
            elif isinstance(raw_cameras, list):
                for ccfg in raw_cameras:
                    if isinstance(ccfg, dict):
                        cid = str(ccfg.get("id", "cam"))
                        try:
                            resolved_cameras[cid] = resolve_camera_config(ccfg)
                        except (ValueError, KeyError, TypeError, OSError):
                            resolved_cameras[cid] = ccfg

            self.cameras_config = resolved_cameras
            self.defaults = {**self.defaults, **config.get("defaults", {})}
            self.multi_camera_config = config.get("multi_camera", {})

            self._logger.info(f"Loaded {len(self.cameras_config)} camera configurations")

        except (yaml.YAMLError, OSError, ValueError) as e:
            self._logger.error(f"Failed to load config: {sanitize_rtsp_url(str(e))}")
            self.cameras_config = {}
            self.multi_camera_config = {}

    def _create_worker(self, camera_id: str, camera_config: dict) -> CameraWorker | None:
        try:
            config = {**self.defaults, **camera_config}
            config["id"] = camera_id
            resolved_config = resolve_camera_config(config)

            worker = CameraWorker(
                camera_id=camera_id,
                camera_config=resolved_config,
                inference_pipeline=self._inference_pipeline,
                detection_processor=self._detection_processor,
            )

            return worker

        except (RuntimeError, ValueError, TypeError, KeyError, OSError) as e:
            self._logger.error(f"Failed to create worker for {camera_id}: {sanitize_rtsp_url(str(e))}")
            return None

    def add_camera(self, camera_id: str, camera_config: dict) -> bool:
        with self._lock:
            if camera_id in self._workers:
                self._logger.warning(f"Camera {camera_id} already exists")
                return False


            if camera_config.get("enforce_admission", False):
                try:
                    from streaming.deployment_readiness import DeploymentReadinessManager

                    dm = DeploymentReadinessManager()
                    adm_res = dm.request_camera_admission(
                        camera_id=camera_id,
                        current_active_cameras=len(self._workers),
                    )
                    if not adm_res.admitted:
                        self._logger.warning(f"Camera {camera_id} admission rejected: {adm_res.reason}")
                        return False
                except (ImportError, Exception) as adm_err:  # noqa: BLE001
                    self._logger.debug(f"Camera admission evaluation notice: {adm_err}")

            worker = self._create_worker(camera_id, camera_config)

            if worker is None:
                return False

            self.cameras_config[camera_id] = camera_config
            self._workers[camera_id] = worker
            self._logger.info(f"Added camera: {camera_id}")
            return True

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            if camera_id not in self._workers:
                self._logger.warning(f"Camera {camera_id} not found")
                return False

            worker = self._workers[camera_id]
            success = worker.stop(timeout=5.0)

            if success:
                self.cameras_config.pop(camera_id, None)
                del self._workers[camera_id]
                self._logger.info(f"Removed camera: {camera_id}")

            return success

    def save_config(self, output_path: str | Path | None = None) -> bool:
        target_path = Path(output_path) if output_path is not None else self.config_path
        with self._lock:
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "defaults": self.defaults,
                    "cameras": self.cameras_config,
                    "multi_camera": self.multi_camera_config,
                }
                with open(target_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, default_flow_style=False)
                self._logger.info(f"Saved camera configuration to {target_path}")
                return True
            except (yaml.YAMLError, OSError, ValueError) as err:
                self._logger.error(f"Failed to save camera config: {err}")
                return False

    def start_all(self) -> int:
        started = 0

        with self._lock:
            for camera_id, camera_config in self.cameras_config.items():
                if not camera_config.get("enabled", True):
                    self._logger.info(f"Skipping disabled camera: {camera_id}")
                    continue

                if camera_id in self._workers:
                    worker = self._workers[camera_id]

                    if not worker.is_running() and worker.start():
                        started += 1
                    continue

                worker = self._create_worker(camera_id, camera_config)

                if worker is not None and worker.start():
                    self._workers[camera_id] = worker
                    started += 1

        if started > 0:
            self._start_health_check()

        self._logger.info(f"Started {started} camera workers")
        return started

    def stop_all(self, timeout: float = 5.0) -> int:
        stopped = 0

        self._stop_health_check()

        with self._lock:
            for worker in self._workers.values():
                if worker.stop(timeout=timeout):
                    stopped += 1

        self._logger.info(f"Stopped {stopped} camera workers")
        return stopped

    def restart_camera(self, camera_id: str) -> bool:
        with self._lock:
            if camera_id not in self._workers:
                self._logger.warning(f"Camera {camera_id} not found")
                return False

            worker = self._workers[camera_id]
            return worker.restart()

    def get_camera_stats(self, camera_id: str) -> dict | None:
        with self._lock:
            if camera_id not in self._workers:
                return None

            return self._workers[camera_id].get_stats()

    def get_all_stats(self) -> dict:
        stats = {}

        with self._lock:
            for camera_id, worker in self._workers.items():
                stats[camera_id] = worker.get_stats()

        return stats

    def get_camera_status(self, camera_id: str) -> dict | None:
        with self._lock:
            if camera_id not in self._workers:
                return None

            worker = self._workers[camera_id]

            return {
                "camera_id": camera_id,
                "connected": worker.is_connected(),
                "running": worker.is_running(),
                "stats": worker.get_stats(),
            }

    def get_all_status(self) -> dict:
        status = {}

        with self._lock:
            for camera_id, worker in self._workers.items():
                status[camera_id] = {
                    "camera_id": camera_id,
                    "connected": worker.is_connected(),
                    "running": worker.is_running(),
                    "stats": worker.get_stats(),
                }

        return status

    def _health_check_loop(self) -> None:
        interval = int(self.multi_camera_config.get("health_check_interval", 30))

        while not self._stop_event.is_set():
            try:
                self._stop_event.wait(interval)

                if self._stop_event.is_set():
                    break

                with self._lock:
                    for camera_id, worker in list(self._workers.items()):
                        if not worker.is_running():
                            self._logger.warning(f"Camera {camera_id} worker crashed")
                            success = worker.restart()

                            if not success:
                                self._logger.error(f"Failed to restart {camera_id}")
                        elif not worker.is_connected():
                            self._logger.warning(f"Camera {camera_id} disconnected")

            except (RuntimeError, ValueError, TypeError, OSError) as e:
                self._logger.error(f"Error in health check: {e!s}")

    def _start_health_check(self) -> None:
        if self._health_check_thread is not None and self._health_check_thread.is_alive():
            return

        self._stop_event.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="camera-health-check",
            daemon=True,
        )
        self._health_check_thread.start()

    def _stop_health_check(self) -> None:
        if self._health_check_thread is None:
            return

        self._stop_event.set()

        if self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)

    def get_worker(self, camera_id: str) -> CameraWorker | None:
        with self._lock:
            return self._workers.get(camera_id)

    def list_cameras(self) -> list[str]:
        with self._lock:
            return list(self._workers.keys())
