import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from core.logger import setup_logger
from services.camera_worker import CameraWorker


class CameraManager:
    """Orchestrates multiple camera workers for multi-camera CCTV surveillance."""

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

        self._workers: Dict[str, CameraWorker] = {}
        self._lock = threading.Lock()
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._load_config()

    def _load_config(self) -> None:
        """Load camera configuration from YAML."""
        if not self.config_path.exists():
            self._logger.error(f"Config file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            self.cameras_config = config.get("cameras", {})
            self.defaults = config.get("defaults", {})
            self.multi_camera_config = config.get("multi_camera", {})

            self._logger.info(f"Loaded {len(self.cameras_config)} camera configurations")

        except Exception as e:
            self._logger.error(f"Failed to load config: {str(e)}")
            self.cameras_config = {}
            self.defaults = {}
            self.multi_camera_config = {}

    def _create_worker(self, camera_id: str, camera_config: dict) -> Optional[CameraWorker]:
        """Create a camera worker."""
        try:
            config = {**self.defaults, **camera_config}
            config["id"] = camera_id

            worker = CameraWorker(
                camera_id=camera_id,
                camera_config=config,
                inference_pipeline=self._inference_pipeline,
                detection_processor=self._detection_processor,
            )

            return worker

        except Exception as e:
            self._logger.error(f"Failed to create worker for {camera_id}: {str(e)}")
            return None

    def add_camera(self, camera_id: str, camera_config: dict) -> bool:
        """Dynamically add a camera."""
        with self._lock:
            if camera_id in self._workers:
                self._logger.warning(f"Camera {camera_id} already exists")
                return False

            worker = self._create_worker(camera_id, camera_config)

            if worker is None:
                return False

            self._workers[camera_id] = worker
            self._logger.info(f"Added camera: {camera_id}")
            return True

    def remove_camera(self, camera_id: str) -> bool:
        """Dynamically remove a camera."""
        with self._lock:
            if camera_id not in self._workers:
                self._logger.warning(f"Camera {camera_id} not found")
                return False

            worker = self._workers[camera_id]
            success = worker.stop(timeout=5.0)

            if success:
                del self._workers[camera_id]
                self._logger.info(f"Removed camera: {camera_id}")

            return success

    def start_all(self) -> int:
        """Start all enabled cameras."""
        started = 0

        with self._lock:
            for camera_id, camera_config in self.cameras_config.items():
                if not camera_config.get("enabled", True):
                    self._logger.info(f"Skipping disabled camera: {camera_id}")
                    continue

                if camera_id in self._workers:
                    worker = self._workers[camera_id]

                    if not worker.is_running():
                        if worker.start():
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
        """Stop all camera workers."""
        stopped = 0

        self._stop_health_check()

        with self._lock:
            for worker in self._workers.values():
                if worker.stop(timeout=timeout):
                    stopped += 1

        self._logger.info(f"Stopped {stopped} camera workers")
        return stopped

    def restart_camera(self, camera_id: str) -> bool:
        """Restart a specific camera."""
        with self._lock:
            if camera_id not in self._workers:
                self._logger.warning(f"Camera {camera_id} not found")
                return False

            worker = self._workers[camera_id]
            return worker.restart()

    def get_camera_stats(self, camera_id: str) -> Optional[dict]:
        """Get statistics for a specific camera."""
        with self._lock:
            if camera_id not in self._workers:
                return None

            return self._workers[camera_id].get_stats()

    def get_all_stats(self) -> dict:
        """Get statistics for all cameras."""
        stats = {}

        with self._lock:
            for camera_id, worker in self._workers.items():
                stats[camera_id] = worker.get_stats()

        return stats

    def get_camera_status(self, camera_id: str) -> Optional[dict]:
        """Get status for a specific camera."""
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
        """Get status for all cameras."""
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
        """Periodic health check for all cameras."""
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

            except Exception as e:
                self._logger.error(f"Error in health check: {str(e)}")

    def _start_health_check(self) -> None:
        """Start background health check thread."""
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
        """Stop background health check thread."""
        if self._health_check_thread is None:
            return

        self._stop_event.set()

        if self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)

    def get_worker(self, camera_id: str) -> Optional[CameraWorker]:
        """Get a specific camera worker."""
        with self._lock:
            return self._workers.get(camera_id)

    def list_cameras(self) -> list[str]:
        """List all camera IDs."""
        with self._lock:
            return list(self._workers.keys())
