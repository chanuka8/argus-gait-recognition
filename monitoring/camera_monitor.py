import json
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from core.logger import setup_logger


class CameraMonitor:
    def __init__(
        self,
        camera_manager: Any,
        stats_dir: str = "outputs/monitoring/camera_stats",
        collection_interval: int = 30,
    ) -> None:
        self.camera_manager = camera_manager
        self.stats_dir = Path(stats_dir)
        self.collection_interval = collection_interval

        self._logger = setup_logger("camera_monitor")
        self._lock = Lock()
        self._stop_event = True
        self._monitor_thread: Thread | None = None

        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self._stats_history: dict[str, list] = {}
        self._alerts: dict[str, list] = {}

    def start(self) -> None:
        if not self._stop_event:
            self._logger.warning("Monitor already running")
            return

        self._stop_event = False
        self._monitor_thread = Thread(
            target=self._monitoring_loop,
            name="camera-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

        self._logger.info("Camera monitor started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event = True

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)

        self._logger.info("Camera monitor stopped")

    def _monitoring_loop(self) -> None:
        while not self._stop_event:
            try:
                time.sleep(self.collection_interval)

                if self._stop_event:
                    break

                self._collect_stats()
                self._check_health()
                self._save_stats()

            except (RuntimeError, ValueError, KeyError, OSError, TypeError) as e:
                self._logger.error(f"Monitoring error: {e!s}")

    def _collect_stats(self) -> None:
        all_stats = self.camera_manager.get_all_stats()
        timestamp = time.monotonic()

        with self._lock:
            for camera_id, stats in all_stats.items():
                if camera_id not in self._stats_history:
                    self._stats_history[camera_id] = []

                self._stats_history[camera_id].append(
                    {
                        "timestamp": timestamp,
                        **stats,
                    }
                )

                if len(self._stats_history[camera_id]) > 1000:
                    self._stats_history[camera_id] = self._stats_history[camera_id][-1000:]

    def _check_health(self) -> None:
        all_status = self.camera_manager.get_all_status()

        with self._lock:
            for camera_id, status in all_status.items():
                if camera_id not in self._alerts:
                    self._alerts[camera_id] = []

                if not status["running"]:
                    self._alerts[camera_id].append(
                        {
                            "type": "worker_down",
                            "timestamp": time.monotonic(),
                            "message": f"Worker crashed for {camera_id}",
                        }
                    )

                    self._logger.error(f"Alert: Worker down for {camera_id}")

                if not status["connected"]:
                    self._alerts[camera_id].append(
                        {
                            "type": "disconnected",
                            "timestamp": time.monotonic(),
                            "message": f"Camera {camera_id} disconnected",
                        }
                    )

                    self._logger.warning(f"Alert: Disconnected {camera_id}")

                stats = status.get("stats", {})

                if stats.get("fps", 0) < 2.0:
                    self._alerts[camera_id].append(
                        {
                            "type": "low_fps",
                            "timestamp": time.monotonic(),
                            "fps": stats.get("fps"),
                            "message": f"Low FPS for {camera_id}",
                        }
                    )

                if stats.get("queue_size", 0) > 8:
                    self._alerts[camera_id].append(
                        {
                            "type": "queue_overflow",
                            "timestamp": time.monotonic(),
                            "queue_size": stats.get("queue_size"),
                            "message": f"Queue overflow for {camera_id}",
                        }
                    )

                if len(self._alerts[camera_id]) > 1000:
                    self._alerts[camera_id] = self._alerts[camera_id][-1000:]

    def _save_stats(self) -> None:
        try:
            with self._lock:
                for camera_id, history in self._stats_history.items():
                    if not history:
                        continue

                    stats_file = self.stats_dir / f"{camera_id}_stats.json"

                    with open(stats_file, "w", encoding="utf-8") as f:
                        json.dump(history[-100:], f, indent=2)

        except (OSError, TypeError, ValueError) as e:
            self._logger.error(f"Failed to save stats: {e!s}")

    def get_camera_health(self, camera_id: str) -> dict[str, Any] | None:
        status = self.camera_manager.get_camera_status(camera_id)

        if status is None:
            return None

        with self._lock:
            alerts = self._alerts.get(camera_id, [])

            return {
                "camera_id": camera_id,
                "status": status,
                "recent_alerts": alerts[-10:],
            }

    def get_all_health(self) -> dict[str, Any]:
        all_status = self.camera_manager.get_all_status()
        health = {}

        with self._lock:
            for camera_id, status in all_status.items():
                alerts = self._alerts.get(camera_id, [])

                health[camera_id] = {
                    "status": status,
                    "recent_alerts": alerts[-10:],
                }

        return health

    def get_alerts(self, camera_id: str | None = None) -> dict[str, list]:
        with self._lock:
            if camera_id:
                return {camera_id: self._alerts.get(camera_id, [])}

            return {cid: alerts.copy() for cid, alerts in self._alerts.items()}

    def clear_alerts(self, camera_id: str | None = None) -> None:
        with self._lock:
            if camera_id:
                if camera_id in self._alerts:
                    self._alerts[camera_id] = []
            else:
                self._alerts = {}

        self._logger.info(f"Cleared alerts for {camera_id or 'all cameras'}")
