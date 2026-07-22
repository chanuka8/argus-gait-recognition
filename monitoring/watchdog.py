import threading
import time
from typing import Any

from monitoring.logging_config import get_logger


class Watchdog:
    def __init__(self, service: Any, config: dict | None = None) -> None:
        self._service = service
        self._config = config or {}
        self._logger = get_logger("watchdog")
        self._error_logger = get_logger("error")

        self._interval: int = int(self._config.get("interval_seconds", 30))
        self._max_queue_warning: int = int(self._config.get("max_queue_warning", 8))
        self._min_fps_warning: float = float(self._config.get("min_fps_warning", 2.0))
        self._auto_restart: bool = bool(self._config.get("auto_restart", True))
        self._max_restarts: int = int(self._config.get("max_restart_attempts", 5))

        self._consecutive_restarts: int = 0
        self._stop_event = threading.Event()

    @staticmethod
    def _collect_resource_usage() -> dict:
        metrics = {
            "cpu_percent": 0.0,
            "ram_percent": 0.0,
            "ram_used_mb": 0.0,
            "gpu_name": "N/A",
            "gpu_memory_used_mb": 0.0,
            "gpu_utilization_percent": 0.0,
        }

        try:
            import psutil
            process = psutil.Process()
            metrics["cpu_percent"] = process.cpu_percent(interval=0.1)
            mem_info = process.memory_info()
            metrics["ram_used_mb"] = round(mem_info.rss / (1024 * 1024), 1)
            metrics["ram_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu_name"] = torch.cuda.get_device_name(0)
                metrics["gpu_memory_used_mb"] = round(
                    torch.cuda.memory_allocated(0) / (1024 * 1024), 1
                )
        except Exception:
            pass

        return metrics

    def run(self) -> None:
        self._logger.info(f"Watchdog initialized (check interval: {self._interval}s).")

        while not self._stop_event.is_set():
            if self._stop_event.wait(self._interval):
                break

            try:
                self.check_health()
            except Exception as error:
                self._error_logger.error(f"Watchdog execution failure: {error}", exc_info=True)

    def check_health(self) -> dict:
        status = self._service.get_status()
        cam_status = status.get("camera", {})
        resources = self._collect_resource_usage()

        cam_alive = cam_status.get("alive", False)
        cam_connected = cam_status.get("connected", False)
        rec_alive = status.get("recognition_alive", False)
        queue_size = cam_status.get("queue_size", 0)
        fps = cam_status.get("fps", 0.0)

        health_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_alive": cam_alive,
            "camera_connected": cam_connected,
            "recognition_alive": rec_alive,
            "fps": fps,
            "queue_size": queue_size,
            "cpu_percent": resources["cpu_percent"],
            "ram_used_mb": resources["ram_used_mb"],
            "gpu_memory_used_mb": resources["gpu_memory_used_mb"],
            "reconnect_count": cam_status.get("reconnect_count", 0),
        }

        self._logger.info(
            f"HEALTH CHECK | CamAlive={cam_alive} | CamConn={cam_connected} | "
            f"RecAlive={rec_alive} | FPS={fps:.1f} | Queue={queue_size} | "
            f"CPU={resources['cpu_percent']}% | RAM={resources['ram_used_mb']}MB"
        )

        if cam_alive and not cam_connected:
            self._logger.warning("Camera service is running but disconnected.")

        if queue_size > self._max_queue_warning:
            self._logger.warning(
                f"Queue size ({queue_size}) exceeded warning threshold ({self._max_queue_warning})."
            )

        if cam_connected and fps < self._min_fps_warning:
            self._logger.warning(
                f"Capture FPS ({fps:.1f}) is below warning threshold ({self._min_fps_warning})."
            )

        if self._auto_restart:
            if not rec_alive:
                self._handle_failure("Recognition worker thread died", restart_target="recognition")
            elif not cam_alive:
                self._handle_failure("Camera service thread died", restart_target="camera")
            else:
                self._consecutive_restarts = 0

        return health_report

    def _handle_failure(self, reason: str, restart_target: str) -> None:
        self._consecutive_restarts += 1

        message = (
            f"CRITICAL HEALTH FAILURE: {reason} | "
            f"Consecutive restarts: {self._consecutive_restarts}/{self._max_restarts}"
        )

        self._error_logger.error(message)
        self._logger.error(message)

        if self._consecutive_restarts > self._max_restarts:
            fatal_message = (
                f"Max consecutive restart attempts ({self._max_restarts}) reached. "
                "Manual intervention required."
            )
            self._error_logger.critical(fatal_message)
            self._logger.critical(fatal_message)
            return

        if restart_target == "recognition":
            self._service.restart_recognition()
        elif restart_target == "camera":
            self._service.restart_camera()

    def stop(self) -> None:
        self._stop_event.set()
