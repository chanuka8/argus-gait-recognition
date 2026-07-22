import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
import yaml

from monitoring.logging_config import get_logger


class CameraService:
    def __init__(self, config: dict | None = None) -> None:
        self._config = config or self._load_config()
        self._logger = get_logger("camera")

        self._source_type: str = self._config.get("type", "usb")
        self._rtsp_url: str = self._config.get("url", "")
        self._device_index: int = int(self._config.get("device_index", 0))
        self._file_path: str = self._config.get("file_path", "")
        self._width: int = int(self._config.get("width", 640))
        self._height: int = int(self._config.get("height", 480))
        self._target_fps: int = int(self._config.get("target_fps", 15))
        self._reconnect_seconds: int = int(self._config.get("reconnect_seconds", 5))
        self._max_reconnect: int = int(self._config.get("max_reconnect_attempts", 0))
        self._max_queue_size: int = int(self._config.get("max_queue_size", 10))

        self._capture: cv2.VideoCapture | None = None
        self._queue: Queue[np.ndarray] = Queue(maxsize=self._max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._frame_count: int = 0
        self._fps: float = 0.0
        self._last_fps_time: float = time.monotonic()
        self._last_fps_count: int = 0
        self._connected: bool = False
        self._reconnect_count: int = 0

    @staticmethod
    def _load_config() -> dict:
        config_path = Path("configs/system.yaml")

        defaults = {
            "type": "usb",
            "url": "",
            "device_index": 0,
            "file_path": "",
            "width": 640,
            "height": 480,
            "target_fps": 15,
            "reconnect_seconds": 5,
            "max_queue_size": 10,
        }

        if not config_path.exists():
            return defaults

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
        except Exception:
            return defaults

        section = data.get("camera", {})

        if not isinstance(section, dict):
            return defaults

        for key, default_value in defaults.items():
            defaults[key] = section.get(key, default_value)

        return defaults

    def _resolve_source(self) -> int | str:
        if self._source_type == "rtsp":
            if not self._rtsp_url:
                raise ValueError("RTSP URL is empty. Set camera.url in configs/system.yaml.")
            return self._rtsp_url
        elif self._source_type == "file":
            if not self._file_path or not Path(self._file_path).exists():
                raise FileNotFoundError(f"Video file not found: {self._file_path}")
            return self._file_path
        else:
            return self._device_index

    def _open_capture(self) -> bool:
        try:
            source = self._resolve_source()
            self._logger.info(f"Opening camera source: {source} (type={self._source_type})")

            self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                self._logger.error(f"Failed to open camera source: {source}")
                self._capture = None
                return False

            if self._source_type == "usb":
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

            if self._source_type == "rtsp":
                self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._connected = True
            self._logger.info(f"Camera connected: {source}")

            return True

        except Exception as error:
            self._logger.error(f"Camera open error: {error}")
            self._capture = None

            return False

    def _close_capture(self) -> None:
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None

            self._connected = False

    def _reconnect(self) -> bool:
        self._close_capture()
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1
            self._reconnect_count += 1

            if 0 < self._max_reconnect < attempt:
                self._logger.error(
                    f"Max reconnect attempts ({self._max_reconnect}) reached. Giving up."
                )
                return False

            self._logger.warning(
                f"Reconnect attempt {attempt} in {self._reconnect_seconds}s..."
            )
            self._stop_event.wait(self._reconnect_seconds)

            if self._stop_event.is_set():
                return False

            if self._open_capture():
                self._logger.info(f"Reconnected after {attempt} attempt(s).")
                return True

        return False

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / self._target_fps if self._target_fps > 0 else 0.0

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            with self._lock:
                cap = self._capture

            if cap is None or not cap.isOpened():
                self._connected = False
                self._logger.warning("Camera disconnected. Attempting reconnect...")

                if not self._reconnect():
                    self._logger.error("Camera reconnection failed permanently.")
                    break

                continue

            ret, frame = cap.read()

            if not ret or frame is None:
                self._connected = False
                self._logger.warning("Frame read failed. Attempting reconnect...")

                if not self._reconnect():
                    break

                continue

            self._connected = True
            self._frame_count += 1

            try:
                self._queue.put_nowait(frame)
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass

                try:
                    self._queue.put_nowait(frame)
                except Full:
                    pass

            now = time.monotonic()
            elapsed = now - self._last_fps_time

            if elapsed >= 1.0:
                self._fps = (self._frame_count - self._last_fps_count) / elapsed
                self._last_fps_time = now
                self._last_fps_count = self._frame_count

            if frame_interval > 0:
                processing_time = time.monotonic() - loop_start
                sleep_time = frame_interval - processing_time

                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._logger.warning("Camera service is already running.")
            return

        self._stop_event.clear()
        self._open_capture()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="ARGUS-CameraService",
            daemon=True,
        )
        self._thread.start()
        self._logger.info("Camera service started.")

    def stop(self) -> None:
        self._logger.info("Stopping camera service...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

        self._close_capture()

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

        self._logger.info("Camera service stopped.")

    def get_frame(self) -> np.ndarray | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    def get_status(self) -> dict:
        return {
            "alive": self.is_alive(),
            "connected": self.is_connected(),
            "fps": round(self._fps, 2),
            "queue_size": self.queue_size,
            "frame_count": self._frame_count,
            "reconnect_count": self._reconnect_count,
            "source_type": self._source_type,
        }
