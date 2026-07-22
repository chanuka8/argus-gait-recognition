import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np

from core.logger import setup_logger
from monitoring.logging_config import get_logger


class CameraWorker:
    """Independent worker for a single camera stream."""

    def __init__(
        self,
        camera_id: str,
        camera_config: dict,
        inference_pipeline,
        detection_processor,
    ) -> None:
        self.camera_id = camera_id
        self.config = camera_config
        self.inference_pipeline = inference_pipeline
        self.detection_processor = detection_processor

        self._logger = get_logger(f"camera.{camera_id}")

        self._source_type = camera_config.get("type", "rtsp")
        self._url = camera_config.get("url", "")
        self._device_index = int(camera_config.get("device_index", 0))
        self._width = int(camera_config.get("width", 640))
        self._height = int(camera_config.get("height", 480))
        self._target_fps = int(camera_config.get("target_fps", 15))
        self._reconnect_interval = int(camera_config.get("reconnect_interval", 5))
        self._max_reconnect = int(camera_config.get("max_reconnect_attempts", 0))
        self._max_queue_size = int(camera_config.get("max_queue_size", 10))

        self._frame_queue = Queue(maxsize=self._max_queue_size)
        self._capture = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "queue_size": 0,
            "connected": False,
            "reconnect_count": 0,
            "uptime_seconds": 0,
            "identities_recognized": 0,
            "last_update": time.monotonic(),
        }

        self._frame_count = 0
        self._last_fps_count = 0
        self._last_fps_time = time.monotonic()
        self._start_time = time.monotonic()
        self._reconnect_count = 0
        self._latency_sum = 0.0
        self._latency_count = 0

    def _resolve_source(self):
        if self._source_type == "rtsp":
            if not self._url:
                raise ValueError(f"Camera {self.camera_id}: RTSP URL is empty")
            return self._url
        elif self._source_type == "file":
            return self.config.get("file_path", "")
        else:
            return self._device_index

    def _open_capture(self) -> bool:
        try:
            source = self._resolve_source()
            self._logger.info(f"Opening camera source: {source}")

            self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                self._logger.error(f"Failed to open camera source: {source}")
                self._capture = None
                return False

            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

            if self._target_fps > 0:
                self._capture.set(cv2.CAP_PROP_FPS, self._target_fps)

            with self._lock:
                self.stats["connected"] = True

            self._logger.info(f"Camera {self.camera_id} connected successfully")
            return True

        except Exception as e:
            self._logger.error(f"Error opening capture: {str(e)}")
            self._capture = None
            return False

    def _close_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

            with self._lock:
                self.stats["connected"] = False

    def _capture_loop(self) -> None:
        """Main frame capture and processing loop."""
        reconnect_attempts = 0

        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                if self._max_reconnect > 0 and reconnect_attempts >= self._max_reconnect:
                    self._logger.error(
                        f"Max reconnect attempts ({self._max_reconnect}) reached. Stopping."
                    )
                    break

                self._logger.warning(
                    f"Camera disconnected. Reconnecting in {self._reconnect_interval}s "
                    f"(attempt {reconnect_attempts + 1})"
                )

                self._stop_event.wait(self._reconnect_interval)

                if not self._open_capture():
                    reconnect_attempts += 1
                    self._reconnect_count += 1

                    with self._lock:
                        self.stats["reconnect_count"] = self._reconnect_count

                    continue

                reconnect_attempts = 0

            try:
                ret, frame = self._capture.read()

                if not ret or frame is None:
                    self._logger.warning("Failed to read frame")
                    self._close_capture()
                    continue

                frame = cv2.resize(frame, (self._width, self._height))

                try:
                    self._frame_queue.put(frame, block=False)
                    self._frame_count += 1
                except Full:
                    with self._lock:
                        self.stats["frames_dropped"] += 1

                elapsed = time.monotonic() - self._last_fps_time
                if elapsed >= 1.0:
                    with self._lock:
                        self.stats["fps"] = (
                            (self._frame_count - self._last_fps_count) / elapsed
                        )
                        self.stats["queue_size"] = self._frame_queue.qsize()
                        self.stats["uptime_seconds"] = time.monotonic() - self._start_time

                    self._last_fps_count = self._frame_count
                    self._last_fps_time = time.monotonic()

                with self._lock:
                    self.stats["frames_captured"] = self._frame_count
                    self.stats["last_update"] = time.monotonic()

            except Exception as e:
                self._logger.error(f"Error in capture loop: {str(e)}")
                self._close_capture()

        self._close_capture()
        self._logger.info("Camera capture loop stopped")

    def get_frame(self, timeout: float = 0.1) -> np.ndarray | None:
        """Get next frame from queue."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_stats(self) -> dict:
        """Get current camera statistics."""
        with self._lock:
            return self.stats.copy()

    def is_connected(self) -> bool:
        """Check if camera is connected."""
        with self._lock:
            return self.stats["connected"]

    def is_running(self) -> bool:
        """Check if worker thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start camera capture thread."""
        if self.is_running():
            self._logger.warning("Camera worker already running")
            return False

        if not self._open_capture():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{self.camera_id}",
            daemon=False,
        )
        self._thread.start()

        self._logger.info(f"Camera worker started: {self.camera_id}")
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop camera capture thread."""
        if not self.is_running():
            self._logger.debug("Camera worker already stopped")
            return True

        self._logger.info("Stopping camera worker...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)

            if self._thread.is_alive():
                self._logger.error("Thread did not stop within timeout")
                return False

        self._close_capture()
        self._logger.info("Camera worker stopped")
        return True

    def restart(self) -> bool:
        """Restart camera worker."""
        self._logger.info("Restarting camera worker...")
        success = self.stop(timeout=5.0)

        if not success:
            self._logger.error("Failed to stop worker before restart")
            return False

        time.sleep(1.0)
        return self.start()
