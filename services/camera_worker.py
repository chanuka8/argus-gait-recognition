import threading
import time
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from typing import Optional

import cv2
import numpy as np

from monitoring.logging_config import get_logger
from security_layer.credentials import sanitize_rtsp_url


def normalize_camera_source(source) -> int | str:
    """Normalize camera source to an integer device index or cleaned string."""
    if isinstance(source, int):
        return source

    normalized = str(source).strip()
    if normalized.isdigit():
        return int(normalized)

    return normalized


class CameraWorker:
    """Independent worker for a single camera stream with real-time JPEG preview buffer."""

    def __init__(
        self,
        camera_id: str,
        camera_config: dict,
        inference_pipeline=None,
        detection_processor=None,
    ) -> None:
        self.camera_id = camera_id
        self.config = camera_config
        self.inference_pipeline = inference_pipeline
        self.detection_processor = detection_processor

        self._logger = get_logger(f"camera.{camera_id}")

        self._source_type = camera_config.get("type", "usb")
        self._url = camera_config.get("url", "")
        self._device_index = int(camera_config.get("device_index", 0))
        self._width = int(camera_config.get("width", 640))
        self._height = int(camera_config.get("height", 480))
        self._target_fps = int(camera_config.get("target_fps", 15))
        self._reconnect_interval = int(camera_config.get("reconnect_interval", 5))
        self._max_reconnect = int(camera_config.get("max_reconnect_attempts", 0))
        self._max_queue_size = int(camera_config.get("max_queue_size", 10))
        self._jpeg_quality = int(camera_config.get("jpeg_quality", 75))
        self._min_jpeg_interval = 1.0 / float(camera_config.get("preview_max_fps", 15))

        self._frame_queue = Queue(maxsize=self._max_queue_size)
        self._capture = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_jpeg: Optional[bytes] = None
        self._last_frame_at: Optional[str] = None
        self._last_jpeg_encode_time: float = 0.0
        self._active_tracks: int = 0

        self.stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "queue_size": 0,
            "connected": False,
            "reconnect_count": 0,
            "uptime_seconds": 0.0,
            "identities_recognized": 0,
            "last_update": time.monotonic(),
        }

        self._frame_count = 0
        self._last_fps_count = 0
        self._last_fps_time = time.monotonic()
        self._start_time = time.monotonic()
        self._reconnect_count = 0

    def _resolve_source(self):
        if self._source_type == "rtsp":
            if not self._url:
                raise ValueError(f"Camera {self.camera_id}: RTSP URL is empty")
            return self._url
        elif self._source_type == "http":
            if not self._url:
                raise ValueError(f"Camera {self.camera_id}: HTTP URL is empty")
            return self._url
        elif self._source_type == "file":
            return self.config.get("file_path", "")
        else:
            return normalize_camera_source(self.config.get("device_index", self._device_index))

    def _open_capture(self) -> bool:
        try:
            source = self._resolve_source()
            safe_source = sanitize_rtsp_url(str(source))
            self._logger.info(f"Opening camera source: {safe_source}")

            self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                self._logger.error(f"Failed to open camera source: {safe_source}")
                self._capture = None
                return False

            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

            if self._target_fps > 0:
                self._capture.set(cv2.CAP_PROP_FPS, self._target_fps)

            # Read initial test frame to guarantee readiness
            ret, frame = self._capture.read()
            if not ret or frame is None or frame.size == 0:
                self._logger.error(f"Failed to read initial frame from camera source: {safe_source}")
                self._capture.release()
                self._capture = None
                return False

            # Process initial frame into JPEG buffer
            frame_resized = cv2.resize(frame, (self._width, self._height))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            success, enc_buf = cv2.imencode(".jpg", frame_resized, encode_param)
            now = time.monotonic()
            iso_now = datetime.now(timezone.utc).isoformat()

            with self._lock:
                self.stats["connected"] = True
                self._frame_count = 1
                self.stats["frames_captured"] = 1
                if success:
                    self._latest_jpeg = enc_buf.tobytes()
                    self._last_frame_at = iso_now
                    self._last_jpeg_encode_time = now

            self._logger.info(f"Camera {self.camera_id} connected and initial frame verified successfully")
            return True

        except Exception as e:
            self._logger.error(f"Error opening capture: {str(e)}")
            if self._capture is not None:
                self._capture.release()
            self._capture = None
            return False

    def _close_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

            with self._lock:
                self.stats["connected"] = False

    def _capture_loop(self) -> None:
        """Main frame capture and preview encoding loop."""
        reconnect_attempts = 0

        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                if self._max_reconnect > 0 and reconnect_attempts >= self._max_reconnect:
                    self._logger.error(f"Max reconnect attempts ({self._max_reconnect}) reached. Stopping.")
                    break

                self._stop_event.wait(self._reconnect_interval)
                if not self._open_capture():
                    reconnect_attempts += 1
                    continue
                reconnect_attempts = 0

            try:
                ret, frame = self._capture.read()

                if not ret or frame is None:
                    self._close_capture()
                    continue

                frame = cv2.resize(frame, (self._width, self._height))
                now = time.monotonic()
                iso_now = datetime.now(timezone.utc).isoformat()

                # Rate-limited preview JPEG encoding
                if now - self._last_jpeg_encode_time >= self._min_jpeg_interval:
                    try:
                        preview_frame = frame
                        if preview_frame.shape[1] > 640:
                            h, w = preview_frame.shape[:2]
                            new_h = int(h * (640 / w))
                            preview_frame = cv2.resize(preview_frame, (640, new_h))

                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
                        success, enc_buf = cv2.imencode(".jpg", preview_frame, encode_param)
                        if success:
                            jpeg_bytes = enc_buf.tobytes()
                            with self._lock:
                                self._latest_jpeg = jpeg_bytes
                                self._last_frame_at = iso_now
                            self._last_jpeg_encode_time = now
                    except Exception as enc_err:
                        self._logger.debug(f"Preview JPEG encode error: {enc_err}")

                try:
                    self._frame_queue.put(frame, block=False)
                    self._frame_count += 1
                except Full:
                    with self._lock:
                        self.stats["frames_dropped"] += 1

                elapsed = now - self._last_fps_time
                if elapsed >= 1.0:
                    with self._lock:
                        self.stats["fps"] = (self._frame_count - self._last_fps_count) / elapsed
                        self.stats["queue_size"] = self._frame_queue.qsize()
                        self.stats["uptime_seconds"] = now - self._start_time

                    self._last_fps_count = self._frame_count
                    self._last_fps_time = now

                with self._lock:
                    self.stats["frames_captured"] = self._frame_count
                    self.stats["last_update"] = now

            except Exception as e:
                self._logger.error(f"Error in capture loop: {str(e)}")
                self._close_capture()

        self._close_capture()
        self._logger.info("Camera capture loop stopped")

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return the latest encoded JPEG frame bytes safely."""
        with self._lock:
            return self._latest_jpeg

    def get_stats(self) -> dict:
        """Get current camera statistics."""
        with self._lock:
            return {
                **self.stats.copy(),
                "last_frame_at": self._last_frame_at,
                "active_tracks": self._active_tracks,
            }

    def is_connected(self) -> bool:
        with self._lock:
            return self.stats["connected"]

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start camera capture thread after verifying readiness."""
        if self.is_running():
            return False

        if not self._open_capture():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 3.0) -> bool:
        """Stop camera capture thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._close_capture()
        return True
