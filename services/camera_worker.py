"""
Single-Owner Camera Worker for ARGUS AI.
Maintains a single cv2.VideoCapture instance per camera worker.
Decouples high-speed continuous capture from detection/gait inference.
Provides low-latency latest-frame slot, real frame statistics, and MJPEG preview streaming.
"""

from datetime import datetime, timezone
from enum import Enum
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.logger import setup_logger
from services.camera_source import parse_backend_flag, sanitize_url


logger = setup_logger("ARGUS.CameraWorker")


class WorkerState(str, Enum):
    DISCOVERED = "DISCOVERED"
    PROBING = "PROBING"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class CameraWorker:
    """
    Thread-safe single-owner Camera Worker.
    Decouples frame capture loop from inference processing loop.
    Supports backward compatibility with legacy camera_config dictionaries and CameraManager API.
    """

    def __init__(
        self,
        camera_id: str,
        source: Any = None,
        zone_id: str = "Z01",
        source_type: str = "auto",
        capture_backend: str = "auto",
        location: str = "Surveillance Zone",
        target_width: int = 640,
        target_height: int = 480,
        target_fps: float = 30.0,
        gait_service: Optional[Any] = None,
        camera_config: Optional[Dict[str, Any]] = None,
        inference_pipeline: Optional[Any] = None,
        detection_processor: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.camera_id = camera_id

        # Handle legacy camera_config dictionary if passed
        if source is None:
            if camera_config and isinstance(camera_config, dict):
                source = camera_config.get("url", camera_config.get("device_index", 0))
                if "type" in camera_config:
                    source_type = camera_config["type"]
                target_width = camera_config.get("width", target_width)
                target_height = camera_config.get("height", target_height)
                target_fps = float(camera_config.get("target_fps", target_fps))
            else:
                source = 0

        self.raw_source = source
        self.zone_id = zone_id
        self.source_type = source_type
        self.capture_backend_requested = capture_backend
        self.location = location
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.gait_service = gait_service
        self.inference_pipeline = inference_pipeline
        self.detection_processor = detection_processor
        self.camera_config = camera_config or {}

        self.sanitized_source = sanitize_url(str(source))
        self.status = WorkerState.STOPPED.value
        self.error_message: Optional[str] = None

        # Process / Capture state
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._lock = threading.Lock()

        # Low-latency Latest Frame Slots
        self._latest_captured_frame: Optional[np.ndarray] = None
        self._latest_captured_seq: int = 0
        self._latest_processed_seq: int = 0

        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_jpeg_bytes: Optional[bytes] = None
        self._last_event: Optional[Dict[str, Any]] = None

        # Real performance metrics
        self.started_at: Optional[str] = None
        self.last_frame_at: Optional[str] = None

        self.captured_frames: int = 0
        self.processed_frames: int = 0
        self.dropped_frames: int = 0

        self.capture_fps: float = 0.0
        self.processing_fps: float = 0.0
        self.processing_latency_ms: float = 0.0
        self.active_tracks: int = 0

    def is_running(self) -> bool:
        """Returns True if the worker is currently starting or active."""
        with self._lock:
            return self.status in (WorkerState.ACTIVE.value, WorkerState.STARTING.value)

    def is_connected(self) -> bool:
        """Returns True if the worker video stream is active and connected."""
        with self._lock:
            return self.status == WorkerState.ACTIVE.value

    def _open_capture(self) -> bool:
        """Opens cv2.VideoCapture using specified backend flag and device/URL."""
        try:
            backend_flag = parse_backend_flag(self.capture_backend_requested)

            # Determine whether source is integer USB index or string URL
            source_val = self.raw_source
            try:
                val_int = int(source_val)
                if val_int >= 0:
                    source_val = val_int
            except (ValueError, TypeError):
                source_val = str(source_val)

            if isinstance(source_val, int) and backend_flag != cv2.CAP_ANY:
                self._capture = cv2.VideoCapture(source_val, backend_flag)
            else:
                self._capture = cv2.VideoCapture(source_val)

            if not self._capture.isOpened():
                self.error_message = f"Failed to open video capture for source: {self.sanitized_source}"
                logger.error(self.error_message)
                self._capture = None
                return False

            if self.target_width > 0:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            if self.target_height > 0:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            if self.target_fps > 0:
                self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

            return True
        except Exception as err:
            self.error_message = f"Exception opening video capture: {err}"
            logger.error(self.error_message)
            self._capture = None
            return False

    def _close_capture(self) -> None:
        """Safely releases cv2.VideoCapture."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def _capture_loop(self) -> None:
        """High-frequency dedicated capture loop reading raw camera frames continuously."""
        fps_counter = 0
        fps_start_time = time.monotonic()

        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                break

            try:
                ret, frame = self._capture.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue

                now_iso = datetime.now(timezone.utc).isoformat()
                fps_counter += 1
                elapsed = time.monotonic() - fps_start_time
                if elapsed >= 1.0:
                    self.capture_fps = round(fps_counter / elapsed, 2)
                    fps_counter = 0
                    fps_start_time = time.monotonic()

                with self._lock:
                    self.captured_frames += 1
                    self._latest_captured_seq += 1
                    self._latest_captured_frame = frame
                    self.last_frame_at = now_iso

                    # Encode initial JPEG bytes for immediate preview availability
                    if self._latest_jpeg_bytes is None:
                        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if buf is not None:
                            self._latest_jpeg_bytes = buf.tobytes()

                if not self._first_frame_event.is_set():
                    self._first_frame_event.set()

            except Exception as err:
                logger.error(f"Error in capture loop for camera {self.camera_id}: {err}")
                break

    def _inference_loop(self) -> None:
        """Inference processing loop consuming the latest unprocessed frame slot."""
        fps_counter = 0
        fps_start_time = time.monotonic()

        while not self._stop_event.is_set():
            target_frame = None
            seq_to_process = 0

            with self._lock:
                if self._latest_captured_frame is not None and self._latest_captured_seq > self._latest_processed_seq:
                    skipped = (self._latest_captured_seq - self._latest_processed_seq) - 1
                    if skipped > 0:
                        self.dropped_frames += skipped

                    target_frame = self._latest_captured_frame.copy()
                    seq_to_process = self._latest_captured_seq
                    self._latest_processed_seq = seq_to_process

            if target_frame is None:
                time.sleep(0.005)
                continue

            t0 = time.monotonic()
            annotated_frame = target_frame
            event_res = None

            if self.gait_service is not None:
                try:
                    _, img_bytes = cv2.imencode(".jpg", target_frame)
                    event_res = self.gait_service.process_image_bytes(img_bytes.tobytes(), camera_id=self.camera_id)

                    if event_res:
                        bbox = event_res.get("bbox", [0, 0, 0, 0])
                        identity = event_res.get("identity", "UNKNOWN")
                        confidence = event_res.get("confidence", 0.0)

                        x1, y1, x2, y2 = bbox
                        if x2 > x1 and y2 > y1:
                            color = (0, 255, 0) if identity != "UNKNOWN" else (0, 0, 255)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{identity} ({confidence:.2f})"
                            cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                except Exception as err:
                    logger.warning(f"Inference processing error for camera {self.camera_id}: {err}")

            t_lat = (time.monotonic() - t0) * 1000.0

            _, jpeg_buf = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = jpeg_buf.tobytes() if jpeg_buf is not None else None

            fps_counter += 1
            elapsed = time.monotonic() - fps_start_time
            if elapsed >= 1.0:
                self.processing_fps = round(fps_counter / elapsed, 2)
                fps_counter = 0
                fps_start_time = time.monotonic()

            with self._lock:
                self.processed_frames += 1
                self.processing_latency_ms = round(t_lat, 2)
                self._latest_annotated_frame = annotated_frame
                if jpeg_bytes:
                    self._latest_jpeg_bytes = jpeg_bytes
                if event_res:
                    self._last_event = event_res
                    self.active_tracks = 1 if event_res.get("identity") != "UNKNOWN" else 0

    def start(self, startup_timeout: float = 5.0) -> bool:
        """Starts worker threads and waits for first real frame handshake."""
        with self._lock:
            if self.status in (WorkerState.ACTIVE.value, WorkerState.STARTING.value):
                logger.warning(f"Worker {self.camera_id} is already starting or active.")
                return True

            self.status = WorkerState.STARTING.value
            self.error_message = None

        if not self._open_capture():
            with self._lock:
                self.status = WorkerState.ERROR.value
            return False

        self._stop_event.clear()
        self._first_frame_event.clear()
        self.started_at = datetime.now(timezone.utc).isoformat()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"cap-{self.camera_id}",
            daemon=True,
        )
        self._capture_thread.start()

        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name=f"inf-{self.camera_id}",
            daemon=True,
        )
        self._inference_thread.start()

        got_frame = self._first_frame_event.wait(timeout=startup_timeout)
        if not got_frame:
            self.error_message = f"Startup timeout: Failed to read first frame within {startup_timeout}s."
            logger.error(self.error_message)
            self.stop()
            with self._lock:
                self.status = WorkerState.ERROR.value
            return False

        with self._lock:
            self.status = WorkerState.ACTIVE.value

        logger.info(f"Camera worker '{self.camera_id}' is ACTIVE in zone '{self.zone_id}'.")
        return True

    def stop(self, timeout: float = 3.0) -> bool:
        """Stops capture and inference threads, releasing VideoCapture."""
        with self._lock:
            if self.status in (WorkerState.STOPPED.value, WorkerState.STOPPING.value):
                return True
            self.status = WorkerState.STOPPING.value

        self._stop_event.set()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=timeout)

        self._close_capture()

        with self._lock:
            self.status = WorkerState.STOPPED.value
            self._latest_captured_frame = None
            self._latest_annotated_frame = None
            self._latest_jpeg_bytes = None

        logger.info(f"Camera worker '{self.camera_id}' stopped cleanly.")
        return True

    def get_jpeg_frame(self) -> Optional[bytes]:
        """Returns the latest pre-encoded JPEG bytes for MJPEG streaming without calling cv2.VideoCapture."""
        with self._lock:
            return self._latest_jpeg_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Backward-compatible alias for to_dict()."""
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Returns detailed status dictionary and real performance statistics."""
        with self._lock:
            uptime = 0.0
            if self.started_at:
                try:
                    dt = datetime.fromisoformat(self.started_at)
                    uptime = (datetime.now(timezone.utc) - dt).total_seconds()
                except Exception:
                    pass

            return {
                "camera_id": self.camera_id,
                "zone_id": self.zone_id,
                "source": self.sanitized_source,
                "source_type": self.source_type,
                "location": self.location,
                "status": self.status,
                "connected": self.status == WorkerState.ACTIVE.value,
                "reconnect_count": 0,
                "uptime_seconds": round(uptime, 2),
                "capture_backend": self.capture_backend_requested,
                "fps": self.capture_fps,
                "capture_fps": self.capture_fps,
                "processing_fps": self.processing_fps,
                "captured_frames": self.captured_frames,
                "frames_captured": self.captured_frames,
                "processed_frames": self.processed_frames,
                "dropped_frames": self.dropped_frames,
                "active_tracks": self.active_tracks,
                "processing_latency_ms": self.processing_latency_ms,
                "started_at": self.started_at,
                "last_frame_at": self.last_frame_at,
                "error": self.error_message,
            }
