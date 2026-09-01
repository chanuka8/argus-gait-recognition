import sys
import threading
import time
from datetime import datetime, timezone
from queue import Full, Queue
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from monitoring.logging_config import get_logger
from security_layer.credentials import sanitize_rtsp_url
from utils.display_renderer import DetectionDisplayRenderer, load_display_config

if TYPE_CHECKING:
    from services.recognition_worker import RecognitionWorker


def normalize_camera_source(source) -> int | str:
    """Normalize camera source to an integer device index or cleaned string."""
    if isinstance(source, int):
        return source

    normalized = str(source).strip()
    if normalized.isdigit():
        return int(normalized)

    lower = normalized.lower()
    if lower.startswith(("webcam:", "usb:")):
        parts = lower.split(":", 1)
        if len(parts) > 1 and parts[1].strip().isdigit():
            return int(parts[1].strip())
    if lower in ("webcam", "usb"):
        return 0

    return normalized


class CameraWorker:
    """Independent worker for a single camera stream with real-time JPEG preview buffer and recognition overlays."""

    def __init__(
        self,
        camera_id: str,
        camera_config: dict,
        inference_pipeline=None,
        detection_processor=None,
        recognition_worker: "RecognitionWorker | None" = None,
        inference_engine: Any | None = None,
        existing_capture: Any | None = None,
        initial_frame: np.ndarray | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.config = camera_config
        self.inference_pipeline = inference_pipeline
        self.detection_processor = detection_processor
        self.recognition_worker = recognition_worker
        self.inference_engine = inference_engine
        self._existing_capture = existing_capture
        self._initial_frame = initial_frame

        self._logger = get_logger(f"camera.{camera_id}")
        self._renderer = DetectionDisplayRenderer(load_display_config())

        raw_type = str(camera_config.get("type") or camera_config.get("source_type") or "webcam").lower()
        if raw_type in ("usb", "webcam", "local", "device"):
            self._source_type = "webcam"
        elif raw_type == "rtsp":
            self._source_type = "rtsp"
        elif raw_type == "http":
            self._source_type = "http"
        elif raw_type == "file":
            self._source_type = "file"
        else:
            self._source_type = "webcam"

        self._url = camera_config.get("url", "")
        self._device_index = int(normalize_camera_source(camera_config.get("device_index", 0)))
        self._width = max(16, int(camera_config.get("width", 640)))
        self._height = max(16, int(camera_config.get("height", 480)))
        self._target_fps = max(0, int(camera_config.get("target_fps", 15)))
        self._reconnect_interval = max(0, int(camera_config.get("reconnect_interval", 5)))
        self._max_reconnect = max(0, int(camera_config.get("max_reconnect_attempts", 0)))
        self._max_queue_size = max(1, int(camera_config.get("max_queue_size", 10)))
        self._jpeg_quality = max(1, min(100, int(camera_config.get("jpeg_quality", 75))))
        preview_max_fps = max(0.1, float(camera_config.get("preview_max_fps", 15)))
        self._min_jpeg_interval = 1.0 / preview_max_fps
        self._startup_timeout = max(0.1, float(camera_config.get("startup_timeout", 10.0)))
        self._startup_retry_interval = max(0.01, float(camera_config.get("startup_retry_interval", 0.3)))

        self._frame_queue = Queue(maxsize=self._max_queue_size)
        self._capture = None
        self._thread = None
        self._is_starting = False
        self._stop_event = threading.Event()
        self._frame_event = threading.Event()
        self._lock = threading.RLock()

        self._latest_jpeg: bytes | None = None
        self._last_frame_at: str | None = None
        self._last_jpeg_encode_time: float = 0.0
        self._active_tracks: int = 0
        self._active_clients: int = 0
        self._recognized_identities: list[str] = []

        self.stats = {
            "source_type": self._source_type,
            "frames_captured": 0,
            "frames_dropped": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "queue_size": 0,
            "connected": False,
            "reconnect_count": 0,
            "uptime_seconds": 0.0,
            "identities_recognized": 0,
            "active_tracks": 0,
            "recognition_active": False,
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
            file_path = self.config.get("file_path", "")
            if not file_path:
                raise ValueError(f"Camera {self.camera_id}: Video file path is empty")
            return file_path
        else:
            source_val = self.config.get("device_index")
            if source_val is None:
                source_val = self.config.get("source", self._device_index)
            return normalize_camera_source(source_val)

    def _wait_for_first_frame(self, safe_source: str) -> bool:
        """Wait for the first valid frame with bounded retry.

        RTSP streams typically require 1-5 seconds to negotiate and buffer the
        first frame after VideoCapture is opened. This method retries read()
        up to startup_timeout seconds, checking _stop_event between retries
        so that a pending stop request can abort startup immediately.

        Returns True if a valid frame was received and the JPEG preview buffer
        was populated, False otherwise.
        """
        self._logger.info(f"Waiting for first frame from {safe_source} (timeout={self._startup_timeout}s)")
        deadline = time.monotonic() + self._startup_timeout
        attempt = 0

        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                self._logger.info(f"Startup cancelled by stop request for {safe_source}")
                return False

            attempt += 1
            ret, frame = self._capture.read()

            if ret and frame is not None and frame.size > 0:
                success = False
                enc_buf = None
                try:
                    frame_resized = cv2.resize(frame, (self._width, self._height))
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
                    success, enc_buf = cv2.imencode(".jpg", frame_resized, encode_param)
                except (cv2.error, OSError, ValueError) as enc_err:
                    self._logger.warning(f"Initial JPEG encode error: {enc_err}")

                now = time.monotonic()
                iso_now = datetime.now(timezone.utc).isoformat()

                with self._lock:
                    self.stats["connected"] = True
                    self._frame_count = 1
                    self.stats["frames_captured"] = 1
                    if success and enc_buf is not None:
                        self._latest_jpeg = enc_buf.tobytes()
                        self._last_frame_at = iso_now
                        self._last_jpeg_encode_time = now

                self._frame_event.set()
                self._logger.info(
                    f"Camera {self.camera_id} first frame received after {attempt} attempt(s) "
                    f"({time.monotonic() - (deadline - self._startup_timeout):.1f}s)"
                )
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sleep_time = min(self._startup_retry_interval, remaining)
            self._stop_event.wait(sleep_time)

        self._logger.error(
            f"Startup timeout: no valid frame from {safe_source} after {self._startup_timeout}s ({attempt} attempts)"
        )
        return False

    def _open_capture(self) -> bool:
        try:
            source = self._resolve_source()
            safe_source = sanitize_rtsp_url(str(source))
            self._logger.info(f"Opening camera source: {safe_source}")


            if self._existing_capture is not None and getattr(self._existing_capture, "isOpened", lambda: False)():
                self._capture = self._existing_capture
                self._existing_capture = None
                self._logger.info(f"Adopted pre-verified camera capture for: {safe_source}")

                try:
                    self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    if self._target_fps > 0:
                        self._capture.set(cv2.CAP_PROP_FPS, self._target_fps)
                except (cv2.error, OSError):
                    pass


                if self._initial_frame is not None and getattr(self._initial_frame, "size", 0) > 0:
                    init_f = self._initial_frame
                    self._initial_frame = None
                    try:
                        frame_resized = cv2.resize(init_f, (self._width, self._height))
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
                        success, enc_buf = cv2.imencode(".jpg", frame_resized, encode_param)
                        now = time.monotonic()
                        iso_now = datetime.now(timezone.utc).isoformat()
                        with self._lock:
                            self.stats["connected"] = True
                            self._frame_count = 1
                            self.stats["frames_captured"] = 1
                            if success and enc_buf is not None:
                                self._latest_jpeg = enc_buf.tobytes()
                                self._last_frame_at = iso_now
                                self._last_jpeg_encode_time = now
                        self._frame_event.set()
                        self._logger.info(f"Camera {self.camera_id} connected instantly using pre-verified frame")
                        return True
                    except (cv2.error, OSError, ValueError) as enc_err:
                        self._logger.warning(f"Initial JPEG encode error from pre-verified frame: {enc_err}")

                if not self._wait_for_first_frame(safe_source):
                    if self._capture is not None:
                        try:
                            self._capture.release()
                        except (RuntimeError, ValueError, TypeError, AttributeError, cv2.error, OSError) as exc:
                            self._logger.warning(f"Error releasing unstarted camera capture: {exc}")
                    self._capture = None
                    return False

                self._logger.info(f"Camera {self.camera_id} connected and ready")
                return True

            if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
                dev_idx = int(source)
                if sys.platform == "win32":
                    self._capture = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
                    if not self._capture.isOpened():
                        self._logger.debug(
                            f"DirectShow open failed for index {dev_idx}, falling back to default backend"
                        )
                        if self._capture is not None:
                            try:
                                self._capture.release()
                            except (RuntimeError, ValueError, TypeError, AttributeError, cv2.error, OSError) as exc:
                                self._logger.warning(f"Error releasing DirectShow camera capture: {exc}")
                        self._capture = cv2.VideoCapture(dev_idx)
                else:
                    self._capture = cv2.VideoCapture(dev_idx)
            else:
                self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                self._logger.error(f"Failed to open camera source: {safe_source}")
                if self._capture is not None:
                    try:
                        self._capture.release()
                    except (RuntimeError, ValueError, TypeError, AttributeError, cv2.error, OSError) as exc:
                        self._logger.warning(f"Error releasing unopened camera capture: {exc}")
                self._capture = None
                return False

            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

            if self._target_fps > 0:
                self._capture.set(cv2.CAP_PROP_FPS, self._target_fps)

            if not self._wait_for_first_frame(safe_source):
                if self._capture is not None:
                    try:
                        self._capture.release()
                    except (RuntimeError, ValueError, TypeError, AttributeError, cv2.error, OSError) as exc:
                        self._logger.warning(f"Error releasing unstarted camera capture: {exc}")
                self._capture = None
                return False

            self._logger.info(f"Camera {self.camera_id} connected and ready")
            return True

        except (RuntimeError, ValueError, TypeError, cv2.error, OSError) as e:
            self._logger.error(f"Error opening capture: {sanitize_rtsp_url(str(e))}")
            if self._capture is not None:
                try:
                    self._capture.release()
                except (RuntimeError, ValueError, TypeError, AttributeError, cv2.error, OSError) as exc:
                    self._logger.warning(f"Error releasing camera capture on open failure: {exc}")
            self._capture = None
            return False

    def _close_capture(self) -> None:
        with self._lock:
            cap = self._capture
            self._capture = None
            self._existing_capture = None
            self._initial_frame = None
            self.stats["connected"] = False
            self._frame_event.set()
            if cap is not None:
                try:
                    cap.release()
                except (RuntimeError, ValueError, TypeError, AttributeError, cv2.error, OSError) as exc:
                    self._logger.warning(f"Error releasing camera capture: {exc}")

    def _render_status_frame(self, message: str = "OFFLINE") -> bytes:
        """Render a clean status placeholder frame when camera is disconnected or reconnecting."""
        try:
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            cv2.putText(
                frame, "ARGUS AI SURVEILLANCE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"CAMERA: {self.camera_id}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"STATUS: {message}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
            )
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            cv2.putText(frame, now_str, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            success, enc_buf = cv2.imencode(".jpg", frame, encode_param)
            return enc_buf.tobytes() if success and enc_buf is not None else b""
        except (cv2.error, OSError, ValueError):
            return b""

    def _render_preview_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Render live CCTV overlays: bounding boxes, track labels, identities, top status, and timestamp."""
        try:
            annotated = frame.copy()
            active_tracks = []
            rec_active = False
            confirmed_ids = []

            if self.recognition_worker is not None:
                rec_active = self.recognition_worker.is_alive()
                active_tracks = self.recognition_worker.cache.get_active_tracks(self.camera_id)
                for res in active_tracks:
                    self._renderer.draw(
                        frame=annotated,
                        box=res.bbox,
                        track_id=res.track_id if res.track_id >= 0 else None,
                        identity=res.identity,
                        score=res.similarity,
                        decision=res.decision,
                        camera_id=self.camera_id,
                        display_state=getattr(res, "display_state", None),
                        is_valid=getattr(res, "is_valid", True),
                        mobility_state=getattr(res, "mobility_state", "STANDARD_WALKING"),
                        gait_eligible=getattr(res, "gait_eligible", True),
                    )
                    if res.status == "CONFIRMED" and res.identity not in ("UNKNOWN", "UNKNOWN_PERSON", ""):
                        confirmed_ids.append(res.identity)
            elif self.inference_engine is not None and hasattr(self.inference_engine, "cache"):
                rec_active = getattr(self.inference_engine, "is_running", lambda: True)()
                active_tracks = self.inference_engine.cache.get_active_tracks(self.camera_id)
                for res in active_tracks:
                    self._renderer.draw(
                        frame=annotated,
                        box=res.bbox,
                        track_id=res.track_id if res.track_id >= 0 else None,
                        identity=res.identity,
                        score=res.similarity,
                        decision=res.decision,
                        camera_id=self.camera_id,
                        display_state=getattr(res, "display_state", None),
                        is_valid=getattr(res, "is_valid", True),
                        mobility_state=getattr(res, "mobility_state", "STANDARD_WALKING"),
                        gait_eligible=getattr(res, "gait_eligible", True),
                    )
                    if res.status == "CONFIRMED" and res.identity not in ("UNKNOWN", "UNKNOWN_PERSON", ""):
                        confirmed_ids.append(res.identity)

            with self._lock:
                self._active_tracks = len(active_tracks)
                self._recognized_identities = list(set(confirmed_ids))
                self.stats["active_tracks"] = self._active_tracks
                self.stats["identities_recognized"] = len(self._recognized_identities)
                self.stats["recognition_active"] = rec_active

            fps_val = self.stats.get("fps", 0.0)
            status_text = f"[{self.camera_id}] FPS: {fps_val:.1f} | LIVE | Tracks: {len(active_tracks)}"
            cv2.putText(annotated, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            cv2.putText(
                annotated,
                now_iso,
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            return annotated
        except (RuntimeError, ValueError, TypeError, cv2.error, OSError) as overlay_err:
            self._logger.debug(f"Overlay rendering error: {overlay_err}")
            return frame

    def register_client(self) -> None:
        """Increment active streaming client counter."""
        with self._lock:
            self._active_clients += 1

    def unregister_client(self) -> None:
        """Decrement active streaming client counter."""
        with self._lock:
            self._active_clients = max(0, self._active_clients - 1)

    def get_active_clients(self) -> int:
        """Get number of currently connected stream clients."""
        with self._lock:
            return self._active_clients

    def _capture_loop(self) -> None:
        """Main frame capture, recognition delegation, and preview encoding loop."""
        reconnect_attempts = 0
        frame_interval = 1.0 / self._target_fps if self._target_fps > 0 else 0.0

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            if self._capture is None or not self._capture.isOpened():
                if self._max_reconnect > 0 and reconnect_attempts >= self._max_reconnect:
                    self._logger.error(f"Max reconnect attempts ({self._max_reconnect}) reached. Stopping.")
                    break

                reconnect_attempts += 1
                self._reconnect_count += 1
                with self._lock:
                    self.stats["reconnect_count"] = self._reconnect_count
                    self._latest_jpeg = self._render_status_frame(f"RECONNECTING ({reconnect_attempts})")

                self._logger.info(f"Reconnecting camera {self.camera_id} (attempt {reconnect_attempts})")
                self._stop_event.wait(self._reconnect_interval)
                if self._stop_event.is_set():
                    break
                if self._open_capture():
                    reconnect_attempts = 0
                    self._logger.info(f"Camera {self.camera_id} reconnected successfully")
                continue

            try:
                ret, frame = self._capture.read()

                if not ret or frame is None:
                    self._logger.warning(f"Frame read failure on camera {self.camera_id}, disconnecting for reconnect")
                    self._close_capture()
                    with self._lock:
                        self._latest_jpeg = self._render_status_frame("SIGNAL LOST")
                    continue

                frame = cv2.resize(frame, (self._width, self._height))
                now = time.monotonic()
                iso_now = datetime.now(timezone.utc).isoformat()

                if self.recognition_worker is not None:
                    self.recognition_worker.put_frame(frame)

                if self.inference_engine is not None and hasattr(self.inference_engine, "put_frame"):
                    self.inference_engine.put_frame(
                        camera_id=self.camera_id,
                        frame=frame,
                        frame_id=self._frame_count,
                        source_type=self._source_type,
                    )

                if now - self._last_jpeg_encode_time >= self._min_jpeg_interval:
                    try:
                        preview_frame = self._render_preview_overlays(frame)
                        if preview_frame.shape[1] > 640:
                            h, w = preview_frame.shape[:2]
                            new_h = int(h * (640 / w))
                            preview_frame = cv2.resize(preview_frame, (640, new_h))

                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
                        success, enc_buf = cv2.imencode(".jpg", preview_frame, encode_param)
                        if success and enc_buf is not None:
                            jpeg_bytes = enc_buf.tobytes()
                            with self._lock:
                                self._latest_jpeg = jpeg_bytes
                                self._last_frame_at = iso_now
                            self._last_jpeg_encode_time = now
                            self._frame_event.set()
                    except (cv2.error, OSError, ValueError) as enc_err:
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

                if frame_interval > 0:
                    processing_time = time.monotonic() - loop_start
                    sleep_time = frame_interval - processing_time
                    if sleep_time > 0:
                        self._stop_event.wait(sleep_time)

            except (RuntimeError, ValueError, TypeError, cv2.error, OSError) as e:
                self._logger.error(f"Error in capture loop: {sanitize_rtsp_url(str(e))}")
                self._close_capture()
                with self._lock:
                    self._latest_jpeg = self._render_status_frame("CAPTURE ERROR")
                self._frame_event.set()

        self._close_capture()
        self._logger.info("Camera capture loop stopped")

    def get_latest_jpeg(self) -> bytes | None:
        """Return the latest encoded JPEG frame bytes safely."""
        with self._lock:
            return self._latest_jpeg

    def wait_for_frame(self, timeout: float = 0.5) -> bytes | None:
        """Wait for the next frame to be encoded and return latest JPEG bytes."""
        self._frame_event.wait(timeout)
        self._frame_event.clear()
        with self._lock:
            return self._latest_jpeg

    def get_stats(self) -> dict:
        """Get current camera statistics including recognition telemetry."""
        rec_stats = {}
        if self.recognition_worker is not None:
            rec_stats = self.recognition_worker.get_stats()

        with self._lock:
            return {
                **self.stats.copy(),
                "last_frame_at": self._last_frame_at,
                "active_tracks": self._active_tracks,
                "active_clients": self._active_clients,
                "recognized_identities": list(self._recognized_identities),
                **rec_stats,
            }

    def is_connected(self) -> bool:
        with self._lock:
            return self.stats["connected"]

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start camera capture thread and recognition worker after verifying readiness."""
        with self._lock:
            if self._is_starting or (self._thread is not None and self._thread.is_alive()):
                return False
            self._is_starting = True
            self._stop_event.clear()

        try:
            if not self._open_capture():
                with self._lock:
                    self._is_starting = False
                return False

            with self._lock:
                if self._stop_event.is_set() or (self._thread is not None and self._thread.is_alive()):
                    self._close_capture()
                    self._is_starting = False
                    return False

                self._thread = threading.Thread(
                    target=self._capture_loop,
                    name=f"camera-{self.camera_id}",
                    daemon=True,
                )
                self._thread.start()
                self._is_starting = False

            if self.recognition_worker is not None:
                self.recognition_worker.start()

            return True
        except (RuntimeError, ValueError, TypeError, OSError):
            with self._lock:
                self._is_starting = False
            self._close_capture()
            return False

    def stop(self, timeout: float = 3.0) -> bool:
        """Stop camera capture thread and recognition worker."""
        self._stop_event.set()
        thread_to_join = None
        with self._lock:
            self._is_starting = False
            if self._thread is not None and self._thread.is_alive():
                thread_to_join = self._thread

        if thread_to_join is not None:
            thread_to_join.join(timeout=timeout)

        if self.recognition_worker is not None:
            self.recognition_worker.stop(timeout=timeout)

        self._close_capture()
        with self._lock:
            self._latest_jpeg = self._render_status_frame("OFFLINE")
        return True

    def restart(self, timeout: float = 3.0) -> bool:
        """Restart camera capture thread and recognition worker cleanly."""
        self.stop(timeout=timeout)
        return self.start()
