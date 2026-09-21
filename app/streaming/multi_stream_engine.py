import threading
import time
from queue import Empty, Full, Queue
from typing import Any

import cv2

from app.security_layer.credentials import (
    CameraTransportSecurityError,
    is_secure_camera_transport_required,
    resolve_camera_config,
    sanitize_rtsp_url,
    validate_camera_transport,
)


class CameraStream:
    def __init__(
        self,
        camera_id: str,
        source: Any = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 5,
        queue_max_size: int = 10,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = max(1, target_fps)
        self.frame_interval = 1.0 / self.target_fps
        self.config = config or {}

        self.queue: Queue = Queue(maxsize=queue_max_size)
        self.cap = None
        self.running = False
        self.thread = None
        self.error = None
        self.frames_read = 0
        self.frames_dropped = 0

    def start(self) -> bool:
        strict = is_secure_camera_transport_required()
        transport_res = validate_camera_transport(
            self.source,
            config=self.config,
            strict_mode=strict,
            enforce=False,
        )
        if strict and not transport_res.get("allowed", False):
            reason = transport_res.get("reason", "Insecure camera transport rejected")
            self.error = f"Camera {self.camera_id}: {reason}"
            raise CameraTransportSecurityError(f"Camera '{self.camera_id}': {reason}")

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            safe_source = sanitize_rtsp_url(str(self.source))
            self.error = f"Camera {self.camera_id}: Failed to open source {safe_source}"
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"cam-{self.camera_id}",
        )
        self.thread.start()

        return True

    def _capture_loop(self) -> None:
        while self.running:
            try:
                ret, frame = self.cap.read()

                if not ret:
                    self.error = f"Camera {self.camera_id}: Failed to read frame"
                    self.running = False
                    break

                self.frames_read += 1

                try:
                    self.queue.put_nowait(frame)
                except Full:
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass

                    try:
                        self.queue.put_nowait(frame)
                    except Full:
                        pass

                    self.frames_dropped += 1

                time.sleep(self.frame_interval)

            except (RuntimeError, ValueError, OSError) as e:
                self.error = f"Camera {self.camera_id}: {e}"
                self.running = False
                break

    def read(self):
        try:
            return True, self.queue.get_nowait()
        except Empty:
            return False, None

    def is_opened(self) -> bool:
        return self.running and self.cap is not None and self.cap.isOpened()

    def stop(self) -> None:
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=3)

        if self.cap is not None:
            self.cap.release()

    def stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "source": sanitize_rtsp_url(str(self.source)),
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "is_running": self.running,
            "error": self.error,
        }


class MultiStreamEngine:
    def __init__(
        self,
        camera_configs: list[dict],
        queue_max_size: int = 10,
    ) -> None:
        self.streams: dict[str, CameraStream] = {}

        for raw_cfg in camera_configs:
            cam_cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
            camera_id = str(cam_cfg.get("id", "cam"))
            try:
                cam_cfg = resolve_camera_config(cam_cfg)
            except CameraTransportSecurityError:
                raise
            except (KeyError, ValueError, TypeError):
                pass

            # Determine source without unsafe fallback to 0 for network cameras
            source = cam_cfg.get("url")
            if source is None:
                cam_type = str(cam_cfg.get("type", "")).strip().lower()
                if cam_type in ("usb", "webcam", "local") or "device_index" in cam_cfg:
                    source = cam_cfg.get("device_index", 0)
                elif cam_type in ("file",) or "file_path" in cam_cfg:
                    source = cam_cfg.get("file_path", "")
                elif "source" in cam_cfg:
                    source = cam_cfg["source"]
                else:
                    if cam_type in ("rtsp", "rtsps", "http", "https") or "host" in cam_cfg:
                        raise CameraTransportSecurityError(
                            f"Camera '{camera_id}': Network camera has no valid URL and cannot fall back to device 0"
                        )
                    source = 0

            strict = is_secure_camera_transport_required()
            transport_res = validate_camera_transport(
                source,
                config=cam_cfg,
                strict_mode=strict,
                enforce=False,
            )
            if strict and not transport_res.get("allowed", False):
                raise CameraTransportSecurityError(
                    f"Camera '{camera_id}': {transport_res.get('reason')}"
                )

            self.streams[camera_id] = CameraStream(
                camera_id=camera_id,
                source=source,
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                target_fps=cam_cfg.get("target_fps", 5),
                queue_max_size=queue_max_size,
                config=cam_cfg,
            )

    def start_all(self) -> dict[str, bool]:
        results = {}

        for camera_id, stream in self.streams.items():
            success = stream.start()
            results[camera_id] = success

            if not success:
                print(f"[WARNING] {stream.error}")

        return results

    def read(self, camera_id: str):
        if camera_id not in self.streams:
            return False, None

        return self.streams[camera_id].read()

    def active_cameras(self) -> list[str]:
        return [cid for cid, stream in self.streams.items() if stream.is_opened()]

    def stop_all(self) -> None:
        for stream in self.streams.values():
            stream.stop()

    def stats(self) -> dict[str, dict]:
        return {cid: stream.stats() for cid, stream in self.streams.items()}
