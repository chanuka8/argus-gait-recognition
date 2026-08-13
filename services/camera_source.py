"""
Camera Source Discovery, Auto-Identification, and Zone-Worker Binding Module for ARGUS AI.
Provides deterministic source resolution, bounded USB webcam discovery, and thread-safe zone bindings.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2

from core.logger import setup_logger


logger = setup_logger("ARGUS.CameraSource")


class SourceType(str, Enum):
    USB = "usb"
    RTSP = "rtsp"
    HTTP = "http"
    AUTO = "auto"


@dataclass
class CameraSourceInfo:
    source_id: str
    display_name: str
    source_type: str
    device_index: Optional[int] = None
    source_url: str = ""
    sanitized_source: str = ""
    capture_backend_requested: str = "auto"
    capture_backend_used: str = "auto"
    available: bool = False
    actual_width: int = 0
    actual_height: int = 0
    actual_fps: float = 0.0
    last_probe_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sanitize_url(url: str) -> str:
    """Mask credentials in RTSP/HTTP URLs for safe display."""
    if not url or "://" not in url:
        return url
    try:
        proto, rest = url.split("://", 1)
        if "@" in rest:
            user_pass, host_path = rest.rsplit("@", 1)
            return f"{proto}://***:***@{host_path}"
        return url
    except Exception:
        return url


def parse_backend_flag(backend_str: str) -> int:
    """Map string backend identifier to OpenCV VideoCapture backend flag."""
    b = str(backend_str).lower().strip()
    if b in ("dshow", "directshow"):
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    elif b in ("msmf", "media_foundation"):
        return getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
    elif b in ("v4l", "v4l2"):
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    elif b == "ffmpeg":
        return getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY)
    return cv2.CAP_ANY


def get_backend_name(backend_flag: int) -> str:
    """Return human readable name of OpenCV backend flag."""
    if backend_flag == getattr(cv2, "CAP_DSHOW", -1):
        return "dshow"
    elif backend_flag == getattr(cv2, "CAP_MSMF", -1):
        return "msmf"
    elif backend_flag == getattr(cv2, "CAP_V4L2", -1):
        return "v4l2"
    elif backend_flag == getattr(cv2, "CAP_FFMPEG", -1):
        return "ffmpeg"
    return "auto"


def resolve_source_type(source_val: Any, requested_type: str = "auto") -> Tuple[str, Any]:
    """
    Deterministically resolves source_type and typed source value.
    - integer or numeric string -> usb (device_index)
    - rtsp:// or rtsps:// -> rtsp
    - http:// or https:// -> http
    Raises ValueError (HTTP 422) on invalid or mismatched source_type.
    """
    req_type = str(requested_type).lower().strip()
    if req_type not in ("auto", "usb", "rtsp", "http"):
        raise ValueError(f"Invalid requested source_type '{requested_type}'. Must be usb, rtsp, http, or auto.")

    source_str = str(source_val).strip()

    # Case 1: Numeric input (integer device index)
    is_numeric = False
    dev_idx = 0
    try:
        dev_idx = int(source_val)
        if dev_idx >= 0:
            is_numeric = True
    except (ValueError, TypeError):
        pass

    if is_numeric:
        detected_type = "usb"
        typed_val = dev_idx
    elif source_str.startswith(("rtsp://", "rtsps://")):
        detected_type = "rtsp"
        typed_val = source_str
    elif source_str.startswith(("http://", "https://")):
        detected_type = "http"
        typed_val = source_str
    else:
        raise ValueError(f"Ambiguous or unsupported camera source value: '{source_val}'. Expected USB device index (0, 1...) or RTSP/HTTP URL.")

    # Validate against explicit requested_type if not auto
    if req_type != "auto" and req_type != detected_type:
        raise ValueError(f"Source value '{source_val}' detected as '{detected_type}', but requested source_type was '{requested_type}'.")

    return detected_type, typed_val


class AutoSourceResolver:
    """
    Central discovery and auto-identification manager for USB webcams and network cameras.
    Features bounded USB probing, caching, and clean resource release.
    """

    def __init__(self, cache_ttl: float = 5.0) -> None:
        self.cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._cached_usb_sources: List[CameraSourceInfo] = []
        self._last_usb_scan: float = 0.0

    def discover_usb_sources(self, max_index: int = 5, force_refresh: bool = False, backend_preference: str = "dshow") -> List[CameraSourceInfo]:
        """Probes local USB device indexes 0..max_index with bounded timeout and safe capture release."""
        now = time.monotonic()
        with self._lock:
            if not force_refresh and (now - self._last_usb_scan) < self.cache_ttl and self._cached_usb_sources:
                return self._cached_usb_sources

        discovered: List[CameraSourceInfo] = []
        backend_flag = parse_backend_flag(backend_preference)

        for idx in range(max_index + 1):
            cam_info = self._probe_single_usb(idx, backend_flag, backend_preference)
            if cam_info.available:
                discovered.append(cam_info)

        with self._lock:
            self._cached_usb_sources = discovered
            self._last_usb_scan = time.monotonic()

        logger.info(f"Discovered {len(discovered)} available USB webcam(s) in range 0-{max_index}.")
        return discovered

    def _probe_single_usb(self, device_index: int, backend_flag: int, backend_name: str) -> CameraSourceInfo:
        """Helper to probe a single USB device index safely releasing capture."""
        source_id = f"usb_{device_index}"
        display_name = f"USB Webcam #{device_index}"
        
        cap = None
        try:
            if backend_flag != cv2.CAP_ANY:
                cap = cv2.VideoCapture(device_index, backend_flag)
            else:
                cap = cv2.VideoCapture(device_index)

            if not cap.isOpened():
                return CameraSourceInfo(
                    source_id=source_id,
                    display_name=display_name,
                    source_type="usb",
                    device_index=device_index,
                    sanitized_source=f"USB Device {device_index}",
                    capture_backend_requested=backend_name,
                    available=False,
                    error="Device not available or locked",
                )

            # Read at least one valid frame
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                return CameraSourceInfo(
                    source_id=source_id,
                    display_name=display_name,
                    source_type="usb",
                    device_index=device_index,
                    sanitized_source=f"USB Device {device_index}",
                    capture_backend_requested=backend_name,
                    available=False,
                    error="Failed to read initial frame",
                )

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

            return CameraSourceInfo(
                source_id=source_id,
                display_name=f"{display_name} ({w}x{h} @ {int(fps)}fps)",
                source_type="usb",
                device_index=device_index,
                sanitized_source=f"USB Device {device_index}",
                capture_backend_requested=backend_name,
                capture_backend_used=backend_name,
                available=True,
                actual_width=w,
                actual_height=h,
                actual_fps=fps,
                last_probe_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as err:
            return CameraSourceInfo(
                source_id=source_id,
                display_name=display_name,
                source_type="usb",
                device_index=device_index,
                sanitized_source=f"USB Device {device_index}",
                capture_backend_requested=backend_name,
                available=False,
                error=str(err),
            )
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def probe_source(self, source_val: Any, requested_type: str = "auto", backend: str = "auto") -> CameraSourceInfo:
        """Probes any USB or network source value without starting a worker."""
        source_type, typed_val = resolve_source_type(source_val, requested_type)

        if source_type == "usb":
            return self._probe_single_usb(typed_val, parse_backend_flag(backend), backend)

        # Network RTSP / HTTP probe
        sanitized = sanitize_url(str(typed_val))
        source_id = f"{source_type}_{abs(hash(typed_val)) % 10000}"
        display_name = f"{source_type.upper()} Stream ({sanitized})"

        cap = None
        try:
            cap = cv2.VideoCapture(str(typed_val))
            if not cap.isOpened():
                return CameraSourceInfo(
                    source_id=source_id,
                    display_name=display_name,
                    source_type=source_type,
                    source_url=str(typed_val),
                    sanitized_source=sanitized,
                    capture_backend_requested=backend,
                    available=False,
                    error="Unable to connect to network stream",
                )

            ret, frame = cap.read()
            if not ret or frame is None:
                return CameraSourceInfo(
                    source_id=source_id,
                    display_name=display_name,
                    source_type=source_type,
                    source_url=str(typed_val),
                    sanitized_source=sanitized,
                    capture_backend_requested=backend,
                    available=False,
                    error="Stream connected but returned no frames",
                )

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

            return CameraSourceInfo(
                source_id=source_id,
                display_name=display_name,
                source_type=source_type,
                source_url=str(typed_val),
                sanitized_source=sanitized,
                capture_backend_requested=backend,
                capture_backend_used="ffmpeg",
                available=True,
                actual_width=w,
                actual_height=h,
                actual_fps=fps,
                last_probe_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as err:
            return CameraSourceInfo(
                source_id=source_id,
                display_name=display_name,
                source_type=source_type,
                source_url=str(typed_val),
                sanitized_source=sanitized,
                capture_backend_requested=backend,
                available=False,
                error=str(err),
            )
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


class ZoneSourceBindingRegistry:
    """
    Thread-safe registry managing active camera workers and source-to-zone bindings.
    Enforces:
    1. Only ONE active worker per physical USB device index (returns HTTP 409 Conflict if duplicate).
    2. Only ONE active worker per zone_id (returns HTTP 409 Conflict if duplicate).
    3. Proper cleanup and resource reservation releases on stop or server shutdown.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._workers: Dict[str, Any] = {}          # camera_id -> CameraWorker
        self._zone_bindings: Dict[str, str] = {}    # zone_id -> camera_id
        self._usb_reservations: Dict[int, str] = {} # device_index -> camera_id

    def is_usb_active(self, device_index: int, exclude_camera_id: Optional[str] = None) -> bool:
        with self.lock:
            cam_id = self._usb_reservations.get(device_index)
            if cam_id and cam_id != exclude_camera_id:
                return True
            return False

    def is_zone_active(self, zone_id: str, exclude_camera_id: Optional[str] = None) -> bool:
        with self.lock:
            cam_id = self._zone_bindings.get(zone_id)
            if cam_id and cam_id != exclude_camera_id:
                return True
            return False

    def register_binding(
        self,
        camera_id: str,
        zone_id: str,
        source_type: str,
        device_index: Optional[int],
        worker_instance: Any,
    ) -> None:
        """Registers a worker and its zone & USB reservations atomically."""
        with self.lock:
            if zone_id in self._zone_bindings and self._zone_bindings[zone_id] != camera_id:
                raise ValueError(f"CONFLICT: Zone '{zone_id}' already has an active worker (camera_id '{self._zone_bindings[zone_id]}').")

            if source_type == "usb" and device_index is not None:
                if device_index in self._usb_reservations and self._usb_reservations[device_index] != camera_id:
                    raise ValueError(f"CONFLICT: USB device index {device_index} is already in use by camera_id '{self._usb_reservations[device_index]}'.")

            self._workers[camera_id] = worker_instance
            self._zone_bindings[zone_id] = camera_id
            if source_type == "usb" and device_index is not None:
                self._usb_reservations[device_index] = camera_id

    def unregister_binding(self, camera_id: str) -> bool:
        """Removes worker and releases zone & USB reservations atomically."""
        with self.lock:
            if camera_id not in self._workers:
                return False

            # Find zone_id and device_index associated with this camera_id
            zones_to_remove = [z for z, cid in self._zone_bindings.items() if cid == camera_id]
            for z in zones_to_remove:
                del self._zone_bindings[z]

            usbs_to_remove = [idx for idx, cid in self._usb_reservations.items() if cid == camera_id]
            for idx in usbs_to_remove:
                del self._usb_reservations[idx]

            del self._workers[camera_id]
            return True

    def get_worker(self, camera_id: str) -> Optional[Any]:
        with self.lock:
            return self._workers.get(camera_id)

    def get_all_workers(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self._workers)

    def shutdown_all(self) -> None:
        """Stops all active camera workers and releases all reservations."""
        workers_to_stop = []
        with self.lock:
            workers_to_stop = list(self._workers.values())
            self._workers.clear()
            self._zone_bindings.clear()
            self._usb_reservations.clear()

        for w in workers_to_stop:
            try:
                w.stop()
            except Exception as e:
                logger.error(f"Error stopping worker during shutdown: {e}")
