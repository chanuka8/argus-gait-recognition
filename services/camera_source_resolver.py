"""
Automatic Camera Source Resolver for ARGUS Surveillance System.
Dynamically resolves available camera hardware (USB webcams, RTSP streams, HTTP streams)
for surveillance zones with thread-safe resource reservation.
"""

import threading
import time
from typing import Any, Dict, List, Optional
import cv2
import yaml

from monitoring.logging_config import get_logger
from services.camera_worker import normalize_camera_source


class CameraSourceResolver:
    """Thread-safe camera source resolution and reservation manager."""

    def __init__(self, config_path: str = "configs/cameras.yaml") -> None:
        self._logger = get_logger("camera_source_resolver")
        self._lock = threading.Lock()
        self._config_path = config_path
        self._reserved_sources: Dict[str, str] = {}  # source_key -> camera_id
        self._registered_cameras: List[Dict[str, Any]] = []
        self._load_registered_cameras()

    def _load_registered_cameras(self) -> None:
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "cameras" in data:
                    self._registered_cameras = list(data["cameras"].values())
        except Exception as e:
            self._logger.warning(f"Could not load camera config from {self._config_path}: {e}")
            self._registered_cameras = []

    def is_source_reserved(self, source_key: str) -> bool:
        with self._lock:
            return source_key in self._reserved_sources

    def reserve_source(self, source_key: str, camera_id: str) -> bool:
        with self._lock:
            if source_key in self._reserved_sources and self._reserved_sources[source_key] != camera_id:
                return False
            self._reserved_sources[source_key] = camera_id
            return True

    def release_source_by_camera_id(self, camera_id: str) -> Optional[str]:
        with self._lock:
            for source_key, cam_id in list(self._reserved_sources.items()):
                if cam_id == camera_id:
                    del self._reserved_sources[source_key]
                    self._logger.info(f"Released source reservation {source_key} for camera {camera_id}")
                    return source_key
            return None

    def release_source(self, source_key: str) -> None:
        with self._lock:
            if source_key in self._reserved_sources:
                del self._reserved_sources[source_key]
                self._logger.info(f"Released source reservation {source_key}")

    def probe_usb_webcam(self, device_index: int) -> bool:
        """Probe if a local USB webcam device index is physically connected and readable."""
        source_key = f"usb:{device_index}"
        if self.is_source_reserved(source_key):
            return False

        try:
            cap = cv2.VideoCapture(device_index)
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return bool(ret and frame is not None and frame.size > 0)
        except Exception as e:
            self._logger.debug(f"USB probe failed for index {device_index}: {e}")
            return False

    def probe_stream(self, url: str) -> bool:
        """Probe if an RTSP or HTTP stream URL is openable."""
        source_key = f"stream:{url}"
        if self.is_source_reserved(source_key):
            return False

        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            cap.release()
            return bool(ret and frame is not None and frame.size > 0)
        except Exception as e:
            self._logger.debug(f"Stream probe failed for {url}: {e}")
            return False

    def resolve_source(
        self,
        camera_id: str,
        requested_source: str = "auto",
        zone_id: Optional[str] = None,
        max_usb_scan: int = 10,
    ) -> Dict[str, Any]:
        """
        Resolves an available camera source according to priority:
        1. Free local USB webcam (probing 0..max_usb_scan-1).
        2. Free registered RTSP/HTTP sources in config.
        3. Explicit source (if requested_source != "auto").
        """
        # If explicit source is requested (not "auto")
        if requested_source and requested_source.strip().lower() != "auto":
            normalized = normalize_camera_source(requested_source)
            if isinstance(normalized, int):
                source_key = f"usb:{normalized}"
                source_type = "usb"
                label = f"USB Webcam {normalized}"
            else:
                source_key = f"stream:{normalized}"
                source_type = "rtsp" if str(normalized).startswith("rtsp://") else "http"
                label = f"{source_type.upper()} Stream {normalized}"

            with self._lock:
                if source_key in self._reserved_sources and self._reserved_sources[source_key] != camera_id:
                    raise RuntimeError(f"Requested source {label} is already in use by active worker {self._reserved_sources[source_key]}")
                self._reserved_sources[source_key] = camera_id

            return {
                "source": str(normalized),
                "resolved_source": str(normalized),
                "resolved_source_type": source_type,
                "resolved_source_label": label,
                "source_key": source_key,
            }

        # Priority 1: Free Local USB Webcam
        for dev_idx in range(max_usb_scan):
            source_key = f"usb:{dev_idx}"
            if self.is_source_reserved(source_key):
                continue

            if self.probe_usb_webcam(dev_idx):
                with self._lock:
                    if source_key in self._reserved_sources:
                        continue
                    self._reserved_sources[source_key] = camera_id

                self._logger.info(f"Auto-resolved USB Webcam {dev_idx} for camera {camera_id}")
                return {
                    "source": str(dev_idx),
                    "resolved_source": str(dev_idx),
                    "resolved_source_type": "usb",
                    "resolved_source_label": f"USB Webcam {dev_idx}",
                    "source_key": source_key,
                }

        # Priority 2: Registered RTSP/HTTP CCTV Streams in configs/cameras.yaml
        for cam_cfg in self._registered_cameras:
            if not cam_cfg.get("enabled", True):
                continue

            url = cam_cfg.get("url") or f"rtsp://{cam_cfg.get('host', '127.0.0.1')}:{cam_cfg.get('port', 554)}{cam_cfg.get('path', '/stream1')}"
            source_key = f"stream:{url}"
            if self.is_source_reserved(source_key):
                continue

            if self.probe_stream(url):
                with self._lock:
                    if source_key in self._reserved_sources:
                        continue
                    self._reserved_sources[source_key] = camera_id

                cam_name = cam_cfg.get("name", "RTSP Camera")
                self._logger.info(f"Auto-resolved {cam_name} ({url}) for camera {camera_id}")
                return {
                    "source": url,
                    "resolved_source": url,
                    "resolved_source_type": "rtsp",
                    "resolved_source_label": f"RTSP CCTV — {cam_name}",
                    "source_key": source_key,
                }

        # If no physical source is reachable
        raise RuntimeError("No free connected or registered camera source is available.")
