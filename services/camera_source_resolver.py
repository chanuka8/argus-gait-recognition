"""
Automatic Camera Source Resolver for ARGUS Surveillance System.

Dynamically resolves available camera hardware (USB webcams, RTSP streams,
and HTTP streams) for surveillance zones with thread-safe resource
reservation and credential-safe logging.
"""

import sys
import threading
from typing import Any, Dict, List, Optional

import cv2
import yaml

from monitoring.logging_config import get_logger
from security_layer.credentials import (
    CredentialManager,
    build_rtsp_url,
    extract_rtsp_credentials,
    sanitize_rtsp_url,
)
from services.camera_worker import normalize_camera_source


class CameraSourceResolver:
    """Thread-safe dynamic camera source resolver with secure credential resolution."""

    def __init__(
        self,
        config_path: str = "configs/cameras.yaml",
        credential_manager: Optional[CredentialManager] = None,
    ) -> None:
        self._logger = get_logger("camera_source_resolver")
        self._lock = threading.Lock()
        self._config_path = config_path
        self._credential_manager = credential_manager or CredentialManager()

        self._reserved_sources: Dict[str, str] = {}

        self._registered_cameras: List[Dict[str, Any]] = []
        self._load_registered_cameras()

    def _load_registered_cameras(self) -> None:
        """Load registered camera definitions from YAML configuration."""
        try:
            with open(self._config_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

            if data and "cameras" in data:
                self._registered_cameras = list(data["cameras"].values())
            else:
                self._registered_cameras = []

        except Exception as exc:
            self._logger.warning(
                f"Could not load camera config from "
                f"{self._config_path}: {exc}"
            )
            self._registered_cameras = []

    def is_source_reserved(self, source_key: str) -> bool:
        """Return True if the source is currently reserved."""
        with self._lock:
            return source_key in self._reserved_sources

    def reserve_source(self, source_key: str, camera_id: str) -> bool:
        """
        Reserve a source for a camera.

        Re-reserving a source for the same camera is allowed.
        A source reserved by another camera cannot be acquired.
        """
        with self._lock:
            existing_camera_id = self._reserved_sources.get(source_key)

            if (
                existing_camera_id is not None
                and existing_camera_id != camera_id
            ):
                return False

            self._reserved_sources[source_key] = camera_id
            return True

    def release_source_by_camera_id(
        self,
        camera_id: str,
    ) -> Optional[str]:
        """
        Release the source reservation belonging to a camera.

        Returns:
            The released source key, or None if no reservation exists.
        """
        with self._lock:
            for source_key, reserved_camera_id in list(
                self._reserved_sources.items()
            ):
                if reserved_camera_id == camera_id:
                    del self._reserved_sources[source_key]

                    self._logger.info(
                        f"Released source reservation {source_key} "
                        f"for camera {camera_id}"
                    )

                    return source_key

            return None

    def release_source(self, source_key: str) -> None:
        """Release a specific source reservation."""
        with self._lock:
            if source_key in self._reserved_sources:
                del self._reserved_sources[source_key]

                self._logger.info(
                    f"Released source reservation {source_key}"
                )

    def probe_usb_webcam(self, device_index: int) -> bool:
        """
        Probe whether a local camera (built-in, integrated, or USB) is connected and readable.

        The camera is opened temporarily, tested using one frame, and
        released safely.
        """
        source_key = f"usb:{device_index}"

        if self.is_source_reserved(source_key):
            return False

        capture = None

        try:
            if sys.platform == "win32":
                capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
                if not capture.isOpened():
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            pass
                    capture = cv2.VideoCapture(device_index)
            else:
                capture = cv2.VideoCapture(device_index)

            if not capture.isOpened():
                return False

            ret, frame = capture.read()

            return bool(
                ret
                and frame is not None
                and frame.size > 0
            )

        except Exception as exc:
            self._logger.debug(
                f"Local camera probe failed for index {device_index}: {exc}"
            )
            return False

        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    self._logger.debug(
                        f"Local camera probe release failed for index "
                        f"{device_index}: {exc}"
                    )
                capture = None

    def probe_stream(self, url: str) -> bool:
        """
        Probe whether an RTSP or HTTP stream is openable and readable.

        Raw stream URLs are never written to logs because they may contain
        usernames and passwords.
        """
        source_key = f"stream:{url}"

        if self.is_source_reserved(source_key):
            return False

        safe_url = sanitize_rtsp_url(str(url))
        capture = None

        try:
            capture = cv2.VideoCapture(url)

            if not capture.isOpened():
                return False

            ret, frame = capture.read()

            return bool(
                ret
                and frame is not None
                and frame.size > 0
            )

        except Exception as exc:
            self._logger.debug(
                f"Stream probe failed for {safe_url}: {exc}"
            )
            return False

        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    self._logger.debug(
                        f"Stream probe release failed for "
                        f"{safe_url}: {exc}"
                    )

    def resolve_source(
        self,
        camera_id: str,
        requested_source: str = "auto",
        zone_id: Optional[str] = None,
        max_usb_scan: int = 10,
        user_id: str = "default_user",
        credential_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve an available camera source automatically with deterministic hardware and stream probing.

        Priority:
        1. Explicitly requested source:
           - Numeric device index / 'webcam' / 'usb' -> Local Webcam ('webcam')
           - RTSP URL ('rtsp://...') or IP address -> RTSP Stream ('rtsp')
           - HTTP/HTTPS URL -> HTTP Stream ('http')
        2. 'auto' or empty source:
           - Priority 1: Free local Webcam device index (0..max_usb_scan-1)
           - Priority 2: Free registered RTSP/HTTP CCTV stream from configs/cameras.yaml
        3. If no usable source is found, raises RuntimeError with clear user message.

        Args:
            camera_id: Unique camera identifier.
            requested_source: Explicit source or "auto".
            zone_id: Optional surveillance zone identifier.
            max_usb_scan: Number of USB device indexes to probe.
            user_id: Authenticated application user ID.
            credential_id: Optional credential ID for authentication.

        Returns:
            Dictionary containing resolved camera source metadata.
        """
        resolved_credential_id = credential_id

        if requested_source and requested_source.strip().lower() != "auto":
            clean_req = str(requested_source).strip()
            normalized = normalize_camera_source(clean_req)

            if isinstance(normalized, int):
                source_key = f"usb:{normalized}"
                source_type = "webcam"
                resolved_source_type = "usb"
                label = f"USB Webcam {normalized}"

                with self._lock:
                    existing_camera_id = self._reserved_sources.get(source_key)
                    if (
                        existing_camera_id is not None
                        and existing_camera_id != camera_id
                    ):
                        raise RuntimeError(
                            f"Requested USB device {normalized} is already in use by active worker {existing_camera_id}"
                        )

                internal_source = str(normalized)
                safe_presentation_source = str(normalized)

            else:
                normalized_str = str(normalized).strip()
                if normalized_str.lower().startswith("rtsp://"):
                    raw_url = normalized_str
                    source_type = "rtsp"
                elif normalized_str.lower().startswith("http://") or normalized_str.lower().startswith("https://"):
                    raw_url = normalized_str
                    source_type = "http"
                else:
                    source_type = "rtsp"
                    if ":" in normalized_str or "/" in normalized_str:
                        raw_url = f"rtsp://{normalized_str}"
                    else:
                        raw_url = f"rtsp://{normalized_str}:554/live"

                extracted_user, extracted_pass, clean_url = extract_rtsp_credentials(raw_url)

                if extracted_pass:
                    if not resolved_credential_id:
                        meta = self._credential_manager.store_credential(
                            owner_user_id=user_id or "default_user",
                            username=extracted_user or "",
                            password=extracted_pass,
                            description=f"Auto-extracted for {camera_id}",
                        )
                        resolved_credential_id = meta["credential_id"]

                    internal_source = build_rtsp_url(clean_url, extracted_user, extracted_pass)
                    safe_presentation_source = clean_url
                    source_key = f"stream:{clean_url}"
                    label = f"{source_type.upper()} Stream {sanitize_rtsp_url(raw_url)}"

                elif resolved_credential_id:
                    if not self._credential_manager.can_access(resolved_credential_id, user_id=user_id):
                        raise RuntimeError(
                            f"User '{user_id}' is not authorized to access credential '{resolved_credential_id}'"
                        )

                    cred_data = self._credential_manager.get_credential(resolved_credential_id, user_id=user_id)
                    if not cred_data:
                        raise RuntimeError(
                            f"Credential '{resolved_credential_id}' was not found in secure store"
                        )

                    internal_source = build_rtsp_url(
                        clean_url,
                        cred_data.get("username"),
                        cred_data.get("password"),
                    )
                    safe_presentation_source = clean_url
                    source_key = f"stream:{clean_url}"
                    label = f"{source_type.upper()} Stream {sanitize_rtsp_url(clean_url)} [credential={resolved_credential_id}]"

                else:
                    internal_source = raw_url
                    safe_presentation_source = clean_url
                    source_key = f"stream:{clean_url}"
                    label = f"{source_type.upper()} Stream {sanitize_rtsp_url(clean_url)}"

                resolved_source_type = source_type
                with self._lock:
                    existing_camera_id = self._reserved_sources.get(source_key)
                    if existing_camera_id is not None and existing_camera_id != camera_id:
                        raise RuntimeError(
                            f"Requested source {label} is already in use by active worker {existing_camera_id}"
                        )

            with self._lock:
                self._reserved_sources[source_key] = camera_id

            return {
                "source": safe_presentation_source,
                "source_type": source_type,
                "resolved_source": internal_source,
                "resolved_source_type": resolved_source_type,
                "resolved_source_label": label,
                "source_key": source_key,
                "credential_id": resolved_credential_id,
                "credential_configured": bool(resolved_credential_id),
            }

        for dev_idx in range(max_usb_scan):
            source_key = f"usb:{dev_idx}"

            if self.is_source_reserved(source_key):
                continue

            if not self.probe_usb_webcam(dev_idx):
                continue

            with self._lock:
                if source_key in self._reserved_sources:
                    continue
                self._reserved_sources[source_key] = camera_id

            self._logger.info(f"Auto-detected Local Webcam {dev_idx} for camera {camera_id}")

            return {
                "source": str(dev_idx),
                "source_type": "webcam",
                "resolved_source": str(dev_idx),
                "resolved_source_type": "usb",
                "resolved_source_label": f"USB Webcam {dev_idx}",
                "source_key": source_key,
                "credential_id": None,
                "credential_configured": False,
            }

        for cam_cfg in self._registered_cameras:
            if not cam_cfg.get("enabled", True):
                continue

            url = cam_cfg.get("url")
            if not url:
                url = (
                    f"rtsp://"
                    f"{cam_cfg.get('host', '127.0.0.1')}:"
                    f"{cam_cfg.get('port', 554)}"
                    f"{cam_cfg.get('path', '/stream1')}"
                )

            raw_url = str(url)
            clean_user, clean_pass, clean_url = extract_rtsp_credentials(raw_url)
            cam_cred_id = cam_cfg.get("credential_id") or resolved_credential_id

            if cam_cred_id and self._credential_manager.can_access(cam_cred_id, user_id=user_id):
                cred = self._credential_manager.get_credential(cam_cred_id, user_id=user_id)
                if cred:
                    internal_url = build_rtsp_url(clean_url, cred.get("username"), cred.get("password"))
                else:
                    internal_url = raw_url
            elif clean_pass:
                internal_url = raw_url
            else:
                internal_url = raw_url

            safe_url = sanitize_rtsp_url(clean_url)
            source_key = f"stream:{clean_url}"

            if self.is_source_reserved(source_key):
                continue

            if not self.probe_stream(internal_url):
                continue

            with self._lock:
                if source_key in self._reserved_sources:
                    continue
                self._reserved_sources[source_key] = camera_id

            cam_name = str(cam_cfg.get("name", "RTSP Camera"))
            source_type = "rtsp" if raw_url.lower().startswith("rtsp://") else "http"

            self._logger.info(f"Auto-detected RTSP stream for {cam_name} ({safe_url}) for camera {camera_id}")

            return {
                "source": clean_url,
                "source_type": source_type,
                "resolved_source": internal_url,
                "resolved_source_type": source_type,
                "resolved_source_label": f"{source_type.upper()} Stream {cam_name} ({safe_url})",
                "source_key": source_key,
                "credential_id": cam_cred_id,
                "credential_configured": bool(cam_cred_id or clean_pass),
            }

        raise RuntimeError("Unable to detect camera source: No connected local webcam or reachable RTSP stream found.")
