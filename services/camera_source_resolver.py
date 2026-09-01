import sys
import threading
from typing import Any

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
    def __init__(
        self,
        config_path: str = "configs/cameras.yaml",
        credential_manager: CredentialManager | None = None,
    ) -> None:
        self._logger = get_logger("camera_source_resolver")
        self._lock = threading.Lock()
        self._config_path = config_path
        self._credential_manager = credential_manager or CredentialManager()

        self._reserved_sources: dict[str, str] = {}
        self._retained_captures: dict[str, tuple[Any, Any]] = {}

        self._registered_cameras: list[dict[str, Any]] = []
        self._load_registered_cameras()

    def _load_registered_cameras(self) -> None:
        try:
            with open(self._config_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)

            if data and "cameras" in data:
                self._registered_cameras = list(data["cameras"].values())
            else:
                self._registered_cameras = []

        except (yaml.YAMLError, OSError, ValueError) as exc:
            self._logger.warning(f"Could not load camera config from {self._config_path}: {exc}")
            self._registered_cameras = []

    def is_source_reserved(self, source_key: str) -> bool:
        with self._lock:
            return source_key in self._reserved_sources

    def reserve_source(self, source_key: str, camera_id: str) -> bool:
        with self._lock:
            existing_camera_id = self._reserved_sources.get(source_key)

            if existing_camera_id is not None and existing_camera_id != camera_id:
                return False

            self._reserved_sources[source_key] = camera_id
            return True

    def pop_retained_capture(self, source_key: str) -> tuple[Any | None, Any | None]:
        with self._lock:
            return self._retained_captures.pop(source_key, (None, None))

    def release_source_by_camera_id(
        self,
        camera_id: str,
    ) -> str | None:
        with self._lock:
            for source_key, reserved_camera_id in list(self._reserved_sources.items()):
                if reserved_camera_id == camera_id:
                    del self._reserved_sources[source_key]
                    retained_cap, _ = self._retained_captures.pop(source_key, (None, None))
                    if retained_cap is not None:
                        try:
                            retained_cap.release()
                        except (cv2.error, OSError):
                            pass

                    self._logger.info(f"Released source reservation {source_key} for camera {camera_id}")

                    return source_key

            return None

    def release_source(self, source_key: str) -> None:
        with self._lock:
            if source_key in self._reserved_sources:
                del self._reserved_sources[source_key]
                retained_cap, _ = self._retained_captures.pop(source_key, (None, None))
                if retained_cap is not None:
                    try:
                        retained_cap.release()
                    except (cv2.error, OSError):
                        pass

                self._logger.info(f"Released source reservation {source_key}")

    def probe_usb_webcam(self, device_index: int, retain: bool = False) -> bool:
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
                        except (cv2.error, OSError):
                            pass
                    capture = cv2.VideoCapture(device_index)
            else:
                capture = cv2.VideoCapture(device_index)

            if not capture.isOpened():
                return False

            ret, frame = capture.read()

            is_valid = bool(ret and frame is not None and frame.size > 0)
            if is_valid and retain:
                with self._lock:
                    old_cap, _ = self._retained_captures.get(source_key, (None, None))
                    if old_cap is not None and old_cap != capture:
                        try:
                            old_cap.release()
                        except (cv2.error, OSError):
                            pass
                    self._retained_captures[source_key] = (capture, frame)

                capture = None

            return is_valid

        except (cv2.error, OSError, ValueError) as exc:
            self._logger.debug(f"Local camera probe failed for index {device_index}: {exc}")
            return False

        finally:
            if capture is not None:
                try:
                    capture.release()
                except (cv2.error, OSError) as exc:
                    self._logger.debug(f"Local camera probe release failed for index {device_index}: {exc}")
                capture = None

    def probe_stream(self, url: str, retain: bool = False) -> bool:
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

            is_valid = bool(ret and frame is not None and frame.size > 0)
            if is_valid and retain:
                with self._lock:
                    old_cap, _ = self._retained_captures.get(source_key, (None, None))
                    if old_cap is not None and old_cap != capture:
                        try:
                            old_cap.release()
                        except (cv2.error, OSError):
                            pass
                    self._retained_captures[source_key] = (capture, frame)
                capture = None

            return is_valid

        except (cv2.error, OSError, ValueError) as exc:
            self._logger.debug(f"Stream probe failed for {safe_url}: {exc}")
            return False

        finally:
            if capture is not None:
                try:
                    capture.release()
                except (cv2.error, OSError) as exc:
                    self._logger.debug(f"Stream probe release failed for {safe_url}: {exc}")

    def resolve_source(
        self,
        camera_id: str,
        requested_source: str = "auto",
        zone_id: str | None = None,
        max_usb_scan: int = 10,
        user_id: str = "default_user",
        credential_id: str | None = None,
    ) -> dict[str, Any]:
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
                    if existing_camera_id is not None and existing_camera_id != camera_id:
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
                        raise RuntimeError(f"Credential '{resolved_credential_id}' was not found in secure store")

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

            if not self.probe_usb_webcam(dev_idx, retain=True):
                continue

            retained_cap, initial_frame = self.pop_retained_capture(source_key)

            with self._lock:
                if source_key in self._reserved_sources:
                    if retained_cap is not None:
                        try:
                            retained_cap.release()
                        except (cv2.error, OSError):
                            pass
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
                "capture": retained_cap,
                "initial_frame": initial_frame,
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
            _clean_user, clean_pass, clean_url = extract_rtsp_credentials(raw_url)
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

            if not self.probe_stream(internal_url, retain=True):
                continue

            retained_cap, initial_frame = self.pop_retained_capture(source_key)

            with self._lock:
                if source_key in self._reserved_sources:
                    if retained_cap is not None:
                        try:
                            retained_cap.release()
                        except (cv2.error, OSError):
                            pass
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
                "capture": retained_cap,
                "initial_frame": initial_frame,
            }

        raise RuntimeError("Unable to detect camera source: No connected local webcam or reachable RTSP stream found.")
