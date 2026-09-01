import re
import socket
import time
from threading import Lock
from typing import Any

import cv2

from monitoring.logging_config import get_logger
from security_layer.credentials import resolve_camera_config


class DiscoveredCamera:
    def __init__(self, host: str, port: int = 554, vendor: str = "generic", onvif_xaddr: str = "") -> None:
        self.host = host
        self.port = port
        self.vendor = vendor
        self.onvif_xaddr = onvif_xaddr
        self.reachable: bool = False
        self.rtsp_valid: bool = False
        self.last_check: float = 0.0
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "vendor": self.vendor,
            "onvif_xaddr": self.onvif_xaddr,
            "reachable": self.reachable,
            "rtsp_valid": self.rtsp_valid,
            "last_check": self.last_check,
            "metadata": self.metadata,
        }


class CameraDiscoveryService:
    def __init__(
        self,
        discovery_config: dict[str, Any] | None = None,
    ) -> None:
        self._logger = get_logger("camera_discovery")
        self._lock = Lock()
        self._config = discovery_config or {}
        self._cameras: dict[str, DiscoveredCamera] = {}

    def discover_from_config(self, cameras_config: dict[str, dict[str, Any]]) -> list[DiscoveredCamera]:
        discovered: list[DiscoveredCamera] = []
        for cam_id, raw_cfg in cameras_config.items():
            cfg_dict = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
            cfg_dict.setdefault("id", cam_id)
            try:
                cam_cfg = resolve_camera_config(cfg_dict)
            except (ValueError, KeyError, TypeError, OSError):
                cam_cfg = cfg_dict

            url = cam_cfg.get("url", "")
            vendor = cam_cfg.get("vendor", "generic")
            host, port = self._parse_rtsp_url(url)
            if not host:
                host = cam_cfg.get("host", "")
            if not port:
                port = cam_cfg.get("port", 554)

            cam = DiscoveredCamera(host=host, port=port, vendor=vendor)
            cam.metadata = {"camera_id": cam_id, "config": cam_cfg}
            cam.last_check = time.monotonic()

            with self._lock:
                self._cameras[cam_id] = cam

            discovered.append(cam)

        self._logger.info(f"Discovered {len(discovered)} cameras from config")
        return discovered

    def parse_ws_discovery_response(self, xml_text: str) -> list[str]:
        xaddrs: list[str] = []
        try:
            matches = re.findall(r"<[\w:]*XAddrs[^>]*>([^<]+)</[\w:]*XAddrs>", xml_text)
            for m in matches:
                for addr in m.split():
                    if addr.startswith("http"):
                        xaddrs.append(addr.strip())
        except (re.error, ValueError, AttributeError) as e:
            self._logger.error(f"WS-Discovery parse error: {e}")
        return xaddrs

    def validate_rtsp_stream(self, rtsp_url: str, timeout_seconds: float = 5.0) -> bool:
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_seconds * 1000))
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret
        except (cv2.error, OSError):
            return False

    def check_host_reachable(self, host: str, port: int = 554, timeout: float = 2.0) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except (OSError, ValueError):
            return False

    def health_check_camera(self, camera_id: str, rtsp_url: str = "") -> dict[str, Any]:
        with self._lock:
            cam = self._cameras.get(camera_id)

        if cam is None:
            return {"camera_id": camera_id, "status": "unknown", "reachable": False, "rtsp_valid": False}

        cam.reachable = self.check_host_reachable(cam.host, cam.port)

        if rtsp_url:
            cam.rtsp_valid = self.validate_rtsp_stream(rtsp_url, timeout_seconds=3.0)

        cam.last_check = time.monotonic()

        return {
            "camera_id": camera_id,
            "status": "healthy" if cam.reachable else "unreachable",
            "reachable": cam.reachable,
            "rtsp_valid": cam.rtsp_valid,
            "host": cam.host,
            "vendor": cam.vendor,
        }

    def get_all_discovered(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {cid: cam.to_dict() for cid, cam in self._cameras.items()}

    @staticmethod
    def _parse_rtsp_url(url: str) -> tuple:
        match = re.search(r"rtsp://(?:[^@]+@)?([^:/]+)(?::(\d+))?", url)
        if match:
            host = match.group(1)
            port = int(match.group(2)) if match.group(2) else 554
            return host, port
        return "", 0

    @staticmethod
    def detect_vendor_from_url(url: str) -> str:
        url_lower = url.lower()
        if "/streaming/channels/" in url_lower:
            return "hikvision"
        elif "realmonitor" in url_lower:
            return "dahua"
        elif "/media/video" in url_lower:
            return "uniview"
        elif "axis-media" in url_lower:
            return "axis"
        return "generic"
