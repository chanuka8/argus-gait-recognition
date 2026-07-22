"""Automatic camera discovery and health validation service."""

import re
import socket
import time
from threading import Lock
from typing import Any, Dict, List, Optional

import cv2

from monitoring.logging_config import get_logger



class DiscoveredCamera:
    """Metadata container for a discovered camera."""

    def __init__(self, host: str, port: int = 554, vendor: str = "generic", onvif_xaddr: str = "") -> None:
        self.host = host
        self.port = port
        self.vendor = vendor
        self.onvif_xaddr = onvif_xaddr
        self.reachable: bool = False
        self.rtsp_valid: bool = False
        self.last_check: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
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
    """Configuration-driven camera discovery and RTSP health validation."""

    def __init__(
        self,
        discovery_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._logger = get_logger("camera_discovery")
        self._lock = Lock()
        self._config = discovery_config or {}
        self._cameras: Dict[str, DiscoveredCamera] = {}

    def discover_from_config(self, cameras_config: Dict[str, Dict[str, Any]]) -> List[DiscoveredCamera]:
        """Discover cameras from a cameras.yaml-style config dict."""
        discovered: List[DiscoveredCamera] = []
        for cam_id, cam_cfg in cameras_config.items():
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

    def parse_ws_discovery_response(self, xml_text: str) -> List[str]:
        """Parse WS-Discovery ProbeMatch responses to extract XAddrs."""
        xaddrs: List[str] = []
        try:
            matches = re.findall(r"<[\w:]*XAddrs[^>]*>([^<]+)</[\w:]*XAddrs>", xml_text)
            for m in matches:
                for addr in m.split():
                    if addr.startswith("http"):
                        xaddrs.append(addr.strip())
        except Exception as e:
            self._logger.error(f"WS-Discovery parse error: {e}")
        return xaddrs

    def validate_rtsp_stream(self, rtsp_url: str, timeout_seconds: float = 5.0) -> bool:
        """Validate that an RTSP URL is reachable and returns frames."""
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_seconds * 1000))
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret
        except Exception:
            return False

    def check_host_reachable(self, host: str, port: int = 554, timeout: float = 2.0) -> bool:
        """TCP connect check to verify camera host is reachable."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def health_check_camera(self, camera_id: str, rtsp_url: str = "") -> Dict[str, Any]:
        """Run full health check on a discovered camera."""
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

    def get_all_discovered(self) -> Dict[str, Dict[str, Any]]:
        """Return all discovered cameras."""
        with self._lock:
            return {cid: cam.to_dict() for cid, cam in self._cameras.items()}

    @staticmethod
    def _parse_rtsp_url(url: str) -> tuple:
        """Extract host and port from an RTSP URL."""
        match = re.search(r"rtsp://(?:[^@]+@)?([^:/]+)(?::(\d+))?", url)
        if match:
            host = match.group(1)
            port = int(match.group(2)) if match.group(2) else 554
            return host, port
        return "", 0

    @staticmethod
    def detect_vendor_from_url(url: str) -> str:
        """Heuristic vendor detection from RTSP URL path."""
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
