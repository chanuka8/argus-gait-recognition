"""Vendor compatibility adapters for enterprise CCTV cameras."""

from typing import Any, Dict

from monitoring.logging_config import get_logger


class CameraAdapter:
    """Base adapter for vendor-specific camera integration."""

    VENDOR = "generic"

    def __init__(self, host: str, port: int = 554, username: str = "", password: str = "", **kwargs: Any) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._logger = get_logger(f"adapter.{self.VENDOR}")

    def get_rtsp_url(self, channel: int = 1, subtype: int = 0) -> str:
        """Return primary RTSP stream URL."""
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"rtsp://{auth}{self.host}:{self.port}"

    def get_camera_config(self, camera_id: str, width: int = 640, height: int = 480, target_fps: int = 15) -> Dict[str, Any]:
        """Build camera config dict compatible with CameraWorker."""
        return {
            "id": camera_id,
            "type": "rtsp",
            "url": self.get_rtsp_url(),
            "width": width,
            "height": height,
            "target_fps": target_fps,
            "vendor": self.VENDOR,
            "enabled": True,
            "reconnect_interval": 5,
            "max_reconnect_attempts": 3,
            "max_queue_size": 10,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Return vendor metadata."""
        return {"vendor": self.VENDOR, "host": self.host, "port": self.port}


class HikvisionAdapter(CameraAdapter):
    """Hikvision RTSP adapter."""

    VENDOR = "hikvision"

    def get_rtsp_url(self, channel: int = 1, subtype: int = 0) -> str:
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"rtsp://{auth}{self.host}:{self.port}/Streaming/Channels/{channel}0{subtype + 1}"

    def get_metadata(self) -> Dict[str, Any]:
        base = super().get_metadata()
        base["onvif_port"] = 80
        base["sdk"] = "ISAPI"
        return base


class DahuaAdapter(CameraAdapter):
    """Dahua RTSP adapter."""

    VENDOR = "dahua"

    def get_rtsp_url(self, channel: int = 1, subtype: int = 0) -> str:
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"rtsp://{auth}{self.host}:{self.port}/cam/realmonitor?channel={channel}&subtype={subtype}"


class UniviewAdapter(CameraAdapter):
    """Uniview RTSP adapter."""

    VENDOR = "uniview"

    def get_rtsp_url(self, channel: int = 1, subtype: int = 0) -> str:
        auth = f"{self.username}:{self.password}@" if self.username else ""
        stream_id = 1 if subtype == 0 else 2
        return f"rtsp://{auth}{self.host}:{self.port}/media/video{stream_id}"


class AxisAdapter(CameraAdapter):
    """Axis RTSP adapter."""

    VENDOR = "axis"

    def get_rtsp_url(self, channel: int = 1, subtype: int = 0) -> str:
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"rtsp://{auth}{self.host}:{self.port}/axis-media/media.amp"


class GenericRTSPAdapter(CameraAdapter):
    """Generic RTSP adapter with custom path."""

    VENDOR = "generic_rtsp"

    def __init__(self, host: str, port: int = 554, username: str = "", password: str = "", path: str = "/stream", **kwargs: Any) -> None:
        super().__init__(host, port, username, password, **kwargs)
        self.path = path

    def get_rtsp_url(self, channel: int = 1, subtype: int = 0) -> str:
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"rtsp://{auth}{self.host}:{self.port}{self.path}"


VENDOR_REGISTRY: Dict[str, type] = {
    "hikvision": HikvisionAdapter,
    "dahua": DahuaAdapter,
    "uniview": UniviewAdapter,
    "axis": AxisAdapter,
    "generic_rtsp": GenericRTSPAdapter,
    "generic": CameraAdapter,
}


def get_adapter(vendor: str, **kwargs: Any) -> CameraAdapter:
    """Factory: return the correct adapter for a vendor name."""
    cls = VENDOR_REGISTRY.get(vendor.lower(), CameraAdapter)
    return cls(**kwargs)
