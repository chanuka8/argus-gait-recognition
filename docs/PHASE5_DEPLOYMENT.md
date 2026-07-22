# ARGUS AI Phase 5 — Enterprise CCTV Deployment Guide

## Components

| Module | Path | Purpose |
|--------|------|---------|
| ONVIFClient | `services/onvif_client.py` | ONVIF SOAP/XML capability discovery, PTZ detection, profile parsing |
| VendorAdapters | `services/vendor_adapters.py` | Adapter pattern for Hikvision, Dahua, Uniview, Axis, Generic RTSP |
| CameraDiscovery | `services/camera_discovery.py` | Config-driven discovery, RTSP validation, health checks |

## Supported Vendors

| Vendor | Adapter | RTSP Format |
|--------|---------|-------------|
| Hikvision | `HikvisionAdapter` | `rtsp://<auth>@host:554/Streaming/Channels/<ch>0<sub>` |
| Dahua | `DahuaAdapter` | `rtsp://<auth>@host:554/cam/realmonitor?channel=<ch>&subtype=<sub>` |
| Uniview | `UniviewAdapter` | `rtsp://<auth>@host:554/media/video<stream>` |
| Axis | `AxisAdapter` | `rtsp://<auth>@host:554/axis-media/media.amp` |
| Generic | `GenericRTSPAdapter` | `rtsp://<auth>@host:554<path>` |

## Configuration

Add vendor field to `configs/cameras.yaml`:

```yaml
cameras:
  entrance_cam:
    id: entrance_cam
    name: Main Entrance
    type: rtsp
    vendor: hikvision
    url: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
    enabled: true
    width: 1920
    height: 1080
    target_fps: 15
```

## ONVIF Discovery

```python
from services.onvif_client import ONVIFClient

client = ONVIFClient("192.168.1.100", username="admin", password="pass")
# Use client.parse_capabilities_response(xml) with SOAP response
# Use client.parse_profiles_response(xml) for stream profiles
```

## Vendor Adapter Usage

```python
from services.vendor_adapters import get_adapter

adapter = get_adapter("hikvision", host="192.168.1.100", username="admin", password="pass")
rtsp_url = adapter.get_rtsp_url(channel=1, subtype=0)
config = adapter.get_camera_config("cam_01", width=1920, height=1080)
```

## Camera Discovery

```python
from services.camera_discovery import CameraDiscoveryService

service = CameraDiscoveryService()
cameras = service.discover_from_config(cameras_yaml_dict)
health = service.health_check_camera("cam_01", rtsp_url="rtsp://...")
```
