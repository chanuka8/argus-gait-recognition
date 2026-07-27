# Services

The `services` package manages persistent system service lifecycle (NSSM Windows service / systemd Linux service), camera hardware acquisition workers, ONVIF camera discovery, and vendor camera stream adapters for ARGUS AI.

## Responsibilities

- Operating as a persistent OS background service process with graceful startup, signal handling, and PID management (`outputs/temporary/argus.pid`).
- Managing RTSP/USB camera acquisition worker threads and auto-reconnecting dropped camera streams.
- Discovering network ONVIF CCTV cameras automatically on local subnets.
- Boundaries: Does not train models or run scientific evaluation benchmarks.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [argus_service.py](file:///e:/ARGUS_AI/services/argus_service.py) | Main background service entrypoint handling process signals (SIGINT/SIGTERM) and PID tracking |
| [camera_discovery.py](file:///e:/ARGUS_AI/services/camera_discovery.py) | Scans local IP subnets to discover active RTSP/USB video stream sources |
| [camera_manager.py](file:///e:/ARGUS_AI/services/camera_manager.py) | Central manager coordinating multi-camera workers, stream status, and health metrics |
| [camera_service.py](file:///e:/ARGUS_AI/services/camera_service.py) | High-level service facade wrapping camera capture, stream reconnects, and frame queueing |
| [camera_worker.py](file:///e:/ARGUS_AI/services/camera_worker.py) | Dedicated background thread worker capturing frames from an individual camera stream |
| [onvif_client.py](file:///e:/ARGUS_AI/services/onvif_client.py) | ONVIF protocol client querying network CCTV camera capabilities and RTSP stream URIs |
| [vendor_adapters.py](file:///e:/ARGUS_AI/services/vendor_adapters.py) | Vendor-specific camera stream adapters (Hikvision, Dahua, Axis, generic RTSP) |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

RTSP/USB Hardware Streams → `services/camera_worker.py` → `services/camera_service.py` → `services/camera_manager.py` → Pipeline Capture Queue.

## Configuration

- [configs/system.yaml](file:///e:/ARGUS_AI/configs/system.yaml): `camera` and `service` sections
- [configs/cameras.yaml](file:///e:/ARGUS_AI/configs/cameras.yaml): RTSP camera configurations

## Public Interfaces

- `ArgusService`: Service entrypoint in [services/argus_service.py](file:///e:/ARGUS_AI/services/argus_service.py).
- `CameraService`: Stream capture service in [services/camera_service.py](file:///e:/ARGUS_AI/services/camera_service.py).
- `CameraManager`: Multi-camera manager in [services/camera_manager.py](file:///e:/ARGUS_AI/services/camera_manager.py).
- `ONVIFClient`: ONVIF camera discovery client in [services/onvif_client.py](file:///e:/ARGUS_AI/services/onvif_client.py).

## Tests

- [tests/test_camera_service.py](file:///e:/ARGUS_AI/tests/test_camera_service.py)
- [tests/test_phase5_cctv.py](file:///e:/ARGUS_AI/tests/test_phase5_cctv.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Deployment Guide](file:///e:/ARGUS_AI/deployment/README.md)
