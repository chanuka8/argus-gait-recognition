# ARGUS AI Phase 1: Production Deployment Implementation Plan

## Current Architecture

```
Input (StreamEngine / cv2.VideoCapture)
  → LiveRecognitionPipeline.run()
    → TrackingStep (YOLOv8 + ByteTrack)
    → SilhouetteStep → LiveGEI → GEI
    → ByGaitLight → 256-d embedding
    → MatchingStep (cosine similarity vs VectorStore gallery)
    → Result rendering + event logging
```

### Key Integration Points
- `streaming/stream_engine.py`: Simple OpenCV VideoCapture wrapper (source=0 or path)
- `pipeline/live_recognition.py`: LiveRecognitionPipeline — main recognition loop
- `core/logger.py`: Basic logging with single file handler, no rotation
- `configs/inference.yaml`: Thresholds, matching policy, crowd control
- `configs/cameras.yaml`: Multi-camera definitions (source, resolution, fps)
- `monitoring/`: Placeholder stubs only

## Files to Create

| File | Purpose |
|------|---------|
| `services/__init__.py` | Package init |
| `services/camera_service.py` | RTSP/USB threaded camera with auto-reconnect |
| `services/argus_service.py` | Background orchestrator service |
| `monitoring/logging_config.py` | Rotating file log handlers |
| `monitoring/watchdog.py` | Health monitor with restart capability |
| `configs/system.yaml` | Unified production config |
| `deployment/install_service.ps1` | Windows service installer (NSSM) |
| `deployment/uninstall_service.ps1` | Windows service uninstaller |
| `deployment/README.md` | Deployment guide |
| `tests/test_camera_service.py` | Camera service tests |
| `tests/test_watchdog.py` | Watchdog tests |
| `tests/test_logging.py` | Logging tests |
| `docs/PHASE1_DEPLOYMENT.md` | Full deployment documentation |

## Files to Modify
- None. All new functionality is additive.

## Execution Flow (Production)

```
argus_service.py (main process)
  ├── logging_config.py (init rotating loggers)
  ├── camera_service.py (threaded RTSP/USB reader)
  │     └── frame_queue (thread-safe)
  ├── LiveRecognitionPipeline (existing, fed frames from queue)
  └── watchdog.py (monitoring thread)
        ├── camera health
        ├── worker health
        ├── system resources
        └── auto-restart on failure
```
