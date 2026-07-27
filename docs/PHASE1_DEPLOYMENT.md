# ARGUS AI Phase 1: Production Deployment Guide

This document specifies the production deployment architecture, setup, management, configuration, and troubleshooting procedures for the **ARGUS AI Gait Recognition Service (Phase 1)**.

---

## 1. System Architecture

ARGUS AI Phase 1 transitions the gait recognition module from a command-driven utility to an **always-running background CCTV surveillance service**.

```
                           +------------------------+
                           |   CCTV RTSP / USB /    |
                           |      Video Source      |
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           |     CameraService      |
                           | (Threaded Capture &    |
                           |    Auto-Reconnect)     |
                           +-----------+------------+
                                       |
                                       v
                            [ Thread-Safe Queue ]
                                       |
                                       v
+-----------------------+  +------------------------+  +-----------------------+
|    Watchdog System    |  |   Recognition Worker   |  |   Rotating Logging    |
| (30s Health Check &  |<->| (YOLOv8 + ByGaitLight  |<->| (system, camera,      |
|     Auto-Restart)     |  |    + VectorStore)      |  |  detection, error)    |
+-----------------------+  +------------------------+  +-----------------------+
```

### Core Architecture Components

1. **CameraService (`services/camera_service.py`)**:
   - Threaded frame reader supporting RTSP stream URLs, USB webcams, and video files.
   - Bounded thread-safe frame queue with newest-frame priority (drops old frames on overflow).
   - Configurable capture FPS and exponential backoff auto-reconnection.

2. **ArgusService Orchestrator (`services/argus_service.py`)**:
   - Master background daemon managing the camera service, recognition worker, and watchdog thread.
   - Provides graceful signal handling (SIGINT/SIGTERM), PID tracking, and GUI/Headless execution modes.

3. **Watchdog System (`monitoring/watchdog.py`)**:
   - Continuous 30-second health evaluation cycle monitoring camera connectivity, queue size, process FPS, CPU, RAM, and GPU memory.
   - Automatic non-blocking recovery: restarts failed recognition or camera threads if anomalies or thread terminations occur.

4. **Production Rotating Logging (`monitoring/logging_config.py`)**:
   - Multi-channel log routing generating 5 separate 10MB rotating log files in `outputs/logs/` (`system.log`, `camera.log`, `detection.log`, `error.log`, `watchdog.log`).

---

## 2. Configuration Guide (`configs/system.yaml`)

All parameters are configured centrally in `configs/system.yaml`:

```yaml
camera:
  type: usb                  # "rtsp", "usb", or "file"
  url: ""                    # e.g. rtsp://admin:pass@192.168.1.100:554/stream1
  device_index: 0            # USB camera device ID
  width: 640
  height: 480
  target_fps: 15
  reconnect_seconds: 5
  max_queue_size: 10

logging:
  log_dir: "outputs/logs"
  max_bytes: 10485760        # 10MB per file
  backup_count: 5
  level: "INFO"

watchdog:
  enabled: true
  interval_seconds: 30
  max_queue_warning: 8
  min_fps_warning: 2.0
  auto_restart: true
  max_restart_attempts: 5

recognition:
  model_path: "runs/exp_001/best_model.pth"
  gallery_dir: "models/live_gallery"
  threshold: 0.85

service:
  name: "ARGUS AI Gait Recognition"
  headless: false
  pid_file: "outputs/argus.pid"
```

---

## 3. Windows Service Setup (NSSM)

### Installation Steps

1. Open PowerShell as **Administrator**.
2. Execute the installation script:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\deployment\install_service.ps1
```

3. Verify service status:

```powershell
nssm status ArgusAiGaitService
```

### Uninstallation

```powershell
.\deployment\uninstall_service.ps1
```

---

## 4. Verification & Testing

Run unit and integration test suites:

```bash
python -m unittest discover -s tests -v
```

---

## 5. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **Camera Disconnected / Black Screen** | RTSP network drop or USB unplugged | Check `outputs/logs/camera.log`. CameraService will auto-reconnect every 5 seconds. |
| **High Queue Size Warnings** | System processing slower than capture FPS | Reduce `target_fps` in `configs/system.yaml` or enable `headless: true`. |
| **Service Fails on Boot** | Python virtual environment not resolved | Ensure `venv/Scripts/python.exe` exists or pass `-PythonPath` explicitly to `install_service.ps1`. |
| **Worker Crashes Repeatedly** | Missing model file or CUDA out of memory | Inspect `outputs/logs/error.log` for Python tracebacks. |
