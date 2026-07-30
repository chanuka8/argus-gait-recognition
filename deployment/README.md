# ARGUS AI Production Deployment Guide

This directory contains scripts, modules, and configurations for deploying ARGUS AI as an always-running background service and native deployment package on Windows environments.

## Overview

ARGUS AI runs as a persistent system background service utilizing **NSSM (Non-Sucking Service Manager)**.

Key Features:
- **Automatic System Boot Start**: Starts automatically when Windows boots up without requiring user login.
- **Crash Recovery**: Automatic restart on worker or process failures.
- **Log Management**: Redirects stdout/stderr to standard rotating log files in `outputs/logs/system/`.
- **Deployment Hardening Suite**: Integrated runtime manifest separation, startup health validation, backend startup summary formatting, build/version metadata tracking, and idempotent graceful shutdown.

> [!NOTE]
> **System Scope**: The system is ready for controlled real-world gait recognition and body-tracking validation using CCTV or recorded video inputs. It is not a CCTV control or camera-management system.

---

## Deployment Hardening Modules

- [runtime_manifest.py](runtime_manifest.py): Build vs runtime asset separator ([runtime_manifest.json](runtime_manifest.json), [runtime_manifest.md](runtime_manifest.md)).
- [startup_validator.py](startup_validator.py): Pre-flight deployment health validator emitting approved status codes.
- [backend_summary.py](backend_summary.py): Formats and logs single-emit backend startup summaries.
- [build_metadata.py](build_metadata.py): Extracts git version, build, configuration fingerprint, and runtime metadata contracts.
- [shutdown_manager.py](shutdown_manager.py): Coordinates graceful, idempotent process teardown upon SIGINT/SIGTERM or stop requests.

---

## Quick Start Installation

### Prerequisites
1. PowerShell with Administrator privileges.
2. NSSM installed (`nssm.exe` in system PATH or placed directly in `deployment/nssm.exe`).
3. Python environment configured (`venv/Scripts/python.exe` or system Python).

### 1. Install as Windows Service

Run PowerShell as **Administrator**:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\deployment\install_service.ps1
```

Or specify custom options:

```powershell
.\deployment\install_service.ps1 -PythonPath "C:\Python311\python.exe" -NssmPath "C:\Tools\nssm.exe"
```

---

## Service Management Commands

Using NSSM:

* **Check Service Status**:
  ```powershell
  nssm status ArgusAiGaitService
  ```

* **Start Service**:
  ```powershell
  nssm start ArgusAiGaitService
  ```

* **Stop Service**:
  ```powershell
  nssm stop ArgusAiGaitService
  ```

* **Restart Service**:
  ```powershell
  nssm restart ArgusAiGaitService
  ```

* **GUI Service Configuration**:
  ```powershell
  nssm edit ArgusAiGaitService
  ```

Using Standard Windows Commands:

```powershell
Get-Service ArgusAiGaitService
Start-Service ArgusAiGaitService
Stop-Service ArgusAiGaitService
```

---

## Uninstallation

Run PowerShell as **Administrator**:

```powershell
.\deployment\uninstall_service.ps1
```

---

## Automated Deployment Smoke Test

Run non-destructive automated deployment smoke testing before service start:

```bash
python scripts/smoke_test_deployment.py
```

---

## Service Log Files

All system logs are continuously written to `outputs/logs/system/` (and camera logs to `outputs/logs/camera/`):
- `outputs/logs/system/system.log`: General system events and startup lifecycle.
- `outputs/logs/camera/camera.log`: RTSP/USB camera status, connections, and reconnect events.
- `outputs/logs/system/detection.log`: Recognition match outputs and tracking statistics.
- `outputs/logs/system/error.log`: Component warnings, tracebacks, and exceptions.
- `outputs/logs/system/watchdog.log`: Periodic health metrics (CPU, RAM, GPU, queue size, FPS) and health warnings.
- `outputs/logs/system/service_stdout.log`: Console stdout captured by NSSM.
- `outputs/logs/system/service_stderr.log`: Console stderr captured by NSSM.
