# ARGUS AI Production Deployment Guide

This directory contains scripts and configurations for deploying ARGUS AI as an always-running background service on Windows environments.

## Overview

ARGUS AI runs as a persistent system background service utilizing **NSSM (Non-Sucking Service Manager)**.

Key Features:
- **Automatic System Boot Start**: Starts automatically when Windows boots up without requiring user login.
- **Crash Recovery**: Automatic restart on worker or process failures.
- **Log Management**: Redirects stdout/stderr to standard rotating log files in `outputs/logs/`.

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

## Service Log Files

All logs are continuously written to `outputs/logs/`:
- `system.log`: General system events and startup lifecycle.
- `camera.log`: RTSP/USB camera status, connections, and reconnect events.
- `detection.log`: Recognition match outputs and tracking statistics.
- `error.log`: Component warnings, tracebacks, and exceptions.
- `watchdog.log`: Periodic health metrics (CPU, RAM, GPU, queue size, FPS) and health warnings.
- `service_stdout.log`: Console stdout captured by NSSM.
- `service_stderr.log`: Console stderr captured by NSSM.
