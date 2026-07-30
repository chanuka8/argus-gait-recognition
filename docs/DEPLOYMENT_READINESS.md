# ARGUS AI Gait Recognition Deployment Readiness Guide

This guide details the deployment readiness features, build/runtime asset separation, startup health validation, backend startup summaries, build metadata tracking, graceful shutdown handlers, diagnostic tools, and automated smoke testing supported in ARGUS AI.

> [!NOTE]
> **System Scope**:
> The system is ready for controlled real-world gait recognition and body-tracking validation using CCTV or recorded video inputs. It is not a CCTV control or camera-management system.

> [!IMPORTANT]
> **Active Inference Backends**:
> - **PyTorch** is the guaranteed reference backend.
> - **ONNX Runtime** is the active optimized CPU deployment backend.
> - **TensorRT** integration is explicitly deferred until CUDA and TensorRT installation and hardware validation are fully complete.

---

## 1. Build / Runtime Asset Separation

ARGUS AI enforces strict separation between build-time development tools and production runtime assets for native Windows deployment packaging.

- **Runtime Required Assets**: [main.py](file:///e:/ARGUS_AI/main.py), [cli.py](file:///e:/ARGUS_AI/cli.py), [VERSION](file:///e:/ARGUS_AI/VERSION), core modules (`core`, `models`, `pipeline`, `storage`, `monitoring`, `deployment`, `streaming`, `security_layer`, `utils`), `configs`, identity gallery (`models/gallery`), and diagnostic script (`scripts/doctor.py`).
- **Build-Only Assets**: `tests`, `training`, `evaluation`, `automation`, `dataconnect`, `Makefile`, `pytest.ini`, export scripts, and documentation generators.
- **Excluded Patterns**: `venv`, `.git`, `.pytest_cache`, `.ruff_cache`, `.vscode`, `.env`, secrets, `__pycache__`, temporary reports.

### Manifest Artifacts
- [deployment/runtime_manifest.py](file:///e:/ARGUS_AI/deployment/runtime_manifest.py)
- [deployment/runtime_manifest.json](file:///e:/ARGUS_AI/deployment/runtime_manifest.json)
- [deployment/runtime_manifest.md](file:///e:/ARGUS_AI/deployment/runtime_manifest.md)

---

## 2. Pre-Flight Startup Health Validation

The extended pre-flight validator executes non-destructive checks before launching live video streams:

```python
from deployment.startup_validator import DeploymentStartupValidator

validator = DeploymentStartupValidator()
summary = validator.validate_startup(raise_on_failure=True)
```

### Approved Status Codes
- `READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING`: All health checks pass cleanly.
- `READY_WITH_WARNINGS`: Non-blocking notices present (e.g. backend fallback active).
- `NOT_READY`: One or more blocking defects identified.
- `UNABLE_TO_VERIFY`: Live network camera access deferred to stream launch.

---

## 3. Structured Backend Startup Summary

Upon backend initialization, a single structured summary block is formatted and logged:

```text
==================================================
ARGUS Backend Startup Summary
==================================================
Requested Backend : auto
Active Backend    : onnxruntime
Provider          : CPUExecutionProvider
Allow Fallback    : true
Fallback Used     : false
Fallback Reason   : None
Attempted Engines : onnxruntime
Model Path        : models/engines/bygait_light.onnx
Startup Status    : READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING
==================================================
```

Implemented in [deployment/backend_summary.py](file:///e:/ARGUS_AI/deployment/backend_summary.py). Emitted exactly once per application lifecycle without exposing credentials or user-home absolute paths.

---

## 4. Build and Version Metadata Contract

Metadata extraction in [deployment/build_metadata.py](file:///e:/ARGUS_AI/deployment/build_metadata.py) aggregates application version, Git commit hash, branch, Python runtime version, model reference, active backend, and configuration SHA-256 fingerprint.

- Operates without network calls.
- Non-fatal Git extraction (reports `UNKNOWN` if Git is unavailable).
- Sanitizes secret credentials and absolute user paths.

---

## 5. Idempotent Graceful Shutdown Manager

[deployment/shutdown_manager.py](file:///e:/ARGUS_AI/deployment/shutdown_manager.py) coordinates process teardown upon `SIGINT` (Ctrl+C), `SIGTERM`, or application stop requests:

1. Signals worker stop events.
2. Drains queues and executes registered cleanup callbacks in reverse order.
3. Joins active worker threads with configurable timeouts.
4. Ensures idempotent execution without process deadlocks or gallery data corruption.

---

## 6. Automated Deployment Smoke Test

Run fast native deployment smoke testing prior to staging:

```bash
python scripts/smoke_test_deployment.py
```

### Verification Scope
- Runtime manifest asset validation
- Startup validator execution
- Inference backend initialization & metadata check
- Synthetic GEI inference execution (shape `(1, 256)` and L2 normalization norm $\approx 1.0$)
- Gallery vector store check
- Smoke test report generation (`outputs/reports/deployment_smoke_test.json`, `outputs/reports/deployment_smoke_test.md`)
- Graceful shutdown sequence verification

### Exit Codes
- `0`: Smoke test passed cleanly.
- `1`: Confirmed deployment defect detected.
- `2`: Internal smoke test setup or invocation failure.

---

## 7. Current Limitations

- **TensorRT & CUDA**: GPU execution providers and engine building remain deferred until hardware installation is verified.
- **Physical RTSP Cameras**: Live RTSP camera streams require network reachability and environment credential configuration.
