# ARGUS AI CCTV Deployment Readiness Guide

This guide details the deployment readiness features, ONNX export workflows, backend selection policies, diagnostic health check commands, externalized configuration validation, and reporting tools supported in ARGUS AI.

> [!IMPORTANT]
> **Active Inference Backends**:
> - **PyTorch** is the guaranteed reference backend.
> - **ONNX Runtime** is the active optimized CPU/GPU deployment backend.
> - **TensorRT** integration is explicitly deferred until CUDA and TensorRT installation and hardware validation are fully complete.

---

## 1. ONNX Model Export & Numerical Parity Validation

Export ByGaitLight PyTorch model checkpoints to ONNX format with automatic structural check, shape validation, and numerical parity verification against PyTorch reference inference.

### Command
```bash
python scripts/export_bygait_onnx.py --output-path models/engines/bygait_light.onnx --precision fp32
```

### Safety Guarantees & Features
- **Atomic File Replacement**: Exports to a temporary `.tmp.onnx` file and replaces target only after export, structural validation (`onnx.checker`), and parity checks pass.
- **Deterministic Output**: Uses fixed seed initialization for inputs and weights.
- **Validation Reports Generated**:
  - `outputs/reports/onnx_validation.json`
  - `outputs/reports/onnx_validation.md`

---

## 2. Inference Backend Selection & Fallback Policy

Backend selection is controlled via `configs/inference.yaml` under `inference_backend`:

```yaml
inference_backend:
  backend: auto          # Options: "auto", "onnxruntime", "pytorch"
  device: auto           # Options: "auto", "cpu", "cuda"
  precision: fp32        # Options: "fp32", "fp16"
  onnx_path: models/engines/bygait_light.onnx
  allow_fallback: true
```

### Selection Policy (`auto`)
$$\text{ONNX Runtime} \longrightarrow \text{PyTorch (Guaranteed Reference Fallback)}$$

1. **ONNX Runtime**: Attempted first. If model exists and `onnxruntime` is available, active backend is set to `onnxruntime`.
2. **PyTorch**: If ONNX model is missing, corrupt, or `onnxruntime` is unavailable, transparent fallback to PyTorch occurs.

### Backend Report Generation
Generates `outputs/reports/backend_report.json` containing `requested_backend`, `active_backend`, `execution_provider`, `fallback_used`, `fallback_reason`, `model_path`, `initialization_result`, and `inference_smoke_test_result`.

---

## 3. Configuration Externalization & Sanitization

All deployment-specific parameters (camera URLs, camera IDs, gallery paths, model paths, batch size, thresholds, logging paths, output paths) are externalized in `configs/`:
- `configs/inference.yaml`
- `configs/cameras.yaml`
- `configs/system.yaml`

### Credential Sanitization
RTSP connection strings in exceptions, logs, and diagnostic reports automatically sanitize secret passwords:
```
rtsp://admin:secret123@192.168.1.100:554/stream1  -->  rtsp://admin:***@192.168.1.100:554/stream1
```

---

## 4. System Health Check CLI (`doctor.py`)

Run non-destructive diagnostic health checks before launching recognition pipelines:

### Command
```bash
python scripts/doctor.py
```

### Verification Scope
- Python version (>= 3.9) and active interpreter path
- Dependencies (`torch`, `onnx`, `onnxruntime`, `cv2`, `numpy`, `yaml`)
- Model artifact presence and structural integrity
- Gallery file safe loading (`allow_pickle=False`) and feature/label count alignment
- Externalized YAML configuration validity
- Writable output directory (`outputs/reports`) and disk space (> 500 MB)
- Logging system initialization

### Reports Generated
- `outputs/reports/health_report.json`
- `outputs/reports/health_report.md`

Returns exit code `0` when healthy, and `1` if blocking defects are detected.

---

## 5. Pipeline Pre-Flight Startup Validation

Call `DeploymentStartupValidator` before starting live video or camera recognition streams:

```python
from deployment.startup_validator import DeploymentStartupValidator

validator = DeploymentStartupValidator()
validator.validate_startup(raise_on_failure=True)
```

---

## 6. Deployment Readiness Reporter

Generates qualitative readiness assessments:
- `outputs/reports/deployment_readiness.json`
- `outputs/reports/deployment_readiness.md`

Qualitative Statuses:
- `READY_FOR_CONTROLLED_CCTV_TESTING`
- `READY_WITH_WARNINGS`
- `NOT_READY`
- `UNABLE_TO_VERIFY`

---

## 7. Current Limitations

- **TensorRT**: TensorRT execution provider and engine building remain deferred until CUDA and TensorRT installation are validated on hardware.
- **Physical RTSP Cameras**: Live RTSP camera streams require network configuration and environment credentials (`ARGUS_CAMERA_XX_USERNAME`/`PASSWORD`).
