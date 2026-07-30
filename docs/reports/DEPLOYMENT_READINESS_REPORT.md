# ARGUS AI — Deployment Readiness Audit Report

**Report Generated:** 2026-07-30T10:37:00+05:30  
**Repository Version / Commit Hash:** `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`  
**Working Tree Status:** Dirty (Uncommitted documentation/report suite additions)  
**Audit Policy:** Zero False Positive Evidence-Based Reporting Policy  
**Target Path:** `docs/reports/DEPLOYMENT_READINESS_REPORT.md`  
**Cross-Reference Document:** [DEPLOYMENT_READINESS.md](../DEPLOYMENT_READINESS.md)

---

## Executive Summary & Official Status Label

> [!IMPORTANT]
> **Approved System Deployment Verdict:**
> **`READY_FOR_CONTROLLED_CCTV_TESTING`**
> 
> *Constraint Note:* The system status **MUST NOT** be simplified or described as "production-ready", "enterprise-ready", "24/7-ready", or "airport-ready". Controlled CCTV field testing is required prior to production deployment.

---

## 1. System Health Doctor Results (`scripts/doctor.py`)

Execution of `scripts/doctor.py` returned an exit code of `0` with zero blocking issues and zero warnings.

| Health Check Category | Test Name | Status | Details / Output | Verification Status |
| :--- | :--- | :---: | :--- | :--- |
| **Python Version** | `python_version` | **PASS** | Python 3.11.9 (`venv/Scripts/python.exe`) | **Verified** |
| **Dependencies** | `import_torch` | **PASS** | PyTorch 2.12.0+cpu imported successfully | **Verified** |
| **Dependencies** | `import_onnx` | **PASS** | Package `onnx` imported successfully | **Verified** |
| **Dependencies** | `import_onnxruntime` | **PASS** | Package `onnxruntime` 1.28.0 imported | **Verified** |
| **Dependencies** | `import_cv2` | **PASS** | OpenCV imported successfully | **Verified** |
| **Dependencies** | `import_numpy` | **PASS** | NumPy imported successfully | **Verified** |
| **Dependencies** | `import_yaml` | **PASS** | PyYAML imported successfully | **Verified** |
| **PyTorch Checkpoint** | `pytorch_checkpoint_exists` | **PASS** | Checkpoint at `runs/exp_001/best_model.pth` found | **Verified** |
| **ONNX Engine File** | `onnx_model_exists` | **PASS** | Engine file at `models/engines/bygait_light.onnx` found | **Verified** |
| **ONNX Integrity** | `onnx_model_integrity` | **PASS** | Structural validation check passed | **Verified** |
| **Backend Init** | `backend_initialization` | **PASS** | Active backend `pytorch` (PyTorch-CPU smoke test PASSED) | **Verified** |
| **Gallery Integrity** | `gallery_integrity` | **PASS** | 13,544 embeddings (124 identities) loaded (`allow_pickle=False`)| **Verified** |
| **Configuration Files**| `configuration_files` | **PASS** | All YAML configs loaded & validated | **Verified** |
| **Storage Writability**| `storage_writability` | **PASS** | `outputs/reports` directory writable | **Verified** |
| **Disk Space** | `disk_space` | **PASS** | Available disk space: 47,834.8 MB (minimum 500 MB required) | **Verified** |
| **Logging System** | `logging_initialization` | **PASS** | Logging subsystem initialized cleanly | **Verified** |

---

## 2. Model Export & ONNX Validation (`scripts/export_bygait_onnx.py`)

| Audit Check | Status | Execution Details | Verification Status |
| :--- | :---: | :--- | :--- |
| **Export Script** | `scripts/export_bygait_onnx.py` | Standalone validation script | **Verified** |
| **Atomic Replacement** | **PASS** | Atomic write to temp file $\rightarrow$ parity check $\rightarrow$ atomic overwrite | **Verified** |
| **Structural Check** | **PASS** | `onnx.checker.check_model()` verified graph structure | **Verified** |
| **Numerical Parity** | **PASS** | Max Absolute Difference: **`0.000000`**, Cosine Similarity: **`1.000000`** | **Verified** |
| **Target Engine File** | `models/engines/bygait_light.onnx` | File size: 505,589 bytes (0.48 MB) | **Verified** |

---

## 3. Backend Execution Readiness

| Backend Name | Availability | Active Status | Execution Provider | Fallback Status |
| :--- | :---: | :---: | :--- | :--- |
| **PyTorch** | **Available** | **Active Default** | `PyTorch-CPU` | Primary Fallback Target |
| **ONNX Runtime** | **Available** | **Active in Benchmark**| `CPUExecutionProvider` | Secondary Fallback Target |
| **TensorRT** | **Unavailable** | Deferred | N/A | Selection Fallback Triggered (`No module named 'tensorrt'`) |

---

## 4. Unable-to-Verify Deployment Items

- **Live RTSP Connectivity:** `UNABLE_TO_VERIFY` (Requires physical network RTSP camera stream).
- **Long-Duration 24/7 Stability:** `UNABLE_TO_VERIFY` (Requires multi-day continuous CCTV field testing).
- **TensorRT GPU Acceleration:** `UNABLE_TO_VERIFY` (Requires NVIDIA GPU and TensorRT execution provider environment).

---
**Status:** `READY_FOR_CONTROLLED_CCTV_TESTING`
