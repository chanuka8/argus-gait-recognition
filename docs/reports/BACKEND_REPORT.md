# ARGUS AI — Inference Backend Audit Report

**Report Generated:** 2026-07-30T10:37:00+05:30  
**Repository Version / Commit Hash:** `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`  
**Working Tree Status:** Dirty (Uncommitted documentation/report suite additions)  
**Audit Policy:** Zero False Positive Evidence-Based Reporting Policy  
**Target Path:** `docs/reports/BACKEND_REPORT.md`

---

## Executive Summary & Runtime Clarification

> [!IMPORTANT]
> **Benchmark vs. Runtime Pipeline Distinction:**
> While ONNX Runtime CPU is verified and benchmarked at sub-millisecond latencies (**0.851 ms**), the default live pipeline configuration requests `auto` or `pytorch` as the primary runtime inference engine. ONNX Runtime benchmark availability does **NOT** imply that ONNX Runtime is hardcoded as the active runtime engine in all deployment scripts.

---

## 1. Backend Selection Policy & Fallback Cascade

The inference subsystem enforces a strict fallback cascade based on requested parameters and environment capability.

```
                  ┌───────────────────────────────┐
                  │   Requested Backend Selection │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │ (auto)                 │ (onnxruntime)          │ (pytorch)
         ▼                        ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Attempt ONNX    │      │ Attempt ONNX    │      │ Initialize      │
│ Runtime Session │      │ Runtime Session │      │ PyTorch Engine │
└────────┬────────┘      └────────┬────────┘      └─────────────────┘
         │ (success)              │ (failure)
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│ Active: ONNX    │      │ Check           │
│ Runtime Engine  │      │ allow_fallback  │
└─────────────────┘      └────────┬────────┘
                                  │
                         ┌────────┴────────┐
                         │ true            │ false
                         ▼                 ▼
                ┌─────────────────┐ ┌─────────────────┐
                │ Active: PyTorch │ │ Raise           │
                │ Fallback Engine │ │ Runtime Error   │
                └─────────────────┘ └─────────────────┘
```

### Configuration Matrix & Rules

| `requested_backend` | `allow_fallback` | Attempted Cascade | Behavior on Primary Failure | Active Backend Result |
| :--- | :---: | :--- | :--- | :--- |
| `auto` | `true` | `['onnxruntime', 'pytorch']` | Fallback to PyTorch | `onnxruntime` (if ONNX valid) or `pytorch` |
| `onnxruntime` | `true` | `['onnxruntime', 'pytorch']` | Fallback to PyTorch | `onnxruntime` (if ONNX valid) or `pytorch` |
| `onnxruntime` | `false` | `['onnxruntime']` | Raise `RuntimeError` | `onnxruntime` (or raise error) |
| `pytorch` | N/A | `['pytorch']` | No fallback allowed | `pytorch` |
| `tensorrt` | `true` | `['tensorrt', 'pytorch']` | Fallback to PyTorch | `pytorch` (Reason: `No module named 'tensorrt'`) |

---

## 2. Environment Provider Discovery & Status

Inspection of installed runtime execution providers:

| Environment Property | Discovered State | Evidence Source | Verification Status |
| :--- | :--- | :--- | :--- |
| **Available ONNX Providers** | `['AzureExecutionProvider', 'CPUExecutionProvider']` | `onnxruntime.get_available_providers()` | **Verified** |
| **Selected ONNX Provider** | `CPUExecutionProvider` | `backend_benchmark` output | **Verified** |
| **PyTorch Execution Target** | `PyTorch-CPU` | `torch.__version__ == 2.12.0+cpu` | **Verified** |
| **TensorRT Module** | **Not Installed** (`ImportError`) | `scripts/benchmark_inference_backends.py` | **Verified** |

---

## 3. Session Reuse & Lifecycle Management

- **Model Instantiation:** Both PyTorch checkpoints and ONNX engine sessions are loaded **once** during component initialization (`__init__`) and reused across frames.
- **Thread Safety:** The ONNX Runtime inference session is configured with single-threaded / multi-threaded CPU execution providers (`CPUExecutionProvider`).
- **Memory Overhead:** Session instantiation consumes $\sim 14.0\text{ ms}$ for ONNX Runtime and $\sim 45.9\text{ ms}$ for PyTorch.

---

## 4. Backend Limitations

1. **TensorRT GPU Acceleration:** TensorRT is currently deferred due to the CPU-only host environment.
2. **Dynamic Batching:** Current ONNX export enforces a fixed input shape of `(1, 1, 64, 64)` for single GEI embedding extraction.

---
**Status:** `VERIFIED - BACKEND SELECTION AUDITED`
