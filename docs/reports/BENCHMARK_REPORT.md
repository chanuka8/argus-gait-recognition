# ARGUS AI — Benchmark Audit Report

**Report Generated:** 2026-07-30T10:37:00+05:30  
**Repository Version / Commit Hash:** `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`  
**Working Tree Status:** Dirty (Uncommitted documentation/report suite additions)  
**Audit Policy:** Zero False Positive Evidence-Based Reporting Policy  
**Target Path:** `docs/reports/BENCHMARK_REPORT.md`

---

## Executive Summary & Performance Warning

> [!WARNING]
> **Embedding-Only FPS vs. Full Pipeline FPS:**
> Embedding-only throughput measures the isolated forward pass of the `ByGaitLight` neural network on pre-cropped $64 \times 64$ GEI tensors. It **MUST NEVER** be presented or interpreted as full CCTV video pipeline FPS, which includes video frame decoding, YOLOv8 person detection, multi-object tracking, silhouette extraction, and gallery search.

---

## 1. Test Hardware & Execution Context

| Property | Value | Source / Evidence |
| :--- | :--- | :--- |
| **CPU Model** | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | `platform.processor()` |
| **CPU Cores** | 8 Physical / 12 Logical | `psutil.cpu_count()` |
| **System RAM** | 7.73 GB | `psutil.virtual_memory()` |
| **GPU Acceleration** | N/A (CPU Execution Only) | `torch.cuda.is_available() == False` |
| **Operating System** | Windows-10-10.0.26200-SP0 | `platform.platform()` |
| **Python Version** | 3.11.9 | `venv/Scripts/python.exe` |

---

## 2. Benchmark Protocol Comparison & Detailed Results

### 2.1 ONNX Runtime Embedding Inference (Isolated Forward Pass)

> **Protocol:** Isolated model forward pass benchmark (`scripts/benchmark_inference_backends.py`)  
> **Input Shape:** `(1, 1, 64, 64)` | **Batch Size:** 1 | **Warm-up Iterations:** 50 | **Timed Iterations:** 500

| Metric | Measured Value | Type | Execution Provider | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Backend** | `onnxruntime` | Configured | `CPUExecutionProvider` | **Verified** |
| **Initialization Latency** | **157.86 ms** | Measured | Engine session instantiation | **Verified** |
| **Mean Latency** | **0.851 ms** | Measured | Mean over 500 iterations | **Verified** |
| **p95 Latency** | **1.002 ms** | Measured | 95th percentile latency | **Verified** |
| **Embedding Throughput** | **1,173.82 FPS** | Derived | $1000 / 0.851\text{ ms}$ | **Verified** |
| **Numerical Parity vs PyTorch**| **PASSED** (Max Diff: 0.000000) | Measured | Cosine Similarity = 1.000000 | **Verified** |

### 2.2 PyTorch Embedding Inference (Isolated Forward Pass)

> **Protocol:** Isolated PyTorch forward pass benchmark (`scripts/benchmark_inference_backends.py`)  
> **Input Shape:** `(1, 1, 64, 64)` | **Batch Size:** 1 | **Warm-up Iterations:** 50 | **Timed Iterations:** 500

| Metric | Measured Value | Type | Execution Provider | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Backend** | `pytorch` | Configured | `PyTorch-CPU` | **Verified** |
| **Initialization Latency** | **45.92 ms** | Measured | Model load & state dict restoration | **Verified** |
| **Mean Latency** | **4.331 ms** | Measured | Mean over 500 iterations | **Verified** |
| **p95 Latency** | **5.834 ms** | Measured | 95th percentile latency | **Verified** |
| **Embedding Throughput** | **230.77 FPS** | Derived | $1000 / 4.331\text{ ms}$ | **Verified** |

### 2.3 Subject-Disjoint Evaluation Feature Extraction Benchmark

> **Protocol Note:** Measured during bulk feature extraction during subject-disjoint test set evaluation (`scripts/evaluate_subject_disjoint.py`). Employs pre-allocated memory buffers across batched dataset iterations.

| Metric | Measured Value | Type | Context / Details | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Latency** | **0.270 ms** | Measured | Bulk extraction over 5,466 items | **Verified** |
| **Throughput** | **3,712.01 FPS** | Derived | Batched feature extraction speed | **Verified** |

### 2.4 Gallery Search Latency

> **Protocol:** Vector similarity search over 13,544 gallery embeddings across 124 identities (`scripts/benchmark.py`).

| Metric | Measured Value | Type | Context | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gallery Load Time** | **12.64 ms** (`0.0126 s`) | Measured | 13,544 embeddings loaded from disk | **Verified** |
| **Gallery Identities** | 124 unique subjects | Measured | Live gallery metadata | **Verified** |
| **Gallery Total Vectors** | 13,544 Float32 256-d vectors | Measured | Live gallery feature matrix | **Verified** |

### 2.5 Full End-to-End CCTV Pipeline Performance

> **Protocol:** Integrated single-person pipeline execution on synthetic video frames (`scripts/benchmark.py`). Includes detection, tracking, silhouette GEI generation, feature extraction, and gallery search.  
> **Timed Iterations:** 10

| Metric | Measured Value | Type | Execution Context | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline Init Latency** | **76.78 ms** | Measured | Pipeline component instantiation | **Verified** |
| **Single Frame Prediction** | **129.24 ms** | Measured | First frame execution | **Verified** |
| **Mean Pipeline Latency** | **90.36 ms** | Measured | Mean over 10 iterations | **Verified** |
| **Min Pipeline Latency** | **85.77 ms** | Measured | Minimum iteration time | **Verified** |
| **Max Pipeline Latency** | **100.01 ms** | Measured | Maximum iteration time | **Verified** |
| **End-to-End Pipeline FPS** | **11.07 FPS** | Derived | $1000 / 90.36\text{ ms}$ | **Verified** |

### 2.6 Crowd Subsystem Overhead

> **Protocol:** Micro-benchmark of crowd intelligence spatial clustering across 200 frames with 30 active tracks/frame (`scripts/benchmark_crowd_performance.py`). Total 6,000 track samples.

| Metric | Measured Value | Type | Context | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Overhead Latency** | **3.314 ms** | Measured | Average per-frame spatial clustering | **Verified** |
| **Median Overhead Latency**| **3.069 ms** | Measured | 50th percentile overhead | **Verified** |
| **p95 Overhead Latency** | **5.062 ms** | Measured | 95th percentile overhead | **Verified** |
| **Memory Growth** | **0 Bytes** | Measured | Zero memory leakage over 6,000 samples | **Verified** |

---

## 3. Resource Utilization Summary

| Resource Metric | Value | Type | Measurement Source | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Process RAM Footprint** | **37.2 MB** | Measured | In-memory process measurement | **Verified** |
| **CPU Utilization** | Utilizes 12 Logical Cores | Measured | PyTorch multi-threaded CPU execution | **Verified** |
| **GPU Utilization** | 0.0% (N/A) | Measured | CPU execution environment | **Verified** |

---

## 4. Benchmark Limitations & Unmeasurable Subsystems

- **Standalone Detector FPS:** `UNABLE_TO_VERIFY` (YOLOv8n detection is benchmarked only as part of the integrated pipeline; no standalone detection script exists).
- **Standalone Tracker FPS:** `UNABLE_TO_VERIFY` (ByteTrack/BoT-SORT tracking is benchmarked only as part of the integrated pipeline).
- **TensorRT GPU Performance:** `UNABLE_TO_VERIFY` (TensorRT execution provider is deferred due to CPU execution environment).

---
**Status:** `VERIFIED - BENCHMARKS AUDITED AND RECONCILED`
