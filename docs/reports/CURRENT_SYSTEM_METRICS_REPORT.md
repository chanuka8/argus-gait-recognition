# ARGUS AI — Complete Master Metrics & Performance Audit Report

**Report Generated:** 2026-07-30T10:37:00+05:30  
**Repository Version / Commit Hash:** `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`  
**Working Tree Status:** Dirty (Uncommitted documentation/report suite additions)  
**Audit Policy:** Zero False Positive Evidence-Based Reporting Policy  
**Target Path:** `docs/reports/CURRENT_SYSTEM_METRICS_REPORT.md`

---

## Modular Report Suite Index

This master audit report provides the high-level system metrics overview. For domain-specific evidence, methodology, and detailed logs, refer to the individual modular audit reports:

- [MODEL_ARCHITECTURE_REPORT.md](MODEL_ARCHITECTURE_REPORT.md) — Neural network topology, layer inventory, parameter breakdown, and FLOPs/MACs profiling.
- [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) — Isolated ONNX/PyTorch embedding latencies, full pipeline FPS, gallery search speed, and crowd overhead.
- [EVALUATION_REPORT.md](EVALUATION_REPORT.md) — Subject-disjoint CASIA-B evaluation (Rank-1, Rank-5, NM/BG/CL breakdowns, Open-set ROC AUC/EER, 11x11 Cross-view matrix).
- [DEPLOYMENT_READINESS_REPORT.md](DEPLOYMENT_READINESS_REPORT.md) — System doctor health checks, ONNX export parity, and deployment status.
- [BACKEND_REPORT.md](BACKEND_REPORT.md) — Inference backend selection policies, execution provider cascades, and fallback handling.
- [TEST_SUMMARY_REPORT.md](TEST_SUMMARY_REPORT.md) — Pytest execution results, pass rates, test categories, and skipped test analysis.
- [SECURITY_INTEGRITY_REPORT.md](SECURITY_INTEGRITY_REPORT.md) — Data integrity controls, `allow_pickle=False` enforcement, RTSP credential masking, and security tests.

---

## Executive Summary

This report presents a complete, empirical audit of the ARGUS AI Gait Recognition System. All reported values were obtained directly from live execution of validation scripts (`scripts/doctor.py`, `scripts/benchmark_inference_backends.py`, `scripts/benchmark.py`, `scripts/benchmark_crowd_performance.py`, `scripts/export_bygait_onnx.py`, `scripts/evaluate_subject_disjoint.py`), test suite runs (`pytest`), configuration files (`configs/*.yaml`, `configs/subject_split.json`), and generated evaluation reports.

No metrics were estimated, fabricated, or assumed. Metrics that could not be measured directly during the audit are explicitly classified as `UNABLE_TO_VERIFY`.

---

## Section 1: System Information

| Metric | Value | Type | Source | Evidence / Log Reference | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Repository Commit** | `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13` | Measured | `git rev-parse HEAD` | Command output | **Verified** |
| **Git Branch** | `main` | Measured | `git branch --show-current` | Command output | **Verified** |
| **Working Tree Status** | Dirty (Report suite additions) | Measured | `git status --short` | Command output | **Verified** |
| **Operating System** | `Windows-10-10.0.26200-SP0` | Measured | `platform.platform()` | `scripts/doctor.py` | **Verified** |
| **Python Version** | `3.11.9` | Measured | `sys.version` | `venv/Scripts/python.exe` | **Verified** |
| **PyTorch Version** | `2.12.0+cpu` | Measured | `torch.__version__` | `outputs/reports/health_report.json` | **Verified** |
| **ONNX Runtime Version** | `1.28.0` | Measured | `onnxruntime.__version__` | `outputs/reports/health_report.json` | **Verified** |
| **CUDA Availability** | `False` (CPU Execution) | Measured | `torch.cuda.is_available()` | System check output | **Verified** |
| **TensorRT Availability** | `False` (Module missing) | Measured | `benchmark_inference_backends.py` | `No module named 'tensorrt'` | **Verified** |
| **CPU Model** | Intel64 Family 6 Model 186 Stepping 2 | Measured | `platform.processor()` | System check output | **Verified** |
| **CPU Cores** | 8 Physical / 12 Logical | Measured | `psutil.cpu_count()` | System check output | **Verified** |
| **System RAM** | 7.73 GB | Measured | `psutil.virtual_memory()` | System check output | **Verified** |

---

## Section 2: Model Information

| Parameter / Property | Value | Type | Source File / Checkpoint | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture Name** | `ByGaitLight` | Configured | [bygait_light.py](../../models/architectures/bygait_light.py) | **Verified** |
| **Embedding Dimension** | `128` | Configured | [bygait_light.py](../../models/architectures/bygait_light.py) | **Verified** |
| **Input Tensor Shape** | `(1, 1, 64, 64)` | Configured | [model_config.yaml](../../configs/model_config.yaml) | **Verified** |
| **Output Representation** | `128-d L2-Normalized Vector` | Configured | [bygait_light.py](../../models/architectures/bygait_light.py) | **Verified** |
| **Normalization Method** | `L2 Normalization` (`F.normalize(x, p=2, dim=1)`) | Configured | [bygait_light.py](../../models/architectures/bygait_light.py) | **Verified** |
| **FLOPs / MACs** | 79.77 M FLOPs / 39.89 M MACs | Measured | `thop` v2.0.20 ($2 \times \text{MACs}$) | **Verified** |
| **Loss Function** | `ArcFace` ($s=30.0, m=0.50$) | Configured | [losses.py](../../training/losses.py) | **Verified** |
| **Training Strategy** | Subject-Disjoint (62 Train / 12 Val / 50 Test) | Configured | [subject_split.json](../../configs/subject_split.json) | **Verified** |
| **Checkpoint Path** | `runs/exp_001/best_model.pth` | Measured | File system check | **Verified** |
| **Total Checkpoint Params** | `190,207` | Measured | `torch.load('runs/exp_001/best_model.pth')` | **Verified** |
| **Trainable Params (Total)** | `189,756` (including classifier) | Measured | Checkpoint parameter count | **Verified** |
| **Backbone Embedder Params** | `126,144` (`0.126 M`) | Measured | Checkpoint `backbone.*` count | **Verified** |
| **PyTorch Checkpoint Size** | `0.73 MB` (`763,893 bytes`) | Measured | `os.path.getsize()` | **Verified** |
| **ONNX Engine File Size** | `0.48 MB` (`505,589 bytes`) | Measured | `models/engines/bygait_light.onnx` | **Verified** |

---

## Section 3: Recognition Metrics

> **Evaluation Dataset:** CASIA-B Gait Dataset  
> **Evaluation Protocol:** Subject-Disjoint Test Set (50 Unseen Subjects: 075 to 124)  
> **Calibrated Operating Threshold:** `0.9913` (Calibrated on 12 Validation Subjects via `min_eer` criterion)  
> **Evidence Source:** `runs/exp_001/evaluation_subject_disjoint/closed_set_eval_report.json` and `open_set_report.json`

### 3.1 Closed-Set Identification Performance (50 Test Subjects, 2,171 Gallery, 3,295 Probes)

| Metric | Value | Type | Protocol & Parameters | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Rank-1 Accuracy** | **86.89%** (`0.86889`) | Measured | Subject-Disjoint Closed-Set | **Verified** |
| **Rank-5 Accuracy** | **93.96%** (`0.93961`) | Measured | Subject-Disjoint Closed-Set | **Verified** |
| **Rank-10 Accuracy** | **95.75%** (`0.95751`) | Measured | Subject-Disjoint Closed-Set | **Verified** |
| **NM (Normal Walking) Rank-1** | **96.82%** (1,065 / 1,100) | Measured | Normal Walking Condition | **Verified** |
| **BG (Bag Carrying) Rank-1** | **91.23%** (999 / 1,095) | Measured | Carrying Bag Condition | **Verified** |
| **CL (Coat Wearing) Rank-1** | **72.64%** (799 / 1,100) | Measured | Wearing Coat Condition | **Verified** |
| **Precision** | **90.02%** (`0.90017`) | Measured | Operating Threshold = `0.9913` | **Verified** |
| **Recall / TAR** | **95.11%** (`0.95110`) | Measured | Operating Threshold = `0.9913` | **Verified** |
| **F1-Score** | **92.49%** (`0.92493`) | Measured | Operating Threshold = `0.9913` | **Verified** |
| **False Acceptance Rate (FAR)** | **69.91%** | Measured | Closed-set forced choice (threshold=0.9913) | **Verified** |
| **False Rejection Rate (FRR)** | **4.89%** | Measured | Closed-set forced choice (threshold=0.9913) | **Verified** |

### 3.2 Open-Set Verification & Identification (Known 075–099 vs Unknown 100–124)

| Metric | Value | Type | Protocol & Parameters | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **ROC AUC** | **0.9150** | Measured | Known (25 subjects) vs Unknown (25 subjects) | **Verified** |
| **Equal Error Rate (EER)** | **16.88%** (`0.1688`) | Measured | EER Threshold = `0.9929` | **Verified** |
| **FAR at Operating Threshold** | **36.75%** | Measured | Known vs Unknown at threshold `0.9913` | **Verified** |
| **FRR at Operating Threshold** | **6.27%** | Measured | Known vs Unknown at threshold `0.9913` | **Verified** |
| **TAR at Operating Threshold** | **93.73%** | Measured | Known vs Unknown at threshold `0.9913` | **Verified** |
| **TNR at Operating Threshold** | **63.25%** | Measured | Known vs Unknown at threshold `0.9913` | **Verified** |
| **Open-Set Precision** | **67.61%** | Measured | Known vs Unknown at threshold `0.9913` | **Verified** |
| **Open-Set F1-Score** | **78.55%** | Measured | Known vs Unknown at threshold `0.9913` | **Verified** |

---

## Section 4: Deployment Metrics & Readiness Audit

| Check / Component | Status | Details / Output | Verification Status |
| :--- | :--- | :--- | :--- |
| **Requested Backend** | `auto` | Configured default | **Verified** |
| **Active Backend** | `pytorch` | Smoke test passed with PyTorch-CPU | **Verified** |
| **ONNX Readiness** | **PASS** | ONNX engine at `models/engines/bygait_light.onnx` verified | **Verified** |
| **PyTorch Readiness** | **PASS** | Checkpoint at `runs/exp_001/best_model.pth` verified | **Verified** |
| **Doctor Status** | **`READY_FOR_CONTROLLED_CCTV_TESTING`** | Exit code: 0, 0 blocking issues, 0 warnings | **Verified** |
| **Startup Validation** | **PASS** | All 16 checks passed cleanly | **Verified** |
| **Gallery Validation** | **PASS** | 13,544 embeddings (`.npy`, `allow_pickle=False`) | **Verified** |
| **Configuration Validation** | **PASS** | All YAML config files parsed and validated | **Verified** |
| **Deployment Readiness Verdict** | **`READY_FOR_CONTROLLED_CCTV_TESTING`** | Full health report status | **Verified** |

---

## Section 5: System Performance & Latency Metrics

> **Test Hardware:** Intel64 Family 6 Model 186 (CPU Only) | **Input Resolution:** `(1, 1, 64, 64)` | **Batch Size:** `1`

| Pipeline Component | Latency (ms) | Throughput (FPS) | Measurement Backend | Evidence Source | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Embedding Forward (ONNX)** | **0.851 ms** (Mean) / 1.002 ms (p95) | **1,173.82 FPS** | ONNX Runtime CPU | `BENCHMARK_REPORT.md` | **Measured** |
| **Embedding Forward (PyTorch)** | **4.331 ms** (Mean) / 5.834 ms (p95) | **230.77 FPS** | PyTorch CPU | `BENCHMARK_REPORT.md` | **Measured** |
| **Embedding Forward (Disjoint Bulk)**| **0.270 ms** | **3,712.01 FPS** | Optimized Engine | `BENCHMARK_REPORT.md` | **Measured** |
| **Gallery Search Latency** | **12.64 ms** | N/A | 13,544 Embeddings Search | `BENCHMARK_REPORT.md` | **Measured** |
| **Full Pipeline (End-to-End)** | **90.36 ms** (Mean) / 85.77 ms (Min) | **11.07 FPS** | PyTorch Pipeline | `BENCHMARK_REPORT.md` | **Measured** |
| **Crowd Intelligence Overhead** | **3.314 ms** (Mean) / 5.062 ms (p95) | N/A (30 tracks/frame) | Crowd Subsystem | `BENCHMARK_REPORT.md` | **Measured** |
| **Person Detection Latency** | `UNABLE_TO_VERIFY` | `UNABLE_TO_VERIFY` | YOLOv8n | Integrated in Pipeline | **UNABLE_TO_VERIFY** |
| **Person Tracking Latency** | `UNABLE_TO_VERIFY` | `UNABLE_TO_VERIFY` | ByteTrack | Integrated in Pipeline | **UNABLE_TO_VERIFY** |
| **System Memory (RAM)** | **37.2 MB** | N/A | Process Footprint | In-memory measurement | **Measured** |

---

## Section 6: Test Coverage & Verification Summary

> **Execution Command:** `.\venv\Scripts\python.exe -m pytest -q --tb=short` (Timestamp: 2026-07-30T10:37:00+05:30)

| Metric | Count | Details / Notes | Verification Status |
| :--- | :---: | :--- | :--- |
| **Total Tests Executed** | **342** | Full pytest suite run | **Verified** |
| **Passed Tests** | **341** | **100.0% Pass Rate** for active tests | **Verified** |
| **Skipped Tests** | **1** | `test_rtsp_reconnection_resilience` (Physical stream unavailable) | **Verified** |
| **Failed Tests** | **0** | Zero failures | **Verified** |
| **Warnings** | **17** | Deprecation warnings (TorchScript ONNX export & ByteTrack) | **Verified** |
| **Unit Test Coverage** | **260 Passed** | `tests/unit/` | **Verified** |
| **Integration Test Coverage** | **65 Passed** | `tests/integration/` | **Verified** |
| **Security Test Coverage** | **16 Passed** | `tests/security/` | **Verified** |

---

## Section 7: Security Audit Summary

| Security Domain | Status | Compliance Details | Verification Status |
| :--- | :--- | :--- | :--- |
| `allow_pickle` Enforcement | **VERIFIED** | `allow_pickle=False` strictly enforced across all gallery loads | **Verified** |
| Credential Sanitization | **VERIFIED** | RTSP connection URLs and logs stripped of inline passwords | **Verified** |
| Live RTSP Stream Security | **UNABLE_TO_VERIFY** | No physical RTSP camera connected during test | **UNABLE_TO_VERIFY** |
| VectorStore Integrity | **VERIFIED** | Input bounds and feature type validation enabled | **Verified** |
| Secrets Management | **VERIFIED** | Zero API keys, passwords, or hardcoded secrets found | **Verified** |

---

## Section 8: Final System Scorecard

> **Rule:** Only **measured values** are included in this summary table.

| Category | Status | Evidence / Metric | Ready | Not Ready | Unable to Verify |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **System Environment** | **PASS** | Python 3.11.9, PyTorch 2.12.0, ONNX 1.28.0 | [x] | [ ] | [ ] |
| **Model Topology & FLOPs** | **PASS** | `ByGaitLight` (79.77M FLOPs / 39.89M MACs, 0.48 MB ONNX)| [x] | [ ] | [ ] |
| **Recognition (Closed-Set)** | **PASS** | Rank-1: 86.89%, Rank-5: 93.96%, Rank-10: 95.75% | [x] | [ ] | [ ] |
| **Recognition (Open-Set)** | **PASS** | ROC AUC: 0.9150, EER: 16.88% | [x] | [ ] | [ ] |
| **ONNX Inference Latency** | **PASS** | 0.851 ms / 1,173.82 FPS (Embedding-only) | [x] | [ ] | [ ] |
| **Pipeline Latency** | **PASS** | 90.36 ms / 11.07 FPS (Full end-to-end) | [x] | [ ] | [ ] |
| **Deployment Readiness** | **PASS** | `doctor.py` Status: `READY_FOR_CONTROLLED_CCTV_TESTING` | [x] | [ ] | [ ] |
| **ONNX Export Parity** | **PASS** | Max Abs Diff: 0.000000, Structural Check Passed | [x] | [ ] | [ ] |
| **Gallery Integrity** | **PASS** | 13,544 Embeddings loaded cleanly (`allow_pickle=False`)| [x] | [ ] | [ ] |
| **Test Suite Coverage** | **PASS** | 341 Passed / 1 Skipped / 0 Failed | [x] | [ ] | [ ] |
| **Security Audit** | **PASS** | `allow_pickle=False` compliant, credentials sanitized | [x] | [ ] | [ ] |
| **Standalone Detector FPS** | **UNABLE_TO_VERIFY** | No standalone detection script available | [ ] | [ ] | [x] |
| **TensorRT Acceleration** | **UNABLE_TO_VERIFY** | No CUDA/TensorRT environment detected | [ ] | [ ] | [x] |

---
**Final Verdict:** `READY_FOR_CONTROLLED_CCTV_TESTING`
