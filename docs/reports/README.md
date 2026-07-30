# ARGUS AI — Audit Reports Directory & Suite Index

**Last Verified Commit:** `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`  
**Working Tree Status:** Dirty (Uncommitted documentation/report suite additions)  
**Audit Policy:** Zero False Positive Evidence-Based Reporting Policy

---

## Overview & Scope Statement

This directory contains the official, evidence-grounded performance and audit report suite for the ARGUS AI Gait Recognition System. All reported values were obtained directly from live test suite executions (`pytest`), diagnostic scripts (`scripts/doctor.py`), benchmark tools, checkpoint inspections, and formal model evaluation exports.

> [!IMPORTANT]
> **Audit Suite Guidelines & Disclaimers:**
> 1. **Evidence Snapshots:** All reports represent empirical snapshots taken at commit `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`.
> 2. **Hardware Dependency:** Inference latencies and frame rates depend on host CPU/GPU hardware and runtime execution providers.
> 3. **Protocol Boundaries:** Recognition metrics apply to the documented CASIA-B subject-disjoint evaluation protocol.
> 4. **Field Validation:** Real-world CCTV performance remains subject to outdoor controlled field testing.

---

## Report Suite Directory Index

| Report Document | Primary Purpose | Verified Commit | Available Formats |
| :--- | :--- | :---: | :---: |
| **[Master Audit Report](CURRENT_SYSTEM_METRICS_REPORT.md)** | Complete system audit summary and scorecard | `d9aefed6c95` | [Markdown](CURRENT_SYSTEM_METRICS_REPORT.md) \| [JSON](CURRENT_SYSTEM_METRICS_REPORT.json) |
| **[Model Architecture Report](MODEL_ARCHITECTURE_REPORT.md)** | Topology, parameter counts, FLOPs/MACs ($79.77\text{M}$), ArcFace | `d9aefed6c95` | [Markdown](MODEL_ARCHITECTURE_REPORT.md) \| [JSON](MODEL_ARCHITECTURE_REPORT.json) |
| **[Benchmark Report](BENCHMARK_REPORT.md)** | ONNX Runtime (0.851 ms), PyTorch, pipeline (90.36 ms), crowd overhead | `d9aefed6c95` | [Markdown](BENCHMARK_REPORT.md) \| [JSON](BENCHMARK_REPORT.json) |
| **[Evaluation Report](EVALUATION_REPORT.md)** | Subject-disjoint Rank-1 (86.89%), Open-Set ROC AUC (0.9150), 11x11 Cross-View | `d9aefed6c95` | [Markdown](EVALUATION_REPORT.md) \| [JSON](EVALUATION_REPORT.json) |
| **[Deployment Readiness Report](DEPLOYMENT_READINESS_REPORT.md)** | Doctor health status (`READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING`), ONNX parity | `d9aefed6c95` | [Markdown](DEPLOYMENT_READINESS_REPORT.md) \| [JSON](DEPLOYMENT_READINESS_REPORT.json) |
| **[Backend Report](BACKEND_REPORT.md)** | Backend selection policy (`auto`), provider discovery, fallback cascade | `d9aefed6c95` | [Markdown](BACKEND_REPORT.md) \| [JSON](BACKEND_REPORT.json) |
| **[Test Summary Report](TEST_SUMMARY_REPORT.md)** | PyTest summary (341 Passed, 1 Skipped, 0 Failed across 342 tests) | `d9aefed6c95` | [Markdown](TEST_SUMMARY_REPORT.md) \| [JSON](TEST_SUMMARY_REPORT.json) |
| **[Security & Integrity Report](SECURITY_INTEGRITY_REPORT.md)** | `allow_pickle=False` controls, RTSP masking, VectorStore security | `d9aefed6c95` | [Markdown](SECURITY_INTEGRITY_REPORT.md) \| [JSON](SECURITY_INTEGRITY_REPORT.json) |
