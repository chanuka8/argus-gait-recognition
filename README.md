# ARGUS AI

![ARGUS AI Gait Recognition Banner](assets/github/Gitrepo_profilepic.png)

A modular spatial-temporal gait recognition, multi-object tracking, and multi-camera surveillance intelligence framework.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/chanuka8/argus-gait-recognition/CI.yaml?branch=main)](https://github.com/chanuka8/argus-gait-recognition/actions/workflows/CI.yaml)
[![Platform: Windows / Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](.)
[![Multi-Camera](https://img.shields.io/badge/multi--camera-supported-blue.svg)](pipeline/multi_camera_recognition.py)
[![Open-Set Recognition](https://img.shields.io/badge/open--set-recognition-blue.svg)](intelligence/open_set_recognizer.py)
[![Crowd Robust](https://img.shields.io/badge/crowd-robust-blue.svg)](intelligence/crowd_robustness_manager.py)
[![Watchlist Ready](https://img.shields.io/badge/watchlist-ready-blue.svg)](intelligence/missing_person_workflow.py)
[![YOLO](https://img.shields.io/badge/YOLO-v8-blue.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pytest](https://img.shields.io/badge/tests-341%20passed-brightgreen.svg)](tests)
[![Status](https://img.shields.io/badge/status-READY__FOR__CONTROLLED__CCTV__TESTING-blue.svg)](docs/DEPLOYMENT_READINESS.md)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)

---

## Deployment Readiness & Architecture

ARGUS AI separates deployment concerns into distinct inference execution backends, pre-flight health validation tools, and externalized YAML configurations. Detailed deployment specifications are maintained in [docs/DEPLOYMENT_READINESS.md](docs/DEPLOYMENT_READINESS.md).

### Backend Status Matrix

- **PyTorch**: `✓ Reference Backend` (Default active backend for model feature extraction and verification).
- **ONNX Runtime**: `✓ Optimized Deployment Backend` (CPU-validated inference engine with atomic export and parity verification).
- **TensorRT**: `Deferred` (Framework implemented; hardware execution deferred pending CUDA + TensorRT environment installation and physical GPU validation).

> **Deployment Notice**: TensorRT is not currently claimed as a production-ready backend. PyTorch serves as the reference engine and ONNX Runtime serves as the optimized CPU deployment engine.

### Backend Selection Policy

Inference backends are selected dynamically via `configs/inference.yaml` or programmatically through `get_inference_backend()` in [models/inference/backend.py](models/inference/backend.py):

- `requested=pytorch` $\rightarrow$ Executes **PyTorch** backend directly without probing ONNX Runtime.
- `requested=onnxruntime` $\rightarrow$ Executes **ONNX Runtime** backend. If unavailable or if session creation fails, falls back to PyTorch (or raises a blocking error if `allow_fallback=false`).
- `requested=auto` $\rightarrow$ Attempts **ONNX Runtime** first, falling back to **PyTorch** if ONNX Runtime is uninstalled or model file is missing.

Every initialized backend exposes an authoritative `.metadata` dictionary returning:
- `requested_backend`: The backend requested by configuration.
- `active_backend`: The actual backend executing inference (`pytorch` or `onnxruntime`).
- `execution_provider`: Active low-level provider (`CPUExecutionProvider`, `PyTorch-CPU`, etc.).
- `allow_fallback`: Boolean flag indicating whether fallback is allowed.
- `fallback_used`: Boolean indicating whether a fallback occurred.
- `fallback_reason`: Detailed reason string if fallback was triggered.
- `attempted_backends`: Exact chain of backends attempted in order (e.g. `["onnxruntime", "pytorch"]`).

---

## ONNX Deployment & Parity Verification

The repository includes a dedicated ONNX export and verification pipeline ([scripts/export_bygait_onnx.py](scripts/export_bygait_onnx.py)):

- **✓ Stable ONNX Export**: Converts `ByGaitLight` PyTorch weights into ONNX format (`models/engines/bygait_light.onnx`) with seed determinism (`seed=42`). Missing checkpoints immediately halt export to prevent saving uninitialized weights.
- **✓ Atomic Replacement**: Exports initially to a temporary file (`bygait_light.onnx.tmp`) and performs replacement only after structural check and parity validation pass, preserving existing destination files byte-for-byte on failure.
- **✓ Numerical Parity Validation**: Compares PyTorch and ONNX embedding outputs using fixed tolerances (`rtol=1e-3`, `atol=1e-4`) and records `max_absolute_diff`.
- **✓ Metadata Report Generation**: Outputs machine-readable JSON (`outputs/reports/onnx_validation.json`) and Markdown (`outputs/reports/onnx_validation.md`) reports containing relative paths only.
- **✓ Pre-Flight Verification**: Validated automatically by system health checks ([scripts/doctor.py](scripts/doctor.py)).

---

## Deployment Health Checks (`scripts/doctor.py`)

[scripts/doctor.py](scripts/doctor.py) provides a non-destructive CLI tool for pre-flight deployment verification.

### Verified Health Checks

- **✓ Configuration**: Validates externalized YAML settings via `ConfigValidator`.
- **✓ Model Availability**: Checks PyTorch checkpoint (`runs/exp_001/best_model.pth`) and ONNX engine (`models/engines/bygait_light.onnx`).
- **✓ Gallery Validation**: Reuses safe vector store checks (`validate_gallery_files` with `allow_pickle=False`, numeric dtypes, non-finite rejection, 2D feature shapes, and label count parity).
- **✓ Backend Readiness**: Executes smoke tests on active inference backend.
- **✓ Output Directory**: Verifies writability of `outputs/reports/` using temporary probe cleanup.
- **✓ Report Generation**: Produces sanitized JSON (`outputs/reports/health_report.json`) and Markdown (`outputs/reports/health_report.md`) reports.

### Non-Destructive Safety Guarantees

`doctor.py` **DOES NOT**:
- Connect to live RTSP cameras or webcams.
- Access external network interfaces.
- Modify model, gallery, or configuration files.
- Install or upgrade Python packages.
- Execute pipeline inference or alter repository state.

---

## Verified System Metrics & Audit Reports

The repository contains an empirical, evidence-grounded audit suite generated under [docs/reports/](docs/reports/README.md).

### Verified System Snapshot

- **Subject-Disjoint Rank-1 Accuracy**: **86.89%** ([EVALUATION_REPORT.md](docs/reports/EVALUATION_REPORT.md))
- **Subject-Disjoint Rank-5 Accuracy**: **93.96%** ([EVALUATION_REPORT.md](docs/reports/EVALUATION_REPORT.md))
- **Open-Set ROC AUC**: **0.9150** | **EER**: **16.88%** ([EVALUATION_REPORT.md](docs/reports/EVALUATION_REPORT.md))
- **ONNX Embedding Latency**: **0.851 ms** / **1,173.82 FPS** (Intel CPU, ONNX Runtime, Batch Size 1, $64 \times 64$, [BENCHMARK_REPORT.md](docs/reports/BENCHMARK_REPORT.md))
- **Full Pipeline Latency**: **90.36 ms** / **11.07 FPS** (Intel CPU, PyTorch, End-to-End single person, [BENCHMARK_REPORT.md](docs/reports/BENCHMARK_REPORT.md))
- **Deployment Readiness Status**: **`READY_FOR_CONTROLLED_CCTV_TESTING`** ([DEPLOYMENT_READINESS_REPORT.md](docs/reports/DEPLOYMENT_READINESS_REPORT.md))

### Modular Audit Suite Links

- **[Master Metrics Audit Report](docs/reports/CURRENT_SYSTEM_METRICS_REPORT.md)** ([JSON](docs/reports/CURRENT_SYSTEM_METRICS_REPORT.json))
- **[Model Architecture Report](docs/reports/MODEL_ARCHITECTURE_REPORT.md)** (FLOPs: 79.77M / MACs: 39.89M, 126K backbone / 190K total params, ArcFace)
- **[Benchmark Report](docs/reports/BENCHMARK_REPORT.md)** (Isolated ONNX/PyTorch embedding, pipeline latency, crowd overhead)
- **[Evaluation Report](docs/reports/EVALUATION_REPORT.md)** (Subject-disjoint Rank-1, Rank-5, NM/BG/CL breakdowns, Open-set ROC AUC, Cross-view matrix)
- **[Deployment Readiness Report](docs/reports/DEPLOYMENT_READINESS_REPORT.md)** (System doctor health status, exit code 0, 16/16 checks passed)
- **[Inference Backend Report](docs/reports/BACKEND_REPORT.md)** (Backend selection policy `auto`, fallback cascade)
- **[Test Summary Report](docs/reports/TEST_SUMMARY_REPORT.md)** (PyTest: 341 Passed, 1 Skipped, 0 Failed across 342 tests)
- **[Security & Data Integrity Report](docs/reports/SECURITY_INTEGRITY_REPORT.md)** (`allow_pickle=False` controls, RTSP masking, VectorStore security)

---

## Capabilities & Implementation Matrix

| Capability | Implementation Status | Default State | Reference Source |
| --- | --- | --- | --- |
| **PyTorch Inference Backend** | Implemented (Reference) | Enabled (Default) | [models/inference/pytorch_backend.py](models/inference/pytorch_backend.py) |
| **ONNX Runtime Backend** | Implemented (Optimized) | Optional | [models/inference/onnx_backend.py](models/inference/onnx_backend.py) |
| **TensorRT Inference Backend** | Framework Implemented (HW Deferred) | Deferred | [models/inference/tensorrt_backend.py](models/inference/tensorrt_backend.py) |
| **System Health CLI (`doctor.py`)** | Implemented | Pre-Flight Tool | [scripts/doctor.py](scripts/doctor.py) |
| **Startup Pipeline Validator** | Implemented | Opt-In / Startup | [deployment/startup_validator.py](deployment/startup_validator.py) |
| **Deployment Readiness Reporter** | Implemented | Automated Tool | [deployment/readiness_reporter.py](deployment/readiness_reporter.py) |
| **Explainable Recognition Reports** | Implemented | Disabled by default | [intelligence/explainable_recognition_report.py](intelligence/explainable_recognition_report.py) |
| **Event Timeline Reconstruction** | Implemented | Disabled by default | [intelligence/event_timeline_reconstructor.py](intelligence/event_timeline_reconstructor.py) |
| **VectorStore Deserialization Security** | Implemented (`allow_pickle=False`) | Enabled | [storage/vector_store.py](storage/vector_store.py) |
| **Secure RTSP Credential Resolution** | Implemented (Fernet & Env Vars) | Enabled | [security_layer/credentials.py](security_layer/credentials.py) |
| **Documentation Synchronization** | Implemented (19 Package Folders) | Pre-Commit Hook | [scripts/sync_folder_readmes.py](scripts/sync_folder_readmes.py) |

---

## Features

### Core Recognition

- **YOLOv8 Person Detection**: Deep learning bounding box localization ([pipeline/detection/person_detector.py](pipeline/detection/person_detector.py)).
- **ByteTrack Multi-Object Tracking**: Track ID assignment using ByteTrack and IoU algorithms ([pipeline/steps/tracking.py](pipeline/steps/tracking.py)).
- **EMA Bounding Box Stabilization**: Exponential Moving Average coordinate filter eliminating detection jitter ([utils/box_stabilizer.py](utils/box_stabilizer.py)).
- **Silhouette Extraction**: Segmented human contour isolation and morphological cleaning ([pipeline/steps/silhouette_step.py](pipeline/steps/silhouette_step.py)).
- **Live GEI Generation**: Rolling sequence aggregation of 30 frames into 2D Gait Energy Images ([pipeline/steps/live_gei.py](pipeline/steps/live_gei.py)).
- **CNN Gait Recognition**: 256-dimensional gait signature extraction via lightweight `ByGaitLight` CNN ([models/architectures/bygait_light.py](models/architectures/bygait_light.py)).
- **Hardened Gallery Storage**: Secure biometric template storage with strict `allow_pickle=False` deserialization ([storage/vector_store.py](storage/vector_store.py)).
- **Cosine Similarity Matching**: Cosine distance evaluation between live embeddings and gallery candidates ([storage/vector_store.py](storage/vector_store.py)).

### Recognition Intelligence

- **Open-Set Recognition**: Three-state identity classification (`KNOWN`, `UNKNOWN`, `UNCERTAIN`) evaluating top-1 similarity thresholds (`known_threshold=0.85`, floor `unknown_threshold=0.70`) and candidate margin constraints (`margin_threshold=0.05`) ([intelligence/open_set_recognizer.py](intelligence/open_set_recognizer.py)).
- **Dual-Modal ReID & Gait Fusion**: Combines gait embeddings with optional appearance (ReID) embeddings using configurable weighted score fusion. Automatically falls back to gait-only recognition when ReID is unavailable or disabled ([intelligence/dual_modal_fusion.py](intelligence/dual_modal_fusion.py)).
- **Track Reliability Score**: Multi-source evidence scoring producing a normalized index in $[0.0, 1.0]$, explicitly decoupling identity confidence from physical track stability ([intelligence/track_reliability_scorer.py](intelligence/track_reliability_scorer.py)).
- **Quality-Aware Recognition**: Area, symmetry, and sharpness evaluation to gate low-quality silhouettes before feature extraction ([pipeline/steps/quality_estimator.py](pipeline/steps/quality_estimator.py), [intelligence/quality_assessment.py](intelligence/quality_assessment.py)).
- **Temporal Verification**: Sliding-window majority vote verification over consecutive frames to prevent transient misclassifications ([pipeline/steps/temporal_gait_verifier.py](pipeline/steps/temporal_gait_verifier.py)).
- **Prediction Smoothing**: Temporal history aggregation preventing rapid state oscillation ([pipeline/steps/temporal_gait_verifier.py](pipeline/steps/temporal_gait_verifier.py)).

### Multi-Camera & Crowd Intelligence

- **Multi-Camera Tracking**: Global track assignment (`GTRACK-XXXX`) and multi-stream trajectory management ([intelligence/cross_camera_tracker.py](intelligence/cross_camera_tracker.py)).
- **Camera Transition Modeling**: Directed topology graph enforcing expected travel-time windows $[T_{min}, T_{max}]$, transition probabilities, entry/exit zones, and candidate tie resolution ([intelligence/camera_transition_model.py](intelligence/camera_transition_model.py)).
- **Spatial-Temporal Camera Topology Auto-Learning**: Learns camera transition statistics from validated cross-camera observations and synchronizes qualified learned routes into active transition models ([intelligence/camera_topology_learner.py](intelligence/camera_topology_learner.py)).
- **Multi-Camera Evidence Fusion**: Accumulates observations across multiple cameras, suppresses duplicate evidence, and produces unified cross-camera identity confidence ([intelligence/multi_camera_evidence_fusion.py](intelligence/multi_camera_evidence_fusion.py)).
- **Cross-Camera Identity Persistence**: Accumulated score decay ($\alpha=0.90$) and duplicate alert suppression across streams ([intelligence/identity_persistence.py](intelligence/identity_persistence.py)).
- **Crowd-Robust Detection**: Adaptive gating, IoU tuning, and threshold adjustments under high spatial density ([intelligence/crowd_robustness_manager.py](intelligence/crowd_robustness_manager.py)).

### Operational Intelligence & Forensic Trace Analysis

- **Explainable Recognition Reports**: Generates JSON, CSV, and Markdown trace reports detailing identity decision logic, similarity scores, candidate margins, track reliability, quality metrics, and deferral flags ([intelligence/explainable_recognition_report.py](intelligence/explainable_recognition_report.py)).
- **Event Timeline Reconstruction**: Cross-camera chronological event trajectory accumulator for global tracks and watchlist targets ([intelligence/event_timeline_reconstructor.py](intelligence/event_timeline_reconstructor.py)).
- **Real-Time Watchlist Integration**: Dynamic target identity registration, priority category routing, and instant match notification triggers ([intelligence/missing_person_workflow.py](intelligence/missing_person_workflow.py)).
- **Crowd Density & Occlusion Analysis**: Real-time person count, area occupancy classification (`LOW`, `MODERATE`, `HIGH`, `SEVERE`), and overlap ratio assessment ([intelligence/crowd_density_estimator.py](intelligence/crowd_density_estimator.py), [intelligence/crowd_occlusion_analyzer.py](intelligence/crowd_occlusion_analyzer.py)).
- **Recognition Deferral & Track Recovery**: Deferring low-confidence or heavily occluded decisions and recovering lost tracks ([intelligence/recognition_deferral_engine.py](intelligence/recognition_deferral_engine.py), [intelligence/track_recovery_manager.py](intelligence/track_recovery_manager.py)).

### Performance, Security & Infrastructure

- **Pluggable Inference Backends**: Unified factory (`get_inference_backend()`) supporting PyTorch (reference), ONNX Runtime (optimized), and TensorRT (deferred) with automatic PyTorch fallback, attempted backend chain reporting, and sanitized log warnings ([models/inference/backend.py](models/inference/backend.py)).
- **Hardened Vector Store**: Complete security remediation enforcing `allow_pickle=False`, rejecting object arrays, and validating numeric dtypes, dimensions, and shape consistency ([storage/vector_store.py](storage/vector_store.py)).
- **Secure RTSP Credential Management**: Fernet-encrypted credential storage, environment variable mapping, per-camera credential resolution, and automatic stream URL sanitization in logs ([security_layer/credentials.py](security_layer/credentials.py)).
- **Automated Documentation Synchronization**: Automated README table synchronization ([scripts/sync_folder_readmes.py](scripts/sync_folder_readmes.py)) covering all 19 package folders including [scripts/README.md](scripts/README.md) (CLI reference, metadata tables, dependency graph, execution order, change impact, safety classification), atomic writes, cross-platform pre-commit hook installer (`scripts/install_git_hooks.py`), and CI freshness check ([.github/workflows/readme_sync_check.yml](.github/workflows/readme_sync_check.yml)).

---

## Technology Stack

- **Python**: Core implementation language (Python 3.11+).
- **PyTorch**: Deep learning backend for `ByGaitLight` CNN feature extraction and vector operations.
- **ONNX Runtime**: Accelerated CPU inference engine for PyTorch exported models.
- **YOLOv8**: Object detection network for high-precision human bounding box localization.
- **OpenCV**: Computer vision operations, silhouette segmentation, image processing, and display rendering.
- **NumPy**: Numerical array computations, vector distance calculation, and GEI accumulation.
- **ByteTrack**: Multi-object tracking algorithm for consistent local track ID assignment.
- **FastAPI & Uvicorn**: Lightweight HTTP REST API server engine and schema definitions ([api/server.py](api/server.py)).
- **cryptography**: Encrypted credential management using Fernet.
- **YAML**: Configuration serialization via PyYAML.

---

## System Architecture

```mermaid
graph TD
    RTSP[Camera Streams / RTSP] --> CS[CameraService / WorkerPool]
    CS --> MSE[MultiStreamEngine]
    MSE --> MCR[MultiCameraRecognition Pipeline]
    
    subgraph Modular Step Pipeline
        MCR --> DET[YOLOv8 Detection]
        DET --> TRK[ByteTrack Tracking]
        TRK --> EMA[EMA Stabilization]
        EMA --> SIL[Silhouette Extraction]
        SIL --> GEI[Live GEI Accumulator]
        GEI --> QUA[Quality Estimator]
        QUA --> FEA[Gait Embedding - ByGaitLight]
        FEA --> APP[Appearance ReID Embedding - OSNet]
        APP --> FUS[Dual-Modal Score Fusion]
        FUS --> MAT[Gallery Search - VectorStore]
    end

    MAT --> OSR[Open-Set Recognition]
    OSR --> TRS[Track Reliability Scorer]
    TRS --> WLM[Real-Time Watchlist Manager]
    WLM --> CIS[Crowd Intelligence System]
    
    subgraph Crowd Intelligence
        CIS --> COA[Crowd Occlusion Analyzer]
        CIS --> RDE[Recognition Deferral]
        CIS --> TRM[Track Recovery]
    end

    CIS --> CCT[Cross-Camera Tracker & Transition Model]
    CCT --> IDP[Identity Persistence Engine]
    IDP --> REP[Detection Output - Reporter / Renderer / Evidence]
```

---

## Security Infrastructure

### Secure RTSP Credential Resolution

ARGUS AI supports secure RTSP camera authentication without storing plaintext credentials inside repository configuration files. Credentials are resolved dynamically at runtime in priority order:

1. **Environment Variables**: Per-camera (`ARGUS_CAMERA_<ID>_USERNAME`, `ARGUS_CAMERA_<ID>_PASSWORD`) or global fallback (`ARGUS_RTSP_USERNAME`, `ARGUS_RTSP_PASSWORD`).
2. **Encrypted Credential Store**: Local credential store (`configs/credentials.enc`) encrypted using Fernet (`cryptography` library).
3. **Legacy Plaintext**: Plaintext fallback (disabled by default; requires `ARGUS_LEGACY_ALLOW_PLAINTEXT_CREDS=true`).

- **Log Sanitization**: Automatic masking of RTSP credentials (`rtsp://***:***@host:port/path`) across all system logs, reports, and CLI outputs.

### Hardened Vector Store Deserialization

Biometric gallery storage in `VectorStore` ([storage/vector_store.py](storage/vector_store.py)) has been fully hardened against arbitrary code execution vulnerabilities:

- **`allow_pickle=False` Enforcement**: All `np.load()` calls strictly prohibit pickle deserialization.
- **Object-Array Rejection**: Rejects any array containing object dtypes (`dtype == object` or `kind == "O"`).
- **Strict Data Validation**: Validates numeric feature dtypes (`np.issubdtype(dtype, np.number)`), 2D feature matrix dimensions `(N, D)`, 1D label vector shape `(N,)`, feature-to-label count parity, and file corruption.

---

## Externalized Configuration

System parameters and deployment settings are externalized across YAML configuration files under [configs/](configs/):

- **[configs/system.yaml](configs/system.yaml)**: Controls thread pool sizes, logging levels, storage output directories, and API binding parameters.
- **[configs/inference.yaml](configs/inference.yaml)**: Controls inference backend selection (`pytorch`, `onnxruntime`, `auto`), fallback policies, open-set thresholds (`known_threshold=0.85`, `unknown_threshold=0.70`), ReID parameters, quality bounds, temporal verification, watchlist routing, explainable reports, and event timeline reconstruction.
- **[configs/cameras.yaml](configs/cameras.yaml)**: Controls RTSP camera endpoints (`host`, `port`, `path`), environment variable credentials (`username_env`, `password_env`), resolution, framerates, worker pool limits, and ONVIF discovery parameters.

> **Architecture Principle**: Recognition thresholds, file paths, backend selection options, and camera stream settings must be controlled through YAML files under `configs/` rather than hardcoded in source logic.

---

## Output Directory Structure

The system uses a standardized output layout:

```text
outputs/
├── reports/
│   ├── health_report.json # System health diagnostic reports
│   ├── deployment_readiness.json # Deployment readiness reports
│   ├── onnx_validation.json # ONNX export parity reports
│   ├── explainable/       # Explainable recognition trace reports (JSON/CSV/MD)
│   ├── timelines/         # Event timeline trajectory exports
│   ├── benchmark/         # Inference backend performance benchmark reports
│   └── evaluation/        # Model evaluation & threshold sweep results
├── logs/
│   ├── system/            # System & camera component log files
│   └── events/            # Recognition and threat alert CSV logs
├── monitoring/            # Watchdog and health monitoring metrics
├── media/
│   ├── detections/        # Detection snapshots and bounding box crops
│   └── snapshots/         # Event snapshot imagery
├── watchlist/             # Watchlist target gallery templates and metadata
└── temporary/             # Transient pipeline processing buffers
```

---

## Project Structure & Documentation Automation

```text
ARGUS_AI/
├── api/                   # FastAPI server implementation and request schemas
├── configs/               # System, inference, camera, and GEI YAML configurations
├── deployment/            # Startup validator and deployment readiness reporter
├── evaluation/            # Open-set, cross-view, dataset split, and leakage evaluators
├── intelligence/          # Open-set recognizer, track reliability, watchlist, crowd intelligence, transition model
├── models/                # ByGaitLight CNN, OSNet ReID backbone, inference backends, gallery storage
├── monitoring/            # Camera health monitor, watchdog daemon, logging infrastructure
├── pipeline/              # Live, multi-camera, video, and folder recognition orchestrators
│   └── steps/             # Modular pipeline steps (detection, tracking, silhouette, GEI, quality)
├── security_layer/        # Security decision engine, audit logging, encrypted credentials
├── services/              # Camera discovery, ONVIF client, worker threads, service manager
├── storage/               # Hardened vector store, evidence manager, dataset loader
├── streaming/             # Multi-stream engine, load balancer, buffer queue, camera scheduler
├── tests/                 # 341 automated tests across unit, integration, and security suites
└── utils/                 # Display renderer, detection reporter, alert manager, box stabilizer
```

### Documentation Automation

ARGUS AI maintains automated folder-level documentation across all 19 package and tool directories:

- **Folder README Sync ([scripts/sync_folder_readmes.py](scripts/sync_folder_readmes.py))**: Automatically scans source files and updates markdown tables between `<!-- BEGIN SYNC: KEY_MODULES -->` comment markers across all package directories.
- **First-Class Scripts Module ([scripts/README.md](scripts/README.md))**: Fully auto-generated and self-maintaining documentation module for the `scripts/` folder. Includes script inventory (43 active scripts), CLI Reference (collapsible tables for 20 CLI-enabled scripts), script metadata table, Mermaid dependency graph, execution pipeline order, change impact outputs, safety classifications, and cross-references.
- **Documentation Index ([docs/README_INDEX.md](docs/README_INDEX.md))**: Central directory of relative links for all package and utility READMEs.
- **Git Pre-Commit Hook (`scripts/install_git_hooks.py`)**: Automatically syncs and stages README changes prior to commits.
- **CI Freshness Workflow (`.github/workflows/readme_sync_check.yml`)**: Read-only GitHub Actions workflow enforcing documentation alignment on PRs and main branch pushes.

---

## Installation & Setup

### Prerequisites

- **Python**: 3.11 or higher
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **GPU** (Optional): CUDA-compatible GPU for PyTorch execution

### Virtual Environment Setup

#### Windows (PowerShell)

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/install_git_hooks.py
```

#### Linux / macOS (Bash)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/install_git_hooks.py
```

---

## Usage

All primary workflows are accessible via [cli.py](cli.py).

### Deployment Pre-Flight Health Check

```bash
python scripts/doctor.py
```

### Multi-Camera Recognition Stream

```bash
python cli.py --mode multi-camera
```

### Video File Recognition

```bash
python cli.py --mode recognize-video --video "path/to/sample.mp4"
```

### Export ONNX Engine & Validate Parity

```bash
python scripts/export_bygait_onnx.py --output-path models/engines/bygait_light.onnx
```

### Run Inference Backend Benchmark

```bash
python scripts/benchmark_inference_backends.py --samples 50
```

### Synchronize Documentation READMEs

```bash
python scripts/sync_folder_readmes.py --check
```

---

## Repository Validation & Testing

Verification is enforced via automated test suites, code linters, documentation checks, and health diagnostics:

```bash
# 1. Deployment health pre-flight check
python scripts/doctor.py

# 2. Linting and code style verification
python -m ruff check .

# 3. Documentation synchronization check
python scripts/sync_folder_readmes.py --check

# 4. Full repository test suite (341 automated tests)
python -m pytest -q
```

### Verified Validation Status

- **✓ Linter Compliance**: `ruff check .` passed with 0 errors.
- **✓ Full Test Suite**: **341 passed**, 1 skipped (342 total test items).
- **✓ Documentation Alignment**: `sync_folder_readmes.py --check` clean across all 19 package & script folders.
- **✓ Deployment Readiness**: `doctor.py` status `READY_FOR_CONTROLLED_CCTV_TESTING` (Exit Code 0).

---

## Project Status

Current qualitative implementation status based on evidence-based health checks:

`READY_FOR_CONTROLLED_CCTV_TESTING`

- [x] YOLOv8 person detection & ByteTrack multi-object tracking
- [x] Silhouette segmentation and Live GEI 30-frame sequence builder
- [x] `ByGaitLight` CNN gait embedding model
- [x] Hardened Vector Store (`allow_pickle=False`, numeric validation)
- [x] Open-Set Recognition (`KNOWN`, `UNKNOWN`, `UNCERTAIN` classification)
- [x] Pluggable Inference Backends (PyTorch reference, ONNX Runtime optimized, TensorRT framework deferred)
- [x] Stable ONNX export, atomic replacement, and numerical parity validation
- [x] Non-destructive deployment health checker CLI ([scripts/doctor.py](scripts/doctor.py))
- [x] Pre-flight pipeline startup validator ([deployment/startup_validator.py](deployment/startup_validator.py))
- [x] Deployment readiness reporter ([deployment/readiness_reporter.py](deployment/readiness_reporter.py))
- [x] Externalized YAML configurations ([configs/](configs/))
- [x] Secure RTSP credential storage & log URL sanitization ([security_layer/credentials.py](security_layer/credentials.py))
- [x] Automated README Documentation Synchronization (`scripts/sync_folder_readmes.py`)
- [x] 341 automated tests passing with 0 failures

---

## Future Roadmap

- **CUDA Installation & Target Hardware Validation**: Finalize CUDA runtime and TensorRT library setup on target GPU deployment environments.
- **TensorRT Engine Generation & Execution**: Validate end-to-end TensorRT engine compilation (`build_tensorrt_engine.py`) and execution parity on target hardware.
- **Docker Deployment & Production Containerization**: Create containerized deployment blueprints for isolated multi-camera worker nodes.
- **Real CCTV Stream Hardware Validation**: Field validation on live RTSP network cameras under real physical surveillance conditions.
- **Long-Duration Continuous Testing**: 24/7 continuous operation testing for long-term memory stability and camera worker pool resilience.

---

## Cross References & Key Documentation

- **Deployment Readiness Specification**: [docs/DEPLOYMENT_READINESS.md](docs/DEPLOYMENT_READINESS.md)
- **Scripts Module Reference**: [scripts/README.md](scripts/README.md)
- **Documentation Index**: [docs/README_INDEX.md](docs/README_INDEX.md)
- **Architecture Specification**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Maintainer

### Chanuka Sandun

Undergraduate in Cyber Security

Developer of the ARGUS AI Gait Recognition Module

- GitHub: [github.com/chanuka8](https://github.com/chanuka8)
- LinkedIn: [linkedin.com/in/chanukasandun](https://www.linkedin.com/in/chanukasandun/)
