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
[![Pytest](https://img.shields.io/badge/tests-313%20passed-brightgreen.svg)](tests)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)

---

## Capabilities & Implementation Matrix

| Capability | Implementation Status | Default State | Reference Source |
| --- | --- | --- | --- |
| **PyTorch Inference Backend** | Implemented | Enabled (Default) | [models/inference/pytorch_backend.py](models/inference/pytorch_backend.py) |
| **ONNX Runtime Backend** | Implemented & CPU-Validated | Optional | [models/inference/onnx_backend.py](models/inference/onnx_backend.py) |
| **TensorRT Inference Backend** | Framework Implemented (HW Validation Pending) | Optional | [models/inference/tensorrt_backend.py](models/inference/tensorrt_backend.py) |
| **Explainable Recognition Reports** | Implemented | Disabled by default | [intelligence/explainable_recognition_report.py](intelligence/explainable_recognition_report.py) |
| **Event Timeline Reconstruction** | Implemented | Disabled by default | [intelligence/event_timeline_reconstructor.py](intelligence/event_timeline_reconstructor.py) |
| **VectorStore Deserialization Security** | Implemented (`allow_pickle=False`) | Enabled | [storage/vector_store.py](storage/vector_store.py) |
| **Secure RTSP Credential Resolution** | Implemented (Fernet & Env Vars) | Enabled | [security_layer/credentials.py](security_layer/credentials.py) |
| **Documentation Synchronization** | Implemented (19 Folder & Script READMEs) | Pre-Commit Hook | [scripts/sync_folder_readmes.py](scripts/sync_folder_readmes.py) |

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

- **Pluggable Inference Backends**: Unified factory (`get_inference_backend()`) supporting PyTorch (reference), ONNX Runtime, and TensorRT with automatic PyTorch fallback, attempted backend chain reporting, and sanitized log warnings ([models/inference/backend.py](models/inference/backend.py)).
- **Hardened Vector Store**: Complete security remediation enforcing `allow_pickle=False`, rejecting object arrays, and validating numeric dtypes, dimensions, and shape consistency ([storage/vector_store.py](storage/vector_store.py)).
- **Secure RTSP Credential Management**: Fernet-encrypted credential storage, environment variable mapping, per-camera credential resolution, and automatic stream URL sanitization in logs ([security_layer/credentials.py](security_layer/credentials.py)).
- **Automated Documentation Synchronization**: Automated README table synchronization (`scripts/sync_folder_readmes.py`) covering all 19 package folders including `scripts/README.md` (CLI reference, metadata tables, dependency graph, execution order, change impact, safety classification), atomic writes, cross-platform pre-commit hook installer (`scripts/install_git_hooks.py`), and CI freshness check ([.github/workflows/readme_sync_check.yml](.github/workflows/readme_sync_check.yml)).

---

## Technology Stack

- **Python**: Core implementation language (Python 3.11+).
- **PyTorch**: Deep learning backend for `ByGaitLight` CNN feature extraction and vector operations.
- **ONNX Runtime**: Optional accelerated inference engine for PyTorch exported models.
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

- **Log Sanitization**: Automatic masking of RTSP credentials (`rtsp://***:***@host:port/path`) across all system logs.

### Hardened Vector Store Deserialization

Biometric gallery storage in `VectorStore` ([storage/vector_store.py](storage/vector_store.py)) has been fully hardened against arbitrary code execution vulnerabilities:

- **`allow_pickle=False` Enforcement**: All `np.load()` calls strictly prohibit pickle deserialization.
- **Object-Array Rejection**: Rejects any array containing object dtypes (`dtype == object` or `kind == "O"`).
- **Strict Data Validation**: Validates numeric feature dtypes (`np.issubdtype(dtype, np.number)`), 2D feature matrix dimensions `(N, D)`, 1D label vector shape `(N,)`, feature-to-label count parity, and file corruption.

---

## Explainability & Event Timeline Reconstruction

### Explainable Recognition Reports

`ExplainableRecognitionReporter` ([intelligence/explainable_recognition_report.py](intelligence/explainable_recognition_report.py)) generates operational trace reports explaining identity decisions.

- **Triggers**: Confirmed identity, deferred recognition, watchlist match, identity change, or manual export.
- **Export Formats**: JSON, CSV, and Markdown saved under `outputs/reports/explainable/`.
- **Privacy & Security**: Excludes raw 256D feature vectors and system credentials.
- **Default State**: Disabled by default (`explainable_reports.enabled: false` in `configs/inference.yaml`).

### Event Timeline Reconstruction

`EventTimelineReconstructor` ([intelligence/event_timeline_reconstructor.py](intelligence/event_timeline_reconstructor.py)) tracks multi-camera spatial-temporal target journeys.

- **Triggers**: Track creation, camera enter/exit, identity change, watchlist match, track recovery, and track close.
- **Export Paths**: Formatted JSON/CSV timelines saved under `outputs/reports/timelines/`.
- **Default State**: Disabled by default (`event_timeline.enabled: false` in `configs/inference.yaml`).

> **Operational Notice**: Explainable reports and event timelines provide internal system diagnostic and audit traces. They do not constitute court-certified or legally binding evidence.

---

## Inference Backends & Benchmark Scope

### Pluggable Inference Framework

The framework ([models/inference/backend.py](models/inference/backend.py)) allows seamless backend selection via `configs/inference.yaml`:

- **PyTorch (`pytorch`)**: Default reference backend executing `ByGaitLight` PyTorch model directly.
- **ONNX Runtime (`onnxruntime`)**: Optional backend executing exported ONNX model (`scripts/export_bygait_onnx.py`).
- **TensorRT (`tensorrt`)**: Optional engine framework (`scripts/build_tensorrt_engine.py`). Hardware execution is pending target CUDA/TensorRT environment validation.
- **Auto Selection (`auto`)**: Attempts TensorRT $\rightarrow$ ONNX Runtime $\rightarrow$ PyTorch, recording selection chains (`attempted_backends`, `selection_fallback_used`, `fallback_reason`).

### Benchmark Scope & Scope Disclaimer

`scripts/benchmark_inference_backends.py` provides backend throughput and numerical parity evaluation against the PyTorch reference:

> **Benchmark Scope Disclaimer**: Benchmark FPS and latency metrics measure core model forward embedding inference on synthetic $64 \times 128$ GEI tensors only (`measurement_scope: "embedding_only_synthetic_gei"`). They exclude video decoding, YOLO person detection, ByteTrack tracking, silhouette segmentation, vector gallery matching, report generation, and multi-stream pipeline overhead. Fallback measurements reflect active backend (PyTorch) execution.

---

## Documentation Automation & Scripts Reference

ARGUS AI maintains automated folder-level documentation across all 19 package and tool directories:

- **Folder README Sync (`scripts/sync_folder_readmes.py`)**: Automatically scans source files and updates markdown tables between `<!-- BEGIN SYNC: KEY_MODULES -->` comment markers across all package directories.
- **First-Class Scripts Module (`scripts/README.md`)**: Automatically generated and self-maintaining documentation module for the `scripts/` folder ([scripts/README.md](scripts/README.md)). Includes script inventory (43 active scripts), CLI Reference (collapsible tables for 20 CLI-enabled scripts), script metadata table, Mermaid dependency graph, execution pipeline order, change impact outputs, safety classifications, and cross-references.
- **Manual Content Preservation**: Preserves all manual prose, headings, architecture diagrams, and data flow sections outside markers.
- **Atomic Writes**: Uses temporary files and `os.replace()` to ensure zero file corruption on interrupted runs.
- **Central Index (`docs/README_INDEX.md`)**: Maintains relative links for all package and utility READMEs.
- **Git Pre-Commit Hook (`scripts/install_git_hooks.py`)**: Automatically syncs and stages README changes prior to commits. Supports both Windows (`venv/Scripts/python.exe`) and POSIX (`venv/bin/python`) environments.
- **CI Freshness Workflow (`.github/workflows/readme_sync_check.yml`)**: Read-only GitHub Actions workflow enforcing documentation alignment on PRs and main branch pushes (`python scripts/sync_folder_readmes.py --check`).

> **Developer Note**: Run `python scripts/install_git_hooks.py` once after cloning to activate local pre-commit documentation synchronization.

---

## Output Directory Structure

The system uses a standardized output layout (migrated via `scripts/migrate_output_layout.py`):

```text
outputs/
├── reports/
│   ├── explainable/       # Explainable recognition trace reports (JSON/CSV/MD)
│   ├── timelines/         # Event timeline trajectory exports
│   ├── benchmark/         # Inference backend performance benchmark reports
│   ├── evaluation/        # Model evaluation & threshold sweep results
│   └── exports/           # Manual forensic report exports
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

## Project Structure

```text
ARGUS_AI/
├── api/                   # FastAPI server implementation and request schemas
├── configs/               # System, inference, camera, and GEI YAML configurations
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
├── tests/                 # 313 automated tests across unit, integration, and security suites
└── utils/                 # Display renderer, detection reporter, alert manager, box stabilizer
```

---

## Installation

### Prerequisites

- **Python**: 3.11 or higher
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **GPU** (Optional): CUDA-compatible GPU for accelerated PyTorch/ONNX execution

### Virtual Environment & Interpreter Setup

The workspace is preconfigured via `.vscode/settings.json` (`"python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"`). Launching a new PowerShell terminal inside VS Code automatically activates the repository virtual environment (`Python 3.11.9`).

#### Automatic Activation (PowerShell)

The tracked activation script (`scripts/activate_venv.ps1`) handles environment setup:
- Resolves workspace paths cleanly.
- Validates the `venv` interpreter (`venv/Scripts/python.exe`).
- Deactivates foreign virtual environments if present.
- Sets prompt context cleanly without side effects.

#### Manual Activation Commands

##### Windows (PowerShell)

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
# Or using the auto-activation script directly:
powershell -ExecutionPolicy Bypass -File scripts/activate_venv.ps1
pip install -r requirements.txt
python scripts/install_git_hooks.py
```

##### Linux / macOS (Bash)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/install_git_hooks.py
```

---

## Configuration

System operations are configured via YAML files in `configs/`:

- **[configs/cameras.yaml](configs/cameras.yaml)**: Defines RTSP camera endpoints (`host`, `port`, `path`), environment variable credentials (`username_env`, `password_env`), resolution, framerates, worker pool limits, and ONVIF discovery parameters.
- **[configs/inference.yaml](configs/inference.yaml)**: Controls matching policy thresholds, open-set recognition, inference backend selection (`pytorch`, `onnxruntime`, `tensorrt`, `auto`), explainable reports, event timeline reconstruction, track reliability scoring, watchlist integration, crowd robustness, crowd intelligence, box stability EMA, ReID settings, GEI quality parameters, temporal verification, and camera transition topology.

---

## Usage

All primary workflows are accessible via [cli.py](cli.py).

### System Health Check

```bash
python cli.py --mode health
```

### Multi-Camera Recognition Stream

```bash
python cli.py --mode multi-camera
```

### Video File Recognition

```bash
python cli.py --mode recognize-video --video "path/to/sample.mp4"
```

### Build Biometric Gallery

```bash
python cli.py --mode gallery
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

## Testing & Quality Assurance

Verification is enforced via automated test suites, bytecode compilation, documentation checks, and code style compliance:

```bash
# 1. Bytecode compilation check
python -m compileall -x "venv|\.venv" .

# 2. Linting and code style verification
ruff check .

# 3. Documentation synchronization check
python scripts/sync_folder_readmes.py --check

# 4. Full repository test suite (313 automated tests)
pytest -q
```

### Current Test & Quality Status

- **Ruff Linting**: Pass (`ruff check .` compliant with 0 errors)
- **Bytecode Compilation**: Pass (`python -m compileall` check clean)
- **Documentation Alignment**: Pass (`sync_folder_readmes.py --check` clean across all 19 package & script folders)
- **Automated Tests**: **313 automated tests** (100% passing across unit, integration, and security suites)
- **Warnings**: 1 non-blocking warning (`ByteTrack` deprecation warning from upstream tracking package)

---

## Development & Code Quality

- **Code Style**: Checked using `ruff`. Run `ruff check .` to verify formatting compliance (`All checks passed!`).
- **Type Annotations**: Comprehensive type hints used across `intelligence/`, `pipeline/`, `services/`, `storage/`, and `models/`.
- **Thread Safety**: Mutable shared state objects use `threading.Lock()` wrappers to guarantee thread-safe operations across multi-camera worker threads.
- **Modular Pipeline Design**: Custom processing steps implement explicit step interfaces to maintain low coupling.

---

## Limitations

- **Gait View Requirements**: Requires clear side or diagonal walking profiles (~30 consecutive frames) to construct valid GEIs.
- **Webcam Constraints**: Stationary upper-body webcam feeds do not supply leg/stride movement signatures required for gait recognition.
- **Legacy Plaintext Fallback**: Embedded plaintext credentials in configuration files are disabled by default and require an explicit legacy override.
- **TensorRT Hardware Validation**: TensorRT engine builder is implemented; hardware execution is pending deployment environment validation.

---

## Project Status

### Implemented

- [x] YOLOv8 person detection & ByteTrack multi-object tracking
- [x] Exponential Moving Average (EMA) bounding box coordinate stabilization
- [x] Silhouette segmentation and Live GEI 30-frame sequence builder
- [x] `ByGaitLight` lightweight CNN gait embedding model
- [x] Hardened Vector Store (`allow_pickle=False`, numeric validation, object array rejection)
- [x] Open-Set Recognition (`KNOWN`, `UNKNOWN`, `UNCERTAIN` classification with margin logic)
- [x] Track Reliability Scorer (`TrackReliabilityScorer` multi-evidence index)
- [x] Real-Time Watchlist Integration (`WatchlistManager` / `MissingPersonWorkflow`)
- [x] Crowd Intelligence System (Crowd Density Estimator, Occlusion Analyzer, Recognition Deferral Engine, Track Recovery)
- [x] `CrossCameraTracker` global track ID management & directed `CameraTransitionModel` topology
- [x] Dual-Modal ReID & Gait Fusion (`intelligence/dual_modal_fusion.py`)
- [x] Multi-Camera Evidence Fusion (`intelligence/multi_camera_evidence_fusion.py`)
- [x] Spatial-Temporal Camera Topology Auto-Learning (`intelligence/camera_topology_learner.py`)
- [x] Explainable Recognition Reports & Event Timeline Reconstruction (`intelligence/`)
- [x] Pluggable Inference Backends (PyTorch reference, ONNX Runtime, TensorRT framework, Auto selection)
- [x] `IdentityPersistence` score decay & alert cooldown suppression
- [x] `QualityEstimator` & `TemporalGaitVerifier` filtering steps
- [x] `MultiStreamEngine`, `WorkerPool`, `LoadBalancer`, and `Watchdog`
- [x] ONVIF discovery & vendor adapters
- [x] Secure RTSP credential storage & log URL sanitization ([security_layer/credentials.py](security_layer/credentials.py))
- [x] Automated README Documentation Synchronization & Pre-Commit Hook (`scripts/sync_folder_readmes.py` covering all 19 folders including `scripts/README.md`)
- [x] 313 automated tests passing with 0 failures

---

## Repository Statistics

- **Primary Language**: Python (100%)
- **Core Packages & Tooling**: `pipeline`, `intelligence`, `models`, `services`, `streaming`, `storage`, `monitoring`, `evaluation`, `security_layer`, `utils`, `api`, `scripts` (12 packages/folders)
- **Automated Tests**: **313 automated tests**
- **Linter Status**: **0 errors** (`ruff check .` compliant)

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
