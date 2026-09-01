# ARGUS AI

![ARGUS AI Gait Recognition Banner](assets/github/Gitrepo_profilepic.png)

### AI-Powered Gait Recognition and Surveillance Intelligence System

ARGUS AI is a modular computer vision and deep learning platform designed for real-time human gait recognition, dual-modal appearance re-identification (ReID), and multi-camera surveillance intelligence. The system extracts biometric walking kinematics from silhouette sequences and appearance features from person crops to perform open-set identity matching, track individuals across camera networks, reconstruct forensic event timelines, and adaptively calibrate through continual learning validation gates.

Built on PyTorch, ONNX Runtime, OpenCV, FastAPI, and React 19, ARGUS AI incorporates an automated hardware-aware compute arbitration layer (`DeviceManager`), a production-grade multi-camera inference engine with fair-share scheduling, an admission controller preventing host resource exhaustion, encrypted RTSP credential resolution, a hardened vector store, and a responsive, resizable surveillance dashboard.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows / Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%2F%2012.6%20Driver-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20.0-blue.svg)](https://onnxruntime.ai/)
[![Tests: 276 Passed](https://img.shields.io/badge/tests-276%20passed-brightgreen.svg)](tests)
[![Frontend: React 19 + Vite](https://img.shields.io/badge/frontend-React%2019%20%2B%20Vite-61DAFB.svg)](frontend)
[![Status: Controlled Testing Ready](https://img.shields.io/badge/status-CONTROLLED%20TESTING%20READY-orange.svg)](docs/reports)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Gait Recognition Pipeline](#core-gait-recognition-pipeline)
3. [Dual-Modal ReID & Appearance Matching](#dual-modal-reid--appearance-matching)
4. [Hardware-Aware Compute Automation](#hardware-aware-compute-automation)
5. [12-Stage Environment Bootstrap](#12-stage-environment-bootstrap)
6. [Verified Development Baseline](#verified-development-baseline)
7. [Compute Verification & CPU Fallback Parity](#compute-verification--cpu-fallback-parity)
8. [Model Architecture & Benchmark Evidence](#model-architecture--benchmark-evidence)
9. [Camera Ingestion & Stream Management](#camera-ingestion--stream-management)
10. [Multi-Camera Inference & Admission Architecture](#multi-camera-inference--admission-architecture)
11. [Continual Learning & Accuracy Validation Gates](#continual-learning--accuracy-validation-gates)
12. [REST API & WebSocket Services](#rest-api--websocket-services)
13. [Frontend Surveillance Dashboard](#frontend-surveillance-dashboard)
14. [Security & Engineering Hardening](#security--engineering-hardening)
15. [Project Structure](#project-structure)
16. [Installation & Windows Setup](#installation--windows-setup)
17. [Running the Application](#running-the-application)
18. [Verification & Testing](#verification--testing)
19. [Troubleshooting Guide](#troubleshooting-guide)
20. [Current Implementation Status](#current-implementation-status)
21. [Known Limitations](#known-limitations)
22. [License & Project Information](#license--project-information)

---

## System Overview

ARGUS AI functions as an end-to-end visual biometric intelligence framework. The primary biometric identifier is human gait—identifying individuals by their dynamic body geometry and walking cadence over consecutive frames without requiring facial visibility or cooperative subject posture. This is augmented with an appearance re-identification (OSNet) feature stream for robust cross-camera tracking and multi-modal fusion.

```mermaid
graph TD
    A[Camera Streams: Webcam / RTSP] --> B[Person Detection: YOLOv8]
    B --> C[Multi-Object Tracking: ByteTrack + EMA]
    C --> D[Silhouette Extraction: UNet ONNX / Otsu]
    C --> E[Appearance Extraction: OSNet-x0.25 512-D]
    D --> F[Cycle-Aware Live GEI: 128x64]
    F --> G[ByGaitLight CNN: HPP part_bins=4]
    G --> H[Gait Embedding: 256-D L2 Norm]
    H --> I[Gait Vector Store Matching]
    E --> J[Appearance Vector Store Matching]
    I --> K[Dual-Modal Fusion & Score Calibration]
    J --> K
    K --> L[Open-Set Decision: KNOWN / UNKNOWN / UNCERTAIN]
    L --> M[Multi-Camera Track Fusion & Forensic Timeline]
    L --> N[Live MJPEG Stream Overlays & WebSocket Alerts]
    L --> O[Operational Embedding Collector]
    O --> P[Continual Learning Validation Gate]
```

### Primary Subsystems

* **Spatial Tracking**: Bounding box localization using YOLOv8, persistent track ID assignment using ByteTrack, and coordinate smoothing via Exponential Moving Average (EMA).
* **Silhouette & GEI Generation**: Neural silhouette segmentation using an ONNX UNet model (with morphological Otsu thresholding as automatic fallback), accumulated into a normalized $128 \times 64$ Gait Energy Image (GEI).
* **Gait Feature Extraction**: Convolutional gait representation using `ByGaitLight` with Horizontal Part Pooling (HPP, `part_bins=4`), producing unit-normalized 256-dimensional embeddings ($\|e\|_2 = 1.0000$).
* **Appearance ReID Extraction**: Deep appearance feature extraction using `OSNet-x0.25`, producing unit-normalized 512-dimensional feature vectors for clothes-consistent short-term re-identification.
* **Dual-Modal Score Fusion**: Calibrated multi-modal fusion combining gait similarity and appearance similarity with dynamic reliability scoring and quality assessment.
* **Biometric Gallery & Database**: Hardened vector stores (`models/live_gallery/`, `models/appearance_gallery/`) and structured SQLite embedding database (`storage/embedding_database.py`) with Firebase Firestore synchronization.
* **Hardware Arbitration**: Centralized runtime device management (`DeviceManager`) providing automatic CUDA GPU acceleration on compatible NVIDIA hardware and deterministic CPU fallback.
* **Multi-Camera Engine & Admission Control**: Fair-share frame scheduling (Deficit Round-Robin + Priority Aging), decoupled per-camera queues with backpressure and stale-frame drop protection, and pre-flight capacity admission control.
* **Continual Learning Subsystem**: Event-date driven background candidate model training (`NNFineTuner`), 50% historical replay mixing to prevent catastrophic forgetting, and multi-gate safety validation (`AccuracyValidationGate`).
* **Responsive Surveillance Frontend**: React 19 single-page application featuring resizable panels (`ResizeHandle`, `useResizablePanel`), adaptive 16:9 CCTV grid, live MJPEG feeds, geospatial mapping, case management, and administrative control.

---

## Core Gait Recognition Pipeline

The implemented gait recognition pipeline processes video streams through sequential stages:

```text
Input Video Frame
      ↓
[Stage 1] Person Detection (YOLOv8, class=0)
      ↓
[Stage 2] Multi-Object Tracking (ByteTrack + EMA Coordinate Smoothing)
      ↓
[Stage 3] Crop Normalization (Aspect-ratio constrained, 85% height target)
      ↓
[Stage 4] Silhouette Segmentation (UNet ONNX / Adaptive Otsu Fallback)
      ↓
[Stage 5] Temporal Cycle Accumulation (Live Gait Energy Image, 128x64)
      ↓
[Stage 6] ByGaitLight CNN Encoding (HPP part_bins=4)
      ↓
[Stage 7] L2 Normalization (256-D Embedding, ||e||₂ = 1.0000)
      ↓
[Stage 8] VectorStore Cosine Similarity Matching (Gallery Comparison)
      ↓
[Stage 9] Open-Set Classification (KNOWN / UNKNOWN / UNCERTAIN)
```

1. **Input Ingestion**: Video frames are captured at native frame rates from local webcams or RTSP network streams via isolated `CameraWorker` threads.
2. **Person Detection**: `PersonDetector` (`pipeline/detection/person_detector.py`) runs YOLOv8 to locate human bounding boxes (`class=0`) with configurable confidence and IoU thresholds.
3. **Multi-Object Tracking**: `TrackingStep` (`pipeline/steps/tracking.py`) maintains identity continuity across occlusions using ByteTrack, applying EMA filtering (`alpha=0.35`) to stabilize bounding box jitter.
4. **Crop Normalization**: Detected bounding boxes are cropped, aspect-ratio constrained, and centered to an $85\%$ relative height target.
5. **Silhouette Segmentation**: `SilhouetteExtractor` (`pipeline/silhouette/extractor.py`) executes `models/weights/silhouette_segmenter.onnx` to isolate binary body masks. If the ONNX engine is unavailable or uncalibrated, it automatically falls back to an adaptive Otsu morphological segmentation strategy.
6. **Live GEI Accumulator**: `LiveGEIStep` (`pipeline/steps/live_gei.py` / `pipeline/gei/stream_gei_builder.py`) aggregates aligned silhouette frames across a temporal window to generate a single 2D Gait Energy Image ($128 \times 64$ pixels).
7. **ByGaitLight Feature Encoding**: `ByGaitLight` (`models/architectures/bygait_light.py`) processes the GEI through convolutional feature extractors and Horizontal Part Pooling (`part_bins=4`), producing a 256-dimensional feature vector normalized via L2 norm ($\|e\|_2 = 1.0000$).
8. **VectorStore Gallery Matching**: `MatchingStep` (`pipeline/steps/matching_step.py`) calculates cosine similarity against enrolled gallery embeddings loaded from `models/live_gallery/`.
9. **Open-Set Decision Logic**: `OpenSetRecognizer` (`intelligence/open_set_recognizer.py`) classifies the match into `KNOWN`, `UNKNOWN`, or `UNCERTAIN` based on calibrated similarity thresholds and top-1/top-2 margin constraints.

---

## Dual-Modal ReID & Appearance Matching

In addition to silhouette-based gait recognition, ARGUS AI incorporates an appearance re-identification pipeline:

* **Backbone**: `OSNet-x0.25` (`models/reid/osnet_backbone.py`, weights: `models/weights/osnet_x0_25.pth`) lightweight omni-scale network for person re-identification.
* **Feature Representation**: 512-dimensional L2-normalized feature embeddings extracted directly from RGB person crops.
* **Appearance Gallery**: Separate appearance vector store (`models/appearance_gallery/`) managed via `VectorStore`.
* **Fusion Layer (`DualModalFusion` / `LearnedFusion`)**: Combines gait similarity score $S_{\text{gait}}$ and appearance similarity score $S_{\text{app}}$:
  $$S_{\text{fused}} = w_{\text{gait}} \cdot S_{\text{gait}} + w_{\text{app}} \cdot S_{\text{app}}$$
  with dynamic weight attenuation based on silhouette quality estimation and temporal track length.

---

## Hardware-Aware Compute Automation

ARGUS AI eliminates hardcoded device assignments by routing all compute queries through a centralized arbitration layer.

### Arbitration Architecture

```text
Host Hardware Detection (OS, CPU, RAM, GPU, Driver Version)
                          ↓
CUDA Capability Detection (Driver API, PyTorch Runtime, Tensor Probing)
                          ↓
Environment Validation (Dependency Health, ONNX Execution Providers)
                          ↓
Authoritative DeviceManager (Singleton Device State)
                          ↓
Downstream Component Binding (YOLO / PyTorch / ONNX / ByGaitLight / OSNet)
```

### Authoritative `DeviceManager` (`automation/device_manager.py`)

`DeviceManager` acts as the single source of truth across the entire system:
* Components resolve their target device by calling `DeviceManager.get_instance().resolve_component_device(requested)`.
* Requesting `'auto'` or `'cuda'` resolves to `'cuda:0'` when CUDA is available and verified healthy; otherwise, it resolves to `'cpu'`.
* PyTorch tensor operations, YOLO detection, ByteTrack tracking, ByGaitLight inference, OSNet inference, and ONNX Runtime sessions strictly follow the authoritative device state.

### Compute Execution Modes

| Subsystem | CUDA Mode (GPU Accelerated) | CPU Mode (Fallback / Validation) |
| :--- | :--- | :--- |
| **Authoritative State** | `EnvironmentState.CUDA_READY` | `EnvironmentState.CPU_READY` |
| **Resolved Device** | `cuda:0` | `cpu` |
| **PyTorch Tensor Device** | `torch.device('cuda:0')` | `torch.device('cpu')` |
| **YOLO PersonDetector** | `cuda:0` | `cpu` |
| **ByteTrack Tracking** | `cuda:0` | `cpu` |
| **ByGaitLight CNN** | `cuda:0` | `cpu` |
| **OSNet ReID Backbone** | `cuda:0` | `cpu` |
| **ONNX Runtime Provider** | `CUDAExecutionProvider` | `CPUExecutionProvider` |

---

## 12-Stage Environment Bootstrap

The bootstrap orchestrator (`automation/bootstrap.py`) performs a deterministic 12-stage discovery and capability verification sequence.

| Stage | Identifier | Verification Action |
| :---: | :--- | :--- |
| **01** | `Operating System` | Detects OS name, version, and architecture (e.g. Windows 10 AMD64). |
| **02** | `Python Runtime` | Validates Python interpreter version (3.11.x 64-bit). |
| **03** | `Hardware Profile` | Probes CPU core count, available RAM, and NVIDIA GPU presence/VRAM. |
| **04** | `NVIDIA Driver` | Queries installed GPU driver version via `nvidia-smi`. |
| **05** | `CUDA Compatibility` | Validates CUDA Driver API level and sets target compute backend. |
| **06** | `PyTorch Validation` | Inspects installed PyTorch build, CUDA support, and tensor probe. |
| **07** | `ONNX Runtime` | Inspects installed ONNX Runtime variant and available execution providers. |
| **08** | `Compute Validation` | Executes a synchronized $1024 \times 1024$ matrix multiplication test on the target device. |
| **09** | `YOLO Validation` | Instantiates `PersonDetector` and validates runtime device assignment. |
| **10** | `ONNX Inference` | Runs active ONNX session inference using `silhouette_segmenter.onnx`. |
| **11** | `ByGaitLight CNN` | Executes a forward pass through `ByGaitLight` and validates `[1, 256]` shape and unit L2 norm. |
| **12** | `Final Validation` | Generates authoritative environment summary and writes `.venv/argus_env_manifest.json`. |

### Idempotent Setup Behavior
The bootstrap validates compatibility before performing actions. If the active environment already contains working CUDA PyTorch (`2.5.1+cu121`) and ONNX Runtime (`1.20.0`), it reports `PyTorch installation required: NO` and skips reinstallation.

---

## Verified Development Baseline

The following hardware and runtime baseline was used during system development and testing:

```text
============================================================
 ARGUS VERIFIED TEST BASELINE
============================================================
Host OS          : Windows 10 (AMD64, Build 19045)
Python Runtime   : 3.11.9 (64-bit)
CPU Topology     : 12 Logical Cores
System Memory    : 7.7 GB Usable RAM
NVIDIA GPU       : NVIDIA GeForce RTX 3050 6GB Laptop GPU
Dedicated VRAM   : 6144 MB
NVIDIA Driver    : 560.94
CUDA Driver API  : 12.6
PyTorch Build    : 2.5.1+cu121 (CUDA 12.1 runtime)
ONNX Runtime     : 1.20.0 (onnxruntime-gpu)
Active Provider  : CUDAExecutionProvider
============================================================
```

> **Note**: While CUDA Driver API 12.6 is installed at the system driver level, PyTorch links against its bundled CUDA 12.1 runtime (`cu121`). These version numbers are distinct and fully compatible.

---

## Compute Verification & CPU Fallback Parity

ARGUS AI has been verified in both standard CUDA acceleration mode and forced CPU fallback mode.

### Standard CUDA Execution (`python -m automation.bootstrap`)

```text
============================================================
 ARGUS COMPUTE ENVIRONMENT (CUDA ACCELERATED)
============================================================
Backend          : CUDA
Device           : cuda:0
GPU              : NVIDIA GeForce RTX 3050 6GB Laptop GPU
VRAM             : 6144 MB
PyTorch          : 2.5.1+cu121
CUDA             : 12.1
YOLO             : CUDA (cuda:0)
ONNX             : CUDA (CUDAExecutionProvider)
ByGaitLight      : CUDA (cuda:0, L2 Norm: 1.0000)
============================================================
[ARGUS] ENVIRONMENT READY
[ARGUS] FULL CUDA ACCELERATION READY
```

### Forced CPU Validation (`python -m automation.bootstrap --force-cpu`)

The `--force-cpu` flag is an operational validation override. It forces `DeviceManager` and downstream components to execute on CPU without modifying physical hardware detection or uninstalling CUDA packages:

```text
============================================================
 ARGUS COMPUTE ENVIRONMENT (FORCED CPU MODE)
============================================================
Backend          : CPU
Device           : cpu
GPU              : NVIDIA GeForce RTX 3050 6GB Laptop GPU
VRAM             : 6144 MB
PyTorch          : 2.5.1+cu121
CUDA             : 12.1
YOLO             : CPU (cpu)
ONNX             : CPU (CPUExecutionProvider)
ByGaitLight      : CPU (cpu, L2 Norm: 1.0000)
============================================================
[ARGUS] ENVIRONMENT READY
[ARGUS] CPU MODE READY
```

---

## Model Architecture & Benchmark Evidence

### Active Baseline vs. Candidate Architecture

* **Active Reference Model (`runs/exp_001/best_model.pth`)**: Baseline `ByGaitLight` model with global average pooling (`part_bins=1`), trained using standard cross-entropy loss.
* **Top Candidate Model (`EXP-003E`)**: `ByGaitLight` with Horizontal Part Pooling (`part_bins=4`), ArcFace classification margin ($m=0.50, s=30.0$), and Batch-Hard Triplet loss ($w=0.25$).

### CASIA-B Subject-Disjoint Ablation Matrix

Evaluated under a strict subject-disjoint partition: Train `001–062` (6,779 sequences), Val `063–074` (1,299 sequences), Test `075–124` (5,466 sequences):

| Experiment | Pooling Strategy | Loss Formulation | Triplet Weight | Rank-1 Accuracy | Rank-5 Accuracy | Normal Walk (NM) | Carrying Bag (BG) | Clothing Change (CL) | ROC-AUC | EER | Open-Set FAR | Calibration Threshold | Impostor Score Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp-001** (Legacy Non-Disjoint)* | Global (1) | Standard CE | ~0.50 | 86.89%* | 93.96%* | 96.82%* | 91.23%* | 72.64%* | 0.9150 | 16.88% | 36.75% | 0.9913 | Saturated near 1.0 |
| **EXP-003A** (Disjoint Base) | Global (1) | Standard CE | 0.50 | 52.78% | 67.10% | 85.82% | 53.15% | 19.36% | 0.7499 | 31.95% | 70.49% | 0.7064 | Compressed `[0.208, 0.984]` |
| **EXP-003B** (HPP Alone) | HPP (4) | Standard CE | 0.50 | 61.43% | 75.63% | 91.55% | 60.55% | 32.18% | 0.8327 | 24.86% | 57.06% | 0.7942 | Compressed `[0.450, 0.995]` |
| **EXP-003C** (ArcFace Alone)| Global (1) | ArcFace | 0.50 | 59.58% | 73.78% | 91.00% | 61.55% | 26.18% | 0.8314 | 25.64% | 47.20% | 0.9927 | Saturated near 1.0 |
| **EXP-003D** (HPP+ArcFace) | HPP (4) | ArcFace | 0.00 | 69.71% | 80.91% | 96.73% | 72.79% | 39.64% | 0.8470 | 23.49% | 60.84% | 0.5287 | Expanded `[0.211, 0.965]` |
| **EXP-003E** (Top Candidate) | HPP (4) | ArcFace | **0.25** | **72.63%** | **82.76%** | **97.00%** | **78.26%** | **42.64%** | **0.8776** | **20.46%** | **62.26%** | **0.4906** | **Desaturated `[-0.60, 0.97]`** |

*\*Note: Exp-001 metrics reflect a historical evaluation where test subjects were supervised during training. On the true subject-disjoint split, the global baseline achieves 52.78% Rank-1. EXP-003E provides a +19.85% absolute improvement over the disjoint baseline.*

---

## Camera Ingestion & Stream Management

ARGUS AI ingests video from local USB/integrated webcams and network RTSP IP cameras via `services/camera_worker.py` and `services/camera_source_resolver.py`.

### Authoritative Camera State Machine

```mermaid
stateDiagram-v2
    [*] --> STANDBY: Worker Initialized
    STANDBY --> CONNECTING: User Calls /cameras/start
    CONNECTING --> CONNECTED: First Valid Frame Acquired
    CONNECTING --> FAILED: Device Unavailable / Timeout
    CONNECTED --> RECONNECTING: Frame Loss / Stream Drop
    RECONNECTING --> CONNECTED: Reconnect Succeeded
    RECONNECTING --> FAILED: Max Retries Exceeded
    CONNECTED --> STOPPED: User Calls /cameras/stop
    FAILED --> STOPPED: Worker Reset
    STOPPED --> STANDBY: Worker Reinitialized
```

> **Lifecycle Invariant**: `STANDBY` is the initial camera state. `FAILED` represents an outcome following an attempted connection, never the default idle state.

### Ingestion Features
* **Safe Device Probing**: Automatically detects local webcams using DirectShow (`CAP_DSHOW`) and MSMF backends on Windows without relying on hardcoded index `0`.
* **Runtime Source Detection**: Source type is hidden during `STANDBY` and dynamically updates to `"Webcam"` or `"RTSP"` upon first frame acquisition.
* **Worker Isolation**: Each camera runs on a dedicated thread with an internal frame buffer, client reference counting, and configurable frame rate throttling (`preview_max_fps`).
* **Live MJPEG Streaming**: Serves real-time video with localized bounding boxes, track IDs, and recognition status over HTTP (`/api/v1/cameras/{id}/stream`).

---

## Multi-Camera Inference & Admission Architecture

To scale across multiple concurrent video feeds without resource exhaustion, ARGUS AI implements a multi-camera pipeline (`streaming/production_multicamera_engine.py` and `streaming/deployment_readiness.py`):

```text
[Camera Feed 1] ──┐
[Camera Feed 2] ──┼─► [Decoupled Ingestion Queues] ─► [Fair-Share Deficit Round-Robin Scheduler]
[Camera Feed N] ──┘                                                  │
                                                                     ▼
                                                       [Dynamic GPU Batch Worker Pool]
                                                                     │
                                                       (ByGaitLight 256D + OSNet 512D)
                                                                     │
                                                                     ▼
                                                     [Cross-Camera Track Aggregator & Fusion]
```

### Key Scaling Mechanisms

1. **Hardware-Aware Admission Controller (`CameraAdmissionController`)**: Pre-flight capacity check evaluating host CPU load, available RAM, and GPU VRAM before admitting new cameras. Evaluates to `ADMITTED`, `ADMITTED_DEGRADED` (reduced target FPS), or `REJECTED`.
2. **Decoupled Bounded Queues**: Per-camera bounded frame buffers with backpressure and automatic stale-frame dropping (`stale_frame_max_age_ms=500.0ms`) preventing memory exhaustion under burst traffic.
3. **Starvation-Free Frame Scheduler (`PersonTrackScheduler`)**: Deficit Round-Robin (DRR) with priority aging ensuring low-traffic cameras receive equal processing opportunities.
4. **Dynamic Batching**: Aggregates appearance crops and gait silhouette sequences across cameras into unified GPU batches (adaptive batch size: 8–32 based on VRAM).
5. **Stream Isolation**: Network failures or disconnects on one camera never degrade, stall, or crash other running camera streams.

---

## Continual Learning & Accuracy Validation Gates

ARGUS AI implements an event-date driven continual learning subsystem designed to safely refine biometric recognition capabilities without risking production degradation:

```text
Live Surveillance Inference
          ↓
[Operational Embedding Collector] (Quality gate: confidence >= threshold, stable track)
          ↓
[Date-Aware Scheduler] (Triggers ONLY when new verified date observations exist)
          ↓
[Training Dataset Builder] (50% Historical Replay Buffer + 50% New Date Observations)
          ↓
[NN Fine-Tuner] (Background transfer learning for ByGaitLight & OSNet backbones)
          ↓
[Candidate Artifact Generation] (Isolated candidate checkpoint: models/candidates/*.pth)
          ↓
[Accuracy Validation Gate] ───► REJECT: Insufficient gain / regression detected
          │
          ▼ PASS (All 5 gates satisfied)
[Model Registry Promotion] (Atomic activation & instant hot-reload)
```

### Continual Learning Subsystem Breakdown

* **Operational Observation Collection (`OperationalEmbeddingCollector`)**: Captures verified high-confidence observations during live CCTV operation without saving raw high-bandwidth video streams.
* **Historical Replay Mixing (`TrainingDatasetBuilder`)**: Combines new date observations with 50% historical baseline embeddings to prevent catastrophic forgetting.
* **Background Neural Network Fine-Tuner (`NNFineTuner`)**: Executes transfer learning on ByGaitLight and OSNet CNN backbones asynchronously in a background thread without blocking active inference. Outputs isolated candidate `.pth` checkpoints with SHA-256 integrity checksums.
* **Multi-Gate Accuracy Gate (`AccuracyValidationGate`)**: Enforces strict safety policies before any candidate model can be promoted:
  1. **Zero FAR Security Gate**: Candidate false accept rate (FAR) must not exceed baseline FAR (0.0% tolerance).
  2. **Confusion-Pair Protection Gate**: 0.0% false accepts permitted on historically confusing subject pairs.
  3. **Catastrophic Forgetting Gate**: Historical true accept rate (TAR) must not degrade beyond calibrated tolerance (0.5%).
  4. **Anti-Churn Gate**: Rejects candidate models whose performance delta is within random noise without meaningful gains.
  5. **Small-Data Uncertainty Policy**: Blocks promotion when statistical sample size is insufficient.
* **Independent Threshold Calibration (`scripts/f1_threshold_calibration_independent_validation.py`)**: Rigorous protocol separating calibration (subjects 101–110) from the independent test set (subjects 051–070). F1 score improvements achieved via threshold sweeps are explicitly documented as calibration optimizations on frozen model weights, not exaggerated as neural network weight learning gains.

---

## REST API & WebSocket Services

The backend API is implemented in FastAPI (`api/server.py`, `api/v1/router.py`, and `api/routes/health.py`) and executed via Uvicorn.

### Route Reference (`/api/v1` and `/health`)

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Root service health check and pipeline loaded state. |
| `GET` | `/status` | Root operational status, device telemetry, and gallery summary. |
| `GET` | `/metrics` | Root system metrics (processed images, videos, active tracks). |
| `GET` | `/health/live` | Production liveness probe (process uptime, PID). |
| `GET` | `/health/ready` | Production readiness probe (worker availability check). |
| `GET` | `/health/system` | Detailed host telemetry (CPU, RAM, GPU, VRAM allocation). |
| `GET` | `/health/cameras` | Per-camera worker connection and health telemetry. |
| `GET` | `/health/workers` | Shared inference worker pool health. |
| `GET` | `/api/v1/health` | API v1 health status, model statuses, and active backend. |
| `GET` | `/api/v1/status` | Operational status, compute backend, thresholds, and gallery count. |
| `GET` | `/api/v1/metrics` | System counters (images, videos, tracks, events). |
| `POST` | `/api/v1/identify/image` | Single-image person detection and gait identification. |
| `POST` | `/api/v1/analyze/video` | Uploaded video analysis and sampled gait recognition. |
| `POST` | `/api/v1/enroll` | Subject biometric enrollment (`person_id` + multi-image upload). |
| `GET` | `/api/v1/events` | In-memory historical recognition event log (JSON). |
| `POST` | `/api/v1/cameras/start` | Start camera worker (`camera_id`, `source`, `location`, `zone_id`). |
| `POST` | `/api/v1/cameras/stop` | Stop active camera worker and release video capture device. |
| `GET` | `/api/v1/cameras` | List all active camera workers with telemetry and frame counters. |
| `GET` | `/api/v1/cameras/{camera_id}` | Retrieve specific camera worker metrics and status. |
| `GET` | `/api/v1/cameras/{camera_id}/stream` | Real-time MJPEG video stream with bounding boxes and overlays. |
| `GET` | `/api/v1/cameras/{camera_id}/snapshot` | Single JPEG snapshot of the latest captured video frame. |
| `POST` | `/api/v1/credentials` | Store encrypted RTSP camera credentials. |
| `GET` | `/api/v1/credentials` | List accessible credentials with masked password fields. |
| `DELETE`| `/api/v1/credentials/{id}` | Delete user-owned credential entry. |
| `POST` | `/api/v1/credentials/{id}/share` | Grant credential access to another user ID. |
| `POST` | `/api/v1/cameras/{id}/credentials` | Store camera-scoped credential. |
| `WS` | `/api/v1/ws/recognition` | Real-time WebSocket feed for recognition events. |
| `WS` | `/api/v1/ws/events` | Real-time WebSocket feed for system security alerts. |

---

## Frontend Surveillance Dashboard

The frontend application (`frontend/`) is built with React 19, Vite, Lucide Icons, and Leaflet.

### Implemented UI Capabilities

* **Responsive & Resizable Dock Architecture**:
  * **`useResizablePanel` Hook**: High-performance, pointer-event-based panel resizing with `requestAnimationFrame` throttling and boundary constraints.
  * **`ResizeHandle` Component**: Fully accessible separator handle supporting mouse dragging, touch interaction, keyboard navigation (`ArrowLeft` / `ArrowRight` steps), double-click reset, and ARIA attributes (`role="separator"`, `aria-valuenow`, `aria-valuemin`, `aria-valuemax`).
  * **`layoutStorage.js` Persistence**: Persists user layout preferences in `localStorage` under `argus_ui_layout` with bounds validation:
    * Dashboard Dock Width: 300px – 640px (Default: 420px).
    * Case Details Panel Width: 240px – 480px (Default: 300px).
    * Admin Split Ratio: 35% – 75% (Default: 60%).
* **CCTV Surveillance Grid (`CctvNetwork.jsx`)**: Responsive 16:9 surveillance feed cards, live MJPEG stream display with automatic reconnection/retry, connection status badges (`STANDBY`, `CONNECTING`, `CONNECTED`, `RECONNECTING`), and worker controls.
* **Gait System Telemetry (`GaitSystemStatus.jsx`)**: Real-time display of backend status, active compute backend (`CUDA` / `CPU`), GPU device name, VRAM allocation, and execution providers.
* **Real-Time Recognition Feed (`RecognitionEvents.jsx`)**: Live WebSocket event feed displaying subject ID, confidence score, camera zone, and timestamp.
* **Forensic Event History (`History.jsx`)**: Searchable and filterable recognition event timeline.
* **Geospatial Map (`Map.jsx`)**: Leaflet map displaying registered camera zone placements and geographic locations.
* **Case Management (`ReportCase.jsx`, `CaseDetails.jsx`)**: Person of interest registration, multi-camera timeline reconstruction, and alert dispatch.
* **Admin Control Center (`AdminDashboard.jsx`)**: User management, security policy configuration, system log inspection (`LogViewer.jsx`), and model registry audit logs.

---

## Security & Engineering Hardening

1. **Safe PyTorch Checkpoint Loading**: All weight loading in `models/inference/pytorch_backend.py` enforces `torch.load(..., weights_only=True)`, preventing arbitrary code execution from untrusted model files.
2. **Hardened Vector Store**: `storage/vector_store.py` enforces `allow_pickle=False` and rejects object-type NumPy arrays (`dtype == object`), mitigating deserialization vulnerabilities.
3. **Encrypted Credentials & Log Masking**: RTSP credentials are encrypted via Fernet (`.credentials.key`) and masked in logs (`rtsp://***:***@host:port`).
4. **Lazy Module Access**: `automation/__init__.py` and `pipeline/steps/__init__.py` use PEP 562 lazy module `__getattr__`, preventing `runpy` `RuntimeWarning: 'automation.bootstrap' found in sys.modules` when running CLI modules.
5. **Non-Blocking Background Warmup**: `GaitService` implements asynchronous background warmup (`warmup_async()`), allowing the FastAPI server to bind and respond to `/health` probes in < 2.0s without blocking on heavyweight model weight loads.

---

## Project Structure

```text
E:\ARGUS_AI
├── api/                        # FastAPI REST routing, server lifecycle, and schemas
│   ├── routes/                 # Production health, status, and readiness probes
│   ├── v1/router.py            # API v1 endpoint implementations
│   ├── schemas.py              # Pydantic request and response models
│   └── server.py               # Application factory, lifespan context, and SPA catch-all
├── assets/                     # Graphical assets and repository banner
├── automation/                 # Hardware detection, arbitration, and bootstrap subsystem
│   ├── bootstrap.py            # Master 12-stage environment discovery and validation
│   ├── cuda_detector.py        # CUDA runtime, driver API, and tensor probe validation
│   ├── device_manager.py       # Authoritative singleton DeviceManager layer
│   ├── dll_manager.py          # Windows DLL search path configuration
│   ├── download_manager.py     # Live progress wheel downloader utility
│   ├── environment_validator.py# Compute capability evaluation and state machine
│   ├── hardware_detector.py    # Hardware profiling (CPU, RAM, GPU, Driver)
│   ├── onnx_manager.py         # ONNX Runtime CPU/GPU compatibility manager
│   └── pytorch_manager.py      # PyTorch build compatibility manager
├── configs/                    # Externalized YAML configuration files
│   ├── cameras.yaml            # Camera stream definitions and worker pool limits
│   ├── continuous_learning.yaml# Continual learning schedule, triggers, and replay ratios
│   ├── detection.yaml          # YOLOv8 detector confidence, IoU, and device settings
│   ├── inference.yaml          # Inference policy, thresholds, and crowd control
│   ├── production.yaml         # Production scaling, admission limits, and VRAM guards
│   └── system.yaml             # Thread limits, storage paths, and logging bindings
├── core/                       # Shared utilities, threshold manager, and logging setup
├── frontend/                   # React 19 + Vite surveillance dashboard application
│   ├── src/
│   │   ├── admin/              # Admin dashboard, user management, policy manager, logs
│   │   ├── components/         # CCTV network, dashboard, case details, map, history
│   │   │   └── common/         # ResizeHandle and accessible UI controls
│   │   ├── contexts/           # AuthContext (Firebase) and GaitContext (WebSocket/State)
│   │   ├── hooks/              # useResizablePanel, useAuth, useGait
│   │   ├── utils/              # layoutStorage, cctvService, geoService, embeddingService
│   │   ├── App.jsx             # React router and protected routes
│   │   └── main.jsx            # Application entry point
│   ├── package.json            # Frontend dependency manifest
│   └── vite.config.js          # Vite development server configuration
├── intelligence/               # Biometric intelligence, fusion, and continual learning
│   ├── accuracy_validation_gate.py      # Multi-gate anti-churn promotion gate
│   ├── background_learning_worker.py    # Background candidate generation thread
│   ├── concurrent_track_manager.py      # Multi-person concurrent track management
│   ├── continual_learning_audit_trail.py# Forensic candidate evaluation audit trail
│   ├── continuous_improvement_engine.py # Continuous learning orchestrator
│   ├── date_aware_learning_scheduler.py # Event-date driven learning job scheduler
│   ├── dual_modal_fusion.py             # Gait + Appearance score fusion
│   ├── nn_fine_tuner.py                 # Background ByGaitLight & OSNet fine-tuning
│   ├── open_set_recognizer.py           # Open-set KNOWN / UNKNOWN / UNCERTAIN decision
│   ├── operational_embedding_collector.py # High-confidence live observation capture
│   └── training_dataset_builder.py      # Replay buffer & balanced dataset assembly
├── models/                     # Deep learning architectures and gallery storage
│   ├── appearance_gallery/     # Active 512-D OSNet appearance embeddings (.npy)
│   ├── architectures/          # ByGaitLight, UNet segmenter, and ArcFace losses
│   ├── candidates/             # Isolated candidate model checkpoints (.pth)
│   ├── live_gallery/           # Active 256-D ByGaitLight gait embeddings (.npy)
│   ├── reid/                   # OSNet-x0.25 lightweight appearance ReID backbone
│   └── weights/                # Model weights (silhouette_segmenter.onnx, osnet_x0_25.pth)
├── monitoring/                 # Structured logging and process metrics
├── pipeline/                   # Modular gait recognition pipeline steps
│   ├── detection/              # PersonDetector (YOLOv8) & DetectionValidator
│   ├── gei/                    # StreamGEIBuilder & cycle accumulation
│   ├── silhouette/             # SilhouetteExtractor (UNet ONNX + Otsu fallback)
│   ├── steps/                  # Lazy-loaded pipeline step modules
│   └── tracking/               # ByteTrack multi-object tracking integration
├── scripts/                    # Automation, diagnostic, evaluation, and benchmark scripts
│   ├── bootstrap_env.ps1       # Windows PowerShell environment bootstrap entry point
│   ├── detect_environment.py   # CLI hardware and compute detector
│   ├── dev.js                  # Unified backend + frontend development orchestrator
│   ├── doctor.py               # Pre-flight deployment health diagnostics
│   ├── f1_threshold_calibration_independent_validation.py # Independent F1 validation
│   ├── test_server_functional_parity.py # Server functional parity verification
│   └── verify_environment.py   # 6-phase environment verification suite
├── security_layer/             # Credential encryption and access control manager
├── services/                   # GaitService, CameraWorker, CameraSourceResolver, RecognitionWorker
├── storage/                    # Hardened VectorStore, SQLite EmbeddingDatabase, Firebase store
├── streaming/                  # Multi-camera engine, admission control, and runtime resilience
├── tests/                      # Automated test suite
│   ├── integration/            # Multi-component integration tests
│   └── unit/                   # Unit test suite (276 tests passed)
├── requirements.txt            # Python dependencies manifest
└── VERSION                     # Project version file (0.1.0)
```

---

## Installation & Windows Setup

### Prerequisites
* **Python**: 3.11.x (64-bit)
* **Node.js**: 18.x+ and npm
* **OS**: Windows 10/11 (AMD64) or Linux (Ubuntu 20.04+)
* **GPU**: NVIDIA GPU with Driver 535.xx+ (for CUDA acceleration; optional for CPU mode)

### 1. Clone & Prepare Virtual Environment

```powershell
# Clone the repository
git clone https://github.com/chanuka8/argus-gait-recognition.git
cd argus-gait-recognition

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Run Automated Environment Bootstrap

Execute the PowerShell bootstrap script to inspect hardware, arbitrate compute devices, and validate dependencies:

```powershell
# Standard environment bootstrap
powershell -ExecutionPolicy Bypass -File ".\scripts\bootstrap_env.ps1"
```

For forced CPU testing:
```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\bootstrap_env.ps1" -ForceCpu
```

### 3. Install Standard Dependencies & Frontend Packages

```powershell
# Install Python backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## Running the Application

### Option 1: Unified Dev Server (Backend + Frontend)

ARGUS AI includes a Node.js orchestrator (`scripts/dev.js`) that boots the FastAPI backend, waits for health readiness at `127.0.0.1:8000`, and starts the React Vite frontend at `localhost:5173`:

```powershell
npm run dev
```

### Option 2: Backend Only (FastAPI)

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```

* **Swagger API Docs**: `http://127.0.0.1:8000/docs`
* **Health Check**: `http://127.0.0.1:8000/api/v1/health`
* **Operational Status**: `http://127.0.0.1:8000/api/v1/status`

### Option 3: Frontend Only

```powershell
npm run dev:frontend
```

---

## Verification & Testing

Execute the diagnostic suite to verify environment health, code correctness, and system readiness:

```powershell
# 1. Complete test discovery (276 tests across tests/)
.\.venv\Scripts\python.exe -u -m unittest discover tests

# 2. Targeted unit test discovery (143 tests across tests/unit/)
.\.venv\Scripts\python.exe -u -m unittest discover tests/unit

# 3. Targeted automation unit tests (13 tests)
.\.venv\Scripts\python.exe -u -m unittest tests/unit/test_automation.py

# 4. Python bytecode compilation check (0 errors)
.\.venv\Scripts\python.exe -m compileall -q -x "\.venv" .

# 5. Full 6-phase environment verification suite
.\.venv\Scripts\python.exe scripts/verify_environment.py

# 6. Pre-flight health doctor check
.\.venv\Scripts\python.exe scripts/doctor.py

# 7. End-to-end server functional parity & API regression test
.\.venv\Scripts\python.exe scripts/test_server_functional_parity.py

# 8. Frontend linter check (ESLint)
cd frontend; npm run lint; cd ..

# 9. Frontend production build validation (Vite)
cd frontend; npm run build; cd ..
```

---

## Troubleshooting Guide

| Symptom | Likely Root Cause | Verification Command | Corrective Action |
| :--- | :--- | :--- | :--- |
| **CUDA Not Detected** | Missing NVIDIA GPU driver or PyTorch CPU build active | `python scripts/detect_environment.py` | Verify GPU in Task Manager; run `python -m automation.bootstrap` to arbitrate PyTorch CUDA build. |
| **ONNX Running on CPU** | Missing `onnxruntime-gpu` package or missing CUDA DLLs | `python scripts/verify_environment.py` | Run `powershell -ExecutionPolicy Bypass -File scripts/bootstrap_env.ps1` to configure ONNX GPU provider. |
| **Camera in Standby / Connecting** | Camera device index unavailable or in use by another app | Check `GET /api/v1/cameras/{id}` | Ensure camera is not open in another application; check `configs/cameras.yaml` device index. |
| **Port 8000 Already in Use** | Orphaned Uvicorn process running in background | `netstat -ano \| findstr :8000` | Stop previous process using Task Manager or `Stop-Process -Id <PID>`. |
| **Forced CPU Mode Remains Active** | Shell session holding `--force-cpu` state in memory | `python scripts/detect_environment.py` | Run `python -m automation.bootstrap` or restart the terminal session. |

---

## Current Implementation Status

The current implementation status is categorized below based strictly on codebase evidence:

### Status Matrix

| Capability / Subsystem | Status | Evidence / Verification Method |
| :--- | :---: | :--- |
| **Hardware Auto-Discovery & Arbitration** | **IMPLEMENTED** | `automation/bootstrap.py`, `DeviceManager` (13/13 unit tests passed) |
| **CUDA GPU Acceleration & CPU Fallback** | **IMPLEMENTED** | `scripts/verify_environment.py` (CUDA verified), `--force-cpu` parity |
| **Person Detection & Box Smoothing** | **IMPLEMENTED** | `PersonDetector` (YOLOv8) + `TrackingStep` (ByteTrack + EMA `alpha=0.35`) |
| **Silhouette Extraction (UNet + Otsu)** | **IMPLEMENTED** | `SilhouetteExtractor` (`silhouette_segmenter.onnx` + morphological Otsu fallback) |
| **Gait Energy Image (GEI) Accumulation** | **IMPLEMENTED** | `LiveGEIStep`, `StreamGEIBuilder` ($128 \times 64$ normalized output) |
| **ByGaitLight Feature Extraction** | **IMPLEMENTED** | `ByGaitLight` CNN (HPP `part_bins=4`, 256-D L2-normalized embedding) |
| **OSNet Appearance ReID** | **IMPLEMENTED** | `OSNet-x0.25` (512-D L2-normalized appearance embeddings) |
| **Dual-Modal Score Fusion** | **IMPLEMENTED** | `DualModalFusion`, `LearnedFusion`, `ScoreCalibrator` |
| **Open-Set Decision Logic** | **IMPLEMENTED** | `OpenSetRecognizer` (`KNOWN`, `UNKNOWN`, `UNCERTAIN` margin boundaries) |
| **Camera Ingestion & State Machine** | **IMPLEMENTED** | `CameraWorker`, `CameraSourceResolver` (`STANDBY` $\rightarrow$ `CONNECTING` $\rightarrow$ `CONNECTED`) |
| **Multi-Camera Fair-Share Scheduling** | **IMPLEMENTED** | `ProductionMultiCameraEngine`, `PersonTrackScheduler` (DRR + Priority Aging) |
| **Hardware Admission Control** | **IMPLEMENTED** | `CameraAdmissionController`, `DeploymentReadinessManager` (RAM/VRAM gating) |
| **RTSP Credential Encryption & Masking** | **IMPLEMENTED** | Fernet encryption (`security_layer/credentials.py`), log masking |
| **Hardened Vector Store & SQLite DB** | **IMPLEMENTED** | `VectorStore` (`allow_pickle=False`), `EmbeddingDatabase` (versioned records) |
| **FastAPI REST API & WebSockets** | **IMPLEMENTED** | `/api/v1/...`, `/health/...`, WebSocket `/ws/recognition` & `/ws/events` |
| **Frontend Surveillance Dashboard** | **IMPLEMENTED** | React 19 SPA, live MJPEG feeds, geospatial mapping, case management |
| **Responsive & Resizable Layout System** | **IMPLEMENTED** | `useResizablePanel`, `ResizeHandle` (keyboard/pointer), `layoutStorage.js` |
| **Continual Learning Observation Capture** | **IMPLEMENTED** | `OperationalEmbeddingCollector`, `ContinuousImprovementEngine` |
| **Background Candidate Fine-Tuning** | **IMPLEMENTED** | `NNFineTuner` (ByGaitLight & OSNet background training with 50% replay) |
| **Accuracy Validation Gate** | **IMPLEMENTED** | `AccuracyValidationGate` (Zero FAR, Anti-Churn, Catastrophic Forgetting gates) |
| **Independent F1 Threshold Validation** | **IMPLEMENTED** | `f1_threshold_calibration_independent_validation.py` (Subject-disjoint protocol) |
| **Automated Test Suite** | **IMPLEMENTED** | **276 tests passed** (`python -m unittest discover tests`) |
| **Multi-Camera Physical Field Trials** | **PARTIALLY IMPLEMENTED** | Synthetic and multi-worker tests verified; large physical field deployment pending |
| **Production-Scale Million-Subject DB** | **PLANNED / FUTURE** | Evaluated on active development gallery; indexing for $10^6$ scale is planned |

---

## Known Limitations

1. **Active Development Gallery Size**: The active test galleries contain development baselines (e.g. 64 gait embeddings and 201 appearance embeddings). Million-identity indexing remains for future production scaling.
2. **Clothing Covariate Sensitivity**: As established in CASIA-B subject-disjoint ablation benchmarks, clothing changes (`CL` Rank-1 = 42.64%) degrade silhouette geometry more significantly than carrying bags (`BG` = 78.26%).
3. **Hardware-Dependent Real-Time Throughput**: Full real-time FPS throughput is dependent on a compatible NVIDIA GPU with CUDA acceleration. While CPU execution is functional, throughput will be lower on CPU-only machines.
4. **Physical Multi-Camera Field Validation**: While unit tests, component smoke tests, and synthetic multi-stream pipelines pass, physical multi-camera trials in unconstrained real-world environments remain ongoing.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Project Information & Maintainer

**Chanuka Sandun**  
Undergraduate in Cybersecurity  
Developer of the ARGUS AI Gait Recognition Framework  
* GitHub: [@chanuka8](https://github.com/chanuka8)  
* LinkedIn: [linkedin.com/in/chanukasandun](https://www.linkedin.com/in/chanukasandun/)
