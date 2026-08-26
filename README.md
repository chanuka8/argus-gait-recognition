# ARGUS AI

![ARGUS AI Gait Recognition Banner](assets/github/Gitrepo_profilepic.png)

### AI-Powered Gait Recognition and Surveillance System

ARGUS AI is a modular computer vision and deep learning platform designed for real-time human gait recognition and multi-camera surveillance intelligence. The system extracts biometric walking patterns from silhouette sequences to perform open-set identity matching, track persons across video streams, and log structured forensic recognition timelines.

Built on PyTorch, ONNX Runtime, OpenCV, and FastAPI, ARGUS AI incorporates an automated hardware-aware compute arbitration layer, centralized runtime device management (`DeviceManager`), a hardened biometric vector store, encrypted camera credential resolution, and a React-based surveillance dashboard.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows / Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%2F%2012.6%20Driver-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20.0-blue.svg)](https://onnxruntime.ai/)
[![Unit Tests: 133 Passed](https://img.shields.io/badge/unit%20tests-133%20passed-brightgreen.svg)](tests/unit)
[![Status: Controlled Testing Ready](https://img.shields.io/badge/status-CONTROLLED%20TESTING%20READY-orange.svg)](docs/reports)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Gait Recognition Pipeline](#core-gait-recognition-pipeline)
3. [Hardware-Aware Compute Automation](#hardware-aware-compute-automation)
4. [12-Stage Environment Bootstrap](#12-stage-environment-bootstrap)
5. [Verified Development Environment](#verified-development-environment)
6. [Compute Verification & CPU Fallback Parity](#compute-verification--cpu-fallback-parity)
7. [Model Architecture & Benchmark Evidence](#model-architecture--benchmark-evidence)
8. [Camera Ingestion & Stream Management](#camera-ingestion--stream-management)
9. [REST API & WebSocket Services](#rest-api--websocket-services)
10. [Frontend Surveillance Dashboard](#frontend-surveillance-dashboard)
11. [Security & Engineering Hardening](#security--engineering-hardening)
12. [Project Structure](#project-structure)
13. [Installation & Windows Setup](#installation--windows-setup)
14. [Running the Application](#running-the-application)
15. [Verification & Testing](#verification--testing)
16. [Troubleshooting Guide](#troubleshooting-guide)
17. [Current Project Validation Status & Limitations](#current-project-validation-status--limitations)
18. [License & Maintainer](#license--maintainer)

---

## System Overview

ARGUS AI is designed as a modular surveillance intelligence engine focused on visual gait biometrics. Rather than relying on facial features, the system identifies individuals by their dynamic body shape and walking kinematics over consecutive video frames.

```mermaid
graph TD
    A[Camera Feed: Webcam / RTSP] --> B[Person Detection: YOLOv8]
    B --> C[Multi-Object Tracking: ByteTrack + EMA]
    C --> D[Silhouette Extraction: UNet ONNX / Otsu]
    D --> E[Cycle-Aware Live GEI: 128x64]
    E --> F[ByGaitLight CNN: HPP part_bins=4]
    F --> G[256-D L2-Normalized Embedding]
    G --> H[VectorStore: Cosine Similarity Matching]
    H --> I[Open-Set Decision: KNOWN / UNKNOWN / UNCERTAIN]
    I --> J[Live Stream Overlays & WebSocket Alerts]
```

### Primary Subsystems
* **Spatial Tracking**: Bounding box localization using YOLOv8, persistent track ID assignment using ByteTrack, and coordinate smoothing via Exponential Moving Average (EMA).
* **Silhouette & GEI Generation**: Neural silhouette segmentation using an ONNX UNet model (with morphological Otsu thresholding as automatic fallback), accumulated into a normalized $128 \times 64$ Gait Energy Image.
* **Feature Extraction**: Convolutional gait representation using `ByGaitLight` with Horizontal Part Pooling (HPP, `part_bins=4`), producing unit-normalized 256-dimensional embeddings.
* **Biometric Gallery**: Secure vector store with `allow_pickle=False` enforcement for gallery matching and subject enrollment.
* **Hardware Arbitration**: Centralized runtime device management (`DeviceManager`) ensuring automatic CUDA acceleration on compatible NVIDIA hardware and validated CPU fallback.
* **Surveillance Ingestion**: Multithreaded camera worker architecture supporting Windows webcams and RTSP IP streams with automatic reconnection logic.

---

## Core Gait Recognition Pipeline

The implemented gait recognition pipeline processes video inputs through sequential stages without ad-hoc branching:

1. **Input Ingestion**: Video frames are captured at native frame rates from local webcams or RTSP network streams via isolated `CameraWorker` threads.
2. **Person Detection**: `PersonDetector` (`pipeline/detection/person_detector.py`) runs YOLOv8 to locate human bounding boxes (`class=0`) with configurable confidence and IoU thresholds.
3. **Multi-Object Tracking**: `TrackingStep` (`pipeline/steps/tracking.py`) maintains identity continuity across occlusions and motion using ByteTrack, applying EMA filtering to stabilize box dimensions.
4. **Crop Normalization**: Detected bounding boxes are cropped, aspect-ratio constrained, and centered to an $85\%$ relative height target.
5. **Silhouette Segmentation**: `SilhouetteExtractor` (`pipeline/silhouette/extractor.py`) executes `models/weights/silhouette_segmenter.onnx` to isolate binary body masks. If the ONNX engine is unavailable or uncalibrated, it automatically falls back to an adaptive Otsu morphological segmentation strategy.
6. **Live GEI Accumulator**: `LiveGEIStep` (`pipeline/steps/live_gei.py`) aggregates aligned silhouette frames across a temporal window to generate a single 2D Gait Energy Image ($128 \times 64$ pixels).
7. **ByGaitLight Feature Encoding**: `ByGaitLight` (`models/architectures/bygait_light.py`) processes the GEI through convolutional feature extractors and Horizontal Part Pooling (`part_bins=4`), producing a 256-dimensional feature vector normalized via L2 norm ($\|e\|_2 = 1.0000$).
8. **VectorStore Gallery Matching**: `MatchingStep` (`pipeline/steps/matching_step.py`) calculates cosine similarity against enrolled gallery embeddings loaded from `models/live_gallery/` or `models/gallery/`.
9. **Open-Set Decision Logic**: `OpenSetRecognizer` (`intelligence/open_set_recognizer.py`) classifies the match into `KNOWN`, `UNKNOWN`, or `UNCERTAIN` based on calibrated similarity thresholds and top-1/top-2 margin constraints.

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
Downstream Component Binding (YOLO / PyTorch / ONNX / ByGaitLight)
```

### Authoritative `DeviceManager` (`automation/device_manager.py`)

`DeviceManager` acts as the single source of truth across the entire system:
* Components resolve their target device by calling `DeviceManager.get_instance().resolve_component_device(requested)`.
* Requesting `'auto'` or `'cuda'` resolves to `'cuda:0'` when CUDA is available and verified healthy; otherwise, it resolves to `'cpu'`.
* PyTorch tensor operations, YOLO detection, ByteTrack tracking, ByGaitLight inference, and ONNX Runtime sessions strictly follow the authoritative device state.

### Compute Execution Modes

| Subsystem | CUDA Mode (GPU Accelerated) | CPU Mode (Fallback / Validation) |
| :--- | :--- | :--- |
| **Authoritative State** | `EnvironmentState.CUDA_READY` | `EnvironmentState.CPU_READY` |
| **Resolved Device** | `cuda:0` | `cpu` |
| **PyTorch Tensor Device** | `torch.device('cuda:0')` | `torch.device('cpu')` |
| **YOLO PersonDetector** | `cuda:0` | `cpu` |
| **ByteTrack Tracking** | `cuda:0` | `cpu` |
| **ByGaitLight CNN** | `cuda:0` | `cpu` |
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

## Verified Development Environment

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
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
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
    CONNECTED --> STANDBY: User Calls /cameras/stop
    FAILED --> STANDBY: Worker Reset
```

> **Lifecycle Constraint**: `STANDBY` is the initial camera state. `FAILED` represents an outcome following an attempted connection, never the default idle state.

### Ingestion Features
* **Safe Device Probing**: Automatically detects local webcams using DirectShow (`CAP_DSHOW`) and MSMF backends on Windows without relying on hardcoded index `0`.
* **Runtime Source Labeling**: Source type is hidden during `STANDBY` and dynamically updates to `"Webcam"` or `"RTSP"` upon first frame acquisition.
* **Worker Isolation**: Each camera runs on a dedicated thread with an internal frame buffer, client reference counting, and configurable frame rate throttling (`preview_max_fps`).
* **Live MJPEG Streaming**: Serves real-time video with localized bounding boxes, track IDs, and recognition status over HTTP (`/api/v1/cameras/{id}/stream`).

---

## REST API & WebSocket Services

The backend API is implemented in FastAPI (`api/server.py` and `api/v1/router.py`) and executed via Uvicorn.

### Route Reference (`/api/v1`)

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health status, pipeline loaded state, and active backend. |
| `GET` | `/api/v1/status` | Operational status, device telemetry, threshold configs, and gallery metrics. |
| `GET` | `/api/v1/metrics` | System counters (processed images, videos, active tracks). |
| `POST` | `/api/v1/identify/image` | Single-image person detection and gait identification. |
| `POST` | `/api/v1/analyze/video` | Uploaded video analysis and sampled gait recognition. |
| `POST` | `/api/v1/enroll` | Subject biometric enrollment (`person_id` + image files). |
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

### Verified `/api/v1/status` Response Payload

```json
{
  "status": "operational",
  "device": "cuda",
  "compute": {
    "backend": "cuda",
    "device": "cuda:0",
    "gpu": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
    "vram_mb": 6144.0,
    "cuda_available": true,
    "pytorch_version": "2.5.1+cu121",
    "cuda_version": "12.1",
    "onnx_provider": "CUDAExecutionProvider"
  },
  "thresholds": {
    "known_threshold": 0.85,
    "unknown_threshold": 0.70,
    "margin_threshold": 0.05
  },
  "gallery": {
    "total_identities": 2,
    "total_embeddings": 24
  },
  "active_cameras": 0
}
```

---

## Frontend Surveillance Dashboard

The frontend application (`frontend/`) is built with React 19, Vite, and Lucide Icons.

### Implemented UI Capabilities
* **CCTV Network (`CctvNetwork.jsx`)**: Displays camera cards, live MJPEG stream feeds, connection state badges (`STANDBY`, `CONNECTING`, `CONNECTED`, `RECONNECTING`), and stream start/stop controls.
* **Gait System Status (`GaitSystemStatus.jsx`)**: Real-time display of backend status, active compute backend (`CUDA` / `CPU`), GPU device name, VRAM usage, and active execution providers.
* **Recognition Events (`RecognitionEvents.jsx`)**: Live WebSocket event feed showing subject ID, match confidence, camera location, and timestamp.
* **Event History (`History.jsx`)**: Searchable and filterable recognition event timeline.
* **Geospatial Map (`Map.jsx`)**: Interactive Leaflet map displaying registered camera zone placements.
* **Case Management (`ReportCase.jsx`, `CaseDetails.jsx`)**: Person of interest registration and alert routing.

---

## Security & Engineering Hardening

1. **Safe PyTorch Checkpoint Loading**: All weight loading in `models/inference/pytorch_backend.py` enforces `torch.load(..., weights_only=True)`, preventing arbitrary code execution from untrusted model files.
2. **Hardened Vector Store**: `storage/vector_store.py` enforces `allow_pickle=False` and rejects object-type NumPy arrays (`dtype == object`), mitigating deserialization vulnerabilities.
3. **Encrypted Credentials & Log Masking**: RTSP credentials are encrypted via Fernet (`.credentials.key`) and masked in logs (`rtsp://***:***@host:port`).
4. **Lazy Module Access**: `automation/__init__.py` uses PEP 562 lazy module `__getattr__`, preventing `runpy` `RuntimeWarning: 'automation.bootstrap' found in sys.modules` when running `python -m automation.bootstrap`.

---

## Project Structure

```text
E:\ARGUS_AI
├── api/                        # FastAPI REST routing, server lifecycle, and schemas
│   ├── v1/router.py            # API v1 endpoint implementations
│   ├── schemas.py              # Pydantic request and response models
│   └── server.py               # Application factory, lifespan context, and CORS
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
│   ├── detection.yaml          # YOLOv8 detector confidence, IoU, and device settings
│   ├── inference.yaml          # Inference policy and quality thresholds
│   └── system.yaml             # Thread limits, storage paths, and logging bindings
├── core/                       # Shared utilities, threshold manager, and logging setup
├── frontend/                   # React 19 + Vite surveillance dashboard application
│   ├── src/                    # UI components, camera views, and WebSocket clients
│   ├── package.json            # Frontend dependency manifest
│   └── vite.config.js          # Vite development server configuration
├── intelligence/               # Open-set classification, ReID fusion, and timeline reconstruction
├── models/                     # Deep learning architectures and gallery storage
│   ├── architectures/          # ByGaitLight, UNet segmenter, and ArcFace loss functions
│   ├── gallery/                # Baseline gallery vectors (.npy)
│   ├── live_gallery/           # Active runtime gallery (24 embeddings, 2 identities)
│   └── weights/                # Model weights (silhouette_segmenter.onnx, yolov8n.pt)
├── monitoring/                 # Structured logging and process metrics
├── pipeline/                   # Modular gait recognition pipeline steps
│   ├── detection/              # PersonDetector (YOLOv8)
│   ├── silhouette/             # SilhouetteExtractor (UNet ONNX + Otsu fallback)
│   └── steps/                  # Tracking, Live GEI, Feature Extraction, Matching
├── scripts/                    # Automation, diagnostic, evaluation, and benchmark scripts
│   ├── bootstrap_env.ps1       # Windows PowerShell environment bootstrap entry point
│   ├── detect_environment.py   # CLI hardware and compute detector
│   ├── dev.js                  # Unified backend + frontend development orchestrator
│   ├── doctor.py               # Pre-flight deployment health diagnostics
│   └── verify_environment.py   # 6-phase environment verification suite
├── security_layer/             # Credential encryption and access control manager
├── services/                   # GaitService, CameraWorker, and CameraSourceResolver
├── storage/                    # Hardened VectorStore implementation
├── streaming/                  # Video capture and streaming utilities
├── tests/                      # Automated test suite
│   └── unit/                   # Unit tests (133 tests passed)
├── requirements.txt            # Python dependencies manifest
└── VERSION                     # Project version file (0.1.0)
```

---

## Installation & Windows Setup

### Prerequisites
* **Python**: 3.11.x (64-bit)
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

### 3. Install Standard Dependencies

```powershell
pip install -r requirements.txt
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

Execute the diagnostic suite to verify environment health and code correctness:

```powershell
# 1. Targeted automation unit tests (13 tests)
.\.venv\Scripts\python.exe -u -m unittest tests/unit/test_automation.py

# 2. Complete unit test discovery (133 tests)
.\.venv\Scripts\python.exe -u -m unittest discover tests/unit

# 3. Python bytecode compilation check
.\.venv\Scripts\python.exe -m compileall -q -x "\.venv" .

# 4. Hardware and compute capability detection
.\.venv\Scripts\python.exe scripts/detect_environment.py

# 5. Full 6-phase environment verification suite
.\.venv\Scripts\python.exe scripts/verify_environment.py

# 6. Pre-flight health doctor check
.\.venv\Scripts\python.exe scripts/doctor.py

# 7. CPU fallback bootstrap validation
.\.venv\Scripts\python.exe -u -m automation.bootstrap --force-cpu

# 8. Standard CUDA bootstrap validation
.\.venv\Scripts\python.exe -u -m automation.bootstrap
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

## Current Project Validation Status & Limitations

### Validation Status Matrix

| Subsystem / Capability | Status | Verified Evidence |
| :--- | :---: | :--- |
| **Environment Auto-Discovery** | **PASS** | `scripts/detect_environment.py` (HEALTHY) |
| **CUDA GPU Acceleration** | **PASS** | `scripts/verify_environment.py` (NVIDIA RTX 3050 6GB verified) |
| **CPU Fallback Mode** | **PASS** | `python -m automation.bootstrap --force-cpu` (All components on CPU) |
| **Authoritative DeviceManager**| **PASS** | `tests/unit/test_automation.py` (13/13 tests passed) |
| **Bytecode Compilation** | **PASS** | `compileall` (0 syntax or compile errors) |
| **Full Unit Test Suite** | **PASS** | `tests/unit` (**133 passed**, 0 failed) |
| **Pre-Flight Health Doctor** | **PASS** | `scripts/doctor.py` (`READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING`) |
| **Person Detection (YOLOv8)** | **PASS** | Validated on `cuda:0` and `cpu` via `PersonDetector` |
| **Silhouette Segmentation** | **PASS** | Validated UNet ONNX inference + Otsu fallback |
| **ByGaitLight CNN Encoder** | **PASS** | Output shape `[1, 256]`, L2 norm `1.0000` verified |
| **Gallery Storage Security** | **PASS** | `allow_pickle=False` validated with 24 embeddings across 2 identities |
| **FastAPI REST Server** | **PASS** | Uvicorn startup verified, `/api/v1/status` operational |
| **Camera Ingestion State Machine**| **PASS** | Standby $\rightarrow$ Connecting $\rightarrow$ Connected lifecycle verified |
| **Controlled Gait Recognition**| **READY** | Ready for staged controlled testing |
| **Multi-Camera Field Validation**| **PENDING** | Multi-camera physical deployment validation pending field trials |
| **Production-Scale Capacity** | **NOT CLAIMED** | Tested on 24 development embeddings; large-scale gallery unmeasured |

### Known Limitations
1. **Development Gallery Baseline**: The active test gallery currently contains 24 embeddings across 2 enrolled identities. Large-scale gallery matching latency and capacity remain to be evaluated.
2. **Field Validation Pending**: While unit tests, component smoke tests, and synthetic video stream pipelines pass, physical multi-camera field trials in unconstrained real-world environments remain pending.
3. **Clothing Covariate Sensitivity**: As established in CASIA-B ablation benchmarks, clothing changes (`CL` accuracy = 42.64%) degrade silhouette geometry more significantly than carrying conditions (`BG` = 78.26%).
4. **Hardware Dependency**: Full real-time FPS throughput is dependent on a compatible NVIDIA GPU with CUDA acceleration. While CPU execution is functional, throughput will be lower on CPU-only machines.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Maintainer

**Chanuka Sandun**  
Undergraduate in Cybersecurity  
Developer of the ARGUS AI Gait Recognition Framework  
* GitHub: [@chanuka8](https://github.com/chanuka8)  
* LinkedIn: [linkedin.com/in/chanukasandun](https://www.linkedin.com/in/chanukasandun/)
