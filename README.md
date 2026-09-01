# ARGUS AI

![ARGUS AI Gait Recognition Banner](assets/github/Gitrepo_profilepic.png)

## AI-Powered Gait Recognition, Dual-Modal Biometrics & Surveillance Intelligence System

ARGUS AI is an enterprise-grade visual biometric intelligence framework designed for real-time human gait recognition, dual-modal appearance re-identification (ReID), multi-camera surveillance intelligence, and date-aware continual learning with multi-gate validation. The system extracts dynamic walking kinematics from silhouette sequences and deep appearance features from person crops to perform open-set identity matching, track individuals across camera networks, reconstruct forensic event timelines, and adaptively calibrate through continual learning validation gates.

Built on PyTorch, ONNX Runtime, OpenCV, FastAPI, and React 19, ARGUS AI incorporates an automated hardware-aware compute arbitration layer (`DeviceManager`), a multi-camera inference engine with fair-share scheduling, an admission controller preventing host resource exhaustion, encrypted RTSP credential resolution, a hardened local vector store, and a responsive, resizable surveillance dashboard.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows / Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%2F%2012.6%20Driver-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20.0-blue.svg)](https://onnxruntime.ai/)
[![Tests: 849 Passed](https://img.shields.io/badge/tests-849%20passed%20(100%25)-brightgreen.svg)](tests)
[![Frontend: React 19 + Vite](https://img.shields.io/badge/frontend-React%2019%20%2B%20Vite-61DAFB.svg)](frontend)
[![Status: Runtime Verified](https://img.shields.io/badge/status-RUNTIME%20VERIFIED-orange.svg)](docs/reports)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Dual-Modal Recognition Pipeline](#dual-modal-recognition-pipeline)
   * [Gait Recognition Pipeline (ByGaitLight 256D)](#gait-recognition-pipeline-bygaitlight-256d)
   * [Appearance ReID Pipeline (OSNet 512D)](#appearance-reid-pipeline-osnet-512d)
   * [Dual-Modal Fusion & Score Calibration](#dual-modal-fusion--score-calibration)
   * [Open-Set Decision Logic](#open-set-decision-logic)
3. [Missing Person Reference Data Flow vs. Live Operational Evidence](#missing-person-reference-data-flow-vs-live-operational-evidence)
   * [User-Added Reference Data Flow (Gallery / Immediate Recognition)](#user-added-reference-data-flow-gallery--immediate-recognition)
   * [Live Operational Evidence Flow (Continual Learning)](#live-operational-evidence-flow-continual-learning)
4. [Firebase Architecture & Failure Boundary](#firebase-architecture--failure-boundary)
   * [Inference Source Separation](#inference-source-separation)
   * [Firebase Data Contract & Canonical Schema](#firebase-data-contract--canonical-schema)
   * [Failure Isolation & Offline Fallback](#failure-isolation--offline-fallback)
5. [Operational Embedding Lifecycle & State Machine](#operational-embedding-lifecycle--state-machine)
6. [Date-Aware Continual Learning & NN Training](#date-aware-continual-learning--nn-training)
   * [Date-Aware Scheduling & Future-Date Protection](#date-aware-scheduling--future-date-protection)
   * [Historical Replay Buffer (Anti-Catastrophic Forgetting)](#historical-replay-buffer-anti-catastrophic-forgetting)
   * [Real Neural Network Fine-Tuning](#real-neural-network-fine-tuning)
   * [Multi-Gate Candidate Validation](#multi-gate-candidate-validation)
   * [Atomic Model Registry Promotion & Instant Rollback](#atomic-model-registry-promotion--instant-rollback)
7. [Hardware-Aware Compute Automation](#hardware-aware-compute-automation)
   * [Arbitration Architecture](#arbitration-architecture)
   * [12-Stage Environment Bootstrap](#12-stage-environment-bootstrap)
   * [Compute Verification & CPU Fallback Parity](#compute-verification--cpu-fallback-parity)
8. [Multi-Camera Surveillance & Ingestion Engine](#multi-camera-surveillance--ingestion-engine)
   * [Camera Lifecycle State Machine](#camera-lifecycle-state-machine)
   * [Multi-Camera Fair-Share Scheduling](#multi-camera-fair-share-scheduling)
   * [Hardware Admission Control](#hardware-admission-control)
9. [Frontend Surveillance Dashboard](#frontend-surveillance-dashboard)
   * [Responsive & Resizable Dock Architecture](#responsive--resizable-dock-architecture)
   * [Live CCTV Surveillance Grid](#live-cctv-surveillance-grid)
   * [Geospatial Mapping & Case Management](#geospatial-mapping--case-management)
10. [REST API & WebSocket Services](#rest-api--websocket-services)
11. [Security & Engineering Hardening](#security--engineering-hardening)
12. [Benchmark Results & Scientific Evidence](#benchmark-results--scientific-evidence)
13. [Project Structure](#project-structure)
14. [Installation & Windows Setup](#installation--windows-setup)
15. [Running the Application](#running-the-application)
16. [Verification & Testing](#verification--testing)
17. [Current Implementation Status](#current-implementation-status)
18. [Known Limitations](#known-limitations)
19. [License & Maintainer](#license--maintainer)

---

## System Overview

ARGUS AI functions as a decoupled, multi-modal visual biometric intelligence platform. The primary biometric identifier is human gait—identifying individuals by their dynamic body geometry and walking cadence over consecutive frames without requiring facial visibility or cooperative subject posture. This is augmented with an appearance re-identification (OSNet) feature stream for robust cross-camera tracking and multi-modal fusion.

```mermaid
graph TD
    A[Camera Streams: Webcam / RTSP] --> B[Person Detection: YOLOv8]
    B --> C[Multi-Object Tracking: ByteTrack + EMA]
    C --> D[Silhouette Extraction: UNet ONNX / Otsu]
    C --> E[Appearance Extraction: OSNet-x0.25 512D]
    D --> F[Cycle-Aware Live GEI: 128x64]
    F --> G[ByGaitLight CNN: HPP part_bins=4]
    G --> H[Gait Embedding: 256D L2 Norm]
    H --> I[Local Gait VectorStore Matching]
    E --> J[Local Appearance VectorStore Matching]
    I --> K[Dual-Modal Fusion & Score Calibration]
    J --> K
    K --> L[Open-Set Decision: KNOWN / UNKNOWN / UNCERTAIN]
    L --> M[Multi-Camera Track Fusion & Forensic Timeline]
    L --> N[Live MJPEG Stream Overlays & WebSocket Alerts]
    L --> O[Operational Embedding Collector: PREDICTED]
    O --> P[Operator Verification: VERIFIED]
    P --> Q[Quality Gate: TRAINING_ELIGIBLE]
    Q --> R[Date-Aware Scheduler & NN Fine-Tuner]
```

### Primary Subsystems

* **Spatial Tracking**: Bounding box localization using YOLOv8, persistent track ID assignment using ByteTrack, and coordinate smoothing via Exponential Moving Average (EMA, $\alpha=0.35$).
* **Silhouette & GEI Generation**: Neural silhouette segmentation using an ONNX UNet model (with morphological Otsu thresholding as automatic fallback), accumulated into a normalized $128 \times 64$ Gait Energy Image (GEI).
* **Gait Feature Extraction**: Convolutional gait representation using `ByGaitLight` with Horizontal Part Pooling (HPP, `part_bins=4`), producing unit-normalized 256-dimensional embeddings ($\|e\|_2 = 1.0000$).
* **Appearance ReID Extraction**: Deep appearance feature extraction using `OSNet-x0.25`, producing unit-normalized 512-dimensional feature vectors for clothes-consistent short-term re-identification.
* **Dual-Modal Score Fusion**: Calibrated multi-modal fusion combining gait similarity and appearance similarity with dynamic reliability scoring and quality assessment.
* **Biometric Gallery & Database**: Hardened local vector stores (`models/live_gallery/`, `models/appearance_gallery/`) and structured SQLite embedding database (`storage/embedding_database.py`) providing zero-latency real-time inference.
* **Asynchronous Firebase Persistence**: Non-blocking Firestore synchronization for biometric metadata, lineage tracking, audit trails, and candidate model records.
* **Date-Aware Continual Learning**: Event-date driven background candidate model training (`NNFineTuner`), 50% historical replay mixing to prevent catastrophic forgetting, future-date contamination protection, and multi-gate safety validation (`CandidateValidator`).
* **Multi-Camera Engine & Admission Control**: Fair-share frame scheduling (Deficit Round-Robin + Priority Aging), decoupled per-camera queues with backpressure and stale-frame drop protection, and pre-flight capacity admission control.
* **Responsive Surveillance Frontend**: React 19 single-page application featuring resizable panels (`ResizeHandle`, `useResizablePanel`), adaptive 16:9 CCTV grid, live MJPEG feeds, geospatial mapping, case management, and administrative control.

---

## Dual-Modal Recognition Pipeline

### Gait Recognition Pipeline (ByGaitLight 256D)

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
[Stage 7] L2 Normalization (256D Embedding, ||e||₂ = 1.0000)
      ↓
[Stage 8] VectorStore Cosine Similarity Matching (Local Gallery Comparison)
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

### Appearance ReID Pipeline (OSNet 512D)

In addition to silhouette-based gait recognition, ARGUS AI incorporates an appearance re-identification pipeline:

* **Backbone**: `OSNet-x0.25` (`models/reid/osnet_backbone.py`, weights: `models/weights/osnet_x0_25.pth`) lightweight omni-scale network for person re-identification.
* **Feature Representation**: 512-dimensional L2-normalized feature embeddings extracted directly from RGB person crops.
* **Appearance Gallery**: Separate appearance vector store (`models/appearance_gallery/`) managed via `VectorStore`.

### Dual-Modal Fusion & Score Calibration

The fusion layer (`DualModalFusion` / `LearnedFusion` in `intelligence/dual_modal_fusion.py`) combines gait similarity score $S_{\text{gait}}$ and appearance similarity score $S_{\text{app}}$:

$$S_{\text{fused}} = w_{\text{gait}} \cdot S_{\text{gait}} + w_{\text{app}} \cdot S_{\text{app}}$$

with dynamic weight attenuation based on silhouette quality estimation, camera viewpoint angle, and temporal track length.

### Open-Set Decision Logic

`OpenSetRecognizer` (`intelligence/open_set_recognizer.py`) evaluates fused similarity scores against dual calibration thresholds:

* **`KNOWN`**: Fused similarity $\ge \tau_{\text{accept}}$ and top-1/top-2 margin $\ge \Delta_{\text{margin}}$.
* **`UNKNOWN`**: Fused similarity $< \tau_{\text{reject}}$.
* **`UNCERTAIN`**: Intermediate similarity score requiring multi-frame temporal consensus or operator review.

---

## Missing Person Reference Data Flow vs. Live Continual Learning Flow

A foundational design rule of ARGUS AI is the strict separation between **User Reference Data** (initial gallery/watchlist targets) and **Live Operational Evidence** (continual learning candidates).

```text
===================================================================================
                       DATA FLOW SEPARATION ARCHITECTURE
===================================================================================

[USER REFERENCE DATA FLOW]                       [LIVE OPERATIONAL EVIDENCE FLOW]
User adds Missing Person / Target                Live CCTV Camera Feed
          ↓                                                ↓
Reference Photos / Videos / GEI                  Person Detection & Tracking
          ↓                                                ↓
Feature Extraction (256D Gait, 512D App)         Feature Extraction (256D Gait, 512D App)
          ↓                                                ↓
Quality Validation (L2 Norm, Finite Check)       Operational Observation (State: PREDICTED)
          ↓                                                ↓
Local EmbeddingDatabase & VectorStore            Operator Confirmation (State: VERIFIED)
(IMMEDIATELY ACTIVE FOR MATCHING)                          ↓
          ↓                                      Quality Gate Check (Quality >= 0.70)
Firebase Persistence (Async Lineage)             (State: TRAINING_ELIGIBLE)
(identity_type: USER_REFERENCE)                            ↓
(training_eligibility: NOT_ELIGIBLE)             Date-Aware Learning Scheduler (Past/Today)
          ↓                                                ↓
[EXCLUDED FROM CONTINUAL LEARNING]               Training Dataset Builder (50% Replay Buffer)
                                                           ↓
                                                 NN Fine-Tuner (ByGaitLight / OSNet)
                                                           ↓
                                                 Candidate Checkpoint (models/candidates/*.pth)
                                                           ↓
                                                 Candidate Multi-Gate Validation (FAR/TAR/Stab)
                                                           ↓
                                                 Atomic Promotion & Live Model Reload
                                                           ↓
                                                 State Transition: TRAINING_CONSUMED
===================================================================================
```

### User-Added Reference Data Flow (Gallery / Immediate Recognition)

1. **Target Registration**: Operator enrolls a missing person or person of interest via `MissingPersonWorkflow` (`intelligence/missing_person_workflow.py`) or `EnrollmentLifecycleManager` (`enrollment/enrollment_lifecycle.py`).
2. **Feature Extraction**: Biometric features are extracted: 256D gait embedding (ByGaitLight) and 512D appearance embedding (OSNet).
3. **Quality Validation**: Vectors are checked for finite values, non-zero norm, and L2 normalization ($\|e\|_2 = 1.0000$).
4. **Local Gallery Insertion**: Enrolled embeddings are inserted into local `EmbeddingDatabase` and saved to `models/live_gallery/` and `models/appearance_gallery/` `.npy` stores for zero-latency live matching.
5. **Firebase Persistence**: Persisted asynchronously to Firestore with:
   * `identity_type = "USER_REFERENCE"`
   * `source_type = "user_reference"`
   * `operational_state = "REFERENCE"`
   * `training_eligibility = "NOT_ELIGIBLE"`
6. **Training Exclusion Guarantee**: User-added reference embeddings are **INITIAL REFERENCE / GALLERY DATA**. They are **EXCLUDED** from continual learning training datasets to prevent overfitting to sparse reference samples.

### Live Operational Evidence Flow (Continual Learning)

1. **Live Observation**: CCTV streams produce observations recorded with `state = PREDICTED`.
2. **Operator Verification**: Human-in-the-loop confirmation transitions observation to `state = VERIFIED`.
3. **Quality & Stability Gate**: Verified observations with $\text{quality\_score} \ge 0.70$, valid dimensions (256D/512D), finite values, and non-zero norms transition to `state = TRAINING_ELIGIBLE`.
4. **Date-Aware Scheduling**: `DateAwareLearningScheduler` scans for eligible observations grouped by capture date.
5. **Training Dataset Assembly**: Combines 50% eligible date observations with 50% historical replay baseline embeddings.
6. **NN Fine-Tuning**: `NNFineTuner` executes real PyTorch gradient descent on ByGaitLight or OSNet.
7. **Candidate Validation**: Evaluates candidate weights against the active baseline across 5 safety gates.
8. **Atomic Promotion**: Promoted candidate is activated in `ModelRegistry` without downtime.
9. **Training Consumption**: Consumed observations transition to `state = TRAINING_CONSUMED` with `consumed_by_model` and `consumed_in_job` metadata, permanently preventing duplicate consumption.

---

## Firebase Architecture & Failure Boundary

### Inference Source Separation

> [!IMPORTANT]
> **Firebase is NOT the real-time inference database.**
> Real-time recognition runs exclusively against the local `EmbeddingDatabase` and memory-mapped `VectorStore` (`.npy` files).

* **Local EmbeddingDatabase + VectorStore**: Sole source for real-time CCTV inference, spatial tracking matching, and live watchlist alerts. Latency: $< 1.5\text{ms}$ per candidate match.
* **Firebase Firestore (`FirebaseEmbeddingStore`)**: Asynchronous store for persistence, identity metadata, embedding provenance, lineage tracking, audit trails, and candidate model records.

### Firebase Data Contract & Canonical Schema

Every persisted embedding conforms to the canonical `FirebaseEmbeddingDocument` contract (`storage/firebase_embedding_store.py`):

| Conceptual Field | Canonical Dataclass Property | Type | Description |
|---|---|---|---|
| `embedding_id` | `embedding_id` | `str` | Deterministic SHA-256 derived identifier (`emb_{modality}_{person}_{ts}_{hash}`). |
| `identity_id` | `person_id` | `str` | Unique subject identifier (e.g. `Missing_Person_101`). |
| `embedding_type` | `modality` | `str` | Modality tag: `"gait"` (256D) or `"appearance"` (512D). |
| `embedding_dimension` | `embedding_dim` | `int` | Exact vector dimension: `256` for gait, `512` for appearance. |
| `vector` | `vector` / `embedding` | `list[float]` | L2-normalized floating-point feature embedding. |
| `model_version` | `model_version` | `str` | Model version used for extraction (e.g. `"v1.0.0"`). |
| `identity_type` | `identity_type` | `str` | `"USER_REFERENCE"` vs `"LIVE_OPERATIONAL"`. |
| `source_type` | `source_type` | `str` | `"user_reference"`, `"live_surveillance"`, `"enrollment"`. |
| `operational_state` | `operational_state` | `str` | `"PREDICTED"`, `"VERIFIED"`, `"TRAINING_ELIGIBLE"`, `"TRAINING_CONSUMED"`, `"REFERENCE"`. |
| `training_eligibility`| `training_eligibility` | `str` | `"NOT_ELIGIBLE"` vs `"ELIGIBLE"`. |
| `observation_date` | `observation_date` / `capture_date` | `str` | ISO date string (`YYYY-MM-DD`). |
| `provenance` | `provenance` | `dict` | Camera ID, track ID, confidence, bounding box metadata. |
| `lineage_id` | `lineage_id` | `str` | Ancestor lineage identifier tracing model and dataset generation. |

### Failure Isolation & Offline Fallback

```text
[Live Camera Inference] ──► [Local VectorStore (.npy)] ──► [Instant Match Result (< 1.5ms)]
                                    │
                                    ▼ (Non-blocking Async Thread)
                         [Firebase Embedding Store]
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                 [Online / Cloud]      [Offline Fallback]
                 Firebase Firestore    Local JSON Store & Retry Queue
```

* **Non-Blocking Execution**: Persistence calls are executed asynchronously or isolated in try-except handlers.
* **Zero Inference Disruption**: Complete network failure or Firestore service outages will **never** stall, delay, or crash live CCTV recognition.
* **Automatic Retry Queue**: Offline transactions are queued (`data/firebase_offline_store.json`) and automatically retried upon connection restoration.
* **Disaster Recovery Rebuild**: `EmbeddingDatabase.rebuild_from_firebase()` enables 100% gallery recovery from cloud snapshots in disaster recovery scenarios.

---

## Operational Embedding Lifecycle & State Machine

Every live observation transitions through an immutable state machine managed by `OperationalEmbeddingCollector` (`intelligence/operational_embedding_collector.py`):

```text
┌────────────────┐
│   PREDICTED    │ ─── Initial state upon live CCTV capture and feature extraction.
└───────┬────────┘
        │
        ▼ (Operator confirmation / High-confidence consensus)
┌────────────────┐
│    VERIFIED    │ ─── Identity verified by operator; awaiting quality evaluation.
└───────┬────────┘
        │
        ▼ (Quality >= 0.70, Finite vector, Valid dimension, Identity != USER_REFERENCE)
┌────────────────┐
│TRAINING_ELIGIBLE│ ─── Available for date-aware continual learning.
└───────┬────────┘
        │
        ▼ (Consumed by NNFineTuner during candidate model training)
┌────────────────┐
│TRAINING_CONSUMED│ ─── Sample consumed; permanently locked from duplicate retraining.
└────────────────┘
```

* **Unverified Guard**: Unverified observations (`PREDICTED`) can **never** enter training datasets.
* **Reference Data Guard**: User reference embeddings (`USER_REFERENCE`) are assigned `training_eligibility = "NOT_ELIGIBLE"` and can **never** enter continual learning training pools.
* **Duplicate Consumption Guard**: Once marked `TRAINING_CONSUMED`, observations cannot be retrained or re-verified.
* **Transition Validation**: Illegal transitions (e.g. `PREDICTED` directly to `TRAINING_CONSUMED`) are strictly rejected.

---

## Date-Aware Continual Learning & NN Training

### Date-Aware Scheduling & Future-Date Protection

`DateAwareLearningScheduler` (`intelligence/date_aware_learning_scheduler.py`) ensures that model updates are organized strictly by chronological capture dates:

* **Event-Date Grouping**: Observations are aggregated by `observation_date` (`YYYY-MM-DD`).
* **Minimum Threshold Gate**: Training is triggered only when a date contains $\ge 10$ eligible embeddings across $\ge 2$ distinct identities (`min_training_embeddings`, `min_identities`).
* **Future-Date Protection Gate**: Jobs for dates in the future ($\text{training\_date} > \text{today}$) are automatically rejected with `status = LearningJobStatus.REJECTED` to prevent timestamp contamination.
* **Idempotency**: Active or completed jobs for a given date and model type are skipped to prevent duplicate execution.

### Historical Replay Buffer (Anti-Catastrophic Forgetting)

`TrainingDatasetBuilder` (`intelligence/training_dataset_builder.py`) constructs balanced training datasets containing:

* **50% New Date Evidence**: Verified, training-eligible observations from the target date.
* **50% Historical Replay Baseline**: Anchor embeddings drawn from the historical baseline gallery to prevent catastrophic forgetting of previously learned identities.

### Real Neural Network Fine-Tuning

ARGUS AI executes **genuine PyTorch neural network training** via `NNFineTuner` (`intelligence/nn_fine_tuner.py`):

```text
Active Weights (.pth) + Training Dataset (50% New + 50% Historical)
                             ↓
                 [PyTorch Gradient Descent]
            Loss: Triplet Loss + ArcFace Margin
                 Optimizer: Adam (lr=1e-4)
                             ↓
              [Parameter Delta Verification]
             changed_tensors > 0, max_param_delta > 0
                             ↓
        Candidate Checkpoint Saved: models/candidates/*.pth
```

* **ByGaitLight (256D)**: Convolutional feature backbone fine-tuning using GEI sequences, Horizontal Part Pooling (`part_bins=4`), and metric learning losses.
* **OSNet (512D)**: Appearance ReID backbone fine-tuning using RGB person crops.
* **Tensor Parameter Delta Verification**: Verifies that actual model weights changed by computing tensor deltas against the baseline:
  * `changed_tensors > 0`
  * `max_param_delta > 0.0`
  * Checkpoint integrity verified with SHA-256 hash.

### Multi-Gate Candidate Validation

Before any candidate model can be promoted to production, `CandidateValidator` (`intelligence/candidate_validator.py`) evaluates it against the active baseline:

1. **FAR Security Gate**: Candidate False Accept Rate must not exceed baseline FAR ($\text{FAR}_{\text{cand}} \le \text{FAR}_{\text{base}}$). Zero security regression allowed.
2. **TAR Performance Gate**: Candidate True Accept Rate must not regress beyond tolerance ($\text{TAR}_{\text{cand}} \ge \text{TAR}_{\text{base}} - 0.005$).
3. **Stability & Dimension Gate**: Output dimensions must strictly match (256D for ByGaitLight, 512D for OSNet) with finite, normalized embeddings.
4. **Anti-Churn Gate**: Rejects candidate models whose performance delta is within random noise without meaningful gains.

### Atomic Model Registry Promotion & Instant Rollback

* **Atomic Promotion**: `ModelRegistry.promote_version()` (`models/model_registry.py`) updates the active model pointer in `models/model_registry.json`. Running workers hot-reload the new weights seamlessly.
* **Automatic Rollback**: If post-promotion monitoring detects anomalies, `ModelRegistry.rollback()` instantly reverts the active model to `previous_production_version` in $< 50\text{ms}$.

---

## Hardware-Aware Compute Automation

### Arbitration Architecture

ARGUS AI eliminates hardcoded device assignments by routing all compute queries through a centralized arbitration layer:

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

`DeviceManager` (`automation/device_manager.py`) acts as the single source of truth across the entire system. Requesting `'auto'` or `'cuda'` resolves to `'cuda:0'` when CUDA is verified healthy; otherwise, it resolves deterministically to `'cpu'`.

### 12-Stage Environment Bootstrap

The bootstrap orchestrator (`automation/bootstrap.py`) performs a deterministic 12-stage discovery sequence:

| Stage | Identifier | Verification Action |
| :---: | :--- | :--- |
| **01** | `Operating System` | Detects OS name, version, and architecture (e.g. Windows 10 AMD64). |
| **02** | `Python Runtime` | Validates Python interpreter version (3.11.x 64-bit). |
| **03** | `Hardware Profile` | Probes CPU core count, available RAM, and NVIDIA GPU presence/VRAM. |
| **04** | `NVIDIA Driver` | Queries installed GPU driver version via `nvidia-smi`. |
| **05** | `CUDA Compatibility` | Validates CUDA Driver API level and sets target compute backend. |
| **06** | `PyTorch Validation` | Inspects installed PyTorch build, CUDA support, and tensor probe. |
| **07** | `ONNX Runtime` | Inspects installed ONNX Runtime variant and available execution providers. |
| **08** | `Compute Validation` | Executes a synchronized $1024 \times 1024$ matrix multiplication test on target device. |
| **09** | `YOLO Validation` | Instantiates `PersonDetector` and validates runtime device assignment. |
| **10** | `ONNX Inference` | Runs active ONNX session inference using `silhouette_segmenter.onnx`. |
| **11** | `ByGaitLight CNN` | Executes forward pass through `ByGaitLight` and validates `[1, 256]` shape and unit L2 norm. |
| **12** | `Final Validation` | Generates authoritative environment summary and writes `.venv/argus_env_manifest.json`. |

### Compute Verification & CPU Fallback Parity

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

## Multi-Camera Surveillance & Ingestion Engine

### Camera Lifecycle State Machine

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

### Multi-Camera Fair-Share Scheduling

To scale across concurrent video feeds without starvation, `ProductionMultiCameraEngine` (`streaming/production_multicamera_engine.py`) implements:

1. **Decoupled Bounded Queues**: Per-camera bounded frame buffers with backpressure and automatic stale-frame dropping (`stale_frame_max_age_ms=500.0ms`).
2. **Deficit Round-Robin (DRR) Scheduler**: `PersonTrackScheduler` prevents high-traffic cameras from monopolizing GPU inference.
3. **Dynamic Batching**: Aggregates appearance crops and gait silhouette sequences across cameras into unified GPU batches (adaptive batch size: 8–32 based on VRAM).
4. **Stream Isolation**: Stream disconnects or network errors on one camera never stall or degrade other running cameras.

### Hardware Admission Control

`CameraAdmissionController` (`streaming/deployment_readiness.py`) runs pre-flight capacity checks before admitting new camera streams:

* **CPU & RAM Guard**: Ensures host CPU utilization $< 85\%$ and available RAM $> 1.0\text{GB}$.
* **VRAM Guard**: Verifies dedicated GPU VRAM availability before expanding batch sizes.
* **Admission Decision**: Evaluates to `ADMITTED`, `ADMITTED_DEGRADED` (reduced target FPS), or `REJECTED`.

---

## Frontend Surveillance Dashboard

The frontend application (`frontend/`) is built with React 19, Vite, Lucide Icons, and Leaflet.

### Responsive & Resizable Dock Architecture

* **`useResizablePanel` Hook**: Pointer-event-based panel resizing with `requestAnimationFrame` throttling and boundary constraints.
* **`ResizeHandle` Component**: Accessible separator handle supporting mouse dragging, touch interaction, keyboard navigation (`ArrowLeft` / `ArrowRight`), double-click reset, and ARIA attributes (`role="separator"`).
* **`layoutStorage.js` Persistence**: Persists user layout preferences in `localStorage` under `argus_ui_layout` with bounds validation:
  * Dashboard Dock Width: 300px – 640px (Default: 420px).
  * Case Details Panel Width: 240px – 480px (Default: 300px).
  * Admin Split Ratio: 35% – 75% (Default: 60%).
* **Responsive Breakpoints**: Seamlessly adapts layout across Desktop ($\ge 1280\text{px}$), Laptop ($1024\text{px} - 1279\text{px}$), Tablet ($768\text{px} - 1023\text{px}$), and Mobile ($< 768\text{px}$).

### Live CCTV Surveillance Grid

* **`CctvNetwork.jsx`**: Responsive 16:9 surveillance feed cards with live MJPEG streams, automatic reconnect retry loops, connection status badges (`STANDBY`, `CONNECTING`, `CONNECTED`, `RECONNECTING`), and worker controls.
* **`GaitSystemStatus.jsx`**: Real-time telemetry displaying compute backend (`CUDA` / `CPU`), GPU device name, VRAM allocation, and execution providers.
* **`RecognitionEvents.jsx`**: Live WebSocket event feed displaying subject ID, confidence score, camera zone, and timestamp.

### Geospatial Mapping & Case Management

* **`Map.jsx`**: Leaflet geospatial map displaying registered camera zone placements and geographic locations.
* **`ReportCase.jsx` & `CaseDetails.jsx`**: Person of interest registration, multi-camera timeline reconstruction, and alert dispatch.
* **`AdminDashboard.jsx`**: User management, security policy configuration, system log inspection (`LogViewer.jsx`), and model registry audit logs.

---

## REST API & WebSocket Services

The backend API is implemented in FastAPI (`api/server.py`, `api/v1/router.py`, and `api/routes/health.py`) and executed via Uvicorn.

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
| `DELETE` | `/api/v1/credentials/{id}` | Delete user-owned credential entry. |
| `POST` | `/api/v1/credentials/{id}/share` | Grant credential access to another user ID. |
| `POST` | `/api/v1/cameras/{id}/credentials` | Store camera-scoped credential. |
| `WS` | `/api/v1/ws/recognition` | Real-time WebSocket feed for recognition events. |
| `WS` | `/api/v1/ws/events` | Real-time WebSocket feed for system security alerts. |

---

## Security & Engineering Hardening

1. **Safe PyTorch Checkpoint Loading**: All weight loading in `models/inference/pytorch_backend.py` and `intelligence/nn_fine_tuner.py` enforces `torch.load(..., weights_only=True)`, preventing arbitrary code execution from untrusted model files.
2. **Hardened Vector Store**: `storage/vector_store.py` enforces `allow_pickle=False` and rejects object-type NumPy arrays (`dtype == object`), mitigating deserialization vulnerabilities.
3. **Encrypted Credentials & Log Masking**: RTSP credentials are encrypted via Fernet (`.credentials.key`) and masked in logs (`rtsp://***:***@host:port`).
4. **Lazy Module Access**: `automation/__init__.py` and `pipeline/steps/__init__.py` use PEP 562 lazy module `__getattr__`, preventing `runpy` `RuntimeWarning: 'automation.bootstrap' found in sys.modules` when running CLI modules.
5. **Non-Blocking Background Warmup**: `GaitService` implements asynchronous background warmup (`warmup_async()`), allowing the FastAPI server to bind and respond to `/health` probes in $< 2.0\text{s}$ without blocking on heavyweight model weight loads.

---

## Benchmark Results & Scientific Evidence

### CASIA-B Subject-Disjoint Ablation Matrix

Evaluated under a strict subject-disjoint partition: Train `001–062` (6,779 sequences), Val `063–074` (1,299 sequences), Test `075–124` (5,466 sequences):

| Experiment | Pooling Strategy | Loss Formulation | Triplet Weight | Rank-1 Accuracy | Rank-5 Accuracy | Normal Walk (NM) | Carrying Bag (BG) | Clothing Change (CL) | ROC-AUC | EER | Open-Set FAR | Calibration Threshold | Impostor Score Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp-001** (Legacy Non-Disjoint)* | Global (1) | Standard CE | ~0.50 | 86.89%* | 93.96%* | 96.82%* | 91.23%* | 72.64%* | 0.9150 | 16.88% | 36.75% | 0.9913 | Saturated near 1.0 |
| **EXP-003A** (Disjoint Base) | Global (1) | Standard CE | 0.50 | 52.78% | 67.10% | 85.82% | 53.15% | 19.36% | 0.7499 | 31.95% | 70.49% | 0.7064 | Compressed `[0.208, 0.984]` |
| **EXP-003B** (HPP Alone) | HPP (4) | Standard CE | 0.50 | 61.43% | 75.63% | 91.55% | 60.55% | 32.18% | 0.8327 | 24.86% | 57.06% | 0.7942 | Compressed `[0.450, 0.995]` |
| **EXP-003C** (ArcFace Alone) | Global (1) | ArcFace | 0.50 | 59.58% | 73.78% | 91.00% | 61.55% | 26.18% | 0.8314 | 25.64% | 47.20% | 0.9927 | Saturated near 1.0 |
| **EXP-003D** (HPP+ArcFace) | HPP (4) | ArcFace | 0.00 | 69.71% | 80.91% | 96.73% | 72.79% | 39.64% | 0.8470 | 23.49% | 60.84% | 0.5287 | Expanded `[0.211, 0.965]` |
| **EXP-003E** (Top Candidate) | HPP (4) | ArcFace | **0.25** | **72.63%** | **82.76%** | **97.00%** | **78.26%** | **42.64%** | **0.8776** | **20.46%** | **62.26%** | **0.4906** | **Desaturated `[-0.60, 0.97]`** |

*\*Note: Exp-001 metrics reflect a historical evaluation where test subjects were supervised during training. On the true subject-disjoint split, the global baseline achieves 52.78% Rank-1. EXP-003E provides a +19.85% absolute improvement over the disjoint baseline.*

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
├── docs/                       # Project documentation and audit reports
├── enrollment/                 # Target identity enrollment and lifecycle manager
├── evaluation/                 # Scientific evaluation metrics (Rank-k, EER, ROC-AUC)
├── events/                     # Event contracts and dispatcher bus
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
│   ├── candidate_validator.py           # Multi-gate candidate validation
│   ├── continual_learning_audit_trail.py# Forensic candidate evaluation audit trail
│   ├── date_aware_learning_scheduler.py # Event-date driven learning job scheduler
│   ├── dual_modal_fusion.py             # Gait + Appearance score fusion
│   ├── missing_person_workflow.py       # Watchlist target registration & case matching
│   ├── nn_fine_tuner.py                 # PyTorch ByGaitLight & OSNet fine-tuning
│   ├── open_set_recognizer.py           # Open-set KNOWN / UNKNOWN / UNCERTAIN decision
│   ├── operational_embedding_collector.py # High-confidence live observation capture
│   └── training_dataset_builder.py      # Replay buffer & balanced dataset assembly
├── models/                     # Deep learning architectures and gallery storage
│   ├── appearance_gallery/     # Active 512D OSNet appearance embeddings (.npy)
│   ├── architectures/          # ByGaitLight, UNet segmenter, and ArcFace losses
│   ├── candidates/             # Isolated candidate model checkpoints (.pth)
│   ├── live_gallery/           # Active 256D ByGaitLight gait embeddings (.npy)
│   ├── model_registry.py       # Atomic model version management & rollback
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
│   ├── sync_folder_readmes.py  # Synchronize package README files
│   ├── verify_environment.py   # 6-phase environment verification suite
│   └── verify_firebase_continual_learning_e2e.py # E2E CL & Firebase verification
├── security_layer/             # Credential encryption and access control manager
├── services/                   # GaitService, CameraWorker, CameraSourceResolver, RecognitionWorker
├── storage/                    # Hardened VectorStore, SQLite EmbeddingDatabase, Firebase store
│   ├── embedding_database.py   # Local SQLite + VectorStore embedding database
│   ├── firebase_embedding_store.py # Non-blocking Firebase persistence & canonical schema
│   ├── lineage_tracker.py      # Embedding lineage & provenance tracking
│   └── vector_store.py         # Hardened NumPy vector store (allow_pickle=False)
├── streaming/                  # Multi-camera engine, admission control, and runtime resilience
├── tests/                      # Automated test suite (849 tests passed)
│   ├── integration/            # Multi-component integration tests (151 passed)
│   └── unit/                   # Unit test suite (698 passed)
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

Execute the test suites to verify environment health, code correctness, and system readiness:

```powershell
# 1. Full automated unit test suite (698 tests passed)
.\.venv\Scripts\python.exe -m pytest tests/unit/ -v

# 2. Integration & root test suites (151 tests passed)
.\.venv\Scripts\python.exe -m pytest tests/integration/ tests/test_*.py -v

# 3. Firebase & continual learning end-to-end standalone verification (7/7 phases passed)
.\.venv\Scripts\python.exe scripts/verify_firebase_continual_learning_e2e.py

# 4. Python bytecode compilation check across all packages (0 errors)
python -m compileall -q api automation core deployment enrollment evaluation events intelligence models monitoring pipeline preprocessing scripts security_layer services storage streaming tests training utils

# 5. Full 6-phase environment verification suite
.\.venv\Scripts\python.exe scripts/verify_environment.py

# 6. Pre-flight health doctor check
.\.venv\Scripts\python.exe scripts/doctor.py

# 7. Package README alignment validation (45 tests passed)
.\.venv\Scripts\python.exe -m pytest tests/unit/test_sync_folder_readmes.py -v
```

---

## Current Implementation Status

The implementation status is categorized below based strictly on codebase inspection and reproducible test execution:

### Status Matrix

| Capability / Subsystem | Status | Evidence / Verification Method |
| :--- | :---: | :--- |
| **Hardware Auto-Discovery & Arbitration** | **IMPLEMENTED** | `automation/bootstrap.py`, `DeviceManager` (13/13 unit tests passed) |
| **CUDA GPU Acceleration & CPU Fallback** | **IMPLEMENTED** | `scripts/verify_environment.py` (CUDA verified), `--force-cpu` parity |
| **Person Detection & Box Smoothing** | **IMPLEMENTED** | `PersonDetector` (YOLOv8) + `TrackingStep` (ByteTrack + EMA $\alpha=0.35$) |
| **Silhouette Extraction (UNet + Otsu)** | **IMPLEMENTED** | `SilhouetteExtractor` (`silhouette_segmenter.onnx` + morphological Otsu fallback) |
| **Gait Energy Image (GEI) Accumulation** | **IMPLEMENTED** | `LiveGEIStep`, `StreamGEIBuilder` ($128 \times 64$ normalized output) |
| **ByGaitLight Feature Extraction (256D)** | **IMPLEMENTED** | `ByGaitLight` CNN (HPP `part_bins=4`, 256D L2-normalized embedding) |
| **OSNet Appearance ReID (512D)** | **IMPLEMENTED** | `OSNet-x0.25` (512D L2-normalized appearance embeddings) |
| **Dual-Modal Score Fusion** | **IMPLEMENTED** | `DualModalFusion`, `LearnedFusion`, `ScoreCalibrator` |
| **Open-Set Decision Logic** | **IMPLEMENTED** | `OpenSetRecognizer` (`KNOWN`, `UNKNOWN`, `UNCERTAIN` margin boundaries) |
| **Missing Person Reference Data Flow** | **IMPLEMENTED** | `MissingPersonWorkflow`, `EmbeddingDatabase` (Reference tagging & CL exclusion) |
| **Firebase Canonical Persistence** | **IMPLEMENTED** | `FirebaseEmbeddingStore` (Canonical schema, deterministic IDs, non-blocking) |
| **Operational Embedding Lifecycle** | **IMPLEMENTED** | `OperationalEmbeddingCollector` (`PREDICTED` $\rightarrow$ `VERIFIED` $\rightarrow$ `ELIGIBLE` $\rightarrow$ `CONSUMED`) |
| **Date-Aware Continual Learning** | **IMPLEMENTED** | `DateAwareLearningScheduler` (Date grouping, future-date rejection) |
| **Real PyTorch NN Fine-Tuning** | **RUNTIME VERIFIED** | `NNFineTuner` (ByGaitLight & OSNet training with tensor delta verification) |
| **Multi-Gate Candidate Validation** | **IMPLEMENTED** | `CandidateValidator` (Zero FAR regression, TAR stability, dimension check) |
| **Atomic Model Registry & Rollback** | **RUNTIME VERIFIED** | `ModelRegistry` (Atomic promotion and $<50\text{ms}$ rollback verified) |
| **Camera Ingestion & State Machine** | **IMPLEMENTED** | `CameraWorker`, `CameraSourceResolver` (`STANDBY` $\rightarrow$ `CONNECTING` $\rightarrow$ `CONNECTED`) |
| **Multi-Camera Fair-Share Scheduling** | **IMPLEMENTED** | `ProductionMultiCameraEngine`, `PersonTrackScheduler` (DRR + Priority Aging) |
| **Hardware Admission Control** | **IMPLEMENTED** | `CameraAdmissionController`, `DeploymentReadinessManager` (RAM/VRAM gating) |
| **RTSP Credential Encryption & Masking** | **IMPLEMENTED** | Fernet encryption (`security_layer/credentials.py`), log masking |
| **Hardened Vector Store & SQLite DB** | **IMPLEMENTED** | `VectorStore` (`allow_pickle=False`), `EmbeddingDatabase` (versioned records) |
| **FastAPI REST API & WebSockets** | **IMPLEMENTED** | `/api/v1/...`, `/health/...`, WebSocket `/ws/recognition` & `/ws/events` |
| **Frontend Surveillance Dashboard** | **IMPLEMENTED** | React 19 SPA, live MJPEG feeds, geospatial mapping, case management |
| **Responsive & Resizable Layout System** | **IMPLEMENTED** | `useResizablePanel`, `ResizeHandle` (keyboard/pointer), `layoutStorage.js` |
| **Automated Test Suite (849 Tests)** | **IMPLEMENTED** | **849 tests passed (100%)** (`698 unit + 151 integration/root`) |
| **Multi-Camera Physical Field Trials** | **PARTIALLY IMPLEMENTED** | Synthetic and multi-worker tests verified; physical multi-camera field trial ongoing |
| **Production-Scale Million-Subject DB** | **PLANNED / FUTURE** | Evaluated on active development gallery; indexing for $10^6$ scale is planned |

---

## Known Limitations

1. **Active Development Gallery Size**: The active test galleries contain development baselines (e.g. 64 gait embeddings and 201 appearance embeddings). Million-identity indexing remains for future production scaling.
2. **Clothing Covariate Sensitivity**: As established in CASIA-B subject-disjoint ablation benchmarks, clothing changes (`CL` Rank-1 = 42.64%) degrade silhouette geometry more significantly than carrying bags (`BG` = 78.26%).
3. **Hardware-Dependent Real-Time Throughput**: Full real-time FPS throughput is dependent on a compatible NVIDIA GPU with CUDA acceleration. While CPU execution is functional, throughput will be lower on CPU-only machines.
4. **Physical Multi-Camera Field Validation**: While unit tests, component smoke tests, and synthetic multi-stream pipelines pass, physical multi-camera trials in unconstrained real-world environments remain ongoing.

---

## License & Maintainer

### License

This project is licensed under the [MIT License](LICENSE).

### Project Maintainer

**Chanuka Sandun**  
Undergraduate in Cybersecurity  
Developer of the ARGUS AI Biometric Surveillance Framework

* GitHub: [@chanuka8](https://github.com/chanuka8)  
* LinkedIn: [linkedin.com/in/chanukasandun](https://www.linkedin.com/in/chanukasandun/)
