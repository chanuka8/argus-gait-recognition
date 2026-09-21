# ARGUS AI

![ARGUS AI Gait Recognition Banner](assets/github/Gitrepo_profilepic.png)

## Biometric Gait Recognition, Dual-Modal Surveillance Intelligence & Continual Learning Framework

ARGUS AI is an advanced biometric research and surveillance intelligence prototype designed for non-invasive human identification at a distance. The platform identifies individuals by extracting dynamic walking kinematics (gait geometry and cadence) from silhouette sequences, supplemented by deep appearance re-identification (ReID) feature streams for short-term identity continuity across multi-camera networks.

Built on **Python 3.11**, **PyTorch**, **ONNX Runtime**, **OpenCV**, **FastAPI**, and **React 19**, ARGUS AI implements a locked, multi-stage gait recognition pipeline, a decoupled camera streaming engine, an isolated missing-person reference enrollment workflow, date-aware continual learning with multi-gate validation, on-disk case dossiers with path-traversal protection, and role-based access control with Argon2id password hashing.

[![Python: 3.11.9](https://img.shields.io/badge/python-3.11.9-blue.svg)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows / Linux](https://img.shields.io/badge/platform-Windows%2011%20%7C%20Linux-lightgrey.svg)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%2F%20Driver%20535%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20.0-blue.svg)](https://onnxruntime.ai/)
[![Tests: 1,030 Passed](https://img.shields.io/badge/tests-1%2C030%20passed%20%7C%200%20failed%20%7C%201%20skipped-brightgreen.svg)](tests)
[![Frontend: React 19 + Vite](https://img.shields.io/badge/frontend-React%2019%20%2B%20Vite-61DAFB.svg)](frontend)
[![Security: Argon2id + RBAC](https://img.shields.io/badge/security-Argon2id%20%7C%20RBAC%20%7C%20Fernet-purple.svg)](security_layer)
[![Status: Advanced Prototype](https://img.shields.io/badge/status-ADVANCED%20PROTOTYPE-orange.svg)](docs)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)

---

> [!NOTE]
> **Operational Status & Scope Notice**:
> ARGUS AI is a **research and development platform and advanced prototype**. It is designed, benchmarked, and validated for local deployment and controlled multi-camera environments. It is **not** a turnkey commercial surveillance product and does **not** claim 100% biometric accuracy or universal CCTV plug-and-play capability. Biometric gait recognition in unconstrained real-world environments remains an active research discipline subject to environmental, clothing, viewpoint, and camera resolution constraints.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Gait Recognition Pipeline](#core-gait-recognition-pipeline)
3. [Dual-Modal Biometrics & Appearance ReID](#dual-modal-biometrics--appearance-reid)
4. [Camera Ingestion & Live Surveillance Engine](#camera-ingestion--live-surveillance-engine)
5. [Camera & Reference Processing Isolation](#camera--reference-processing-isolation)
6. [Missing Person Reference Media Processing](#missing-person-reference-media-processing)
7. [Storage Architecture](#storage-architecture)
8. [Date-Aware Continual Learning & Model Management](#date-aware-continual-learning--model-management)
9. [Case Dossier Management](#case-dossier-management)
10. [Authentication, RBAC & Security Engineering](#authentication-rbac--security-engineering)
11. [Hardware-Aware Compute Automation](#hardware-aware-compute-automation)
12. [Multi-Camera Scheduling & Admission Control](#multi-camera-scheduling--admission-control)
13. [Frontend Surveillance Dashboard](#frontend-surveillance-dashboard)
14. [REST API & WebSocket Services](#rest-api--websocket-services)
15. [Research Benchmark Results](#research-benchmark-results)
16. [Project Structure](#project-structure)
17. [Hardware & Software Environment](#hardware--software-environment)
18. [Installation & Windows Setup](#installation--windows-setup)
19. [Running the System](#running-the-system)
20. [Verification & Testing Commands](#verification--testing-commands)
21. [Current Implementation Status Matrix](#current-implementation-status-matrix)
22. [Research Boundaries & Current Limitations](#research-boundaries--current-limitations)
23. [Security Notes & Responsible Disclosure](#security-notes--responsible-disclosure)
24. [License & Maintainer](#license--maintainer)

---

## System Architecture

ARGUS AI follows a decoupled, service-oriented architecture where video ingestion, neural inference, background reference enrollment, and continual learning execute asynchronously across independent worker threads.

```
+-----------------------------------------------------------------------------------+
|                               SURVEILLANCE FRONTEND                               |
|            React 19 SPA | Live MJPEG Grid | Case Dossier Modal | Leaflet Map       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ (HTTP REST / WebSocket Events)
+-----------------------------------------------------------------------------------+
|                                FASTAPI API LAYER                                  |
|   Session Auth (Argon2id) | RBAC & Anti-BOLA | Rate Limiting | Video Streaming    |
+-----------------------------------------------------------------------------------+
                         │                                     │
                         ▼                                     ▼
+───────────────────────────────────+       +───────────────────────────────────────+
|      LIVE CCTV CAMERA WORKER      |       |      REFERENCE JOB MANAGER (ASYNC)    |
| Dedicated Thread | DirectShow/RTSP|       |  Isolated Worker Pool | Job Recovery  |
| Frame Ring-Buffer | Software Pacing|      |  Bounded RAM Streaming | Checkpoints  |
+───────────────────────────────────+       +───────────────────────────────────────+
                         │                                     │
                         ▼                                     ▼
+───────────────────────────────────────────────────────────────────────────────────+
|                         LOCKED BIOMETRIC RECOGNITION PIPELINE                     |
|  YOLOv8n (conf=0.40, IoU=0.45) ──► EMA Smoothing (α=0.35) ──► ByteTrack Tracking  |
|  Silhouette Extractor (UNet ONNX / Otsu) ──► Rolling GEI (64x128, window=15)       |
|  ByGaitLight CNN (HPP 4-bin) ──► 256-D L2-Normalized Gait Embedding Vector        |
|  Local Cosine Similarity ──► Four-Tier Open-Set Policy ──► Temporal Voting (10/3) |
+───────────────────────────────────────────────────────────────────────────────────+
                         │                                     │
                         ▼                                     ▼
+───────────────────────────────────+       +───────────────────────────────────────+
|     LOCAL-FIRST PRIMARY STORAGE   |       |    ASYNCHRONOUS DURABLE PERSISTENCE   |
| Hardened VectorStore (.npy)       |       |  Firebase Firestore (Admin SDK)       |
| SQLite EmbeddingDatabase (<1.5ms) |──────►|  Canonical Schema | Lineage Auditing  |
| data/runtime/cases/{id}_{name}/   |       |  Offline Fallback Queue (data/*.json) |
+───────────────────────────────────+       +───────────────────────────────────────+
```

### Continual Learning Loop (Date-Aware & Validated)

```
Operational CCTV Observations (State: PREDICTED)
                     │
                     ▼
Human-in-the-Loop Operator Confirmation (State: VERIFIED)
                     │
                     ▼
Quality & Dimensionality Filter (Quality >= 0.70, State: TRAINING_ELIGIBLE)
                     │
                     ▼
Date-Aware Learning Scheduler (Group by Date, Reject Future Timestamps)
                     │
                     ▼
Training Dataset Builder (50% New Date Evidence + 50% Historical Replay Baseline)
                     │
                     ▼
PyTorch Neural Network Fine-Tuning (NNFineTuner: Tensor Delta Verification)
                     │
                     ▼
Multi-Gate Candidate Validator (FAR <= Base, TAR >= Base - 0.005, Stable 256D)
                     │
                     ▼
Atomic Model Registry Promotion (State: TRAINING_CONSUMED, Zero-Downtime Hot Reload)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
[Promotion Retained]   [Instant Rollback (<50ms)]
```

---

## Core Gait Recognition Pipeline

The ARGUS AI gait recognition pipeline is strictly locked in configuration and code (`configs/inference.yaml`, `configs/gei.yaml`, `configs/detection.yaml`, and `pipeline/`):

```
Camera / Video / Reference Media
               ↓
    Person Detection (YOLOv8n)
               ↓
EMA Bounding-Box Stabilisation (α = 0.35)
               ↓
       ByteTrack Tracking
               ↓
 Silhouette Extraction / Normalisation (UNet ONNX / Otsu)
               ↓
      Rolling GEI (64 × 128)
               ↓
          ByGaitLight CNN
               ↓
256-D L2-Normalised Gait Embedding
               ↓
       Cosine Similarity
               ↓
    Open-Set Decision Policy
               ↓
    Temporal Majority Voting
               ↓
       Recognition Result
```

### Validated Pipeline Parameters (Source of Truth)

| Stage | Implementation Component | Configuration / Specification | Evidence / Source Code |
| :--- | :--- | :--- | :--- |
| **Detection** | `PersonDetector` (`pipeline/detection/`) | YOLOv8n, `class=0` (person), confidence $\ge 0.40$, IoU $\ge 0.45$, input $640 \times 640$ | `configs/detection.yaml` |
| **Smoothing** | `TrackingStep` (`pipeline/steps/tracking.py`) | Exponential Moving Average (EMA) with smoothing factor $\alpha = 0.35$ to eliminate jitter | `configs/inference.yaml` |
| **Tracking** | `ByteTrack` (`pipeline/steps/tracking.py`) | Multi-object association with track state machine and lost-track recovery | `pipeline/steps/tracking.py` |
| **Silhouette** | `SilhouetteExtractor` (`pipeline/silhouette/`) | Neural UNet segmenter (`silhouette_segmenter.onnx`), automatic morphological Otsu fallback | `pipeline/silhouette/extractor.py` |
| **GEI Builder** | `LiveGEIStep` (`pipeline/steps/live_gei.py`) | Rolling Gait Energy Image window $= 15$ frames, minimum $= 10$ frames, resolution: **64 width $\times$ 128 height** | `configs/gei.yaml` |
| **Backbone** | `ByGaitLight` (`models/architectures/`) | Lightweight 3-block CNN with Horizontal Part Pooling (HPP, `part_bins=4`), input $1 \times 128 \times 64$ | `models/architectures/bygait_light.py` |
| **Embedding** | `ByGaitLight.forward()` | **256-dimensional float32 vector**, strictly L2-normalised ($\|e\|_2 = 1.0000 \pm 10^{-5}$) | `models/architectures/bygait_light.py` |
| **Similarity** | `VectorStore` / `MatchingStep` | Dot product / Cosine similarity against enrolled gallery templates in `models/galleries/live_gallery/` | `storage/vector_store.py` |
| **Decision** | `OpenSetRecognizer` (`intelligence/`) | **Four-Tier Policy**: `KNOWN` ($\ge 0.92$), `UNCERTAIN` ($0.85 - 0.92$), `REVIEW_REQUIRED` ($0.70 - 0.85$), `UNKNOWN` ($< 0.70$) | `configs/inference.yaml` |
| **Temporal Consensus**| `MatchingPolicy` (`configs/inference.yaml`) | Rolling voting history $= 10$ frames, minimum stable votes required $= 3$ | `configs/inference.yaml` |

> [!IMPORTANT]
> **No Facial Biometrics in Gait Pipeline**:
> The gait pipeline operates strictly on whole-body binary silhouette sequences and dynamic kinematics. Face recognition is not part of this biometric feature pipeline, ensuring functionality at long camera distances, low resolution, or when faces are covered, turned away, or obscured.

---

## Dual-Modal Biometrics & Appearance ReID

In addition to kinematic gait representation, ARGUS AI integrates an appearance re-identification (ReID) feature stream for short-term identity association:

* **Appearance Backbone**: `OSNet-x0.25` (`models/reid/osnet_backbone.py`, weights: `models/model_store/weights/osnet_x0_25.pth`) lightweight omni-scale network.
* **Feature Representation**: 512-dimensional L2-normalized feature vector extracted from RGB person crops.
* **Appearance Gallery**: Separate appearance vector store (`models/galleries/appearance_gallery/`) managed via `VectorStore`.
* **Dual-Modal Score Fusion**:
  $$S_{\text{fused}} = w_{\text{gait}} \cdot S_{\text{gait}} + w_{\text{app}} \cdot S_{\text{app}}$$
  with dynamic weight attenuation based on silhouette quality, bounding-box aspect ratio, and track length.

---

## Camera Ingestion & Live Surveillance Engine

ARGUS AI isolates camera capture from recognition processing to guarantee responsive, high-fps live previews even under heavy inference loads.

```
Camera Hardware / RTSP Stream
              │
              ▼
    [CameraWorker Thread]  ◄── Independent OS thread, OpenCV VideoCapture
              │
              ├──► Exposes Latest Valid Frame (Live MJPEG Stream @ ~30 FPS)
              │
              ▼ (Non-blocking Bounded Queue)
   [RecognitionWorker Thread] ◄── Decoupled worker, batching, and model inference
              │
              ▼
   Track Updates & WebSocket Recognition Events
```

### Camera Lifecycle & Startup Optimization

The camera subsystem supports local USB/webcams (via Windows DirectShow or Linux V4L2) and IP cameras (via RTSP / ONVIF).

* **Decoupled Architecture**: `CameraWorker` acquires frames into a bounded ring buffer. `RecognitionWorker` attaches asynchronously. The live MJPEG preview stream starts immediately upon the first decoded frame without waiting for heavyweight model loading.
* **Disconnect / Reconnect Resilience**: Thread-safe reconnect loops automatically attempt stream re-acquisition with exponential backoff on frame drops or network timeouts.
* **Software Pacing**: Configurable target FPS pacing ensures low CPU usage when acquiring high-frequency camera streams.

### Validated Camera Startup Benchmarks (10-Cycle Stress Test)

Validated on the target development environment across 10 complete camera startup/shutdown cycles:

| Metric | Measured Duration | Engineering Context & Source of Truth |
| :--- | :---: | :--- |
| **Minimum Startup** | **590.33 ms** | Fastest warm-driver camera handle acquisition |
| **Median / P50** | **913.94 ms** | Typical user-visible start-stream to first-frame duration |
| **Average** | **1,107.88 ms** | Mean duration across 10 consecutive stress cycles |
| **P95** | **2,199.18 ms** | 95th percentile under host load |
| **Maximum Startup** | **2,679.01 ms** | Cold-driver acquisition and device renegotiation |
| **Application Adoption Path** | **≈ 31.06 ms** | Internal Python/FastAPI worker adoption when source is pre-verified |
| **Resource Leaks** | **0 Observed** | Zero worker thread, capture handle, or memory leaks across all cycles |

> [!NOTE]
> **Understanding Startup Latency**:
> The application-side worker adoption latency is approximately **31 ms**. However, the total end-to-end latency to first live frame (median **≈ 914 ms**) is governed by Windows DirectShow driver initialization, USB controller bus negotiation, and camera hardware firmware spin-up.

---

## Camera & Reference Processing Isolation

A core architectural invariant of ARGUS AI is the **strict isolation** between real-time camera streaming and missing-person reference job processing.

* Reference processing jobs execute in dedicated background threads and **never** access or block `CameraWorker`, `RecognitionWorker`, or live camera capture pipelines.
* Live surveillance operations continue uninterrupted at native framerates while long reference videos are processed in the background.

### 12 Verified Integration Scenarios

The isolation and resilience boundaries were verified through integration test suite `tests/integration/backend/test_camera_reference_isolation.py` (12/12 passed):

1. **Camera OFF + Photo Upload**: Reference photo upload creates background job, extracts 256-D embedding, and enrolls identity while CCTV is completely stopped.
2. **Camera OFF + Video Upload**: Reference video upload processes end-to-end through detection, tracking, GEI, and ByGaitLight while CCTV is stopped.
3. **Camera ON + Photo Upload**: Enrolling a reference photo while multiple CCTV cameras are actively streaming incurs zero frame drops or stream stutter.
4. **Camera ON + Video Upload**: Background video enrollment executes concurrently with live camera recognition without inference lockups.
5. **Camera Disconnect During Reference Processing**: Forcibly dropping the CCTV camera connection mid-video-processing does **not** corrupt or fail the running reference job.
6. **Camera Reconnect During Reference Processing**: Reconnecting the camera stream while a reference job is actively running does **not** interrupt job execution.
7. **Camera Stop During Reference Processing**: An operator stopping CCTV cameras mid-job does not interfere with the reference enrollment queue.
8. **Camera Start While Processing**: Starting new camera streams while a reference job is running executes safely with clean resource allocation.
9. **Corrupt Video Rejection**: Uploading truncated, malformed, or corrupt video containers is rejected with an explicit `INVALID_VIDEO` status without crashing workers.
10. **Empty Video Rejection**: Zero-byte files or zero-frame video streams are caught at validation and rejected with `INVALID_VIDEO`.
11. **Idempotent Duplicate Prevention**: Re-submitting identical video or photo media uses SHA-256 content hashing to prevent duplicate gallery pollution.
12. **Numerical Embedding Validation**: Synthetic malformed embeddings containing `NaN`, `Inf`, non-256 dimensions, or non-finite norms are caught and rejected prior to database persistence.

---

## Missing Person Reference Media Processing

The reference processing engine (`services/missing_person_processor.py` and `services/reference_job_manager.py`) provides offline, camera-independent biometric enrollment from user-submitted media.

```
Uploaded Media (Photo or Video)
              ↓
  File & Container Validation (Decodability, Resolution >= 32x32)
              ↓
       Person Detection (YOLOv8n)
              ↓
      ByteTrack Association (Temporal Track Continuity)
              ↓
  Target Track Isolation Policy (Prominence Ratio >= 2.5)
              ↓
  Silhouette Extraction (UNet ONNX / Otsu Morphological Fallback)
              ↓
    Rolling GEI Generation (64 × 128 Normalized)
              ↓
  ByGaitLight Feature Extraction (256-D Gait Embedding)
              ↓
 Numerical Validation (256-D, Finite Float32, L2-Norm = 1.0)
              ↓
 Local Gallery & SQLite Persistence + Async Firestore Sync
```

### Safety & Robustness Guarantees

* **Strict Input Validation**: Rejects videos smaller than $32 \times 32$ pixels or with fewer than 10 frames (`min_gait_frames`).
* **Clear Error Differentiation**: The system explicitly differentiates between:
  * `INVALID_VIDEO`: Corrupt header, unreadable codec, or un-decodable stream.
  * `NO_PERSON_DETECTED`: Valid video container, but zero human figures detected.
  * `INSUFFICIENT_GAIT_SEQUENCE`: Detected person track is shorter than the required minimum gait cycle.
  * `AMBIGUOUS_MULTIPLE_PERSONS`: Multiple individuals detected without a dominant foreground subject (prominence ratio $< 2.5$).
* **Bounded Memory Processing**: Frames are decoded and processed iteratively via generator pipelines rather than loading entire uncompressed video sequences into RAM.
* **Durable Checkpointing & Job Recovery**: Video processing creates intermediate state checkpoints on disk (`data/runtime/reference_jobs/{job_id}_checkpoint.json`). If the host service is terminated, interrupted jobs automatically resume from the last completed processing phase.

---

## Storage Architecture

ARGUS AI employs a **local-first** storage architecture. Real-time inference relies exclusively on local, zero-latency stores, while cloud storage is used asynchronously for durable backup and cross-station synchronization.

```
Live Inference Loop (<1.5ms) ──► Local EmbeddingDatabase (SQLite)
                                  Local VectorStore (.npy files)
                                            │
                                            ▼ (Async Background Thread)
                               Firebase Firestore Store
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                    Cloud Firestore               Local Fallback Queue
                    (Online State)          (data/runtime/firebase_offline_store.json)
```

### 1. Local Embedding Database & VectorStore (Primary Inference Path)

* **Hardened VectorStore** (`storage/vector_store.py`): Memory-mapped `.npy` array storage for active biometric templates (`models/galleries/live_gallery/` and `models/galleries/appearance_gallery/`). Enforces `allow_pickle=False` and rejects object-type NumPy arrays. Match evaluation latency is **$< 1.5\text{ms}$** per candidate.
* **SQLite Embedding Database** (`storage/embedding_database.py`): Local relational metadata database storing identity records, template associations, operational status (`ACTIVE`, `DISABLED`, `ARCHIVED`), and extraction provenance.

### 2. Firebase Firestore (Asynchronous Durable Persistence)

* **Decoupled Synchronization**: Firestore calls execute on detached threads. Network latency, disconnects, or cloud outages **never** stall, delay, or crash the live CCTV recognition loop.
* **Canonical Embedding Schema**: Every persisted embedding conforms to `FirebaseEmbeddingDocument`:
  * `embedding_id`: Deterministic SHA-256 hash derived from modality, person ID, timestamp, and vector content.
  * `identity_type`: `"USER_REFERENCE"` (gallery watchlist) vs `"LIVE_OPERATIONAL"` (CCTV evidence).
  * `operational_state`: `"PREDICTED"`, `"VERIFIED"`, `"TRAINING_ELIGIBLE"`, `"TRAINING_CONSUMED"`, or `"REFERENCE"`.
  * `training_eligibility`: Explicitly `"NOT_ELIGIBLE"` for reference gallery samples to prevent training set contamination.
* **Offline Resilience**: When unconfigured or offline, synchronization events buffer into `data/runtime/firebase_offline_store.json` and drain automatically upon reconnection.

---

## Date-Aware Continual Learning & Model Management

Continual learning in ARGUS AI is **date-aware and eligibility-driven**. The system does **not** blindly retrain every 24 hours. Instead, it aggregates operational observations, requires human-in-the-loop verification, enforces anti-forgetting replay buffers, and gates all model updates behind safety benchmarks.

```
[Operational CCTV Observation]
              │
              ▼
    State: PREDICTED  (Captured by OperationalEmbeddingCollector)
              │
              ▼ (Human-in-the-Loop Operator Confirmation)
    State: VERIFIED   (Confirmed real-world identity match)
              │
              ▼ (Quality >= 0.70, Finite Norm, Valid Dimensions, Identity != USER_REFERENCE)
 State: TRAINING_ELIGIBLE
              │
              ▼ (DateAwareLearningScheduler: Group by Capture Date, Reject Future Dates)
 Assemble Training Dataset (50% New Date Evidence + 50% Historical Replay Baseline)
              │
              ▼ (PyTorch Gradient Descent via NNFineTuner)
 Candidate Model Checkpoint (models/model_store/candidates/*.pth)
              │
              ▼ (CandidateValidator: 5 Safety Gates)
   Validation Gate Evaluation
         ├── FAIL ──► Candidate Rejected & Logged to Audit Trail
         └── PASS ──► ModelRegistry Atomic Promotion
                            │
                            ▼
               State: TRAINING_CONSUMED (Permanently Locked from Retraining)
               Live Workers Hot-Reload Weights (Instant Rollback Available < 50ms)
```

### Continual Learning Components

* **`OperationalEmbeddingCollector`** (`intelligence/operational_embedding_collector.py`): Captures high-confidence CCTV observations in `PREDICTED` state.
* **Reference Data Exclusion Guard**: Reference watchlist embeddings (`USER_REFERENCE`) are tagged `training_eligibility = "NOT_ELIGIBLE"` and are strictly excluded from training pools to avoid overfitting on sparse enrollment data.
* **`DateAwareLearningScheduler`** (`intelligence/date_aware_learning_scheduler.py`): Groups eligible samples by chronological date (`YYYY-MM-DD`). Requires $\ge 10$ eligible embeddings across $\ge 2$ distinct subjects before triggering training. Automatically rejects future-dated records ($\text{date} > \text{today}$).
* **`TrainingDatasetBuilder`** (`intelligence/training_dataset_builder.py`): Implements an anti-catastrophic forgetting replay buffer by pairing 50% new operational evidence with 50% historical baseline templates.
* **`NNFineTuner`** (`intelligence/nn_fine_tuner.py`): Performs genuine PyTorch gradient descent on `ByGaitLight` (Triplet Loss + ArcFace margin). Calculates parameter deltas (`changed_tensors > 0`, `max_param_delta > 0.0`) to verify genuine weight updates.
* **`CandidateValidator`** (`intelligence/candidate_validator.py`): Enforces 5 strict validation gates before promotion:
  1. *False Accept Rate (FAR) Gate*: Candidate FAR must not exceed baseline FAR ($\text{FAR}_{\text{cand}} \le \text{FAR}_{\text{base}}$).
  2. *True Accept Rate (TAR) Gate*: Candidate TAR must not regress beyond tolerance ($\text{TAR}_{\text{cand}} \ge \text{TAR}_{\text{base}} - 0.005$).
  3. *Stability & Shape Gate*: Output dimensionality must strictly match (256-D) with finite, normalized embeddings.
  4. *Anti-Churn Gate*: Rejects trivial parameter updates within random noise thresholds.
  5. *Integrity Gate*: Verifies SHA-256 checkpoint hashing.
* **`ModelRegistry`** (`models/model_registry.py`): Manages version manifests in `models/model_registry.json`. Supports atomic production promotion and sub-50ms instant rollback to the previous production checkpoint.

---

## Case Dossier Management

The Case Dossier capability (`services/case_dossier_manager.py`) organizes investigation cases and missing-person records into structured on-disk directories.

### Filesystem Layout

```
data/runtime/cases/{case_id}_{person_name}/
├── case_details.json         # Structured dossier metadata, GPS coordinates, file index
├── media/                    # Associated reference photographs and reference videos
│   ├── reference_photo_01.jpg
│   └── walking_clip_cctv.mp4
└── biometrics/               # Extracted biometric feature manifests and gallery sync logs
    └── biometrics_manifest.json
```

### Technical Capabilities

* **Deterministic Naming & Sanitization**: Standardized folder naming `{case_id}_{person_name}` with character sanitization (`[\\/*?:"<>| \t\n\r]+` replaced by `_`) to eliminate filesystem injection vulnerabilities.
* **Path Traversal Defense**: All file access endpoints resolve canonical paths and enforce `target.relative_to(folder_resolved)`. Traversal attempts (e.g. `../../etc/passwd` or `..\..\Windows`) are rejected with `404 Not Found` and logged to security telemetry.
* **HTTP 206 Partial Content Video Streaming**: `serve_dossier_file` implements byte-range request handling (`Range: bytes=start-end`), enabling seamless scrubbing and seeking in HTML5 video players within the frontend.
* **Atomic On-Disk Metadata Writes**: `case_details.json` is updated via temporary file swap (`.tmp_{pid}_{time}`) to prevent metadata corruption during unexpected process terminations.
* **Frontend Dossier Modal** (`frontend/src/components/CaseDossierModal.jsx`): Interactive React modal providing case status filters, media playback, biometrics inspection, and synchronized dossier refresh.

---

## Authentication, RBAC & Security Engineering

ARGUS AI implements defense-in-depth security across backend endpoints, authentication flows, and credential storage.

### 1. Password Hashing with Argon2id

Passwords are encrypted using **Argon2id** (`security_layer/password_hasher.py`), the winner of the Password Hashing Competition (PHC), configured to robust memory and time cost parameters:
* Memory cost: $65,536\text{ KB}$ (64 MB)
* Time cost: $3$ iterations
* Parallelism: $4$ threads
* Hash length: $32$ bytes
* Transparent upgrade logic: Automatically re-hashes credentials on login if stored under legacy formats.

### 2. Session Management & RBAC

* **Session Tokens**: In-memory, thread-safe `SessionStore` (`security_layer/auth.py`) with sliding idle timeout ($30\text{ minutes}$), maximum session lifetime ($8\text{ hours}$), and maximum concurrent session limits.
* **Role Hierarchy**:
  * `ROOT_ADMIN`: Full administrative control, operator user management, system reconfiguration.
  * `ADMIN`: Camera management, case creation, reference job execution, policy configuration.
  * `INVESTIGATOR`: Case inspection, live CCTV monitoring, recognition event viewing, dossier access.
* **Anti-BOLA / IDOR Protection**: Server-side verification ensures operators can only access jobs, cases, and credentials within their authorized scope.

### 3. Credential & Media Protection

* **RTSP URL Sanitization & Credential Encryption**: RTSP camera credentials are encrypted with Fernet symmetric encryption (`security_layer/credentials.py`) and masked in logs (`rtsp://***:***@host:port`).
* **Safe PyTorch Checkpoint Loading**: Enforces `torch.load(..., weights_only=True)` across all inference and training loaders, preventing arbitrary code execution from untrusted model binaries.
* **Hardened Deserialization**: NumPy vector loading rejects arbitrary pickle deserialization (`allow_pickle=False`).
* **Secret Hygiene**: Zero passwords, tokens, private keys, or Firebase service-account credentials are committed to the Git repository.

---

## Hardware-Aware Compute Automation

ARGUS AI dynamically adapts to host hardware through a centralized arbitration layer (`DeviceManager` in `automation/device_manager.py`), avoiding hardcoded device assignments.

### 12-Stage Environment Bootstrap Sequence

The bootstrap orchestrator (`automation/bootstrap.py`) executes a deterministic discovery routine during startup:

| Stage | Subsystem Checked | Validation Action |
| :---: | :--- | :--- |
| **01** | Operating System | Probes OS name, kernel, and CPU architecture (Windows 11 AMD64) |
| **02** | Python Interpreter | Verifies 64-bit Python 3.11.x runtime |
| **03** | Host Hardware Profile | Measures available CPU cores, usable host RAM, and detected GPUs |
| **04** | NVIDIA Driver API | Queries GPU driver version and driver capability via `nvidia-smi` |
| **05** | CUDA Compatibility | Checks CUDA Driver API level against PyTorch CUDA requirements |
| **06** | PyTorch CUDA Probe | Performs tensor allocation and CUDA device synchronization |
| **07** | ONNX Runtime Providers| Validates `CUDAExecutionProvider` / `CPUExecutionProvider` priority |
| **08** | Compute Matrix Test | Executes synchronized $1024 \times 1024$ matrix multiplication on target device |
| **09** | YOLO Detection Probe | Instantiates `PersonDetector` and validates device binding |
| **10** | ONNX Inference Probe | Executes test inference pass using `silhouette_segmenter.onnx` |
| **11** | ByGaitLight Probe | Executes forward pass through ByGaitLight; verifies $[1, 256]$ shape and unit norm |
| **12** | Manifest Generation | Generates authoritative hardware manifest at `.venv/argus_env_manifest.json` |

### CUDA vs. CPU Parity

When a compatible NVIDIA GPU is present, computation routes to `cuda:0`. If unavailable, the system deterministically falls back to `cpu` mode across all subsystems with identical algorithmic logic.

---

## Multi-Camera Scheduling & Admission Control

To support concurrent video feeds without resource starvation, the streaming subsystem (`streaming/`) implements:

* **Deficit Round-Robin (DRR) Scheduling**: `PersonTrackScheduler` prevents individual high-density camera streams from starving lower-density cameras during GPU batch inference.
* **Bounded Per-Camera Queues**: Individual camera queues drop stale frames ($> 500\text{ms}$ latency) when backpressure develops, maintaining real-time alignment.
* **Hardware Admission Controller**: Pre-flight capacity checks verify host CPU ($< 85\%$), RAM ($> 1.0\text{GB}$ free), and GPU VRAM before admitting additional concurrent camera feeds.

---

## Frontend Surveillance Dashboard

The frontend application (`frontend/`) is built with React 19, Vite, and Lucide Icons.

* **Live CCTV Grid** (`CctvNetwork.jsx`): Responsive multi-camera feed cards featuring live MJPEG streams, reconnect status badges (`STANDBY`, `CONNECTING`, `CONNECTED`, `RECONNECTING`), and worker controls.
* **Resizable Dock Architecture**: Pointer-event-based resizable panels (`ResizeHandle`, `useResizablePanel`) with keyboard navigation (`ArrowLeft` / `ArrowRight`) and persisted layout boundaries (`layoutStorage.js`).
* **Case Dossier Viewer** (`CaseDossierModal.jsx`): Dedicated modal displaying structured case details, photo/video inventory, Range-request media players, biometrics metadata, and dossier resync.
* **Geospatial Mapping** (`Map.jsx`): Interactive Leaflet map visualizing registered camera locations and geographical event sightings.
* **Telemetry & Alerts** (`GaitSystemStatus.jsx` & `RecognitionEvents.jsx`): Real-time WebSocket telemetry displaying GPU device name, VRAM utilization, active execution providers, and live match events.

---

## REST API & WebSocket Services

The backend exposes a structured REST API and WebSocket services via FastAPI:

### API Endpoints Overview

| Category | Method | Route | Description | Auth Required |
| :--- | :--- | :--- | :--- | :---: |
| **Auth** | `POST` | `/api/v1/auth/login` | Authenticate operator with username & password | No |
| **Auth** | `POST` | `/api/v1/auth/logout` | Invalidate current operator session token | Yes |
| **Auth** | `GET` | `/api/v1/auth/me` | Retrieve profile and RBAC role of authenticated operator | Yes |
| **Auth** | `POST` | `/api/v1/auth/verify-password`| Asynchronously verify current password via Argon2id | Yes |
| **Auth** | `POST` | `/api/v1/auth/change-password`| Change operator password with Argon2id re-hashing | Yes |
| **Auth** | `GET` | `/api/v1/auth/operators` | List registered system operators (Admin only) | Admin |
| **Camera** | `POST` | `/api/v1/cameras/start` | Start camera worker (`camera_id`, `source`, `location`) | Yes |
| **Camera** | `POST` | `/api/v1/cameras/stop` | Stop camera worker and release video capture device | Yes |
| **Camera** | `GET` | `/api/v1/cameras` | List active camera workers and streaming telemetry | Yes |
| **Camera** | `GET` | `/api/v1/cameras/{id}/stream` | Live MJPEG video stream with bounding-box overlays | Yes |
| **Camera** | `GET` | `/api/v1/cameras/{id}/snapshot`| Single JPEG snapshot from the latest captured frame | Yes |
| **Reference**| `POST` | `/api/v1/enroll` | Upload reference photo/video for background processing | Yes |
| **Reference**| `GET` | `/api/v1/cases/jobs` | List background reference media processing jobs | Yes |
| **Reference**| `GET` | `/api/v1/cases/jobs/{job_id}` | Retrieve specific reference job status and progress | Yes |
| **Dossier** | `GET` | `/api/v1/cases/dossiers` | List all on-disk case dossiers with file counts | Yes |
| **Dossier** | `GET` | `/api/v1/cases/dossiers/{id}` | Retrieve complete structured dossier for a specific case | Yes |
| **Dossier** | `GET` | `/api/v1/cases/dossiers/{id}/files/{path}` | Stream dossier file (supports HTTP 206 video Range) | Yes |
| **Dossier** | `POST` | `/api/v1/cases/dossiers/sync` | Force resynchronization of on-disk case dossiers | Yes |
| **Analysis** | `POST` | `/api/v1/identify/image` | Single-image person detection and gait identification | Yes |
| **Analysis** | `POST` | `/api/v1/analyze/video` | Video file upload and sampled gait analysis | Yes |
| **Telemetry**| `GET` | `/health` | Root service health check and loaded status | No |
| **Telemetry**| `GET` | `/health/live` | Process liveness probe (PID, uptime) | No |
| **Telemetry**| `GET` | `/health/ready` | Worker readiness check | No |
| **Telemetry**| `GET` | `/health/system` | Detailed host telemetry (CPU, RAM, GPU, VRAM) | No |
| **WebSocket**| `WS` | `/ws/recognition` | Real-time recognition event stream (JSON) | Yes |
| **WebSocket**| `WS` | `/ws/events` | Real-time system alert and security event stream | Yes |

---

## Research Benchmark Results

Scientific evaluation of the ByGaitLight architecture on the **CASIA-B** gait database under a strict **subject-disjoint** protocol (Train: `001–062`, Val: `063–074`, Test: `075–124`):

| Experiment | Pooling Strategy | Loss Formulation | Triplet Wt | Rank-1 Acc | Rank-5 Acc | Normal Walk (NM) | Carrying Bag (BG) | Clothing Change (CL) | ROC-AUC | EER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-003A** (Baseline) | Global (1) | Standard CE | 0.50 | 52.78% | 67.10% | 85.82% | 53.15% | 19.36% | 0.7499 | 31.95% |
| **EXP-003B** (HPP Alone) | HPP (4) | Standard CE | 0.50 | 61.43% | 75.63% | 91.55% | 60.55% | 32.18% | 0.8327 | 24.86% |
| **EXP-003C** (ArcFace Alone)| Global (1) | ArcFace | 0.50 | 59.58% | 73.78% | 91.00% | 61.55% | 26.18% | 0.8314 | 25.64% |
| **EXP-003D** (HPP + ArcFace)| HPP (4) | ArcFace | 0.00 | 69.71% | 80.91% | 96.73% | 72.79% | 39.64% | 0.8470 | 23.49% |
| **EXP-003E** (Locked Model) | **HPP (4)** | **ArcFace** | **0.25** | **72.63%** | **82.76%** | **97.00%** | **78.26%** | **42.64%** | **0.8776** | **20.46%** |

### Benchmark Analysis & Reality of Gait Biometrics

* **Covariate Impact**: Under standard walking conditions (`NM`), ByGaitLight achieves **97.00%** accuracy. When subjects carry bags (`BG`), accuracy drops to **78.26%**. When clothing changes significantly (`CL` - heavy coats, jackets), silhouette geometry changes substantially, reducing Rank-1 accuracy to **42.64%**.
* **Prototype Context**: Real-world surveillance footage presents additional challenges including unconstrained viewing angles, camera perspective distortions, dynamic shadows, and physical occlusions.

---

## Project Structure

```
ARGUS_AI/
├── .agents/                    # AGENTS.md instructions for AI coding assistants
├── .github/workflows/          # CI: test suite, lint, README-sync verification
├── .qodo/                      # Qodo AI agent/workflow configuration
├── api/                        # FastAPI REST routers, Pydantic schemas, server lifespan
├── assets/                     # Graphical assets and repository banner
├── automation/                 # Hardware detection, arbitration (DeviceManager), 12-stage bootstrap
├── configs/                    # YAML configurations (cameras, detection, gei, inference, system)
│   └── secrets/                # Gitignored credentials (e.g. firebase-service-account.json)
├── core/                       # Core system coordinator, shared utilities, logging configuration
├── data/                       # Gitignored local data, split by lifecycle
│   ├── datasets/                #   Research datasets (CASIA-B raw/cache/processed)
│   └── runtime/                 #   Live operational state (cases, galleries DB, enrollment, jobs)
│       └── cases/                #   Structured on-disk Case Dossiers ({case_id}_{person_name}/)
├── dataconnect/                # Firebase Data Connect schema
├── deployment/                 # Service shutdown management and environment manifests
├── docs/                       # Architectural documentation, reports, and README index
├── enrollment/                 # Target identity enrollment managers and lifecycle hooks
├── evaluation/                 # Scientific evaluation scripts (Rank-k, EER, ROC-AUC, threshold sweep)
├── events/                     # Event bus contracts, telemetry dispatchers
├── frontend/                   # React 19 + Vite surveillance dashboard application
│   ├── src/
│   │   ├── admin/              # User management, system policy, log viewers
│   │   ├── components/         # CCTV network grid, CaseDossierModal, Map, Telemetry
│   │   ├── contexts/           # AuthContext, GaitContext
│   │   └── hooks/              # useResizablePanel, useAuth, useGait
├── intelligence/               # Biometric intelligence, OpenSetRecognizer, Continual Learning
│   ├── candidate_validator.py  # Multi-gate candidate model validator (FAR/TAR/Anti-churn)
│   ├── date_aware_learning_scheduler.py # Event-date grouped learning job scheduler
│   ├── dual_modal_fusion.py    # Gait + Appearance score fusion
│   ├── nn_fine_tuner.py        # PyTorch ByGaitLight / OSNet gradient fine-tuner
│   ├── operational_embedding_collector.py # CCTV evidence collector (PREDICTED -> VERIFIED)
│   └── training_dataset_builder.py # 50% replay buffer and balanced dataset generator
├── models/                     # Deep learning architectures, model registry, and galleries
│   ├── architectures/          # ByGaitLight, UNet segmenter definitions
│   ├── live_gallery/           # Active 256-D ByGaitLight gait embeddings (.npy)
│   ├── appearance_gallery/     # Active 512-D OSNet appearance embeddings (.npy)
│   ├── candidates/             # Isolated candidate model weights (.pth)
│   ├── model_registry.py       # Atomic model version management and instant rollback
│   └── weights/                # Base weights (silhouette_segmenter.onnx, yolov8n.pt)
├── monitoring/                 # Structured logging, metrics collectors, telemetry
├── outputs/                    # Gitignored run artifacts: logs, benchmarks, reports, media
├── pipeline/                   # Modular gait recognition steps (Detection, Tracking, GEI, Matching)
├── preprocessing/              # Video frame extractors, silhouette binarization
├── runs/                       # Gitignored training run outputs and checkpoints
├── security_layer/             # Argon2id password hashing, SessionStore, RBAC, Fernet encryption
├── services/                   # Background services, CameraWorker, MissingPersonProcessor, CaseDossier
├── storage/                    # Local VectorStore (.npy), SQLite EmbeddingDatabase, Firebase store
├── streaming/                  # Multi-camera DRR scheduling, frame ring-buffers, admission control
├── tests/                      # Automated test suite (1,460 passed, 0 failed, 1 skipped)
│   ├── integration/            # Multi-component & isolation tests (12 camera/ref scenarios)
│   └── unit/                   # Unit test suite
├── tools/                      # Operational CLI tools, benchmarks, maintenance scripts, migrations
├── training/                   # Model training routines, loss functions, CASIA-B loaders
├── utils/                      # File I/O, image processing, geometry math utilities
├── cli.py                      # Unified CLI management entry point
├── main.py                     # Command-line system runner
├── Makefile                    # Make command targets
├── package.json                # Root dev-orchestration scripts (npm run dev/build/lint)
├── requirements.txt            # Python backend dependencies
└── VERSION                     # Project version file (0.1.0)
```

---

## Hardware & Software Environment

### Validated Development Environment

The following development environment represents the tested baseline on which all benchmarks and test suites were executed:

* **Operating System**: Windows 11 Pro (x86_64 AMD64)
* **Python Runtime**: Python 3.11.9 (64-bit)
* **Graphics Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB GDDR6 VRAM)
* **NVIDIA Driver**: 535.xx+ (CUDA 12.1 / 12.6 driver capability)
* **Host Memory**: 8 GB usable system RAM
* **Storage**: NVMe M.2 Solid State Drive
* **Node.js Environment**: Node.js v18+ and npm v9+

*(Note: CPU-only execution is fully supported via automatic fallback, though video inference framerates will be proportionally lower).*

---

## Installation & Windows Setup

### 1. Clone & Prepare Virtual Environment

Open Windows PowerShell in the desired directory:

```powershell
# Clone the repository
git clone https://github.com/chanuka8/argus-gait-recognition.git
cd argus-gait-recognition

# Create Python 3.11 virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python Dependencies & Bootstrap Environment

```powershell
# Upgrade pip and install wheel
python -m pip install --upgrade pip setuptools wheel

# Install backend dependencies
pip install -r requirements.txt

# Run automated hardware discovery and environment bootstrap
powershell -ExecutionPolicy Bypass -File ".\tools\bootstrap_env.ps1"
```

### 3. Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### 4. Optional: Firebase Configuration

Firebase persistence is **optional**. When unconfigured, ARGUS runs in hermetic offline mode without errors. To enable cloud synchronization:
1. Place your Firebase Admin SDK service account key JSON at:
   `configs/secrets/firebase-service-account.json`
2. Set the environment variable in PowerShell:
   ```powershell
   $env:FIREBASE_SERVICE_ACCOUNT_PATH="E:\ARGUS_AI\configs\secrets\firebase-service-account.json"
   ```

---

## Running the System

### Option 1: Unified Development Server (Backend + Frontend)

ARGUS AI includes a unified Node.js dev orchestrator (`tools/dev.js`) that starts the FastAPI server, awaits readiness, and starts the React Vite development server:

```powershell
npm run dev
```

* **Frontend UI**: `http://localhost:5173`
* **FastAPI Backend**: `http://127.0.0.1:8000`
* **Interactive OpenAPI Docs (Swagger)**: `http://127.0.0.1:8000/docs`

### Option 2: Backend Only

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```

### Option 3: Frontend Only

```powershell
npm run dev:frontend
```

### Option 4: Unified Command-Line Interface (CLI)

```powershell
# System health check
.\.venv\Scripts\python.exe cli.py --mode health

# Documentation integrity check
.\.venv\Scripts\python.exe cli.py --mode docs-check

# Live webcam recognition with auto-enrollment watcher
.\.venv\Scripts\python.exe cli.py --mode live
```

---

## Verification & Testing Commands

All verification commands are repository-relative and executable from `E:\ARGUS_AI`:

```powershell
# 1. Full Pytest Suite (1,030 passed, 0 failed, 1 hardware-dependent test skipped)
.\.venv\Scripts\python.exe -m pytest -v

# 2. Camera & Reference Media Isolation Integration Tests (12 scenarios)
.\.venv\Scripts\python.exe -m pytest tests/integration/backend/test_camera_reference_isolation.py -v

# 3. Reference Job Recovery & Checkpointing Integration Tests (17 tests)
.\.venv\Scripts\python.exe -m pytest tests/unit/backend/test_job_recovery.py -v

# 4. Continual Learning & Accuracy Validation Suite
.\.venv\Scripts\python.exe -m pytest tests/unit/backend/test_continual_learning_accuracy_validation.py -v

# 5. Backend Code Quality & Linter (Ruff)
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .

# 6. Python Bytecode Compilation Verification across all modules
.\.venv\Scripts\python.exe -m compileall -q api automation core deployment enrollment evaluation events frontend intelligence models monitoring pipeline preprocessing scripts security_layer services storage streaming tests tools training utils

# 7. Frontend Linter & Production Build
npm --prefix frontend run lint
npm --prefix frontend run build

# 8. Git Whitespace & Formatting Check
git diff --check
```

---

## Current Implementation Status Matrix

The following classification separates features verified in the codebase from research boundaries:

| Subsystem / Capability | Implementation Classification | Evidence / Source of Truth |
| :--- | :---: | :--- |
| **YOLOv8n Person Detection & EMA Smoothing** | **Implemented & Verified** | `pipeline/detection/person_detector.py`, `TrackingStep` ($\alpha=0.35$) |
| **ByteTrack Multi-Object Tracking** | **Implemented & Verified** | `pipeline/steps/tracking.py` |
| **Silhouette Extraction (UNet + Otsu Fallback)**| **Implemented & Verified** | `pipeline/silhouette/extractor.py`, `models/model_store/weights/silhouette_segmenter.onnx` |
| **Rolling GEI Generation (64 × 128, window=15)** | **Implemented & Verified** | `pipeline/steps/live_gei.py`, `configs/gei.yaml` |
| **ByGaitLight 256-D L2-Normalized Embedding** | **Implemented & Verified** | `models/architectures/bygait_light.py` (HPP `part_bins=4`) |
| **Appearance ReID (OSNet-x0.25 512-D)** | **Implemented & Verified** | `models/reid/osnet_backbone.py` |
| **Four-Tier Open-Set Decision Policy** | **Implemented & Verified** | `intelligence/open_set_recognizer.py`, `configs/inference.yaml` |
| **Temporal Majority Voting (10 frames / 3 min)**| **Implemented & Verified** | `configs/inference.yaml` (`min_stable_votes=3`, `history_size=10`) |
| **Decoupled Camera Capture & Live Preview** | **Implemented & Verified** | `services/camera_worker.py` (median startup ≈ 914 ms, adoption ≈ 31 ms) |
| **Camera / Reference Job Isolation** | **Implemented & Verified** | `test_camera_reference_isolation.py` (12/12 integration tests passed) |
| **Offline Reference Job Recovery & Checkpoints**| **Implemented & Verified** | `services/reference_job_manager.py`, `test_job_recovery.py` (17 tests) |
| **On-Disk Case Dossiers & HTTP 206 Streaming** | **Implemented & Verified** | `services/case_dossier_manager.py`, `CaseDossierModal.jsx` |
| **Path Traversal Protection (`relative_to`)** | **Implemented & Verified** | `CaseDossierManager.get_dossier_file()`, `api/v1/router.py` |
| **Argon2id Password Hashing & RBAC** | **Implemented & Verified** | `security_layer/password_hasher.py`, `security_layer/authorization.py` |
| **Local-First SQLite & VectorStore Storage** | **Implemented & Verified** | `storage/vector_store.py` (`allow_pickle=False`), `EmbeddingDatabase` |
| **Asynchronous Firestore Persistence** | **Implemented & Verified** | `storage/firebase_embedding_store.py`, `data/runtime/firebase_offline_store.json` |
| **Date-Aware Continual Learning & Replay** | **Implemented & Verified** | `DateAwareLearningScheduler`, `TrainingDatasetBuilder` (50% replay) |
| **PyTorch NN Fine-Tuning & Multi-Gate Gating** | **Implemented & Verified** | `NNFineTuner`, `CandidateValidator` (FAR/TAR/Anti-churn) |
| **Atomic Model Registry & Rollback** | **Implemented & Verified** | `ModelRegistry` (Hot-reload, rollback $< 50\text{ms}$) |
| **Automated Hardware Arbitration (CUDA/CPU)** | **Implemented & Verified** | `automation/device_manager.py`, 12-stage bootstrap |
| **Multi-Camera Fair-Share Ingestion (DRR)** | **Implemented & Verified** | `streaming/production_multicamera_engine.py` |
| **Full Pytest Test Suite** | **Tested & Verified** | **1,030 passed, 0 failed, 1 skipped** (hardware-dependent webcam) |
| **Physical Multi-Camera Citywide Deployment** | **Research / Boundary** | Laboratory and local network verified; municipal-scale testing not claimed |
| **Million-Identity Sub-Millisecond Indexing** | **Future Work** | Evaluated on development gallery templates; FAISS-IVF/HNSW scaling planned |

---

## Research Boundaries & Current Limitations

To maintain scientific integrity and realistic expectations, the following limitations are explicitly noted:

1. **Biometric Covariate Variations**: As demonstrated in the CASIA-B benchmarks, changes in clothing (e.g. heavy winter coats vs. athletic shorts) substantially alter silhouette geometry and lower single-frame recognition accuracy. Multi-frame temporal consensus and appearance fusion mitigate, but do not completely eliminate, this covariate effect.
2. **Camera Hardware & Driver Startup Overhead**: Total camera initialization time is primarily governed by Windows DirectShow driver negotiation and USB bus synchronization (median ≈ 914 ms), rather than application code (pre-verified adoption path ≈ 31 ms).
3. **Physical Hardware Dependency in Tests**: One test in the automated suite is skipped when executing in environments lacking a physical DirectShow USB webcam (`test_auto_camera_detection.py`).
4. **Human-in-the-Loop Continual Learning**: Continual learning requires operator validation of operational observations before samples can transition to `TRAINING_ELIGIBLE`. Autonomous self-training on unverified observations is deliberately prevented to prevent model drift and data poisoning.
5. **Local Workstation Resource Footprint**: Real-time multi-camera batching requires an NVIDIA GPU with at least 4–6 GB VRAM. When running on CPU-only hardware, frame sampling and queue throttling are automatically activated by the admission controller.

---

## Security Notes & Responsible Disclosure

* **Defensive Engineering**: ARGUS AI enforces input boundaries, parameter bounds checking, path traversal sanitization, and cryptographically sound password hashing (Argon2id).
* **Zero Committed Credentials**: The repository includes zero secret keys, service-account certificates, credentials files, or runtime tokens. Template configurations (`.env.example`) provide dummy placeholder structures only.
* **Biometric Privacy Considerations**: Biometric gait signatures are treated as sensitive identification data. Local vector stores use restricted file permissions, and cloud synchronization requires authenticated service accounts.

---

## License & Maintainer

### License

This project is licensed under the [MIT License](LICENSE).

### Project Maintainer & Lead Researcher
<br>

**Chanuka Sandun**  
Undergraduate in Cybersecurity  
Developer & Lead Architect of ARGUS AI

* **GitHub**: [@chanuka8](https://github.com/chanuka8)
* **LinkedIn**: [linkedin.com/in/chanukasandun](https://www.linkedin.com/in/chanukasandun/)
