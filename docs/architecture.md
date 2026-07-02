# System Architecture Specification

This document details the high-level architecture, subsystem boundaries, data flows, and design decisions governing the ARGUS Gait Recognition Module.

---

## 1. High-Level Architecture Overview

The module is structured as a decoupled, layered pipeline processing framework. This approach separates real-time ingestion, feature engineering, neural network inference, database retrieval, and risk management policies into distinct, manageable steps.

The architecture is divided into five core layers:
1. **Ingestion Layer:** Captures raw frames from cameras or files using OpenCV and routes them through thread-safe buffers.
2. **Processing Layer:** Detects individuals, tracks their movements across frames, extracts crop contours, and accumulates silhouettes into a temporal Gait Energy Image (GEI).
3. **Inference & Matching Layer:** Feeds aggregated GEIs through the `ByGaitLight` CNN to output feature representations, and executes vector scans against a local NumPy store.
4. **Intelligence & Risk Layer:** Analyzes similarity scores, verifies confidence level categories, and implements threat classifications.
5. **Interface Layer:** Provides CLI entry points, REST endpoints (FastAPI), and visual feedback overlays.

```
       +-----------------------------------------------------------+
       |                     INTERFACE LAYER                       |
       |  FastAPI Server (api/) | Command CLI (cli.py) | Overlays  |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |                    INTELLIGENCE & RISK                    |
       |   ConfidenceScorer | SecurityEngine | csv SecurityLogger  |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |                INFERENCE & MATCHING LAYER                 |
       |   ByGaitLight CNN | Cosine Similarity | VectorStore (npy) |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |                     PROCESSING LAYER                      |
       |   YOLOv8 Detect | ByteTrack | Silhouette Step | LiveGEI   |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |                     INGESTION LAYER                       |
       |    StreamEngine (OpenCV) | BufferQueue | FrameDropper     |
       +-----------------------------------------------------------+
```

---

## 2. Module Responsibility Matrix

The system functionality is distributed across the following modules:

| Subsystem / Module | Primary Class / Component | Responsibilities |
| :--- | :--- | :--- |
| [core/boot.py](file:///e:/ARGUS_AI/core/boot.py) | `BootManager` | Loads config YAMLs, validates folder directories, performs CPU/GPU RAM check. |
| [core/context.py](file:///e:/ARGUS_AI/core/context.py) | `SystemContext` | Global safe dictionary tracking runtime status, settings, and hardware load. |
| [core/system_monitor.py](file:///e:/ARGUS_AI/core/system_monitor.py) | `SystemMonitor` | Queries system CPU cores, RAM limits, and VRAM memory status. |
| [pipeline/steps/](file:///e:/ARGUS_AI/pipeline/steps/) | `DetectionStep`, `TrackingStep`, `SilhouetteStep`, `LiveGEI`, `FeatureExtractionStep`, `MatchingStep` | Single-responsibility pipeline nodes that perform detection, tracking, silhouette extraction, GEI averaging, feature extraction, and database cosine matching. |
| [streaming/](file:///e:/ARGUS_AI/streaming/) | `BufferQueue`, `FrameDropper` | Manages thread-locked raw frame caching and drops intermediate frames if pipeline processing lags behind camera inputs. |
| [intelligence/](file:///e:/ARGUS_AI/intelligence/) | `ConfidenceScorer` | Maps matching similarity values to score ranges (`HIGH`, `MEDIUM`, `LOW`) and checks if matches are trusted. |
| [security_layer/](file:///e:/ARGUS_AI/security_layer/) | `SecurityEngine`, `SecurityLogger` | Compares identity states, assigns severity ratings (`INFO`, `MEDIUM`, `HIGH`), decisions (`ALLOW`, `REVIEW_REQUIRED`, `SECURITY_ALERT`), and writes logs to a local CSV file. |
| [storage/](file:///e:/ARGUS_AI/storage/) | `VectorStore`, `CacheManager`, `DataManager` | Manages NumPy vector save/loads, LRU cache memory pipelines, and JSON files CRUD utilities. |
| [enrollment/](file:///e:/ARGUS_AI/enrollment/) | `EnrollmentManager`, `FolderWatcher` | Validates folders, processes enrollment directories, extracts embeddings, and appends them to the vector store. |
| [evaluation/](file:///e:/ARGUS_AI/evaluation/) | `SplitEvaluator`, `EvaluationVisualizer` | Executes performance validations and plots accuracy and confidence graphs using matplotlib. |

---

## 3. System Entry Points

There are three primary ways to interact with the module:
1. **Surveillance Script ([scripts/test_live_recognition.py](file:///e:/ARGUS_AI/scripts/test_live_recognition.py)):** Launches the real-time processing loop that processes a live webcam feed, tracks individuals, matches their gaits, and displays tracking bounding boxes with overlays.
2. **API Gateway ([api/server.py](file:///e:/ARGUS_AI/api/server.py)):** Starts a FastAPI server hosting POST endpoints for target identification and directory-based person registration.
3. **Command router ([cli.py](file:///e:/ARGUS_AI/cli.py)):** Simplifies execution of live feeds, servers, evaluations, and hardware diagnostic routines via command line flags.

---

## 4. System Data Flow

The flow of data through the system during a real-time matching cycle is outlined below:

```
[Camera Input] 
      │ 
      ▼
(OpenCV Mat Frame) ──> [BufferQueue] (Pushed if queue not full)
      │
      ▼
[YOLOv8 Detector] ──> (Bounding Boxes)
      │
      ▼
[ByteTrack Association] ──> (Active Track IDs + Box Coordinates)
      │
      ▼
[Cropping & Background Subtraction] ──> (Binary Silhouettes)
      │
      ▼
[LiveGEI Aggregator] ──> (Accumulated 128x64 Average GEI Image)
      │
      ▼
[ByGaitLight CNN] ──> (256-Dimensional Feature Embedding)
      │
      ▼
[Cosine Similarity Matcher] ──> (Closest Profile ID + Score Value)
      │
      ▼
[ConfidenceScorer & SecurityEngine] ──> (Severity Code + Decision Code)
      │
      ▼
[Overlay Rendering & CSV Log]
```

---

## 5. Security Evaluation Layer

The security layer performs threat assessment based on identification confidence and profile matching:
- **Identified Subject with High Confidence ($\ge 0.92$):**
  - Severity: `INFO`
  - Decision: `ALLOW` (Green bounding box overlay)
- **Identified Subject with Low Confidence ($0.70 \le \text{score} < 0.85$):**
  - Severity: `MEDIUM`
  - Decision: `REVIEW_REQUIRED` (Orange bounding box overlay)
- **Unregistered or Unknown Pattern ($< 0.70$):**
  - Severity: `HIGH`
  - Decision: `SECURITY_ALERT` (Red bounding box overlay)

**Audit Logging:** All threat outputs are appended to a CSV file (`outputs/security_logs/security_events.csv`) with timestamps, track IDs, matched identities, confidence scores, severity ratings, and decisions.

---

## 6. Architecture Selection Rationale

1. **Pipeline Step Design Pattern:** Breaking the video stream steps into self-contained classes makes the code easily maintainable. Each step has its own scope and can be tested individually.
2. **Decoupling Training from Inference Runtime:** The model training subsystem is completely separated from the live camera matching pipeline. During runtime matching, the system loads pretrained checkpoint weights in evaluation mode (`model.eval()`), which removes training overhead.
3. **NumPy-Based Gallery Operations:** Instead of deploying heavy database engines, gallery embeddings are stored as flat NumPy arrays (`.npy`). Calculating cosine similarity is reduced to a matrix dot product operation using vectorized NumPy, which completes in under 10 milliseconds.
4. **Producer-Consumer Streaming Design:** Ingesting camera frames in a separate thread and storing them in a thread-locked queue decoupling the frame capture rate from model prediction speed, preventing UI freezes.
