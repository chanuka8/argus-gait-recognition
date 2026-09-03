# ARGUS AI — Firebase Integration Audit & Inventory

## 1. Executive Overview

This audit establishes the explicit boundary between the **ARGUS Local Real-Time Inference Pipeline** and the **Firebase Asynchronous Persistence & Synchronization Layer**. 

### Core Architectural Principle
* **Real-Time Inference Source of Truth**: Local VectorStore, in-memory galleries, local file persistence, and local PyTorch/TensorFlow execution.
* **Asynchronous Persistence & Audit Layer**: Firebase Firestore (biometric embeddings, operator accounts, identity lineage, model registry, audit logs) and Firebase Storage (model artifacts, controlled case evidence).
* **Strict Non-Blocking Invariant**: Firebase is **never** on the synchronous per-frame critical path. If Firebase is offline, degraded, or experiencing network latency, camera streaming, inference, person detection, silhouette extraction, GEI computation, and local vector matching continue with 0ms added latency.

---

## 2. Module Persistence Inventory

| Module | Subsystem / Location | Current Persistence | Firebase Required? | Firebase Purpose | Real-time Critical Path? | Priority | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Operator Authentication** | `security_layer/auth.py` | Firestore `admins`/`investigators` + memory SessionStore | **REQUIRED** | Production authoritative source for administrator & investigator credentials and roles | No (Login / Session validation) | HIGH | **VERIFIED** |
| **Gait Feature Extraction** | `pipeline/steps/feature_extraction.py` | Local PyTorch weights (`runs/exp_001/best_model.pth`) | **NOT REQUIRED** | Pure local CNN inference (ByGaitLight 256D) | Yes (15-30 FPS) | N/A | **VERIFIED** |
| **Appearance Feature Extractor** | `intelligence/appearance_embedding.py` | Local PyTorch weights (`models/weights/osnet_x0_25.pth`) | **NOT REQUIRED** | Pure local ReID CNN inference (OSNet 512D) | Yes (Inference interval) | N/A | **VERIFIED** |
| **Local Vector Gallery** | `storage/vector_store.py` | Local `.npy` / `.json` in `models/live_gallery/` | **NOT REQUIRED** | Real-time cosine similarity gallery search | Yes (< 5ms search) | N/A | **VERIFIED** |
| **Biometric Embedding Store** | `storage/firebase_embedding_store.py` | Firestore `biometric_embeddings` + `data/firebase_offline_store.json` | **REQUIRED** | Long-term cloud persistence, cross-node synchronization, provenance, lineage, and disaster recovery | No (Asynchronous / Buffered) | HIGH | **VERIFIED** |
| **Subject Enrollment** | `services/gait_service.py` -> `EmbeddingDatabase` | Local Person JSON (`data/embedding_db/persons/`) + Firebase | **REQUIRED** | Multi-node subject synchronization, identity management, and cloud backup | No (Enrollment event) | HIGH | **VERIFIED** |
| **Camera Workers & Streaming** | `streaming/camera_worker.py` | In-memory frame buffers | **NOT REQUIRED** | Local MJPEG frame capture and encoding | Yes (Hardware I/O) | N/A | **VERIFIED** |
| **Camera Metadata & Node State** | `api/v1/router.py`, `streaming/` | Memory + Firestore `active_cameras` | **OPTIONAL** | Cluster-wide camera registry and deployment records | No | MEDIUM | **VERIFIED** |
| **Recognition Event Logging** | `services/gait_service.py` | Memory event ring buffer + Firestore `detections` | **OPTIONAL** | Centralized surveillance audit and multi-investigator dashboards | No (Async batching) | MEDIUM | **VERIFIED** |
| **Date-Aware Learning Scheduler** | `intelligence/date_aware_learning_scheduler.py` | Local `data/learning_jobs.json` + Firestore `learning_jobs` | **OPTIONAL** | Cross-cluster continual learning coordination | No (Scheduled batch) | MEDIUM | **VERIFIED** |
| **Model Registry** | `models/model_registry.py` | Local `models/model_registry.json` + Firestore `model_registry` | **REQUIRED** | Model governance, candidate tracking, checksums, atomic promotion, rollback metadata | No (Model deployment) | HIGH | **VERIFIED** |
| **Continual Learning Audit Trail** | `intelligence/continual_learning_audit_trail.py` | Local `data/continual_learning_audit_trail.json` + Firestore `audit_logs` | **REQUIRED** | Tamper-evident forensic audit trail of retraining, metrics, and promotions | No (Post-training) | HIGH | **VERIFIED** |
| **Candidate Model Validator** | `intelligence/candidate_validator.py` | In-memory evaluation vs holdout datasets | **NOT REQUIRED** | Local mathematical validation and safety gating | No (Post-training) | N/A | **VERIFIED** |
| **Controlled Evidence Storage** | `storage/firebase_embedding_store.py` | Firebase Storage (`cases/{case_id}/`) | **OPTIONAL** | Controlled case evidence export and temporary investigation media | No (Manual action) | LOW | **VERIFIED** |

---

## 3. Technical Justifications

### 3.1 Why Modules are NOT REQUIRED to Use Firebase
1. **Real-Time Gait Recognition Pipeline (Detection -> Tracking -> Silhouette -> GEI -> ByGaitLight -> VectorStore)**:
   Surveillance frames arrive at 15 to 30 FPS per camera. Querying a remote cloud database over WAN/HTTP introduces 50ms to 2000ms latency and creates an external failure dependency. ARGUS executes all feature extraction and gallery matching entirely in local GPU/CPU memory.
2. **Local Frame Streaming (MJPEG & Snapshots)**:
   Video feeds are generated directly by camera workers (`streaming/camera_worker.py`) using OpenCV and in-memory frame buffers. Video streams are delivered directly to the browser via authenticated HTTP streams (`multipart/x-mixed-replace`) and authenticated snapshot polling, completely bypassing external cloud services.

### 3.2 Why Modules are REQUIRED to Use Firebase
1. **Operator Authentication (`security_layer/auth.py`)**:
   Enforces single source of truth for operator access, role-based privileges (`Root Admin`, `admin`, `investigator`), and password hash verification. Ensures multi-node deployments share consistent access control.
2. **Biometric Embedding Store (`storage/firebase_embedding_store.py`)**:
   Provides durable, multi-node cloud persistence for enrolled identities, provenance metadata, embedding lineage, and full disaster recovery without relying solely on local disk storage.
3. **Model Registry & Audit Trail (`models/model_registry.py`, `intelligence/continual_learning_audit_trail.py`)**:
   Prevents unvalidated models from entering production, tracks candidate validation metrics, ensures atomic promotion and rollback capabilities, and logs tamper-evident records.
