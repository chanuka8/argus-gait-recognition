# ARGUS AI — Firebase Architecture & Responsibility Boundary

## 1. Architectural Principles

ARGUS AI strictly separates **Real-Time Edge/Local Inference** from **Cloud-Based Asynchronous Persistence & Synchronization**.

```mermaid
flowchart TD
    subgraph LocalEdge["REAL-TIME LOCAL INFERENCE (Authoritative & Zero-Latency)"]
        Cam["Camera Feed (Webcam / USB / RTSP)"] --> Det["Person Detection (YOLO / OpenCV)"]
        Det --> Track["ByteTrack Tracking Engine"]
        Track --> Sil["Silhouette Extractor (64x128)"]
        Sil --> GEI["Gait Energy Image (GEI)"]
        GEI --> CNN["ByGaitLight CNN (PyTorch)"]
        CNN --> Emb["256D Gait Embedding"]
        Track -.-> OSNet["OSNet-x0.25 (PyTorch)"]
        OSNet -.-> AppEmb["512D Appearance Embedding"]
        Emb --> VecStore["Local VectorStore (models/live_gallery/)"]
        AppEmb --> VecStore
        VecStore --> Match["Cosine Similarity & Dual-Modal Fusion"]
        Match --> RealtimeAlert["Real-time UI Recognition (< 10ms)"]
    end

    subgraph AsyncSync["ASYNCHRONOUS PERSISTENCE & SYNC (Non-Blocking)"]
        Match --> LocalDB["Local EmbeddingDatabase (data/embedding_db/)"]
        LocalDB --> RetryQ["In-Memory Retry Queue (100 max)"]
        RetryQ --> OfflineFile["Offline Store (data/firebase_offline_store.json)"]
        RetryQ --> AdminSDK["Firebase Admin SDK (auth, firestore, storage)"]
        AdminSDK --> Firestore["Google Cloud Firestore (argus-17702)"]
        AdminSDK --> GCS["Firebase Storage (argus-17702.firebasestorage.app)"]
    end

    subgraph Governance["MODEL GOVERNANCE & AUDIT"]
        ModelReg["Model Registry (models/model_registry.py)"] --> Firestore
        AuditTrail["Continual Learning Audit Trail"] --> Firestore
    end
```

---

## 2. Real-Time Local Boundary

1. **Hardware Independence**: The real-time loop communicates directly with local cameras (USB index or RTSP URL).
2. **Inference Latency Target**: Frame-to-identity latency remains under 35ms on GPU and under 80ms on CPU.
3. **Local VectorStore**: Enrolled subject galleries are held in-memory as normalized NumPy arrays (`float32`).
4. **Zero WAN Dependency**: If internet connectivity is completely severed:
   - Camera feeds run normally.
   - Person detection and tracking run normally.
   - Gait silhouettes and GEIs are generated normally.
   - Recognition events fire immediately.
   - Local disk persists person records and offline queue records.

---

## 3. Asynchronous Persistence Boundary

Firebase operations occur entirely outside the frame-processing loop:
1. **Enrollment**: When an operator enrolls a subject, the embeddings are first saved to the local `VectorStore` and `data/embedding_db/persons/{id}.json`. After local persistence is verified, embeddings are dispatched asynchronously to Firebase.
2. **Offline Queueing & Automatic Sync**:
   - If Firebase Admin SDK is offline or returns a network timeout, the write is buffered into an in-memory queue and written to `data/firebase_offline_store.json`.
   - When connectivity resumes, `FirebaseEmbeddingStore.process_retry_queue()` flushes queued documents idempotently.
3. **Deterministic ID Idempotency**:
   - Document IDs are deterministically generated using `generate_deterministic_id(person_id, modality, capture_timestamp, track_id, camera_id)`.
   - Retries overwrite or merge cleanly without creating duplicate records.

---

## 4. Failure Domain Isolation

| Failure Scenario | Local Pipeline Impact | Firebase Impact | Recovery Mechanism |
| :--- | :--- | :--- | :--- |
| **No Internet Connection** | None. Local inference continues at full FPS. | Enters `offline` mode. Writes queued locally. | Automatic retry upon connection restoration. |
| **Missing Service Account** | None. | Safe fallback to offline mode. | Credentials placed in `config/firebase-service-account.json`. |
| **Firestore Network Timeout** | None. Inference loop never waits for HTTP calls. | Request marked for retry. | Backoff retry queue processed on background worker. |
| **Local Disk Crash** | Local cache lost. | Full cloud copy preserved. | `FirebaseEmbeddingStore.rebuild_local_from_firebase()` restores entire gallery. |

---

## 5. Dual-Modal Vector Separation

To prevent dimensionality corruption:
- **Gait Modality**: strictly 256D normalized vector (`embedding_dim=256`, `modality="gait"`).
- **Appearance Modality**: strictly 512D normalized vector (`embedding_dim=512`, `modality="appearance"`).
- **Schema Validation**: `FirebaseEmbeddingDocument.validate_schema()` rejects any document where `len(vector) != embedding_dim` or where vectors contain NaN/Inf or zero-norm.

---

## 6. Model Promotion Consistency and Recovery

ARGUS AI implements:
> **"Thread-safe atomic local model promotion with durable asynchronous Firebase synchronization and reconciliation."**

### 6.1 Authoritative Local State vs Cloud Mirror
* **Local Model Registry (`models/model_registry.json`)**: Authoritative for runtime inference. `LOCAL ACTIVE MODEL = runtime authority`.
* **Firestore Model Registry Mirror (`model_registry` collection)**: Asynchronous cloud mirror and governance state. Cloud outages never stop edge model inference.

### 6.2 Transactional Promotion State Machine
* **Lifecycle**:
  `CANDIDATE -> VALIDATED -> PROMOTION_PENDING -> LOCAL_COMMITTED -> CLOUD_SYNC_PENDING -> SYNCHRONIZED`
* **Cloud Failure Path**:
  `LOCAL_COMMITTED -> CLOUD_SYNC_FAILED -> RECONCILIATION_PENDING -> SYNCHRONIZED`
* **Rollback Lifecycle**:
  `ACTIVE -> ROLLBACK_PENDING -> LOCAL_ROLLBACK_COMMITTED -> CLOUD_SYNC_PENDING -> SYNCHRONIZED`

### 6.3 Local Atomic Commit
1. Acquires internal `threading.RLock()`.
2. Validates candidate status (`VALIDATED`), file existence, and architecture compatibility.
3. Increments `registry_revision`.
4. Writes new registry state to `.tmp` file and flushes with `os.fsync()`.
5. Atomically replaces destination using `tmp.replace(self.registry_file)`.

### 6.4 Durable Synchronization Intent (Outbox)
Cloud sync intent is durably persisted to `data/model_sync_outbox.json`:
* Fields: `event_id`, `model_version`, `model_type`, `desired_status`, `operation`, `registry_revision`, `created_at`, `attempt_count`, `last_attempt_at`, `next_retry_at`, `status`, `checksum_sha256`, `error_info`.
* Survives process crashes and restarts.

### 6.5 Idempotent Cloud Sync & Optimistic Concurrency
* Synchronization uses deterministic document IDs: `{model_type}_{model_version}`.
* Active pointer `{model_type}_active_pointer` performs optimistic concurrency checking against `registry_revision` to prevent stale writers from overwriting newer cloud state.
* Exponential backoff retry for failed cloud writes; marks `RECONCILIATION_REQUIRED` upon retry exhaustion.

### 6.6 Reconciliation (`reconcile_with_firebase()`)
1. Reads local authoritative model state.
2. Reads pending outbox events.
3. Local runtime state always wins (`LOCAL > CLOUD`).
4. Repairs Firestore mirror and active pointers.
5. Marks outbox events `SYNCHRONIZED`.

### 6.7 Failure Scenarios Matrix
* **Scenario A (Local Write Succeeds, Cloud Fails)**: Local remains new production model; outbox marks event pending/retrying; inference continues uninterrupted.
* **Scenario B (Local Write Fails)**: Local active model remains previous version; operation raises safely; zero Firebase active state created.
* **Scenario C (Crash Recovery)**: Process crash after local commit reloads outbox from disk on restart; reconciliation resumes automatically.
* **Scenario D (Idempotent Retry)**: Duplicate retries update existing records without creating duplicates or corrupting active state.
* **Scenario E (Concurrent Promotion)**: Thread lock serializes promotions; registry file remains valid JSON.
* **Scenario F (Successive Promotions)**: Promotion v2 followed by v3 converges to v3 locally and in cloud.
* **Scenario G (Rollback During Pending Sync)**: Rollback atomically committed locally; cloud mirror converges to restored version.

---

## 7. Physical Camera Validation

ARGUS AI strictly separates **Automated Security Tests** from **Physical Hardware Stream Validation**:

| Component / Path | Verification Tier | Result | Measured Metrics / Details |
| :--- | :--- | :---: | :--- |
| **Local Webcam (Index 0)** | Physical Hardware Probe | **PASS** | Backend: `CAP_DSHOW`, Resolution: `640x480`, Measured FPS: `19.97`, Frames Captured: 10, Reconnect Test: `PASS` |
| **USB Webcam (Index 1)** | Physical Hardware Probe | **NOT EXECUTED** | No secondary USB video class hardware connected in current environment |
| **Live RTSP Camera Stream** | Network Stream Probe | **NOT EXECUTED** | No external live RTSP stream URL configured (`ARGUS_TEST_RTSP_URL` unset) |
| **End-to-End Camera -> Gait** | Pipeline Stage Execution | **PASS** | Transport: `PASS`, Decoding: `PASS`, ByGaitLight (256D): `PASS`, OSNet ReID (512D): `PASS`, VectorStore: `PASS` |

### Deployment Workstation Hardware Validation Procedure
To execute physical hardware probe on deployment surveillance stations:
```bash
# 1. Probe local webcam (index 0) and any secondary USB webcam
.\.venv\Scripts\python.exe scripts/validate_camera_hardware.py --webcam-index 0 --usb-index 1

# 2. Probe live RTSP surveillance feed
.\.venv\Scripts\python.exe scripts/validate_camera_hardware.py --rtsp-url rtsp://user:pass@camera_ip:554/live

# 3. Verify End-to-End Camera -> ByGaitLight Pipeline
.\.venv\Scripts\python.exe scripts/verify_camera_to_gait_pipeline.py
```
