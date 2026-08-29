# ARGUS AI — Pre-Firebase Upgrade Baseline Record

**Baseline Date:** 2026-08-29  
**Repository:** `chanuka8/argus-gait-recognition` (`ARGUS AI`)  
**Active Branch:** `main`  
**Baseline HEAD Commit:** `ba5b166 docs: update thesis audit reports, quality documentation, and forensic audits`  
**Remote Tracking Branch:** `origin/main` (`https://github.com/chanuka8/argus-gait-recognition.git`)  
**Synchronization State:** `0 0` (0 ahead, 0 behind, Up-to-date)  
**Working Tree State:** `CLEAN` (0 uncommitted files, 0 untracked files)

---

## 1. Quality & Test Baseline

| Check / Suite | Status | Execution Metrics |
|---|---|---|
| **Pytest Unit & Integration Suite** | **PASS** | 642 / 642 tests passing (100% pass rate in 104.95s) |
| **Ruff Linter** | **PASS** | 0 errors, 0 warnings across all Python packages |
| **Python Bytecode Compilation (`compileall`)** | **PASS** | 0 compilation errors across 399 Python source files |
| **Frontend ESLint** | **PASS** | 0 errors, 0 warnings (`eslint .`) |
| **Frontend Vite Production Build** | **PASS** | 1832 modules transformed; 0 warnings; largest chunk 365.46 kB < 500 kB threshold |
| **YAML Configuration Integrity** | **PASS** | 15 / 15 YAML files valid |
| **JSON Configuration Integrity** | **PASS** | 252 / 252 JSON files valid |
| **Folder Documentation Sync** | **PASS** | 20 / 20 package READMEs aligned with architecture index |

---

## 2. Architecture Subsystem State (Pre-Upgrade)

### A. Artificial Intelligence & Feature Extraction
- **Gait Recognition Backbone:** `ByGaitLight` ONNX / PyTorch architecture producing 256-dimensional gait embeddings from extracted silhouettes.
- **Appearance Re-Identification:** `OSNet-x0.25` lightweight deep ReID backbone generating 512-dimensional appearance embeddings.
- **Dual-Modal Decision Fusion:** `DualModalFusionEngine` utilizing calibrated confidence scoring and adaptive modal weighting.
- **Continuous Learning Engine:** `ContinuousImprovementEngine` with drift detection, candidate validation, and neural network fine-tuning workers.

### B. Camera & RTSP Lifecycle
- **Ingestion Pipelines:** Threaded `CameraWorker` managing DirectShow webcam auto-detection and secure authenticated RTSP video streams.
- **Resilience & Watchdogs:** Automatic stream reconnection, frame drop recovery, and MJPEG preview multiplexing.
- **Source Resolution:** `CameraSourceResolver` orchestrating dynamic hardware device routing and credential retrieval.

### C. Storage & Gallery Management
- **Vector Store:** `VectorStore` in-memory and disk persistence for enrolled gait and appearance embeddings.
- **Embedding Database:** `EmbeddingDatabase` SQLite backend for historical gallery records and operational embeddings.
- **Firebase Store Adapter:** `FirebaseEmbeddingStore` stub adapter prepared for upcoming cloud persistence integration.

### D. Security & Encryption
- **Credential Storage:** Fernet symmetric key encryption for RTSP camera passwords and sensitive runtime secrets.
- **Access Control:** User authorization filters and role-based permissions in API routes and frontend routers.

### E. API & Network Services
- **Backend Framework:** FastAPI with modular API v1 routers (`/api/v1/cameras`, `/api/v1/enroll`, `/api/v1/events`, `/api/v1/system`).
- **Real-Time Streaming:** WebSocket telemetry channels (`/api/v1/ws/recognition`) broadcasting live recognition events.

### F. Frontend Portal
- **Framework:** React 19 + Vite 7 with Tailwind / Vanilla CSS design system.
- **State Management:** Decoupled context architectures (`AuthContext`, `GaitContext`) with dedicated hooks (`useAuth`, `useGait`) to eliminate Fast Refresh warnings.
- **Production Bundling:** Rollup `manualChunks` vendor splitting (`vendor-firebase`, `vendor-leaflet`, `vendor-react`, `vendor-router`, `vendor-icons`).

---

## 3. Git Baseline & Commit History

```text
ba5b166 (HEAD -> main, origin/main) docs: update thesis audit reports, quality documentation, and forensic audits
066641d feat(core): harden API schemas, CLI commands, and recognition worker lifecycle
1fcc607 feat(models): add model registry, video quality gate, and image enhancement
397d3a7 feat(storage): implement SQLite embedding database and Firebase vector store adapter
8f1ddd0 feat(intelligence): add continuous learning engine, OSNet ReID fusion, and drift detection
69ded99 feat(evaluation): update calibration rigor benchmarks and track-level breakdown scripts
f5fec5d refactor(frontend): decouple context hooks for fast refresh and optimize rollup chunking
4428171 fix: harden runtime inference and lifecycle handling
```

---

## 4. Artifact Lifecycle Accounting

### Generated Artifacts Retained (Version Controlled)
1. `evaluation/results/cmc_curves.png` (Benchmark Cumulative Matching Characteristics)
2. `evaluation/results/confusion_matrices.png` (Benchmark Multi-Class Confusion Matrix)
3. `evaluation/results/fusion_weight_sweep.png` (Benchmark Weight Optimization Curve)
4. `evaluation/results/roc_curves.png` (Benchmark Receiver Operating Characteristic)
5. `evaluation/results/ARGUS_metrics_report.md` (Validated Metrics Documentation)
6. `evaluation/results/comprehensive_metrics.json` (Numerical Benchmark Results)

### Runtime Test Artifacts Reset
- `models/appearance_gallery/gallery_features.npy` (Reset to pristine template)
- `models/appearance_gallery/gallery_labels.npy` (Reset to pristine template)
- `models/appearance_gallery/gallery_metadata.json` (Reset to pristine template)

---

## 5. Upgrade Readiness Verdict

- **Firebase Backend Migration:** `NOT STARTED (Ready to begin)`
- **Production Performance Optimization:** `NOT STARTED (Ready to begin)`
- **Repository Baseline Status:** `IMMUTABLE & CLEAN`
