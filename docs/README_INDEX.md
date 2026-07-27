# ARGUS AI Documentation Index

Welcome to the central documentation index for the ARGUS AI CCTV Gait Recognition & Biometric Surveillance Engine.

This repository maintains modular, package-level documentation across all core subsystems, along with automated README synchronization.

---

## 1. System Overview & Getting Started

- **[Root Repository README](file:///e:/ARGUS_AI/README.md)**: Main architecture, quickstart guide, core features, and system requirements.
- **[Production Deployment Guide](file:///e:/ARGUS_AI/deployment/README.md)**: Windows background service setup with NSSM, log management, and service control.

---

## 2. Package Folder Documentation

Below is the complete list of package-level documentation files for all major components in the ARGUS AI codebase:

| Package Folder | Description | Link |
|---|---|---|
| **`api/`** | FastAPI REST API endpoints, response/request schemas, and routes | [api/README.md](file:///e:/ARGUS_AI/api/README.md) |
| **`configs/`** | Declarative YAML & JSON configuration manifests | [configs/README.md](file:///e:/ARGUS_AI/configs/README.md) |
| **`core/`** | System boot lifecycle, global context, orchestrator, and health check | [core/README.md](file:///e:/ARGUS_AI/core/README.md) |
| **`enrollment/`** | Target identity enrollment, gallery updaters, and folder watcher | [enrollment/README.md](file:///e:/ARGUS_AI/enrollment/README.md) |
| **`evaluation/`** | Scientific metrics (Rank-k, EER, ROC-AUC), protocol splits, and visualizer | [evaluation/README.md](file:///e:/ARGUS_AI/evaluation/README.md) |
| **`events/`** | In-memory event bus, dispatcher, and event data contracts | [events/README.md](file:///e:/ARGUS_AI/events/README.md) |
| **`intelligence/`** | Open-set recognition, dual-modal fusion, track reliability, crowd intelligence, and watchlist | [intelligence/README.md](file:///e:/ARGUS_AI/intelligence/README.md) |
| **`models/`** | ByGaitLight CNN model architecture, checkpoints, and feature galleries | [models/README.md](file:///e:/ARGUS_AI/models/README.md) |
| **`monitoring/`** | Camera health monitor, system watchdog, multi-channel rotating logger, and GPU tuner | [monitoring/README.md](file:///e:/ARGUS_AI/monitoring/README.md) |
| **`pipeline/`** | Live camera, video file, and multi-camera CCTV execution pipelines | [pipeline/README.md](file:///e:/ARGUS_AI/pipeline/README.md) |
| **`preprocessing/`** | Silhouette extraction (Otsu + morphology), GEI synthesis, dataset building, and augmentation | [preprocessing/README.md](file:///e:/ARGUS_AI/preprocessing/README.md) |
| **`security_layer/`** | Security decision engine (ALLOW/SECURITY_ALERT/REVIEW_REQUIRED), audit logger, and credentials | [security_layer/README.md](file:///e:/ARGUS_AI/security_layer/README.md) |
| **`services/`** | Argus OS background service, RTSP/USB camera acquisition workers, and ONVIF discovery | [services/README.md](file:///e:/ARGUS_AI/services/README.md) |
| **`storage/`** | Evidence snapshot persistence, retention policy enforcement, lineage tracking, and vector store | [storage/README.md](file:///e:/ARGUS_AI/storage/README.md) |
| **`streaming/`** | Stream acquisition engine, thread-safe ring buffers (`BufferQueue`), load balancer, and worker pool | [streaming/README.md](file:///e:/ARGUS_AI/streaming/README.md) |
| **`tests/`** | Unit and integration test suite documentation | [tests/README.md](file:///e:/ARGUS_AI/tests/README.md) |
| **`training/`** | PyTorch model trainer, loss functions (Triplet + Cross-Entropy), data loaders, and callbacks | [training/README.md](file:///e:/ARGUS_AI/training/README.md) |
| **`utils/`** | Detection reporter, OpenCV HUD renderer, EMA box stabilizer, prediction smoother, and alert manager | [utils/README.md](file:///e:/ARGUS_AI/utils/README.md) |

---

## 3. Automated Documentation Synchronization

Folder documentation alignment is automatically enforced and maintained:

1. **Local Pre-Commit Hook**: Automatically runs `python scripts/sync_folder_readmes.py` before every commit, updating and staging README files.
2. **CI Freshness Check**: The GitHub Actions workflow `.github/workflows/readme_sync_check.yml` verifies README freshness using `python scripts/sync_folder_readmes.py --check`.
3. **Manual Verification**: Developers can run `python scripts/sync_folder_readmes.py --check` or `python scripts/sync_folder_readmes.py` at any time.
