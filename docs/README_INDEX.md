# ARGUS AI Documentation Index

Welcome to the central documentation index for the ARGUS AI CCTV Gait Recognition & Biometric Surveillance Engine.

This repository maintains modular, package-level documentation across all core subsystems, along with automated README synchronization.

---

## 1. System Overview & Audit Reports

- **[Root Repository README](../README.md)**: Main architecture, quickstart guide, core features, and system requirements.
- **[Audit Reports Directory Index](reports/README.md)**: Complete evidence-based performance, evaluation, benchmark, and security audit report suite.
- **[Production Deployment Guide](../ops/deployment/README.md)**: Windows background service setup with NSSM, log management, and service control.

---

## 2. Package Folder Documentation

Below is the complete list of package-level documentation files for all major components in the ARGUS AI codebase:

**`app/`** — the running application (API, core, business logic, security):

| Package Folder | Description | Link |
|---|---|---|
| **`app/api/`** | FastAPI REST API endpoints, response/request schemas, and routes | [app/api/README.md](../app/api/README.md) |
| **`app/core/`** | System boot lifecycle, global context, orchestrator, and health check | [app/core/README.md](../app/core/README.md) |
| **`app/enrollment/`** | Target identity enrollment, gallery updaters, and folder watcher | [app/enrollment/README.md](../app/enrollment/README.md) |
| **`app/events/`** | In-memory event bus, dispatcher, and event data contracts | [app/events/README.md](../app/events/README.md) |
| **`app/intelligence/`** | Open-set recognition, dual-modal fusion, track reliability, crowd intelligence, and watchlist | [app/intelligence/README.md](../app/intelligence/README.md) |
| **`app/monitoring/`** | Camera health monitor, system watchdog, multi-channel rotating logger, and GPU tuner | [app/monitoring/README.md](../app/monitoring/README.md) |
| **`app/pipeline/`** | Live camera, video file, and multi-camera CCTV execution pipelines | [app/pipeline/README.md](../app/pipeline/README.md) |
| **`app/security_layer/`** | Security decision engine (ALLOW/SECURITY_ALERT/REVIEW_REQUIRED), audit logger, and credentials | [app/security_layer/README.md](../app/security_layer/README.md) |
| **`app/services/`** | Argus OS background service, RTSP/USB camera acquisition workers, and ONVIF discovery | [app/services/README.md](../app/services/README.md) |
| **`app/storage/`** | Evidence snapshot persistence, retention policy enforcement, lineage tracking, and vector store | [app/storage/README.md](../app/storage/README.md) |
| **`app/streaming/`** | Stream acquisition engine, thread-safe ring buffers (`BufferQueue`), load balancer, and worker pool | [app/streaming/README.md](../app/streaming/README.md) |
| **`app/utils/`** | Detection reporter, OpenCV HUD renderer, EMA box stabilizer, prediction smoother, and alert manager | [app/utils/README.md](../app/utils/README.md) |

**`ml_platform/`** — the research/ML platform (models, training, evaluation):

| Package Folder | Description | Link |
|---|---|---|
| **`ml_platform/evaluation/`** | Scientific metrics (Rank-k, EER, ROC-AUC), protocol splits, and visualizer | [ml_platform/evaluation/README.md](../ml_platform/evaluation/README.md) |
| **`ml_platform/models/`** | ByGaitLight CNN model architecture, checkpoints, and feature galleries | [ml_platform/models/README.md](../ml_platform/models/README.md) |
| **`ml_platform/preprocessing/`** | Silhouette extraction (Otsu + morphology), GEI synthesis, dataset building, and augmentation | [ml_platform/preprocessing/README.md](../ml_platform/preprocessing/README.md) |
| **`ml_platform/training/`** | PyTorch model trainer, loss functions (Triplet + Cross-Entropy), data loaders, and callbacks | [ml_platform/training/README.md](../ml_platform/training/README.md) |

**`ops/`** — deployment, hardware automation, and operator tooling:

| Package Folder | Description | Link |
|---|---|---|
| **`ops/deployment/`** | Service shutdown management, runtime manifests, and Windows service install | [ops/deployment/README.md](../ops/deployment/README.md) |
| **`ops/automation/`** | Hardware detection, arbitration (DeviceManager), and 12-stage bootstrap | [ops/automation/README.md](../ops/automation/README.md) |
| **`ops/tools/`** | Operational CLI tools, benchmarks, maintenance scripts, and migrations | [ops/tools/README.md](../ops/tools/README.md) |

**Shared, ungrouped:**

| Package Folder | Description | Link |
|---|---|---|
| **`configs/`** | Declarative YAML & JSON configuration manifests | [configs/README.md](../configs/README.md) |
| **`tests/`** | Unit and integration test suite documentation | [tests/README.md](../tests/README.md) |

---

## 3. Automated Documentation Synchronization

Folder documentation alignment is automatically enforced and maintained:

1. **Local Pre-Commit Hook**: Automatically runs `python ops/tools/maintenance/sync_folder_readmes.py` before every commit, updating and staging README files.
2. **CI Freshness Check**: The GitHub Actions workflow `.github/workflows/readme_sync_check.yml` verifies README freshness using `python ops/tools/maintenance/sync_folder_readmes.py --check`.
3. **Manual Verification**: Developers can run `python ops/tools/maintenance/sync_folder_readmes.py --check` or `python ops/tools/maintenance/sync_folder_readmes.py` at any time.
