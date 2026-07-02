# ARGUS Structure Migration Recommendation

**Date:** 2026-06-12  
**Author:** ARGUS Migration Analyser  
**Status:** ANALYSIS ONLY — NO FILES MODIFIED  
**Project Root:** `E:\ARGUS_AI`

---

## Table of Contents

1. [Current Working Structure Summary](#1-current-working-structure-summary)
2. [Proposed Target Structure Summary](#2-proposed-target-structure-summary)
3. [Exact Module Mapping](#3-exact-module-mapping)
4. [Runtime-Critical Files](#4-runtime-critical-files-that-must-not-be-moved-before-submission)
5. [Files Safe to Rename or Relocate](#5-files-safe-to-rename-or-relocate)
6. [Files That Overlap in Responsibility](#6-files-that-overlap-in-responsibility)
7. [Import Risks If Files Are Moved](#7-import-risks-if-files-are-moved)
8. [CLI Risks](#8-cli-risks)
9. [API Risks](#9-api-risks)
10. [Training / Evaluation Risks](#10-trainingevaluation-risks)
11. [Gallery Path Risks](#11-gallery-path-risks)
12. [Data Path Risks](#12-data-path-risks)
13. [Migration Strategy Options](#13-recommended-migration-strategy)
14. [Risk Scores](#14-risk-score-for-each-option)
15. [Recommended Option for Final Year Submission](#15-recommended-option-for-final-year-submission)
16. [Step-by-Step Migration Plan](#16-step-by-step-migration-plan-if-migration-is-recommended)
17. [Final Recommendation](#17-final-recommendation)

---

## 1. Current Working Structure Summary

The current ARGUS project is a working, verified gait recognition system with 19 top-level Python packages/modules and 4 additional non-code directories. The structure evolved organically during development.

```
ARGUS_AI/                        # CURRENT STRUCTURE (WORKING + VERIFIED)
│
├── main.py                      # System entry point (imports core.system)
├── cli.py                       # Full CLI (16 modes, subprocess-based)
├── create_project.py            # One-time scaffolding script (not runtime)
├── __init__.py                  # Root package marker
├── VERSION                      # Version file (read by API)
├── Makefile                     # Build helper
├── requirements.txt             # Dependencies
├── requirements-dev.txt         # Dev dependencies
├── README.md                    # Documentation
├── .env.example                 # Environment template
│
├── configs/                     # YAML configuration files
│   ├── base.yaml
│   ├── inference.yaml
│   ├── train.yaml
│   ├── logging.yaml
│   ├── mode_config.yaml
│   ├── auto_train.yaml
│   └── gpu_profiles.yaml
│
├── core/                        # Boot, config, logging, orchestration
│   ├── boot.py
│   ├── config.py
│   ├── context.py
│   ├── exceptions.py
│   ├── health_check.py
│   ├── logger.py
│   ├── orchestrator.py
│   ├── system.py
│   └── system_monitor.py
│
├── api/                         # FastAPI server + routes
│   ├── server.py
│   ├── schemas.py
│   └── routes/
│       ├── enrollment.py
│       ├── inference.py
│       └── status.py
│
├── pipeline/                    # All pipeline logic (inference, live, video, folder)
│   ├── base_pipeline.py
│   ├── cache_engine.py
│   ├── folder_recognition.py
│   ├── inference_pipeline.py
│   ├── live_recognition.py
│   ├── pipeline_factory.py
│   ├── speed_controller.py
│   ├── video_recognition.py
│   └── steps/
│       ├── detection.py
│       ├── feature_extraction.py
│       ├── live_gei.py
│       ├── matching_step.py
│       ├── silhouette_step.py
│       └── tracking.py
│
├── models/                      # Model architectures + weights + gallery
│   ├── architectures/
│   │   ├── bygait_light.py
│   │   ├── gait_encoder.py      # [PLACEHOLDER]
│   │   └── losses.py
│   ├── weights/
│   │   └── yolov8n.pt
│   ├── gallery/
│   │   ├── gallery_features.npy
│   │   ├── gallery_labels.npy
│   │   └── gallery_metadata.json
│   ├── active/                  # [EMPTY]
│   ├── candidates/              # [EMPTY]
│   └── rollback/                # [EMPTY]
│
├── preprocessing/               # CASIA-B preprocessing pipeline
│   ├── augmentation.py
│   ├── casia_extractor.py
│   ├── dataset_builder.py
│   ├── gei_builder.py
│   ├── silhouette_extractor.py
│   └── skeleton_extractor.py   # [PLACEHOLDER]
│
├── training/                    # Training engine
│   ├── callbacks.py
│   ├── checkpointer.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── loss_functions.py
│   ├── optimizer.py
│   └── trainer.py
│
├── evaluation/                  # Model evaluation
│   ├── evaluator.py
│   ├── metrics.py
│   └── visualizer.py
│
├── enrollment/                  # Person enrollment workflow
│   ├── enrollment_manager.py
│   ├── enrollment_queue.py
│   ├── enrollment_validator.py
│   ├── folder_watcher.py
│   └── gallery_updater.py
│
├── storage/                     # Data persistence layer
│   ├── cache_manager.py
│   ├── data_manager.py
│   ├── dataset_loader.py
│   ├── lineage_tracker.py
│   └── vector_store.py
│
├── streaming/                   # Camera/stream input
│   ├── buffer_queue.py
│   ├── frame_dropper.py
│   └── stream_engine.py
│
├── security_layer/              # Security evaluation + logging
│   ├── security_engine.py
│   └── security_logger.py
│
├── events/                      # Event bus system
│   ├── dispatcher.py
│   ├── event_bus.py
│   └── event_types.py
│
├── intelligence/                # Confidence scoring + decision logic
│   ├── alert_manager.py         # [PLACEHOLDER]
│   ├── confidence_scorer.py
│   ├── decision_engine.py       # [PLACEHOLDER]
│   └── policy_engine.py         # [PLACEHOLDER]
│
├── utils/                       # Shared utilities
│   ├── alert_manager.py
│   ├── event_logger.py
│   ├── helpers.py
│   ├── io_utils.py
│   ├── math_utils.py
│   ├── prediction_smoother.py
│   ├── queue_utils.py
│   ├── video.py
│   └── zip_streamer.py
│
├── automation/                  # [ALL PLACEHOLDER FILES]
│   ├── auto_trainer.py
│   ├── lifecycle_controller.py
│   ├── model_promoter.py
│   ├── model_validator.py
│   ├── rollback_manager.py
│   └── training_queue.py
│
├── monitoring/                  # [ALL PLACEHOLDER FILES]
│   ├── crash_guard.py
│   ├── gpu_tuner.py
│   ├── metrics_collector.py
│   └── performance_profiler.py
│
├── registry/                    # [ALL PLACEHOLDER FILES]
│   ├── experiment_tracker.py
│   └── model_registry.py
│
├── scripts/                     # Entry-point scripts (26 files)
│   ├── preprocess_casia.py
│   ├── train_model.py
│   ├── build_gallery.py
│   ├── evaluate_model.py
│   ├── benchmark.py
│   ├── system_check.py
│   ├── run_folder_recognition.py
│   ├── run_video_recognition.py
│   ├── test_live_recognition.py
│   ├── test_*.py                # (14 test/demo scripts)
│   └── start_system.bat/.sh
│
├── tests/                       # Unit + integration tests
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_evaluation.py
│   │   ├── test_model_arch.py
│   │   ├── test_pipeline.py
│   │   ├── test_preprocessing.py
│   │   ├── test_training.py
│   │   └── test_vector_store.py
│   └── integration/
│       ├── test_end_to_end.py
│       └── test_enrollment_flow.py
│
├── data/                        # Dataset & input data
│   ├── casia_b_raw.zip          # 728 MB source ZIP
│   ├── casia_processed/
│   │   ├── gei/                 # Generated GEI images
│   │   ├── silhouettes/
│   │   └── skeletons/
│   ├── new_input/               # Enrollment input folders
│   ├── casia_cache/
│   ├── embeddings/
│   ├── faiss_store/
│   └── incremental_cache/
│
├── runs/                        # Training output
│   ├── exp_001/
│   └── tensorboard/
│
├── outputs/                     # Runtime output (logs, reports, charts)
│   ├── debug/
│   ├── eval_reports/
│   ├── evaluation_charts/
│   ├── events/
│   ├── reports/
│   ├── security_logs/
│   └── videos/
│
└── docs/                        # Project documentation (25 files)
```

**Key facts about current structure:**
- **19 Python packages** at top level
- **~90 Python source files** (excluding `venv/`)
- **3 fully placeholder packages:** `automation/`, `monitoring/`, `registry/`
- **Gallery data lives inside `models/gallery/`** (not `data/`)
- **No `datasets/` package** exists
- **No `matching/` package** exists — matching lives in `pipeline/steps/matching_step.py`
- **No root-level `config.py`** — configuration is in `core/config.py`
- **No `run.py`** at root level
- **`api/` package** exists (not in proposed structure)
- **`events/`, `intelligence/`, `automation/`, `monitoring/`, `registry/`** are current packages not in proposed structure

---

## 2. Proposed Target Structure Summary

The proposed target structure consolidates the system into **14 functional packages** plus root-level files, removing several current packages and adding new ones:

**New packages in proposed structure (not in current):**
- `datasets/` — casia_loader.py, casia_subset.py, casia_preprocessor.py, cache_manager.py
- `matching/` — embedding_matcher.py, threshold.py

**Root-level files added:**
- `run.py` — full system runner
- `config.py` — global config (currently `core/config.py`)

**Current packages REMOVED from proposed:**
- `api/` — entire FastAPI server
- `events/` — event bus system
- `intelligence/` — confidence scorer, etc.
- `automation/` — auto-training placeholders
- `monitoring/` — performance monitoring placeholders
- `registry/` — model/experiment registry placeholders
- `utils/` — utility modules

**Significant renames in proposed:**
- `evaluation/visualizer.py` → `evaluation/charts.py`
- `evaluation/evaluator.py` → `evaluation/benchmark.py`
- `training/trainer.py` stays, but `training/train.py` added
- `security_layer/security_engine.py` → `security_layer/security_manager.py`
- `security_layer/security_logger.py` → `security_layer/event_logger.py`
- `pipeline/` gains `detector.py`, `tracker.py`, `gei_pipeline.py`, `gait_pipeline.py`
- `streaming/` restructured: `stream_reader.py`, `frame_buffer.py`, `fps_controller.py`, `stream_manager.py`

---

## 3. Exact Module Mapping

### 3.1 Root-Level Files

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `main.py` | `main.py` | **KEEP** | None |
| `cli.py` | `cli.py` | **KEEP** | None |
| *(does not exist)* | `run.py` | **NEW** | Low — would be new file |
| *(does not exist)* | `config.py` | **NEW** | ⚠️ MEDIUM — conflicts with `core/config.py` |
| `create_project.py` | *(not in proposed)* | **REMOVE** | None (not runtime) |
| `__init__.py` | *(not listed)* | **KEEP** | None |
| `VERSION` | *(not listed)* | **KEEP** | None |
| `requirements.txt` | `requirements.txt` | **KEEP** | None |
| `README.md` | `README.md` | **KEEP** | None |

---

### 3.2 configs/

| Current Path | Proposed Path | Action | Notes |
|---|---|---|---|
| `configs/base.yaml` | *(not listed)* | **KEEP** | Referenced by `core/config.py` |
| `configs/inference.yaml` | `configs/inference.yaml` | **KEEP** | ✅ Direct match |
| `configs/train.yaml` | `configs/training.yaml` | **RENAME** | ⚠️ Name change |
| *(does not exist)* | `configs/casia_b.yaml` | **NEW** | Low risk |
| *(does not exist)* | `configs/system.yaml` | **NEW** | Low risk |
| `configs/logging.yaml` | *(not in proposed)* | **REMOVE** | Low risk — not imported in code |
| `configs/mode_config.yaml` | *(not in proposed)* | **REMOVE** | Low risk |
| `configs/auto_train.yaml` | *(not in proposed)* | **REMOVE** | Low risk |
| `configs/gpu_profiles.yaml` | *(not in proposed)* | **REMOVE** | Low risk |

---

### 3.3 core/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `core/system.py` | `core/system.py` | **KEEP** | None |
| `core/logger.py` | `core/logger.py` | **KEEP** | None |
| `core/config.py` | `core/config_loader.py` | **RENAME** | 🔴 HIGH — imported by `core/boot.py` and indirectly by everything |
| *(does not exist)* | `core/utils.py` | **NEW** | Would need to absorb `utils/` content |
| `core/boot.py` | *(not in proposed)* | **REMOVE/MERGE** | 🔴 HIGH — `main.py` → `core.system` → `core.boot` |
| `core/orchestrator.py` | *(not in proposed)* | **REMOVE/MERGE** | 🔴 HIGH — runtime orchestrator |
| `core/context.py` | *(not in proposed)* | **REMOVE/MERGE** | MEDIUM — used by orchestrator |
| `core/exceptions.py` | *(not in proposed)* | **REMOVE/MERGE** | Low |
| `core/health_check.py` | *(not in proposed)* | **REMOVE/MERGE** | MEDIUM — used by boot |
| `core/system_monitor.py` | *(not in proposed)* | **REMOVE/MERGE** | MEDIUM — used by boot + orchestrator |

---

### 3.4 datasets/ (NEW — does not exist currently)

| Current Path | Proposed Path | Action | Notes |
|---|---|---|---|
| *(does not exist)* | `datasets/casia_loader.py` | **NEW** | |
| *(does not exist)* | `datasets/casia_subset.py` | **NEW** | |
| `preprocessing/casia_extractor.py` | `datasets/casia_preprocessor.py` | **MOVE+RENAME** | MEDIUM |
| `storage/cache_manager.py` | `datasets/cache_manager.py` | **MOVE** | Low — not heavily imported |

---

### 3.5 preprocessing/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `preprocessing/silhouette_extractor.py` | `preprocessing/silhouette_extractor.py` | **KEEP** | None |
| `preprocessing/gei_builder.py` | `preprocessing/gei_builder.py` | **KEEP** | None |
| `preprocessing/augmentation.py` | `preprocessing/augmentation.py` | **KEEP** | None |
| *(does not exist)* | `preprocessing/frame_sampler.py` | **NEW** | |
| *(does not exist)* | `preprocessing/normalization.py` | **NEW** | |
| `preprocessing/dataset_builder.py` | *(not in proposed)* | **MOVE** | 🔴 HIGH — used by `scripts/preprocess_casia.py` |
| `preprocessing/casia_extractor.py` | *(→ datasets/)* | **MOVE** | See datasets section |
| `preprocessing/skeleton_extractor.py` | *(not in proposed)* | **REMOVE** | None — placeholder |

---

### 3.6 models/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `models/architectures/bygait_light.py` | `models/architectures/bygait_light.py` | **KEEP** | None |
| `models/architectures/losses.py` | `models/losses.py` | **MOVE** | MEDIUM — imported by `training/loss_functions.py` |
| `models/weights/yolov8n.pt` | `models/weights/best_model.pth` | ⚠️ **CONFLICT** | Proposed lists `best_model.pth`; current has `yolov8n.pt` here. Best model is in `runs/exp_001/` |
| `models/gallery/` | *(→ data/gallery/)* | **MOVE** | 🔴 HIGH — hardcoded in `VectorStore` |
| `models/architectures/gait_encoder.py` | *(not in proposed)* | **REMOVE** | None — placeholder |
| `models/active/` | *(not in proposed)* | **REMOVE** | None — empty |
| `models/candidates/` | *(not in proposed)* | **REMOVE** | None — empty |
| `models/rollback/` | *(not in proposed)* | **REMOVE** | None — empty |

---

### 3.7 matching/ (NEW — does not exist currently)

| Current Path | Proposed Path | Action | Notes |
|---|---|---|---|
| `pipeline/steps/matching_step.py` | `matching/embedding_matcher.py` | **MOVE+RENAME** | 🔴 HIGH — imported by 4+ files |
| *(does not exist)* | `matching/threshold.py` | **NEW** | |

---

### 3.8 enrollment/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `enrollment/enrollment_manager.py` | `enrollment/person_enroller.py` | **RENAME** | 🔴 HIGH — imported by API route |
| `enrollment/gallery_updater.py` | `enrollment/gallery_builder.py` | **RENAME** | MEDIUM — imported by enrollment_manager |
| `enrollment/enrollment_validator.py` | *(not in proposed)* | **MERGE** | MEDIUM — imported by enrollment_manager |
| `enrollment/enrollment_queue.py` | *(not in proposed)* | **MERGE** | Low |
| `enrollment/folder_watcher.py` | *(not in proposed)* | **MERGE** | Low |
| *(does not exist)* | `enrollment/dataset_builder.py` | **NEW or MOVE** | Could absorb `preprocessing/dataset_builder.py` |

---

### 3.9 pipeline/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `pipeline/inference_pipeline.py` | `pipeline/inference_pipeline.py` | **KEEP** | None |
| `pipeline/folder_recognition.py` | `pipeline/folder_recognition.py` | **KEEP** | None |
| `pipeline/video_recognition.py` | `pipeline/video_recognition.py` | **KEEP** | None |
| `pipeline/live_recognition.py` | *(not in proposed)* | **MERGE** | 🔴 HIGH — this IS the live system |
| `pipeline/steps/detection.py` | `pipeline/detector.py` | **MOVE+RENAME** | MEDIUM |
| `pipeline/steps/tracking.py` | `pipeline/tracker.py` | **MOVE+RENAME** | 🔴 HIGH — imported by live + video |
| `pipeline/steps/feature_extraction.py` | *(not in proposed)* | **MERGE** | 🔴 HIGH — used by inference + enrollment |
| `pipeline/steps/live_gei.py` | *(→ pipeline/gei_pipeline.py?)* | **MOVE** | MEDIUM |
| `pipeline/steps/silhouette_step.py` | *(not in proposed)* | **MERGE** | HIGH — used by live + video |
| `pipeline/steps/matching_step.py` | *(→ matching/)* | **MOVE** | HIGH |
| `pipeline/base_pipeline.py` | *(not in proposed)* | **MERGE** | Low |
| `pipeline/cache_engine.py` | *(not in proposed)* | **MERGE** | Low |
| `pipeline/pipeline_factory.py` | *(not in proposed)* | **MERGE** | Low |
| `pipeline/speed_controller.py` | *(not in proposed)* | **MERGE** | Low |
| *(does not exist)* | `pipeline/gait_pipeline.py` | **NEW** | |
| *(does not exist)* | `pipeline/gei_pipeline.py` | **NEW** | |
| *(does not exist)* | `pipeline/training_pipeline.py` | **NEW** | |

---

### 3.10 streaming/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `streaming/stream_engine.py` | `streaming/stream_reader.py` | **RENAME** | 🔴 HIGH — imported by `live_recognition.py` |
| `streaming/buffer_queue.py` | `streaming/frame_buffer.py` | **RENAME** | Low |
| `streaming/frame_dropper.py` | `streaming/fps_controller.py` | **RENAME** | Low |
| *(does not exist)* | `streaming/stream_manager.py` | **NEW** | |

---

### 3.11 security_layer/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `security_layer/security_engine.py` | `security_layer/security_manager.py` | **RENAME** | 🔴 HIGH — imported by live + video |
| `security_layer/security_logger.py` | `security_layer/event_logger.py` | **RENAME** | MEDIUM — imported by security_engine |
| *(does not exist)* | `security_layer/confidence_scorer.py` | **NEW or MOVE** from `intelligence/` |
| *(does not exist)* | `security_layer/alert_rules.py` | **NEW** | |

---

### 3.12 storage/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `storage/vector_store.py` | `storage/gallery_store.py` | **RENAME** | 🔴 HIGH — imported by 6+ modules |
| `storage/cache_manager.py` | `storage/cache_store.py` | **RENAME** | Low |
| *(does not exist)* | `storage/embedding_store.py` | **NEW** | |
| `storage/data_manager.py` | *(not in proposed)* | **REMOVE/MERGE** | Low |
| `storage/dataset_loader.py` | *(not in proposed)* | **REMOVE/MERGE** | Low |
| `storage/lineage_tracker.py` | *(not in proposed)* | **REMOVE/MERGE** | Low |

---

### 3.13 training/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `training/trainer.py` | `training/trainer.py` | **KEEP** | None |
| `training/dataset.py` | `training/dataset.py` | **KEEP** | None |
| `training/loss_functions.py` | `training/losses.py` | **RENAME** | MEDIUM |
| *(does not exist)* | `training/train.py` | **NEW** | |
| `training/callbacks.py` | *(not in proposed)* | **MERGE** | Low |
| `training/checkpointer.py` | *(not in proposed)* | **MERGE** | MEDIUM — used by trainer |
| `training/dataloader.py` | *(not in proposed)* | **MERGE** | MEDIUM — used by trainer |
| `training/optimizer.py` | *(not in proposed)* | **MERGE** | MEDIUM — used by trainer |

---

### 3.14 evaluation/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `evaluation/evaluator.py` | `evaluation/benchmark.py` | **RENAME** | MEDIUM — imported by `scripts/evaluate_model.py` |
| `evaluation/metrics.py` | `evaluation/metrics.py` | **KEEP** | None |
| `evaluation/visualizer.py` | `evaluation/charts.py` | **RENAME** | Low — imported by scripts only |

---

### 3.15 scripts/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `scripts/preprocess_casia.py` | `scripts/preprocess_casia.py` | **KEEP** | None |
| `scripts/train_model.py` | `scripts/train.py` | **RENAME** | MEDIUM — referenced in `cli.py` |
| `scripts/build_gallery.py` | `scripts/build_gallery.py` | **KEEP** | None |
| `scripts/evaluate_model.py` | *(not in proposed)* | **MERGE/REMOVE** | MEDIUM — referenced in `cli.py` |
| `scripts/benchmark.py` | *(not in proposed)* | **MERGE** | MEDIUM — referenced in `cli.py` |
| `scripts/run_folder_recognition.py` | `scripts/run_folder_recognition.py` | **KEEP** | None |
| `scripts/run_video_recognition.py` | `scripts/run_video_recognition.py` | **KEEP** | None |
| `scripts/system_check.py` | *(not in proposed)* | **MERGE** | MEDIUM — referenced in `cli.py` |
| `scripts/test_*.py` (14 files) | *(not in proposed)* | **MERGE/MOVE to tests/** | MEDIUM — referenced in `cli.py` |
| *(does not exist)* | `scripts/run_system.py` | **NEW** | |

---

### 3.16 tests/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `tests/unit/test_preprocessing.py` | `tests/test_preprocessing.py` | **FLATTEN** | MEDIUM |
| `tests/unit/test_model_arch.py` | `tests/test_model.py` | **FLATTEN+RENAME** | MEDIUM |
| `tests/unit/test_pipeline.py` | *(→ test_matching.py?)* | **FLATTEN+RENAME** | MEDIUM |
| `tests/integration/test_end_to_end.py` | `tests/test_integration.py` | **FLATTEN+RENAME** | MEDIUM |
| `tests/unit/test_evaluation.py` | *(not directly mapped)* | **MERGE** | Low |
| `tests/unit/test_training.py` | *(not directly mapped)* | **MERGE** | Low |
| `tests/unit/test_vector_store.py` | *(not directly mapped)* | **MERGE** | Low |
| `tests/integration/test_enrollment_flow.py` | *(not directly mapped)* | **MERGE** | Low |
| *(does not exist)* | `tests/test_security.py` | **NEW** | |

---

### 3.17 Packages That Exist in Current but NOT in Proposed (Full Removal)

| Current Package | Action | Contents | Risk |
|---|---|---|---|
| `api/` | **NOT IN PROPOSED** | Working FastAPI server, 3 route files, schemas | 🔴 **CRITICAL** — API is a verified working feature |
| `events/` | **NOT IN PROPOSED** | EventBus, dispatcher, event types | MEDIUM — used at runtime |
| `intelligence/` | **NOT IN PROPOSED** | ConfidenceScorer (used by scripts) | MEDIUM |
| `automation/` | **NOT IN PROPOSED** | All placeholders | None |
| `monitoring/` | **NOT IN PROPOSED** | All placeholders | None |
| `registry/` | **NOT IN PROPOSED** | All placeholders | None |
| `utils/` | **NOT IN PROPOSED** | AlertManager, EventLogger, PredictionSmoother, helpers, etc. | 🔴 **CRITICAL** — heavily used by live + video |

---

### 3.18 data/ and docs/

| Current Path | Proposed Path | Action | Risk |
|---|---|---|---|
| `data/casia_b_raw.zip` | `data/casia_b_raw.zip` | **KEEP** | None |
| `data/casia_processed/` | `data/casia_processed/` | **KEEP** | None |
| `data/casia_processed/gei/` | `data/casia_processed/gei/` | **KEEP** | None |
| `data/casia_processed/silhouettes/` | `data/casia_processed/silhouettes/` | **KEEP** | None |
| `data/new_input/` | `data/new_input/persons/` | **RENAME subdir** | MEDIUM |
| `data/casia_cache/` | *(not in proposed)* | **REMOVE** | Low |
| `data/embeddings/` | `data/casia_processed/embeddings/` | **MOVE** | Low |
| `data/faiss_store/` | *(not in proposed)* | **REMOVE** | Low |
| `data/incremental_cache/` | *(not in proposed)* | **REMOVE** | Low |
| `models/gallery/` | `data/gallery/` | **MOVE** | 🔴 HIGH — hardcoded in VectorStore |
| *(does not exist)* | `data/gallery/gallery_meta.json` | **RENAME** from `gallery_metadata.json` | MEDIUM |

---

## 4. Runtime-Critical Files That Must NOT Be Moved Before Submission

These files are on the **critical execution path** of verified features. Moving them would break imports and/or runtime behavior:

| File | Why Critical | Used By |
|---|---|---|
| `cli.py` | Entry point for ALL modes including `--mode demo` | User, `python cli.py --mode *` |
| `main.py` | System entry point | Direct execution |
| `core/system.py` | Boot chain root | `main.py` |
| `core/boot.py` | Boot sequence | `core/system.py` |
| `core/orchestrator.py` | Runtime controller | `core/system.py` |
| `core/config.py` | Configuration loader | `core/boot.py` |
| `core/logger.py` | Logging (used everywhere) | 10+ modules |
| `pipeline/live_recognition.py` | Live webcam recognition | `scripts/test_live_recognition.py`, `cli.py --mode live` |
| `pipeline/video_recognition.py` | Video file recognition | `scripts/run_video_recognition.py` |
| `pipeline/folder_recognition.py` | Folder batch recognition | `scripts/run_folder_recognition.py` |
| `pipeline/inference_pipeline.py` | Single-image inference | API `/identify`, benchmark, folder recognition |
| `pipeline/steps/matching_step.py` | Cosine matching core | 4+ pipeline files |
| `pipeline/steps/tracking.py` | YOLO+ByteTrack | live_recognition, video_recognition |
| `pipeline/steps/feature_extraction.py` | GEI→embedding | inference_pipeline, enrollment |
| `pipeline/steps/silhouette_step.py` | Silhouette extraction | live_recognition, video_recognition |
| `pipeline/steps/live_gei.py` | Live GEI buffer | live_recognition, video_recognition |
| `models/architectures/bygait_light.py` | CNN architecture | trainer, evaluator, all pipelines |
| `models/gallery/*` | Gallery data files | VectorStore → all pipelines |
| `models/weights/yolov8n.pt` | YOLO weights | detection, tracking |
| `storage/vector_store.py` | Gallery load/save | 6+ modules |
| `security_layer/security_engine.py` | Security evaluation | live_recognition, video_recognition |
| `api/server.py` | FastAPI app | `cli.py --mode api`, uvicorn |
| `api/routes/*.py` | API endpoints | `api/server.py` |
| `enrollment/enrollment_manager.py` | Enrollment flow | API route, scripts |
| `training/trainer.py` | Training engine | `scripts/train_model.py` |
| `evaluation/evaluator.py` | Evaluation engine | `scripts/evaluate_model.py` |
| `utils/alert_manager.py` | Alert system | live_recognition |
| `utils/event_logger.py` | Event logging | live_recognition, video_recognition |
| `utils/prediction_smoother.py` | Prediction stability | live_recognition, video_recognition |
| `runs/exp_001/best_model.pth` | Trained model checkpoint | All inference pipelines |
| `configs/base.yaml` | Base configuration | `core/config.py` |

**Total: 30+ runtime-critical files/paths**

---

## 5. Files Safe to Rename or Relocate

These files are either **placeholders**, **unused**, or have **zero runtime imports**:

| File | Reason Safe |
|---|---|
| `create_project.py` | One-time scaffolding, never imported |
| `automation/*` (6 files) | All placeholders — `"""ARGUS module. Implementation will be added step by step."""` |
| `monitoring/*` (4 files) | All placeholders |
| `registry/*` (2 files) | All placeholders |
| `intelligence/alert_manager.py` | Placeholder |
| `intelligence/decision_engine.py` | Placeholder |
| `intelligence/policy_engine.py` | Placeholder |
| `models/architectures/gait_encoder.py` | Placeholder |
| `preprocessing/skeleton_extractor.py` | Placeholder |
| `models/active/` | Empty directory |
| `models/candidates/` | Empty directory |
| `models/rollback/` | Empty directory |
| `data/embeddings/` | Empty (only `.gitkeep`) |
| `data/faiss_store/` | Not referenced in code |
| `data/incremental_cache/` | Not referenced in active code |
| `configs/logging.yaml` | Not imported by any code |
| `configs/mode_config.yaml` | Not imported by active code |
| `configs/auto_train.yaml` | Not imported by active code |
| `configs/gpu_profiles.yaml` | Not imported by active code |
| `docs/*` | Documentation — safe to reorganize |

---

## 6. Files That Overlap in Responsibility

| Overlap Area | File A (Current) | File B (Current) | Notes |
|---|---|---|---|
| **Losses** | `models/architectures/losses.py` (TripletLoss) | `training/loss_functions.py` (wraps TripletLoss + CrossEntropy) | Functional dependency, not duplication |
| **Alert/Event Logging** | `utils/alert_manager.py` (CSV alerts) | `utils/event_logger.py` (CSV recognition log) | Complementary but similar pattern |
| **Security Logging** | `security_layer/security_logger.py` (CSV security events) | `utils/event_logger.py` | Different severity scope, same pattern |
| **Confidence Scoring** | `intelligence/confidence_scorer.py` (level/trusted) | `security_layer/security_engine.py` (severity/decision) | Both evaluate scores — different consumers |
| **Cache Management** | `storage/cache_manager.py` | `pipeline/cache_engine.py` | Different cache layers |
| **Dataset Handling** | `preprocessing/dataset_builder.py` (CASIA ZIP→GEI) | `training/dataset.py` (PyTorch Dataset) | Different pipeline stages |
| **Gallery Storage** | `storage/vector_store.py` | `enrollment/gallery_updater.py` | Updater wraps VectorStore |

---

## 7. Import Risks If Files Are Moved

### Cross-module import dependency graph (critical chains):

```
cli.py
  └── subprocess → scripts/*.py
      └── from pipeline.live_recognition import LiveRecognitionPipeline
      └── from pipeline.video_recognition import VideoRecognitionPipeline
      └── from pipeline.folder_recognition import FolderRecognitionPipeline
      └── from pipeline.inference_pipeline import InferencePipeline
      └── from training.trainer import Trainer
      └── from evaluation.evaluator import SplitEvaluator
      └── from preprocessing.dataset_builder import CasiaGEIDatasetBuilder

api/server.py
  └── from api.routes.enrollment → from enrollment.enrollment_manager
  └── from api.routes.inference → from pipeline.inference_pipeline
  └── from api.routes.status → from storage.vector_store

pipeline/live_recognition.py (MOST COMPLEX — 12 imports from 7 packages):
  └── models.architectures.bygait_light
  └── pipeline.steps.live_gei
  └── pipeline.steps.matching_step
  └── pipeline.steps.silhouette_step
  └── pipeline.steps.tracking
  └── security_layer.security_engine
  └── storage.vector_store
  └── streaming.stream_engine
  └── utils.alert_manager
  └── utils.event_logger
  └── utils.prediction_smoother
```

### High-risk import chains that would break:

| If Moved | Breaks | Count of Broken Imports |
|---|---|---|
| `pipeline/steps/matching_step.py` | `pipeline/inference_pipeline.py`, `pipeline/live_recognition.py`, `pipeline/video_recognition.py`, `evaluation/evaluator.py` | **4** |
| `storage/vector_store.py` | `pipeline/inference_pipeline.py`, `pipeline/live_recognition.py`, `pipeline/video_recognition.py`, `api/routes/status.py`, `enrollment/gallery_updater.py`, `scripts/build_gallery.py`, `scripts/benchmark.py` | **7** |
| `pipeline/steps/tracking.py` | `pipeline/live_recognition.py`, `pipeline/video_recognition.py` | **2** |
| `streaming/stream_engine.py` | `pipeline/live_recognition.py` | **1** |
| `security_layer/security_engine.py` | `pipeline/live_recognition.py`, `pipeline/video_recognition.py` | **2** |
| `utils/alert_manager.py` | `pipeline/live_recognition.py` | **1** |
| `utils/event_logger.py` | `pipeline/live_recognition.py`, `pipeline/video_recognition.py` | **2** |
| `utils/prediction_smoother.py` | `pipeline/live_recognition.py`, `pipeline/video_recognition.py` | **2** |
| `core/config.py` | `core/boot.py` → everything | **ALL** |
| `enrollment/enrollment_manager.py` | `api/routes/enrollment.py` | **1** |
| `models/architectures/bygait_light.py` | `pipeline/steps/feature_extraction.py`, `pipeline/live_recognition.py`, `pipeline/video_recognition.py`, `training/trainer.py`, `evaluation/evaluator.py`, `scripts/build_gallery.py` | **6** |

**Total unique import-breaking risks: 50+ import statements across 25+ files**

---

## 8. CLI Risks

`cli.py` uses `subprocess.run` to invoke scripts by **hardcoded relative path**:

| CLI Mode | Script Path (Hardcoded) | Risk If Moved |
|---|---|---|
| `--mode health` | `scripts/system_check.py` | Script references `core`, `configs`, etc. |
| `--mode preprocess` | `scripts/preprocess_casia.py` | Imports `preprocessing.dataset_builder` |
| `--mode train` | `scripts/train_model.py` | Imports `training.trainer` |
| `--mode gallery` | `scripts/build_gallery.py` | Imports `models.architectures`, `storage.vector_store` |
| `--mode evaluate` | `scripts/evaluate_model.py` | Imports `evaluation.evaluator` |
| `--mode benchmark` | `scripts/benchmark.py` | Imports `pipeline.inference_pipeline`, `storage.vector_store` |
| `--mode live` | `scripts/test_live_recognition.py` | Imports `pipeline.live_recognition` |
| `--mode api` | `uvicorn api.server:app` | Module path `api.server` |
| `--mode tests` | `unittest discover -s tests` | Test discovery path `tests/` |
| `--mode integration-tests` | `pytest tests/integration` | Directory path |
| `--mode security-test` | `scripts/test_security_layer.py` | Imports security layer |
| `--mode demo` | Calls all above sequentially | **ALL RISKS COMBINED** |

> [!CAUTION]
> `cli.py --mode demo` is the **most fragile** path. It chains 7 CLI modes sequentially. ANY import break in ANY script will cascade to a demo failure. The demo is likely used for your final year presentation.

**Risk Level: 🔴 CRITICAL**

---

## 9. API Risks

The API (`api/` package) is **NOT included in the proposed structure at all**. This is a significant omission:

- `api/server.py` provides FastAPI endpoints for `/health`, `/metrics`, `/identify`, `/enroll`
- It is invoked by `cli.py --mode api` via `uvicorn api.server:app`
- The module path `api.server` is the uvicorn entry point — it cannot be changed without updating the CLI
- API routes import from `enrollment.enrollment_manager`, `pipeline.inference_pipeline`, `storage.vector_store`

> [!WARNING]
> If the proposed structure is adopted literally (removing `api/`), the entire API feature is lost. This is a working, verified feature that should be preserved.

**Risk Level: 🔴 CRITICAL — API would be completely lost**

---

## 10. Training/Evaluation Risks

| Component | Current Import Chain | Risk |
|---|---|---|
| Training | `cli.py` → `scripts/train_model.py` → `training.trainer.Trainer` → `models.architectures.bygait_light`, `training.checkpointer`, `training.dataloader`, `training.optimizer` | MEDIUM — `checkpointer`, `dataloader`, `optimizer` not in proposed |
| Evaluation | `cli.py` → `scripts/evaluate_model.py` → `evaluation.evaluator.SplitEvaluator` → `evaluation.metrics`, `models.architectures.bygait_light`, `pipeline.steps.matching_step` | MEDIUM — `matching_step` would move |
| Benchmark | `cli.py` → `scripts/benchmark.py` → `pipeline.inference_pipeline`, `storage.vector_store` | MEDIUM — vector_store rename |
| Visualizer | `cli.py` → `scripts/test_visualizer.py` → `evaluation.visualizer` | Low — rename only |

Specific issues:
1. `training/checkpointer.py`, `training/dataloader.py`, `training/optimizer.py` are NOT in the proposed structure but are actively used by `Trainer`
2. `evaluation/evaluator.py` → `evaluation/benchmark.py` rename would break `scripts/evaluate_model.py`
3. `pipeline/steps/matching_step.py` → `matching/embedding_matcher.py` would break evaluator imports

---

## 11. Gallery Path Risks

Gallery data is currently stored at `models/gallery/` and the `VectorStore` class has this as default:

```python
class VectorStore:
    def __init__(self, gallery_dir: str = "models/gallery"):
```

The proposed structure moves gallery to `data/gallery/`. This affects:

| Consumer | Current Path Used | Would Break? |
|---|---|---|
| `storage/vector_store.py` | `models/gallery` (default) | ✅ YES |
| `scripts/build_gallery.py` | Uses `VectorStore()` default | ✅ YES |
| `pipeline/inference_pipeline.py` | Uses `VectorStore()` default | ✅ YES |
| `pipeline/live_recognition.py` | Uses `VectorStore()` default | ✅ YES |
| `pipeline/video_recognition.py` | Uses `VectorStore()` default | ✅ YES |
| `api/routes/status.py` | Uses `VectorStore()` default | ✅ YES |
| `enrollment/gallery_updater.py` | Uses `VectorStore()` default | ✅ YES |

Additionally, the proposed renames:
- `gallery_metadata.json` → `gallery_meta.json`
- This filename is hardcoded in `VectorStore`

> [!CAUTION]
> Moving gallery from `models/gallery/` to `data/gallery/` would break **every feature** that uses the gallery: inference, live recognition, video recognition, enrollment, API metrics, and benchmarks.

**Risk Level: 🔴 CRITICAL**

---

## 12. Data Path Risks

| Path Reference | Where Hardcoded | Proposed Change | Risk |
|---|---|---|---|
| `data/casia_b_raw.zip` | `preprocessing/dataset_builder.py`, `scripts/preprocess_casia.py` | No change | ✅ Safe |
| `data/casia_processed/gei` | `training/trainer.py`, `training/dataset.py`, `evaluation/evaluator.py`, `scripts/build_gallery.py` | No change | ✅ Safe |
| `data/new_input/` | `enrollment/*`, `tests/conftest.py` | → `data/new_input/persons/` | ⚠️ MEDIUM |
| `models/gallery/` | `storage/vector_store.py` (default) | → `data/gallery/` | 🔴 CRITICAL |
| `runs/exp_001/best_model.pth` | `pipeline/steps/feature_extraction.py`, `pipeline/live_recognition.py`, `pipeline/video_recognition.py`, `evaluation/evaluator.py` | Unchanged | ✅ Safe |
| `models/weights/yolov8n.pt` | `pipeline/steps/detection.py`, `pipeline/steps/tracking.py` | Proposed: `models/weights/best_model.pth` | ⚠️ Conflict |
| `outputs/*` | `core/logger.py`, `utils/alert_manager.py`, `utils/event_logger.py`, `security_layer/security_logger.py`, `scripts/evaluate_model.py`, `scripts/benchmark.py` | Not in proposed | ⚠️ Would need separate handling |
| `configs/base.yaml` | `core/config.py` | Not listed in proposed configs | ⚠️ MEDIUM |

---

## 13. Recommended Migration Strategy

### Option A: No Migration — Keep Current Structure

**Description:** Freeze the current structure. Submit as-is. Document the current architecture clearly in `docs/architecture.md`.

**Advantages:**
- Zero risk of breaking anything
- All verified features remain working
- No testing overhead
- Maximum time for polish, documentation, and presentation prep

**Disadvantages:**
- Structure has 3 empty placeholder packages
- Some module names don't match proposed "clean" layout
- Slightly more complex than proposed

---

### Option B: Soft Migration Using Wrapper/Alias Files

**Description:** Keep all current files in place. Create **alias modules** at the proposed paths that re-export from the current locations. Remove placeholder packages.

Example: Create `matching/embedding_matcher.py` that does:
```python
from pipeline.steps.matching_step import MatchingStep as EmbeddingMatcher
```

**Advantages:**
- Current code keeps working with zero import changes
- Project appears to follow cleaner structure
- Can be done incrementally
- Easy to roll back

**Disadvantages:**
- Creates duplication in project tree
- Alias files may confuse assessors if inspected closely
- Extra maintenance burden
- Doesn't actually clean the structure

---

### Option C: Full Migration

**Description:** Move, rename, and restructure all files to match the proposed target structure. Update all imports. Update `cli.py` script paths. Update all hardcoded data paths.

**Advantages:**
- Clean, academic-quality structure
- Matches proposal document

**Disadvantages:**
- **50+ import statements** need updating across **25+ files**
- **Gallery path** hardcoded in 7+ locations needs updating
- **`cli.py`** needs rewriting (hardcoded script paths)
- **API would be removed** (not in proposed structure) or needs to be re-added
- `utils/` package removal requires redistributing 10 utility files
- Full re-testing of all features required
- **HIGH probability** of introducing regressions
- Multiple days of work + testing

---

## 14. Risk Score for Each Option

| Option | Implementation Risk | Testing Overhead | Regression Risk | Time Required | Overall Risk |
|---|---|---|---|---|---|
| **A: No Migration** | 🟢 None | 🟢 None | 🟢 None | 🟢 0 hours | 🟢 **VERY LOW** |
| **B: Soft Migration** | 🟡 Low | 🟡 Low | 🟡 Low | 🟡 4-8 hours | 🟡 **LOW-MEDIUM** |
| **C: Full Migration** | 🔴 Very High | 🔴 Very High | 🔴 Very High | 🔴 16-24 hours | 🔴 **CRITICAL** |

### Risk Quantification for Option C:

| Risk Category | Affected Files | Severity |
|---|---|---|
| Import breakage | 25+ files, 50+ import statements | 🔴 Critical |
| CLI breakage | 16 modes, all subprocess paths | 🔴 Critical |
| API loss | Entire `api/` package dropped | 🔴 Critical |
| Gallery path breakage | 7+ consumers | 🔴 Critical |
| Test failures | All unit + integration tests | 🔴 Critical |
| Utils redistribution | 10 files across 3+ target packages | 🟠 High |
| Training pipeline breakage | Checkpointer/dataloader/optimizer removed | 🟠 High |
| Streaming rename breakage | Live recognition | 🟠 High |

---

## 15. Recommended Option for Final Year Submission

> [!IMPORTANT]
> **RECOMMENDED: Option A — Do NOT migrate before submission.**

### Reasoning:

1. **The current structure is already working and verified.** Every feature (`cli.py --mode demo`, live recognition, API, enrollment, gallery matching, evaluation, benchmark, tests, security layer) has been tested and confirmed working.

2. **The proposed structure has critical gaps:**
   - Missing `api/` — a verified feature would be lost
   - Missing `utils/` — AlertManager, EventLogger, PredictionSmoother are actively used by live+video pipelines
   - Missing `events/` — EventBus is used at runtime
   - Missing `training/checkpointer.py`, `training/dataloader.py`, `training/optimizer.py` — would break training
   - `models/weights/best_model.pth` listed, but actual file is `yolov8n.pt` (different purpose)

3. **Risk-to-reward ratio is extremely unfavorable.** The cosmetic benefit of a "cleaner" folder layout does not justify the risk of breaking a working system days before submission.

4. **Assessors evaluate working systems, not folder names.** A working demo with clear documentation is worth far more than a "pretty" folder structure that crashes.

5. **The current structure is already well-organized.** It has clear separation of concerns with 19 focused packages. The proposed structure reduces this to 14 by merging concerns — which is not necessarily better.

### What to do instead:

- **Remove the 3 fully-placeholder packages** (`automation/`, `monitoring/`, `registry/`) — they add no value and may raise questions
- **Clean up remaining placeholder files** (e.g., `gait_encoder.py`, `skeleton_extractor.py`)
- **Update `docs/architecture.md`** to accurately reflect the current working structure
- **Keep the proposed target structure** as a `docs/future_architecture.md` for the "future work" section of your report

---

## 16. Step-by-Step Migration Plan (If Migration Is Recommended)

> [!WARNING]
> Migration is NOT recommended before submission. This plan is provided only for completeness.

If a migration were to be undertaken (post-submission, or if significant time remains):

### Phase 0: Preparation (2 hours)
1. Create a full backup / git branch
2. Run all tests and CLI modes to establish baseline
3. Document all test results as reference

### Phase 1: Remove Placeholders (30 min)
1. Delete `automation/`, `monitoring/`, `registry/` packages
2. Delete placeholder files: `gait_encoder.py`, `skeleton_extractor.py`, `decision_engine.py`, `policy_engine.py`, `alert_manager.py` (in intelligence/)
3. Verify no imports reference these files

### Phase 2: Create New Packages (1 hour)
1. Create `datasets/` package with `__init__.py`
2. Create `matching/` package with `__init__.py`
3. Move `pipeline/steps/matching_step.py` → `matching/embedding_matcher.py`
4. Update ALL imports of `pipeline.steps.matching_step` (4+ files)

### Phase 3: Rename Files (2 hours)
1. `core/config.py` → `core/config_loader.py` — update `core/boot.py`
2. `storage/vector_store.py` → `storage/gallery_store.py` — update 7+ files
3. `security_layer/security_engine.py` → `security_layer/security_manager.py` — update 2 files
4. `security_layer/security_logger.py` → `security_layer/event_logger.py` — update 1 file
5. `streaming/stream_engine.py` → `streaming/stream_reader.py` — update 1 file
6. `evaluation/evaluator.py` → `evaluation/benchmark.py` — update scripts
7. `evaluation/visualizer.py` → `evaluation/charts.py` — update scripts
8. `training/loss_functions.py` → `training/losses.py` — no external imports

### Phase 4: Move Gallery Data (1 hour)
1. Move `models/gallery/` → `data/gallery/`
2. Rename `gallery_metadata.json` → `gallery_meta.json`
3. Update `VectorStore` default path
4. Update all 7+ consumers

### Phase 5: Flatten Pipeline Steps (2 hours)
1. Move `pipeline/steps/detection.py` → `pipeline/detector.py`
2. Move `pipeline/steps/tracking.py` → `pipeline/tracker.py`
3. Redistribute remaining steps into pipeline modules
4. Update all imports in live_recognition, video_recognition

### Phase 6: Redistribute Utils (2 hours)
1. Move `utils/alert_manager.py` → decide target (security_layer? core?)
2. Move `utils/event_logger.py` → `security_layer/event_logger.py`
3. Move `utils/prediction_smoother.py` → `pipeline/` or `core/`
4. Move remaining utils into `core/utils.py`
5. Update all imports

### Phase 7: Update CLI (1 hour)
1. Update all subprocess paths in `cli.py`
2. Rename referenced scripts

### Phase 8: Preserve API (1 hour)
1. Keep `api/` package as-is (it's not in proposed but it works)
2. Update internal imports if dependencies moved

### Phase 9: Update Tests (2 hours)
1. Flatten `tests/unit/` and `tests/integration/` into `tests/`
2. Rename test files to match proposed
3. Update test imports

### Phase 10: Full Regression Test (3 hours)
1. Run `python cli.py --mode demo`
2. Test all 16 CLI modes individually
3. Run `python -m pytest tests/ -v`
4. Test API: `python cli.py --mode api` + curl endpoints
5. Test live recognition with webcam
6. Verify gallery operations

### Estimated Total: 16-24 hours with testing

---

## 17. Final Recommendation

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   RECOMMENDATION:  DO NOT MIGRATE BEFORE SUBMISSION          ║
║                                                              ║
║   Keep the current working structure.                        ║
║   Remove placeholder packages only.                          ║
║   Update docs/architecture.md to match reality.              ║
║   Include the proposed structure as "future work".           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Summary of Decision:

| Question | Answer |
|---|---|
| Can the current structure be safely migrated? | Technically yes, but with **very high risk** |
| Should it be migrated before submission? | **NO** |
| What is the biggest risk? | **50+ import chain breakages** + gallery path breakage |
| What would break first? | `cli.py --mode demo` (chains all modes) |
| What is the proposed structure missing? | `api/`, `utils/`, `events/`, training sub-modules |
| Is the current structure good enough? | **YES** — it's clean, modular, and working |
| Recommended action? | Remove placeholders, update docs, submit as-is |

### Safe Pre-Submission Cleanup (Optional, Low-Risk):

1. ✅ Delete `automation/`, `monitoring/`, `registry/` (all placeholders)
2. ✅ Delete individual placeholder files (`gait_encoder.py`, `skeleton_extractor.py`, etc.)
3. ✅ Delete `create_project.py` (scaffolding, not runtime)
4. ✅ Update `docs/architecture.md` to reflect the actual working structure
5. ✅ Add `docs/future_architecture.md` showing the proposed clean structure as planned improvement

---

*End of migration recommendation report.*  
*Generated: 2026-06-12 | Analysis only — no files were modified.*
