# ARGUS Recommended Cleanup Plan

**Audit Date:** 2026-06-11
**Project Root:** `E:\ARGUS_AI`

> [!IMPORTANT]
> This document is **analysis only**. No changes have been made to any files.
> All recommendations are categorized by risk and effort.

---

## Priority 1 — Critical Structural Fixes (No Code Change Required)

These are missing infrastructure items that will cause silent import failures or confusion.

### 1.1 Add Missing `__init__.py` to `models/`

**Problem:** `models/` is actively imported as `from models.architectures.bygait_light import ByGaitLight` in 6+ files. However, the top-level `models/` directory has **no `__init__.py`**. This works today only because Python 3.3+ supports namespace packages, but it is non-standard, fragile, and may break tools like pytest, mypy, or IDE indexing.

**Recommendation:** Create `models/__init__.py` (empty or with docstring).

**Risk:** Very low. **Effort:** Trivial.

---

### 1.2 Add Missing `__init__.py` to `tests/`

**Problem:** `tests/` has no `__init__.py`. While pytest can often discover tests without it, the presence of `tests/unit/__init__.py` and `tests/integration/__init__.py` (both empty) implies a package-based test structure was intended. Without `tests/__init__.py`, relative imports within the test suite will fail.

**Recommendation:** Create `tests/__init__.py` (empty).

**Risk:** Very low. **Effort:** Trivial.

---

## Priority 2 — Safe Removals (Dead Code)

These modules contain implementations that are **never imported** anywhere in the project. They are candidates for removal after confirming no future plans depend on them.

| File | Reason | Risk |
|---|---|---|
| `events/dispatcher.py` | `EventDispatcher` wraps `EventBus` but is never used — direct `EventBus` usage already works | Very Low |
| `enrollment/enrollment_queue.py` | `EnrollmentQueue` is fully implemented but unused — `folder_watcher.py` handles queuing directly | Very Low |
| `preprocessing/casia_extractor.py` | Legacy extractor; replaced by `preprocessing/dataset_builder.py` + `CasiaGEIDatasetBuilder` | Low |
| `utils/video.py` | `VideoReader` thin wrapper around `cv2.VideoCapture`; replaced by `streaming/stream_engine.py` | Very Low |
| `utils/io_utils.py` | `ensure_directory` / `save_image` utilities that no current module imports | Very Low |

> **Note:** Before deleting `preprocessing/casia_extractor.py`, verify no future scripts are planned to use it. It uses `utils/zip_streamer.py` (which itself is only used by `casia_extractor.py`), so both could be cleaned up together.

---

## Priority 3 — Implement or Wire Up Partially-Done Modules

These files have real implementations but are **not connected** to the active pipeline. They should be wired in before the project reaches production maturity.

### 3.1 `models/architectures/losses.py` — `TripletLoss`

**Status:** Fully implemented. **Problem:** Never used — training currently uses `CrossEntropyLoss`.

**Recommendation:** Wire `TripletLoss` into `training/loss_functions.py` when metric learning / triplet training is required. `training/loss_functions.py` is already a placeholder for this purpose.

---

### 3.2 `preprocessing/augmentation.py` — `Augmentation`

**Status:** Implemented (`horizontal_flip`, `gaussian_noise`). **Problem:** Never used in `training/dataset.py`.

**Recommendation:** Import and apply `Augmentation` in `training/dataset.py` `__getitem__` when preparing training transforms. This is a direct complement to the existing dataset pipeline.

---

### 3.3 `events/event_types.py` — `EventType` Enum

**Status:** Fully defined enum. **Problem:** Never imported — the event bus is wired in scripts but raw strings are used for event names.

**Recommendation:** Replace raw string event names in `events/event_bus.py` usage with `EventType` enum members. This makes event names type-safe and discoverable.

---

### 3.4 `enrollment/enrollment_queue.py` — `EnrollmentQueue`

**Status:** Implemented. **Problem:** `folder_watcher.py` processes enrollments synchronously within a polling loop, which blocks on large batches.

**Recommendation:** Integrate `EnrollmentQueue` as an async producer/consumer between `folder_watcher.py` (producer) and `enrollment_manager.py` (consumer). This would improve throughput for high-volume enrollment scenarios.

---

## Priority 4 — Required Implementations (Grouped by Layer)

These are placeholders with no implementation at all. Listed in recommended build order.

### Layer A — Foundations (Build First)

| File | Description | Depends On |
|---|---|---|
| `utils/helpers.py` | General utility functions (string ops, date formatting, etc.) | Nothing |
| `utils/math_utils.py` | Math helpers (distance metrics, normalization) | Nothing |
| `utils/queue_utils.py` | Thread-safe queue helpers | Nothing |
| `training/loss_functions.py` | Wrap `TripletLoss` from `models/architectures/losses.py` | `models/architectures/losses.py` |
| `training/callbacks.py` | Training callbacks (early stop, LR scheduler, logging hooks) | `training/trainer.py` |

---

### Layer B — Core Architecture (Build Second)

| File | Description | Depends On |
|---|---|---|
| `pipeline/base_pipeline.py` | Abstract base class for inference and training pipelines | `core/logger.py` |
| `storage/data_manager.py` | CRUD abstraction for file/model/dataset storage | `core/logger.py` |
| `storage/dataset_loader.py` | Dataset loading from disk with caching | `storage/data_manager.py` |
| `storage/cache_manager.py` | LRU / disk-backed embedding cache | `storage/data_manager.py` |
| `storage/lineage_tracker.py` | Track which data and models produced each result | `storage/data_manager.py` |

---

### Layer C — Pipeline Extensions (Build Third)

| File | Description | Depends On |
|---|---|---|
| `pipeline/cache_engine.py` | Cache recent embeddings per track ID to avoid redundant inference | `storage/cache_manager.py`, `pipeline/base_pipeline.py` |
| `pipeline/speed_controller.py` | Adaptive FPS throttle based on load | `monitoring/metrics_collector.py` |
| `pipeline/pipeline_factory.py` | Factory for creating InferencePipeline or LiveRecognitionPipeline | `pipeline/base_pipeline.py` |
| `streaming/buffer_queue.py` | Thread-safe async frame buffer between capture and inference | `utils/queue_utils.py` |
| `streaming/frame_dropper.py` | Drop frames when queue is full under high load | `streaming/buffer_queue.py` |

---

### Layer D — Intelligence Layer (Build Fourth)

| File | Description | Depends On |
|---|---|---|
| `intelligence/confidence_scorer.py` | Normalize and calibrate raw similarity scores | `utils/math_utils.py` |
| `intelligence/alert_manager.py` | Replaces `utils/alert_manager.py` with richer logic and event integration | `intelligence/confidence_scorer.py`, `events/event_bus.py` |
| `intelligence/decision_engine.py` | Evaluate results against security rules | `intelligence/policy_engine.py`, `intelligence/alert_manager.py` |
| `intelligence/policy_engine.py` | Load and execute rule-based access/alert policies | `events/event_types.py` |

---

### Layer E — Model Lifecycle (Build Fifth)

| File | Description | Depends On |
|---|---|---|
| `preprocessing/skeleton_extractor.py` | Pose estimation (MediaPipe or OpenPose) for skeleton GEI | External: mediapipe |
| `models/architectures/gait_encoder.py` | Deeper transformer or LSTM-based encoder | `models/architectures/bygait_light.py` |
| `evaluation/visualizer.py` | Plot CMC curves, confusion matrices, ROC curves | `evaluation/metrics.py` |
| `monitoring/metrics_collector.py` | Collect inference FPS, latency, memory usage | `core/system_monitor.py` |
| `monitoring/performance_profiler.py` | Profile bottlenecks per pipeline step | `monitoring/metrics_collector.py` |
| `monitoring/crash_guard.py` | Watchdog for process crash recovery | `monitoring/metrics_collector.py` |
| `monitoring/gpu_tuner.py` | Dynamic batch size / precision selection based on VRAM | `monitoring/metrics_collector.py` |

---

### Layer F — Registry & Automation (Build Last)

| File | Description | Depends On |
|---|---|---|
| `registry/model_registry.py` | Store, version, and retrieve trained model weights | `storage/data_manager.py` |
| `registry/experiment_tracker.py` | Track training runs, hyperparams, and metrics | `registry/model_registry.py` |
| `automation/training_queue.py` | Queue management for scheduled retraining jobs | `utils/queue_utils.py` |
| `automation/model_validator.py` | Run evaluation suite as a gate before promotion | `evaluation/evaluator.py` |
| `automation/model_promoter.py` | Promote validated model to active deployment | `registry/model_registry.py` |
| `automation/rollback_manager.py` | Restore a previous model version on failure | `registry/model_registry.py` |
| `automation/lifecycle_controller.py` | Orchestrate full auto-training → validate → promote cycle | All automation modules |
| `automation/auto_trainer.py` | Trigger retraining on new data detected in enrollment | `automation/lifecycle_controller.py` |
| `cli.py` | Command-line interface for running modes, querying status | All core modules |

---

## Priority 5 — Test Stubs to Implement

All seven test files below are placeholders. They should be implemented in parallel with the corresponding module implementations.

| Test File | Module to Test |
|---|---|
| `tests/conftest.py` | Common fixtures (temp dirs, mock galleries, model stubs) |
| `tests/unit/test_evaluation.py` | `evaluation/evaluator.py`, `evaluation/metrics.py` |
| `tests/unit/test_pipeline.py` | `pipeline/inference_pipeline.py`, `pipeline/live_recognition.py` |
| `tests/unit/test_preprocessing.py` | `preprocessing/dataset_builder.py`, `preprocessing/gei_builder.py` |
| `tests/unit/test_vector_store.py` | `storage/vector_store.py` |
| `tests/integration/test_end_to_end.py` | Full inference pipeline from image to identity |
| `tests/integration/test_enrollment_flow.py` | Enrollment from folder detection to gallery update |

---

## Priority 6 — Duplicate / Overlapping Module Responsibilities

These overlaps should be resolved through consolidation during implementation.

| Conflict | Description | Recommendation |
|---|---|---|
| `utils/alert_manager.py` vs `intelligence/alert_manager.py` | Both are intended to handle alert generation. `utils/alert_manager.py` is active; `intelligence/alert_manager.py` is a placeholder. | Implement `intelligence/alert_manager.py` as the canonical module. Migrate `live_recognition.py` to import from there. Archive or remove `utils/alert_manager.py`. |
| `core/system_monitor.py` vs `monitoring/metrics_collector.py` | `core/system_monitor.py` gives a one-shot CPU/RAM/VRAM snapshot. `monitoring/metrics_collector.py` is planned for real-time continuous metrics. | Keep `core/system_monitor.py` for boot diagnostics only. Implement `monitoring/metrics_collector.py` for runtime collection. Avoid scope overlap. |
| `preprocessing/casia_extractor.py` vs `preprocessing/dataset_builder.py` | Both target CASIA-B raw data extraction. `dataset_builder.py` is the active implementation. `casia_extractor.py` appears to be an older iteration. | Verify if `casia_extractor.py` provides anything not in `dataset_builder.py`. If not, remove it. |
| `utils/video.py` vs `streaming/stream_engine.py` | `VideoReader` in `utils/video.py` is a thin OpenCV wrapper. `StreamEngine` in `streaming/stream_engine.py` is the active implementation with resolution control. | Remove `utils/video.py`. All video reading should go through `StreamEngine`. |
| `enrollment/enrollment_queue.py` and direct synchronous polling in `folder_watcher.py` | Both handle the sequencing of enrollment jobs. | Integrate `EnrollmentQueue` into `folder_watcher.py` for async decoupling, or remove the queue if synchronous polling suffices. |

---

## Summary Action Table

| Action | File(s) | Priority | Risk |
|---|---|---|---|
| Create `models/__init__.py` | `models/` | 🔴 P1 | Very Low |
| Create `tests/__init__.py` | `tests/` | 🔴 P1 | Very Low |
| Remove dead `EventDispatcher` | `events/dispatcher.py` | 🟡 P2 | Very Low |
| Remove or consolidate `casia_extractor.py` | `preprocessing/casia_extractor.py` | 🟡 P2 | Low |
| Remove `utils/video.py` | `utils/video.py` | 🟡 P2 | Very Low |
| Remove `utils/io_utils.py` | `utils/io_utils.py` | 🟡 P2 | Very Low |
| Wire `TripletLoss` → `training/loss_functions.py` | `training/loss_functions.py` | 🟠 P3 | Low |
| Wire `Augmentation` → `training/dataset.py` | `preprocessing/augmentation.py` | 🟠 P3 | Low |
| Implement `utils/helpers.py`, `math_utils.py`, `queue_utils.py` | Layer A | 🟠 P4 | None |
| Implement `pipeline/base_pipeline.py` + `storage/` layer | Layer B | 🟠 P4 | Medium |
| Implement all `intelligence/` modules | Layer D | 🟠 P4 | Medium |
| Implement all `automation/` modules | Layer F | 🔵 P4 | High (complex) |
| Migrate alerts from `utils/` to `intelligence/` | Cross-package | 🟡 Long-term | Low |
| Write all test stubs | P5 | 🟠 Ongoing | None |

---

*Report generated by read-only ARGUS Empty File Audit. No files were modified.*
