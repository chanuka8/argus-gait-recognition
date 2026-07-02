# Remaining Placeholder Completion Plan

**Document Path:** `docs/remaining_placeholder_completion_plan.md`  
**Phase:** Planning for Final Project Submission  
**Scope:** Gait Recognition Module Audit & Subsystem Integration Strategy  

---

## 1. Overview & Strategy

This document outlines the implementation strategy for the remaining placeholder files in the `ARGUS` repository. As this is a final-year academic project, the primary objectives for the submission are:
1.  **High Demonstration Value:** Implementing features that show strong academic depth (e.g., comparative models, alternative preprocessors).
2.  **Safety and Stability:** Avoiding modifications to working real-time tracking, background subtraction, REST API endpoints, and validation pipelines.
3.  **Scoped Delivery:** Excluding enterprise MLOps (AutoML, deployment validators, rollback systems, and advanced hardware tuners) which are out-of-scope for the core gait recognition biometrics module.

---

## 2. File Classification Matrix

Below is the assessment of all 20 remaining placeholder files (17 source modules and 3 test modules):

| File Path | Subsystem | Risk Level | Recommendation | Purpose & FYP Viva Justification |
| :--- | :--- | :--- | :--- | :--- |
| [gait_encoder.py](file:///e:/ARGUS_AI/models/architectures/gait_encoder.py) | `models/` | **Low** | **Fill Now** | **Recommended:** Implements an alternative CNN architecture (e.g., a deeper network or resnet-based block) to compare accuracy against the default `ByGaitLight` model, which is a key requirement for the thesis comparative analysis. |
| [test_end_to_end.py](file:///e:/ARGUS_AI/tests/integration/test_end_to_end.py) | `tests/` | **Low** | **Fill Now** | **Recommended:** Provides automated validation to verify that raw frames route successfully through tracking, silhouette extraction, embedding generation, and gallery retrieval. |
| [test_enrollment_flow.py](file:///e:/ARGUS_AI/tests/integration/test_enrollment_flow.py) | `tests/` | **Low** | **Fill Now** | **Recommended:** Verifies the file system integration of `FolderWatcher` and the REST API `/enroll` route. |
| [conftest.py](file:///e:/ARGUS_AI/tests/conftest.py) | `tests/` | **Low** | **Fill Now** | **Recommended:** Defines fixtures (mock frames, target directories) to clean up unit tests. |
| [skeleton_extractor.py](file:///e:/ARGUS_AI/preprocessing/skeleton_extractor.py) | `preprocessing/` | **Medium** | **Future Work** | **Optional:** Integrates coordinate pose tracking (e.g., MediaPipe or YOLO-pose keypoints) to compare performance against silhouette-based GEIs. Highly valued, but has high dependency overhead. |
| [alert_manager.py](file:///e:/ARGUS_AI/intelligence/alert_manager.py) | `intelligence/` | **Low** | **Skip Now** | **Not Needed:** This is a duplicate file. Standard system alert triggers are already fully implemented and managed by the active [alert_manager.py](file:///e:/ARGUS_AI/utils/alert_manager.py) helper. |
| [decision_engine.py](file:///e:/ARGUS_AI/intelligence/decision_engine.py) | `intelligence/` | **Low** | **Skip Now** | **Not Needed:** The security engine class [security_engine.py](file:///e:/ARGUS_AI/security_layer/security_engine.py) already handles risk evaluations and logs alerts in real-time. |
| [policy_engine.py](file:///e:/ARGUS_AI/intelligence/policy_engine.py) | `intelligence/` | **Low** | **Skip Now** | **Not Needed:** Access rules and camera active times are outside the scope of the core gait recognition algorithm. |
| [crash_guard.py](file:///e:/ARGUS_AI/monitoring/crash_guard.py) | `monitoring/` | **High** | **Future Work** | **Not Needed:** Implements background recovery loops for process crashes. Adding these features introduces background threads that can interfere with PyTorch execution. |
| [gpu_tuner.py](file:///e:/ARGUS_AI/monitoring/gpu_tuner.py) | `monitoring/` | **Medium** | **Future Work** | **Not Needed:** Runs automatic GPU batch and profile optimizations. Unnecessary since the local system runs on CPU and the light models have a small footprint. |
| [metrics_collector.py](file:///e:/ARGUS_AI/monitoring/metrics_collector.py) | `monitoring/` | **Low** | **Future Work** | **Not Needed:** Gathers long-term telemetry metrics. Resource metrics are already gathered by `system_monitor.py` and exposed via the API `/metrics` endpoint. |
| [performance_profiler.py](file:///e:/ARGUS_AI/monitoring/performance_profiler.py) | `monitoring/` | **Low** | **Future Work** | **Not Needed:** Profiles step durations. Step timings are already monitored by `benchmark.py` and diagnostic run output logs. |
| [experiment_tracker.py](file:///e:/ARGUS_AI/registry/experiment_tracker.py) | `registry/` | **Low** | **Future Work** | **Not Needed:** Track hyperparameters and epoch results (such as with MLflow). Training metrics are already saved as JSONs by `checkpointer.py` and logged to txt by `callbacks.py`. |
| [model_registry.py](file:///e:/ARGUS_AI/registry/model_registry.py) | `registry/` | **Low** | **Future Work** | **Not Needed:** Manages versioned checkpoints in cloud deployments. Simple local weight files (`best_model.pth`) are sufficient for project validation. |
| [auto_trainer.py](file:///e:/ARGUS_AI/automation/auto_trainer.py) | `automation/` | **Medium** | **Future Work** | **Not Needed:** Implements automated retraining schedulers. Schedulers pose risks of locking the GPU or CPU when multiple clients request lookups. |
| [lifecycle_controller.py](file:///e:/ARGUS_AI/automation/lifecycle_controller.py) | `automation/` | **Medium** | **Future Work** | **Not Needed:** Swaps neural model weights dynamically at runtime. The current model load in `live_recognition.py` is sufficient. |
| [model_promoter.py](file:///e:/ARGUS_AI/automation/model_promoter.py) | `automation/` | **Low** | **Future Work** | **Not Needed:** Promotes models automatically by comparing verification scores. Unnecessary for a single-investigator project. |
| [model_validator.py](file:///e:/ARGUS_AI/automation/model_validator.py) | `automation/` | **Low** | **Future Work** | **Not Needed:** Runs compliance checks before deployment, which is redundant with Pytest unit testing. |
| [rollback_manager.py](file:///e:/ARGUS_AI/automation/rollback_manager.py) | `automation/` | **Medium** | **Future Work** | **Not Needed:** Reverts active models to previous stable checkpoints on failure. |
| [training_queue.py](file:///e:/ARGUS_AI/automation/training_queue.py) | `automation/` | **Low** | **Future Work** | **Not Needed:** Schedules background training tasks. Retraining is handled offline using the standalone training script. |

---

## 3. Recommended Implementation Plan

To maximize evaluation scores while minimizing risk, we recommend implementing the selected **4 modules** in the following order:

```
[ conftest.py ] (Define test directories and fake frame arrays)
       │
       ▼
[ gait_encoder.py ] (Implement secondary comparative CNN model)
       │
       ▼
[ test_end_to_end.py ] (Verify pipelines from tracking to matches)
       │
       ▼
[ test_enrollment_flow.py ] (Verify FolderWatcher & REST enrollment API)
```

### Step 1: `tests/conftest.py`
*   **Action:** Implement fixture methods.
*   **Fixture Details:**
    *   Provide temporary folders for model checkpoints and validation files.
    *   Provide mock NumPy arrays representing target frames ($128 \times 64$) to run tests without needing local CASIA-B directories.

### Step 2: `models/architectures/gait_encoder.py`
*   **Action:** Implement `GaitEncoder` or `ByGaitDeep` CNN.
*   **Implementation Strategy:**
    *   Build a PyTorch network class containing 5 convolutional layers with batch normalization (to compare with `ByGaitLight`'s non-normalized layers).
    *   Ensure the class outputs a 128-dimensional embedding to keep it compatible with `MatchingStep` and `VectorStore`.

### Step 3: `tests/integration/test_end_to_end.py`
*   **Action:** Implement test cases.
*   **Tests to write:**
    *   Configure mock video tracks, run them through detection, tracking, silhouette extraction steps, and query the vector store for matches.

### Step 4: `tests/integration/test_enrollment_flow.py`
*   **Action:** Implement test cases.
*   **Tests to write:**
    *   Start a mock watcher directory, add temporary subject files, and verify that `FolderWatcher` registers them and updates `VectorStore` correctly.

---

## 4. Final Recommendation

For the final project submission, focus on **Model Comparative Analysis** and **Automated Pipeline Verification**:
1.  **Fill Now:** Implement comparative architectures in [gait_encoder.py](file:///e:/ARGUS_AI/models/architectures/gait_encoder.py) and complete the integration test files ([conftest.py](file:///e:/ARGUS_AI/tests/conftest.py), [test_end_to_end.py](file:///e:/ARGUS_AI/tests/integration/test_end_to_end.py), [test_enrollment_flow.py](file:///e:/ARGUS_AI/tests/integration/test_enrollment_flow.py)).
2.  **Skip Now (Future Work):** The remaining 16 placeholders represent out-of-scope enterprise features. Keeping these as placeholders is appropriate for an academic project, as they can be documented in the thesis "Future Work" section to demonstrate long-term system scaling plans.
