# Phase 1 — Repository and Environment Audit

## 1.1 Git State

| Property | Value |
|---|---|
| **Branch** | `main` |
| **Latest Commit** | `4db1632` — "Fix evaluation pipeline and baseline audit issues" |
| **Untracked Files** | `docs/CLEAN_RETRAINING_READINESS_REPORT.md`, `docs/THESIS_ARCHITECTURE_COMPLIANCE_REPORT.md` |
| **Uncommitted Changes** | None (clean working tree) |
| **Remote** | `chanuka8/argus-gait-recognition` (GitHub) |

## 1.2 Python Environment

| Property | Value |
|---|---|
| **Python Version** | 3.14.6 |
| **Virtual Environment** | `e:\ARGUS_AI\venv` (active) |
| **Installed Packages (venv)** | numpy 2.5.1, opencv-python 5.0.0.93, psutil 7.2.2, packaging 26.2, pip 26.1.2, setuptools 83.0.0, wheel 0.47.0 |

> [!WARNING]
> The venv only contains minimal packages (numpy, opencv, psutil). PyTorch, ultralytics, supervision, FastAPI, and other core dependencies listed in `requirements.txt` are **not installed** in the current venv. Full pipeline execution requires installing all dependencies from `requirements.txt`.

## 1.3 Main Dependencies (requirements.txt)

| Package | Min Version | Purpose |
|---|---|---|
| torch | ≥2.0.0 | Deep learning framework |
| torchvision | ≥0.15.0 | Vision transforms/datasets |
| ultralytics | ≥8.0.0 | YOLOv8 person detection |
| supervision | ≥0.18.0 | ByteTrack multi-target tracking |
| opencv-python | ≥4.8.0 | Image processing, video I/O |
| numpy | ≥1.24.0 | Numerical arrays |
| fastapi | ≥0.100.0 | REST API server |
| uvicorn | ≥0.22.0 | ASGI server |
| pydantic | ≥2.0.0 | Data validation |
| pyyaml | ≥6.0.0 | YAML config parsing |
| matplotlib | ≥3.7.0 | Visualization |
| psutil | ≥5.9.0 | System monitoring |
| tqdm | ≥4.65.0 | Progress bars |
| pytest | ≥8.0.0 | Test framework |
| ruff | ≥0.3.0 | Linter |
| black | ≥24.0.0 | Code formatter |
| mypy | ≥1.8.0 | Type checker |

## 1.4 Model Checkpoint Paths

| File | Size | Location |
|---|---|---|
| `best_model.pth` | 770 KB | `runs/exp_001/best_model.pth` |
| `last_model.pth` | 770 KB | `runs/exp_001/last_model.pth` |
| `best_model_ce_726.pth` | 567 KB | `runs/exp_001/best_model_ce_726.pth` (legacy) |
| `yolov8n.pt` | 6.5 MB | `models/weights/yolov8n.pt` |

## 1.5 Dataset Paths

| Asset | Path | Status |
|---|---|---|
| Raw CASIA-B archive | `data/casia_b_raw.zip` | Present (763 MB) |
| Processed GEIs | `data/casia_processed/gei/` | Present (124 subjects) |
| Processed silhouettes | `data/casia_processed/silhouettes/` | Present |
| Processed skeletons | `data/casia_processed/skeletons/` | Present |
| Auto-enrollment data | `data/auto_enrollment/` | Present |

## 1.6 Configuration Files

| File | Purpose |
|---|---|
| `configs/inference.yaml` | Matching thresholds, crowd control, display, reporting |
| `configs/system.yaml` | Camera, logging, watchdog, recognition, service config |
| `configs/cameras.yaml` | Multi-camera CCTV definitions (3 cameras configured) |
| `configs/subject_split.json` | Subject-disjoint split: Train 001-062, Val 063-074, Test 075-124 |
| `configs/gallery_probe_manifest.json` | Gallery/probe assignment manifest |
| `configs/detection.yaml` | Detection parameters |
| `configs/gei.yaml` | GEI generation parameters |
| `configs/train.yaml` | Training parameters |
| `configs/auto_train.yaml` | Auto-training parameters |
| `ruff.toml` | Linter configuration |
| `.env.example` | Environment variable template |

## 1.7 Gallery Assets

| File | Size | Location |
|---|---|---|
| `gallery_features.npy` | 13.9 MB | `models/gallery/` |
| `gallery_labels.npy` | 163 KB | `models/gallery/` |
| `gallery_metadata.json` | 18 KB | `models/gallery/` |

## 1.8 Evaluation Outputs

| File | Location |
|---|---|
| `closed_set_eval_report.json` | `runs/exp_001/evaluation_subject_disjoint/` |
| `open_set_report.json` | `runs/exp_001/evaluation_subject_disjoint/` |
| `cross_view_report.json` + `.md` + `.csv` | `runs/exp_001/evaluation_subject_disjoint/` |
| `threshold_calibration.json` | `runs/exp_001/evaluation_subject_disjoint/` |
| `inference_benchmark.json` | `runs/exp_001/evaluation_subject_disjoint/` |
| `metrics.json` (50 epochs) | `runs/exp_001/` |

## 1.9 Component Classification

| Directory/Module | Classification | Evidence |
|---|---|---|
| `models/architectures/bygait_light.py` | **Implemented and verified** | Full CNN architecture, trained checkpoint exists |
| `models/architectures/losses.py` | **Implemented and verified** | BatchHardTripletLoss, JointGaitLoss, ArcMarginProduct |
| `models/architectures/gait_encoder.py` | **Placeholder** | 64-byte file with docstring only |
| `pipeline/live_recognition.py` | **Implemented and verified** | 763-line full pipeline |
| `pipeline/multi_camera_recognition.py` | **Implemented and verified** | 1022-line multi-camera pipeline |
| `pipeline/video_recognition.py` | **Implemented** | Video file processing pipeline |
| `pipeline/steps/tracking.py` | **Implemented and verified** | YOLOv8 + ByteTrack |
| `pipeline/steps/silhouette_step.py` | **Implemented and verified** | Otsu + morphology + contour extraction |
| `pipeline/steps/live_gei.py` | **Implemented and verified** | Rolling-window GEI builder |
| `pipeline/steps/matching_step.py` | **Implemented and verified** | Cosine similarity matching |
| `pipeline/steps/centroid_matching_step.py` | **Implemented and verified** | Centroid + margin + top-k |
| `training/trainer.py` | **Implemented and executed** | 50-epoch run confirmed by metrics.json |
| `training/dataset.py` | **Implemented and verified** | GEIDataset with scan/resize/normalize |
| `evaluation/evaluator.py` | **Implemented and executed** | Subject-disjoint evaluation with leakage checks |
| `evaluation/leakage_validator.py` | **Implemented and verified** | Subject, gallery/probe, calibration disjointness checks |
| `evaluation/metrics.py` | **Implemented and verified** | Rank-k, CMC, biometric rates, ROC-AUC, EER |
| `evaluation/cross_view_evaluator.py` | **Implemented and executed** | Cross-view matrix generated |
| `evaluation/open_set_evaluator.py` | **Implemented and executed** | Open-set report generated |
| `evaluation/threshold_calibrator.py` | **Implemented and executed** | Threshold calibration on val set |
| `enrollment/enrollment_manager.py` | **Implemented** | Gait + appearance enrollment |
| `enrollment/auto_enrollment_service.py` | **Implemented** | Automated video-based enrollment |
| `security_layer/security_engine.py` | **Implemented** | Confidence-based decision engine |
| `security_layer/security_logger.py` | **Implemented** | Thread-safe CSV audit logger |
| `intelligence/cross_camera_tracker.py` | **Implemented but not integration-tested** | Global track ID, transitions |
| `intelligence/identity_persistence.py` | **Implemented but not integration-tested** | Score accumulation, alert suppression |
| `intelligence/missing_person_workflow.py` | **Implemented but not integration-tested** | Watchlist, alert, evidence triggers |
| `intelligence/reid_cache.py` | **Implemented but not integration-tested** | TTL-based embedding cache |
| `intelligence/confidence_scorer.py` | **Implemented** | Confidence scoring |
| `intelligence/alert_manager.py` | **Placeholder** | 64-byte stub |
| `intelligence/decision_engine.py` | **Placeholder** | 64-byte stub |
| `intelligence/policy_engine.py` | **Placeholder** | 64-byte stub |
| `streaming/multi_stream_engine.py` | **Implemented** | Multi-stream ingestion |
| `streaming/worker_pool.py` | **Implemented** | Thread pool management |
| `streaming/load_balancer.py` | **Implemented** | Load balancing logic |
| `streaming/camera_scheduler.py` | **Implemented** | Camera scheduling |
| `storage/vector_store.py` | **Implemented and verified** | Gallery save/load (NumPy + JSON) |
| `storage/evidence_manager.py` | **Implemented** | Evidence snapshot/retention |
| `storage/lineage_tracker.py` | **Implemented** | Data lineage tracking |
| `monitoring/watchdog.py` | **Implemented** | Health check watchdog |
| `monitoring/camera_monitor.py` | **Implemented** | Camera health monitoring |
| `monitoring/logging_config.py` | **Implemented** | Structured logging configuration |
| `monitoring/crash_guard.py` | **Placeholder** | 64-byte stub |
| `monitoring/gpu_tuner.py` | **Placeholder** | 64-byte stub |
| `monitoring/metrics_collector.py` | **Placeholder** | 64-byte stub |
| `monitoring/performance_profiler.py` | **Placeholder** | 64-byte stub |
| `automation/*` | **All placeholders** | All 64-byte stubs (auto_trainer, lifecycle_controller, model_promoter, model_validator, rollback_manager, training_queue) |
| `services/camera_service.py` | **Implemented** | Camera lifecycle service |
| `services/camera_worker.py` | **Implemented** | Per-camera worker threads |
| `services/camera_discovery.py` | **Implemented** | ONVIF camera discovery |
| `services/onvif_client.py` | **Implemented** | ONVIF protocol client |
| `services/vendor_adapters.py` | **Implemented** | Camera vendor adapters |
| `api/server.py` | **Implemented** | FastAPI server scaffold |
| `deployment/` | **Configuration only** | PowerShell service install/uninstall scripts |
| `utils/prediction_smoother.py` | **Implemented and verified** | Voting-based temporal smoothing |
| `utils/box_stabilizer.py` | **Implemented** | EMA bounding box smoothing |
| `utils/detection_reporter.py` | **Implemented** | JSONL/CSV detection reporting |
| `utils/display_renderer.py` | **Implemented** | CCTV-style overlay renderer |
| `utils/alert_manager.py` | **Implemented** | Alert evaluation and cooldown |

## 1.10 Reproducibility Risks

| Risk | Severity | Description |
|---|---|---|
| **Incomplete venv** | HIGH | Only numpy/opencv installed; full requirements not in active venv |
| **Training used all 124 subjects** | **CRITICAL** | `metrics.json` shows `num_classes: 124`, `samples: 13544`. The model was trained on ALL subjects including test subjects 075-124. This means the current `best_model.pth` has **subject leakage** for closed-set classification. |
| **Evaluation uses embedding-based matching** | MEDIUM | The evaluator uses the backbone for embedding extraction (not classifier head), which partially mitigates leakage but does not eliminate it |
| **No clean subject-disjoint checkpoint** | HIGH | No checkpoint trained exclusively on subjects 001-074 exists in the repository |
| **CASIA-B raw data required** | LOW | Raw ZIP present locally (763 MB) but not in git |
| **Hardware-dependent** | LOW | Model trained on CPU, evaluation on CPU |
| **No random seed in dataloader split** | MEDIUM | `random_split` in `dataloader.py` has no seed parameter |

## 1.11 CI/CD

| Property | Value |
|---|---|
| **CI System** | GitHub Actions |
| **Workflow** | `.github/workflows/CI.yaml` |
| **Python Version (CI)** | 3.11 |
| **Steps** | Checkout → Install deps → Ruff lint → pytest tests/ |
| **Status** | Last merged PR passed CI |
