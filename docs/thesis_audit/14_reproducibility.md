# Phase 13 — Reproducibility and Evidence Chain

## 14.1 Reproducibility Requirements

| Requirement | Status | Evidence | Action Needed |
|---|---|---|---|
| **Source code** | ✅ Available | GitHub repository `chanuka8/argus-gait-recognition` | None |
| **Dependencies specified** | ✅ Available | `requirements.txt` with version pins | None |
| **Dataset obtainable** | ⚠️ Conditional | CASIA-B requires academic request from CASIA | Document dataset acquisition process |
| **Local dataset present** | ✅ Available | `data/casia_b_raw.zip` (763 MB) + `data/casia_processed/gei/` | None |
| **Subject split documented** | ✅ Available | `configs/subject_split.json` | None |
| **Training configuration** | ✅ Available | `metrics.json` + `configs/train.yaml` | None |
| **Training code** | ✅ Available | `training/trainer.py`, `training/dataset.py`, `training/dataloader.py` | None |
| **Evaluation code** | ✅ Available | `evaluation/evaluator.py`, `evaluation/metrics.py` | None |
| **Model checkpoint** | ⚠️ Leaky | `runs/exp_001/best_model.pth` (trained on all 124 subjects) | Retrain on subjects 001-074 |
| **Evaluation results** | ⚠️ Preliminary | `runs/exp_001/evaluation_subject_disjoint/` | Re-run with clean checkpoint |
| **Random seed** | ❌ Not set | No seed in `dataloader.py::random_split`, no seed in `trainer.py` | Add seed control |
| **Hardware specification** | ⚠️ Partial | CPU confirmed; specific hardware not documented | Document CPU model, RAM |
| **CI/CD pipeline** | ✅ Available | `.github/workflows/CI.yaml` | None |
| **Linter passing** | ✅ Available | Ruff configured and run in CI | None |
| **Tests available** | ✅ Available | `tests/` directory with 13+ test files | None |
| **Gallery/probe protocol** | ✅ Available | `evaluation/gallery_probe_builder.py` with CASIA-B standard protocol | None |

## 14.2 Complete File Evidence Chain

### Training Evidence Chain

```
[Input] data/casia_b_raw.zip (763 MB raw CASIA-B)
  ↓ preprocessing/casia_extractor.py + silhouette_extractor.py + gei_builder.py
[Processed] data/casia_processed/gei/{001-124}/*.png (13,544 GEI images)
  ↓ training/dataset.py::GEIDataset + training/dataloader.py::build_dataloaders
[DataLoader] 80/20 random split, batch_size=16
  ↓ training/trainer.py::GaitClassifier (ByGaitLight + ArcMarginProduct)
[Training] 50 epochs, Adam lr=1e-4, CosineAnnealing, ce_arcface loss
  ↓ 
[Output] runs/exp_001/best_model.pth (770 KB)
[Output] runs/exp_001/metrics.json (50 epoch training history)
```

### Evaluation Evidence Chain

```
[Input] runs/exp_001/best_model.pth
[Input] configs/subject_split.json (test subjects 075-124)
  ↓ evaluation/dataset_split.py + evaluation/gallery_probe_builder.py
[Protocol] Gallery: nm-01..04 (test subjects), Probe: nm-05/06, bg-01/02, cl-01/02
  ↓ evaluation/leakage_validator.py::assert_subject_disjointness()
[Verified] No subject overlap between splits
  ↓ evaluation/evaluator.py (embedding extraction + cosine matching)
[Output] closed_set_eval_report.json (Rank-1: 86.89%)
  ↓ evaluation/cross_view_evaluator.py
[Output] cross_view_report.md (Cross-view avg: 71.17%)
  ↓ evaluation/open_set_evaluator.py
[Output] open_set_report.json (ROC-AUC: 0.915)
  ↓ evaluation/threshold_calibrator.py (val subjects 063-074 only)
[Output] threshold_calibration.json (θ=0.9913 at min EER)
```

### Live Pipeline Evidence Chain

```
[Input] Camera/video feed
  ↓ streaming/stream_engine.py
[Frame] BGR numpy array
  ↓ pipeline/steps/tracking.py (YOLOv8n + ByteTrack)
[Tracked] Detections with track IDs
  ↓ utils/box_stabilizer.py (EMA smoothing)
[Stabilized] Stable bounding boxes
  ↓ pipeline/steps/silhouette_step.py (Otsu + morphology)
[Silhouette] 64×128 binary mask
  ↓ pipeline/steps/live_gei.py (rolling window, 15 frames)
[GEI] 64×128 averaged silhouette
  ↓ models/architectures/bygait_light.py (forward pass)
[Embedding] 256-dim L2-normalized vector
  ↓ pipeline/steps/matching_step.py (cosine similarity)
[Match] Top-1 identity + score
  ↓ pipeline/live_recognition.py::_adaptive_decision()
[Decision] {CONFIRMED, VERIFIED, REVIEW, LOW_CONF, UNKNOWN}
  ↓ utils/prediction_smoother.py (temporal voting)
[Stable] Final identity label
  ↓ security_layer/security_engine.py + security_logger.py
[Audit] CSV log entry
  ↓ utils/display_renderer.py + utils/detection_reporter.py
[Output] CCTV overlay + JSONL/CSV reports
```

## 14.3 Configuration Traceability

| Config File | Used By | Parameters Controlled |
|---|---|---|
| `configs/inference.yaml` | `live_recognition.py`, `multi_camera_recognition.py` | Matching thresholds, crowd control, display, reporting, box stability |
| `configs/system.yaml` | `services/argus_service.py`, `services/camera_service.py` | Camera, logging, watchdog, recognition, service params |
| `configs/cameras.yaml` | `multi_camera_recognition.py` | Camera definitions, stream URLs, priorities |
| `configs/subject_split.json` | `evaluation/dataset_split.py`, `evaluation/evaluator.py` | Subject partition: train/val/test |
| `configs/train.yaml` | `training/trainer.py` | Training hyperparameters |
| `configs/detection.yaml` | `pipeline/steps/detection.py` | Detection confidence, NMS thresholds |
| `configs/gei.yaml` | `pipeline/steps/live_gei.py` | GEI frame count, image size |
| `ruff.toml` | CI pipeline | Linter rules |

## 14.4 Thesis Figure Traceability

| Potential Thesis Figure | Source Data | Source File | Reproducible? |
|---|---|---|---|
| Training loss/accuracy curves | `runs/exp_001/metrics.json` | `training/trainer.py` | Yes (with clean retrain) |
| Cross-view accuracy matrix | `cross_view_matrix.csv` | `evaluation/cross_view_evaluator.py` | Yes |
| CMC curve | `closed_set_eval_report.json::cmc_curve` | `evaluation/metrics.py` | Yes |
| ROC curve | `open_set_scores.json` | `evaluation/open_set_evaluator.py` | Yes |
| Condition-wise bar chart | `closed_set_eval_report.json::condition_wise_accuracy` | `evaluation/evaluator.py` | Yes |
| Pipeline architecture diagram | Code structure | Manual/Mermaid | N/A |
| Model architecture diagram | `bygait_light.py` | Manual/Mermaid | N/A |
| CCTV overlay screenshot | Live pipeline output | `utils/display_renderer.py` | Yes (requires camera) |
