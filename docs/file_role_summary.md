# ARGUS File Role Summary

**Date:** 2026-06-13  
**Total Files Analyzed:** 29

---

## Classification Legend

| Tag | Meaning |
|---|---|
| 🟢 CORE | Essential runtime file — system breaks without it |
| 🔵 SCRIPT | CLI-callable standalone script — entry point for a specific task |
| 🟡 SUPPORT | Utility / helper — used by core files but not executed directly |
| ⚪ CONFIG | Entry point / configuration / orchestration |

---

## Entry Points

| # | File | Tag | Role |
|---|---|---|---|
| 1 | `cli.py` | ⚪ CONFIG | **Central CLI dispatcher.** 22 modes. All user-facing commands route through here. Spawns scripts via `subprocess.run()`. Handles background process management for the live + watcher parallel mode. |

---

## Pipeline — Core Runtime

| # | File | Tag | Role |
|---|---|---|---|
| 2 | `pipeline/live_recognition.py` | 🟢 CORE | **Live camera recognition pipeline.** Integrates all subsystems: tracking, silhouette, GEI, ByGaitLight, matching, smoother, alerts, security. Reads from webcam via StreamEngine. Uses `models/live_gallery`. Threshold: 0.85. |
| 3 | `pipeline/video_recognition.py` | 🟢 CORE | **Video file recognition pipeline.** Structurally identical to live, but reads from a video file. Produces frame-level CSV reports. Uses `models/live_gallery`. Threshold: 0.85. |
| 4 | `pipeline/folder_recognition.py` | 🟢 CORE | **Folder-based batch recognition.** Reads GEI images from a folder and matches each against gallery. Uses `InferencePipeline` which loads `models/gallery` (training gallery). No status filtering. |
| 5 | `pipeline/inference_pipeline.py` | 🟢 CORE | **Single-image inference wrapper.** Combines `FeatureExtractionStep` + `MatchingStep` + `VectorStore`. Used by folder recognition. ⚠ Uses `models/gallery`, not `models/live_gallery`. |

---

## Pipeline Steps — Core Components

| # | File | Tag | Role |
|---|---|---|---|
| 6 | `pipeline/steps/tracking.py` | 🟢 CORE | **Person detection + tracking.** YOLOv8n for person detection (class=0), ByteTrack for persistent ID assignment across frames. Used by live, video, and enrollment pipelines. |
| 7 | `pipeline/steps/silhouette_step.py` | 🟢 CORE | **Silhouette extraction.** Converts person crop → grayscale → Gaussian blur → Otsu threshold → resize (64×128) → morphological cleanup. Produces binary silhouette mask. |
| 8 | `pipeline/steps/live_gei.py` | 🟢 CORE | **Live GEI builder.** Rolling buffer of silhouette frames (max 15). `build()` computes pixel-wise mean → Gait Energy Image. Used in live/video recognition and enrollment video processing. |
| 9 | `pipeline/steps/feature_extraction.py` | 🟢 CORE | **Gait embedding extractor.** Loads ByGaitLight model, reads GEI images, normalizes to binary silhouette (with adaptive thresholding fallback), runs through CNN → 256-dim L2-normalized embedding. Used in enrollment and inference. |
| 10 | `pipeline/steps/appearance_feature_extraction.py` | 🟢 CORE | **Appearance embedding extractor.** Non-CNN feature extraction: HSV color histogram (32×32×8 bins) + Canny edge shape projections. L2-normalized. Used for photo-based enrollment. No ByGaitLight dependency. |
| 11 | `pipeline/steps/matching_step.py` | 🟢 CORE | **Gallery matching engine.** Cosine similarity between query embedding and gallery embeddings. Filters DISABLED/ARCHIVED identities via `_is_active()`. Returns best match + score above threshold, or "UNKNOWN". |

---

## Model Architecture

| # | File | Tag | Role |
|---|---|---|---|
| 12 | `models/architectures/bygait_light.py` | 🟢 CORE | **ByGaitLight CNN.** 3-block Conv2d→BN→ReLU→MaxPool (1→32→64→128 channels), AdaptiveAvgPool → Linear(128, 256) → L2-normalize. Input: [B, 1, 128, 64] grayscale GEI. Output: [B, 256] normalized embedding. |

---

## Storage

| # | File | Tag | Role |
|---|---|---|---|
| 13 | `storage/vector_store.py` | 🟢 CORE | **Gallery persistence.** Saves/loads gallery_features.npy, gallery_labels.npy, gallery_metadata.json. Normalizes metadata entries to include status/enabled/embeddings/updated_at fields. Used by all gallery updaters and recognition pipelines. |

---

## Enrollment

| # | File | Tag | Role |
|---|---|---|---|
| 14 | `enrollment/auto_enrollment_service.py` | 🟢 CORE | **Auto enrollment orchestrator.** Scans `data/new_input/`, processes videos → GEI, classifies images vs videos, routes to gait vs appearance enrollment. Supports watch mode (polling). Skips `_disabled_*` folders and already-enrolled identities. |
| 15 | `enrollment/enrollment_manager.py` | 🟢 CORE | **Enrollment coordinator.** Provides `enroll_gait_person()` and `enroll_appearance_person()`. Collects images, runs feature extraction, delegates to gallery updaters. Validates folder before gait enrollment. |
| 16 | `enrollment/gallery_updater.py` | 🟢 CORE | **Gait gallery writer.** Appends embeddings + labels to `models/live_gallery/`. Preserves existing status when re-enrolling. Status-aware metadata entries. |
| 17 | `enrollment/appearance_gallery_updater.py` | 🟢 CORE | **Appearance gallery writer.** Appends embeddings + labels to `models/appearance_gallery/`. Tags metadata with `source: "PHOTO"`. |
| 18 | `enrollment/enrollment_validator.py` | 🟡 SUPPORT | **Enrollment quality gate.** Validates person folder: ≥5 images, minimum 64×64 resolution, all images readable. Used by gait enrollment only. |

---

## Scripts — Standalone Entry Points

| # | File | Tag | Role |
|---|---|---|---|
| 19 | `scripts/run_auto_enrollment.py` | 🔵 SCRIPT | Entry point for auto enrollment. Supports `--watch`, `--force`, `--scan-interval`, `--gei-frames`, `--video-stride` flags. |
| 20 | `scripts/run_folder_recognition.py` | 🔵 SCRIPT | Entry point for folder recognition. Accepts `--folder`, `--threshold`, `--output`. |
| 21 | `scripts/run_video_recognition.py` | 🔵 SCRIPT | Entry point for video recognition. Accepts `--video`, `--model`, `--threshold`, `--show`, `--output`, `--max-frames`. |
| 22 | `scripts/remove_gallery_identity.py` | 🔵 SCRIPT | Removes an identity from gallery. Filters numpy arrays by label, pops metadata, saves. |
| 23 | `scripts/set_gallery_identity_status.py` | 🔵 SCRIPT | Sets identity status to ACTIVE/DISABLED/ARCHIVED. Updates metadata enabled flag and timestamp. |
| 24 | `scripts/build_gallery.py` | 🔵 SCRIPT | Builds CASIA-B training/evaluation gallery from `data/casia_processed/gei/`. Saves to `models/gallery/`. |
| 25 | `scripts/train_model.py` | 🔵 SCRIPT | Trains ByGaitLight model. Delegates to `training.trainer.Trainer`. Accepts `--epochs`, `--batch-size`, `--lr`, `--max-classes`. |

---

## Security / Utilities

| # | File | Tag | Role |
|---|---|---|---|
| 26 | `security_layer/security_engine.py` | 🟡 SUPPORT | **Security decision engine.** Evaluates each recognition event → severity (INFO/MEDIUM/HIGH) + decision (ALLOW/REVIEW_REQUIRED/SECURITY_ALERT). Logs via SecurityLogger. |
| 27 | `utils/alert_manager.py` | 🟡 SUPPORT | **Alert system.** Logs UNKNOWN_PERSON and LOW_CONFIDENCE events to `outputs/events/alerts.csv`. |
| 28 | `utils/prediction_smoother.py` | 🟡 SUPPORT | **Temporal smoothing.** Per-track deque of last N predictions, returns majority vote. Prevents flickering between identities. |
| 29 | `utils/event_logger.py` | 🟡 SUPPORT | **Event audit trail.** Logs every recognition event (timestamp, track_id, identity, score) to `outputs/events/recognition_log.csv`. |

---

## File Dependency Graph (Simplified)

```
cli.py
  ├── scripts/test_live_recognition.py
  │     └── pipeline/live_recognition.py ─────────────────────────┐
  │           ├── pipeline/steps/tracking.py                      │
  │           ├── pipeline/steps/silhouette_step.py               │
  │           ├── pipeline/steps/live_gei.py                      │
  │           ├── pipeline/steps/matching_step.py ←───────────────┤
  │           ├── models/architectures/bygait_light.py            │
  │           ├── storage/vector_store.py ←───────────────────────┤
  │           ├── streaming/stream_engine.py                      │
  │           ├── utils/prediction_smoother.py                    │
  │           ├── utils/event_logger.py                           │
  │           ├── utils/alert_manager.py                          │
  │           └── security_layer/security_engine.py               │
  │                                                               │
  ├── scripts/run_auto_enrollment.py                              │
  │     └── enrollment/auto_enrollment_service.py                 │
  │           ├── enrollment/enrollment_manager.py                │
  │           │     ├── enrollment/enrollment_validator.py        │
  │           │     ├── enrollment/gallery_updater.py ────────────┤
  │           │     ├── enrollment/appearance_gallery_updater.py  │
  │           │     ├── pipeline/steps/feature_extraction.py ─────┤
  │           │     └── pipeline/steps/appearance_feature_extraction.py
  │           ├── pipeline/steps/tracking.py                      │
  │           ├── pipeline/steps/silhouette_step.py               │
  │           ├── pipeline/steps/live_gei.py                      │
  │           └── storage/vector_store.py ────────────────────────┘
  │
  ├── scripts/run_video_recognition.py
  │     └── pipeline/video_recognition.py
  │           └── (same dependencies as live_recognition.py)
  │
  └── scripts/run_folder_recognition.py
        └── pipeline/folder_recognition.py
              └── pipeline/inference_pipeline.py
                    ├── pipeline/steps/feature_extraction.py
                    ├── pipeline/steps/matching_step.py
                    └── storage/vector_store.py
```
