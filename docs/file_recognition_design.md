# ARGUS File-Based Gait Recognition — Design Report

> **Step 15A — Analysis Only**
> Generated: 2026-06-11
> Status: ANALYSIS COMPLETE — NO CODE CHANGES MADE

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Audit](#2-current-architecture-audit)
3. [Module Reuse Analysis](#3-module-reuse-analysis)
4. [New Capability Requirements](#4-new-capability-requirements)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Recommended New Files](#6-recommended-new-files)
7. [Recommended CLI Commands](#7-recommended-cli-commands)
8. [Implementation Path](#8-implementation-path)
9. [Risk Assessment](#9-risk-assessment)
10. [Final Architecture](#10-final-architecture)

---

## 1. Executive Summary

The goal is to add two new recognition modes to ARGUS:

| Mode | Input | Pipeline |
|------|-------|----------|
| **Video File Recognition** | `.mp4` / `.avi` / video path | Video → Detection → Tracking → Silhouette → GEI → Recognition |
| **Image Folder Recognition** | Folder of pre-computed GEI images | GEI Images → Recognition |

The current system only supports:
- **Live webcam** (`LiveRecognitionPipeline`) — full detection → tracking → silhouette → GEI → recognition loop from camera
- **Single-image inference** (`InferencePipeline`) — one pre-computed GEI image → embedding → gallery match

### Key Findings

- **Video file recognition** can reuse ~80% of `LiveRecognitionPipeline` logic. The primary difference is the frame source (file vs. webcam) and the output target (report vs. live display).
- **Image folder recognition** can reuse ~90% of `InferencePipeline` logic. It simply needs to iterate over a folder of GEI images and aggregate results.
- **Zero existing modules need modification.** All new functionality can be added as new files alongside existing code.
- **Estimated effort:** 2–3 implementation steps (low complexity).

---

## 2. Current Architecture Audit

### 2.1 Live Recognition Pipeline

**File:** [`pipeline/live_recognition.py`](file:///e:/ARGUS_AI/pipeline/live_recognition.py)
**Class:** `LiveRecognitionPipeline`

| Aspect | Detail |
|--------|--------|
| Frame source | `StreamEngine` (webcam, hardcoded `source=0`) |
| Detection + Tracking | `TrackingStep` (YOLOv8n + ByteTrack) |
| Silhouette | `SilhouetteStep.extract_from_crop()` |
| GEI building | `LiveGEI` (rolling buffer, 15 frames) |
| Embedding | `ByGaitLight` model, inline `_gei_to_embedding()` |
| Matching | `MatchingStep.match()` against `VectorStore` gallery |
| Smoothing | `PredictionSmoother` (majority-vote over history) |
| Security | `SecurityEngine.evaluate()` |
| Events | `EventLogger.log()` |
| Alerts | `AlertManager.evaluate()` |
| Output | `cv2.imshow()` windows (live display), `cv2.waitKey()` loop |

**Coupling issues for reuse:**
- `StreamEngine` is instantiated in `__init__` with no option to pass a video file path
- `cv2.imshow()` calls are scattered throughout (not optional)
- The `run()` method is a monolithic while-loop tied to camera input
- Model loading is duplicated (also in `FeatureExtractionStep`)

### 2.2 Inference Pipeline

**File:** [`pipeline/inference_pipeline.py`](file:///e:/ARGUS_AI/pipeline/inference_pipeline.py)
**Class:** `InferencePipeline`

| Aspect | Detail |
|--------|--------|
| Input | Single image path (expects a pre-computed GEI image) |
| Feature extraction | `FeatureExtractionStep` (loads `ByGaitLight`, reads grayscale image) |
| Matching | `MatchingStep.match()` against `VectorStore` gallery |
| Output | `dict` with `identity` and `score` |

**Coupling issues for reuse:**
- Only handles a single image at a time
- No batch/folder processing
- No aggregation or summary reporting
- Otherwise clean and reusable

### 2.3 GEI Builder

Two separate implementations exist:

| Module | Location | Usage |
|--------|----------|-------|
| `GEIBuilder` | [`preprocessing/gei_builder.py`](file:///e:/ARGUS_AI/preprocessing/gei_builder.py) | Offline dataset building (CASIA-B preprocessing) |
| `LiveGEI` | [`pipeline/steps/live_gei.py`](file:///e:/ARGUS_AI/pipeline/steps/live_gei.py) | Real-time streaming GEI from webcam |

Both are functionally equivalent: accumulate silhouette frames → average → scale to uint8. `GEIBuilder` uses `max_frames=30` and `add_frame()`. `LiveGEI` uses `max_frames=15` and `add()`. Either can be reused for video file processing.

### 2.4 Gallery Matching

**File:** [`pipeline/steps/matching_step.py`](file:///e:/ARGUS_AI/pipeline/steps/matching_step.py)
**Class:** `MatchingStep`

- Cosine similarity matching
- Threshold-based UNKNOWN detection (default 0.75)
- **Fully reusable as-is** — no source-specific coupling

### 2.5 Silhouette Extraction

Two separate implementations exist:

| Module | Location | Method |
|--------|----------|--------|
| `SilhouetteStep` | [`pipeline/steps/silhouette_step.py`](file:///e:/ARGUS_AI/pipeline/steps/silhouette_step.py) | Otsu thresholding on cropped person (used in live pipeline) |
| `SilhouetteExtractor` | [`preprocessing/silhouette_extractor.py`](file:///e:/ARGUS_AI/preprocessing/silhouette_extractor.py) | Background subtraction MOG2 (stateful, used in preprocessing) |

For **video file recognition**, `SilhouetteStep.extract_from_crop()` is the correct choice — it operates on per-person crops from detection, matching the live pipeline approach.

### 2.6 CLI Integration

**File:** [`cli.py`](file:///e:/ARGUS_AI/cli.py)

- Uses `argparse` with `--mode` flag
- 16 existing modes: `health`, `preprocess`, `train`, `gallery`, `evaluate`, `benchmark`, `live`, `api`, `tests`, `integration-tests`, `security-test`, `confidence-test`, `visualizer-test`, `streaming-test`, `full-check`, `demo`
- Each mode calls a function that runs a subprocess via `scripts/`
- **Pattern is clear:** add new mode → add new function → add new script

### 2.7 Supporting Modules

| Module | File | Reusable? |
|--------|------|-----------|
| `StreamEngine` | [`streaming/stream_engine.py`](file:///e:/ARGUS_AI/streaming/stream_engine.py) | **Yes** — `cv2.VideoCapture(source)` already accepts file paths via the `source` parameter |
| `TrackingStep` | [`pipeline/steps/tracking.py`](file:///e:/ARGUS_AI/pipeline/steps/tracking.py) | **Yes, unchanged** — frame-agnostic |
| `DetectionStep` | [`pipeline/steps/detection.py`](file:///e:/ARGUS_AI/pipeline/steps/detection.py) | **Yes, unchanged** — frame-agnostic |
| `FeatureExtractionStep` | [`pipeline/steps/feature_extraction.py`](file:///e:/ARGUS_AI/pipeline/steps/feature_extraction.py) | **Yes, unchanged** — path-agnostic |
| `VectorStore` | [`storage/vector_store.py`](file:///e:/ARGUS_AI/storage/vector_store.py) | **Yes, unchanged** |
| `PredictionSmoother` | [`utils/prediction_smoother.py`](file:///e:/ARGUS_AI/utils/prediction_smoother.py) | **Yes** — useful for video, skip for folder |
| `EventLogger` | [`utils/event_logger.py`](file:///e:/ARGUS_AI/utils/event_logger.py) | **Yes, unchanged** |
| `SecurityEngine` | [`security_layer/security_engine.py`](file:///e:/ARGUS_AI/security_layer/security_engine.py) | **Yes, unchanged** |
| `PipelineFactory` | [`pipeline/pipeline_factory.py`](file:///e:/ARGUS_AI/pipeline/pipeline_factory.py) | **Needs update** — register new pipeline modes |
| `ByGaitLight` | [`models/architectures/bygait_light.py`](file:///e:/ARGUS_AI/models/architectures/bygait_light.py) | **Yes, unchanged** |

---

## 3. Module Reuse Analysis

### A. Modules Reusable UNCHANGED

| Module | Used By |
|--------|---------|
| `TrackingStep` | Video file recognition |
| `SilhouetteStep` | Video file recognition |
| `LiveGEI` | Video file recognition |
| `MatchingStep` | Both video and folder recognition |
| `FeatureExtractionStep` | Folder recognition |
| `VectorStore` | Both |
| `ByGaitLight` | Both (via existing steps) |
| `PredictionSmoother` | Video file recognition |
| `EventLogger` | Both |
| `SecurityEngine` | Both (optional for folder) |
| `AlertManager` | Both (optional for folder) |
| `SpeedController` | Video file recognition (optional) |

### B. New Modules Required

| New Module | Purpose |
|------------|---------|
| `pipeline/video_recognition.py` | Video file → full recognition pipeline |
| `pipeline/folder_recognition.py` | Image folder → batch recognition pipeline |
| `scripts/run_video_recognition.py` | CLI entry point for video mode |
| `scripts/run_folder_recognition.py` | CLI entry point for folder mode |

### C. Can Video-File Recognition Reuse `live_recognition.py`?

**No — not directly.** But it can reuse ~80% of the same components.

Reasons it cannot directly subclass or call `LiveRecognitionPipeline`:
1. `__init__` hardcodes `StreamEngine()` with `source=0` (webcam)
2. `run()` has an infinite `while True` loop with `cv2.waitKey()` for quit — video files need finite iteration with end-of-file detection
3. `cv2.imshow()` calls are embedded inline — video file processing should support headless (no display) mode
4. Live pipeline prints "Camera not found" — semantically wrong for files

**Recommendation:** Create a new `VideoRecognitionPipeline` class that reuses the same step objects (`TrackingStep`, `SilhouetteStep`, `LiveGEI`, `MatchingStep`) but with a video-file-aware run loop. This is safer than modifying `LiveRecognitionPipeline` because it avoids regression risk on the working live system.

### D. Can Image-Folder Recognition Reuse `inference_pipeline.py`?

**Partially — yes.** `InferencePipeline.predict(image_path)` is the core building block.

The folder recognizer can:
1. Instantiate `InferencePipeline` once (loads model + gallery once)
2. Iterate over all GEI images in the folder
3. Call `pipeline.predict(path)` for each
4. Aggregate and report results

**Recommendation:** Create a thin `FolderRecognitionPipeline` that wraps `InferencePipeline` and adds folder iteration + result aggregation. This avoids modifying `InferencePipeline` entirely.

### E. Expected Implementation Effort

| Component | Estimated Lines | Complexity |
|-----------|----------------|------------|
| `VideoRecognitionPipeline` | ~180–250 | Medium — reuses all steps, new run loop |
| `FolderRecognitionPipeline` | ~80–120 | Low — wraps InferencePipeline |
| `scripts/run_video_recognition.py` | ~30–40 | Low |
| `scripts/run_folder_recognition.py` | ~25–35 | Low |
| CLI updates (`cli.py`) | ~30 lines added | Low |
| `PipelineFactory` update | ~6 lines added | Trivial |
| Config updates | ~8 lines | Trivial |
| **Total** | **~360–480 lines** | **Low–Medium** |

### F. Risks of Introducing File-Based Recognition

| Risk | Severity | Mitigation |
|------|----------|------------|
| Regression in live pipeline | **HIGH** | Never modify `live_recognition.py` — add new files only |
| Model loading duplication | LOW | Video pipeline can use same `_load_model()` pattern; not ideal but safe |
| Memory pressure on long videos | MEDIUM | Process frames sequentially; clear GEI buffers after recognition; cap buffer sizes |
| ByteTrack state contamination | LOW | Fresh `TrackingStep()` per video file; no shared state with live |
| Gallery loading overhead | LOW | Gallery is loaded once at init; same pattern as live/inference |
| cv2.imshow on headless server | LOW | Make display optional via `--show` flag; default to headless for video |
| Inconsistent recognition thresholds | LOW | Share defaults from config; pass same threshold to both pipelines |
| Video codec compatibility | LOW | OpenCV handles common codecs; add validation in `__init__` |

---

## 4. New Capability Requirements

### 4.1 Video File Recognition

```
Input:  Path to video file (.mp4, .avi, .mov, .mkv)
Output: Console summary + optional CSV report + optional live display
```

**Processing steps:**
1. Open video file via `cv2.VideoCapture(video_path)`
2. Read frames sequentially
3. For each frame: detect → track → crop → silhouette → accumulate GEI
4. When GEI ready: embed → match against gallery
5. Apply prediction smoothing per track
6. Log events, evaluate security
7. After all frames: print summary report
8. Optional: display frames with annotations via `--show` flag

### 4.2 Image Folder Recognition

```
Input:  Path to folder containing GEI images (.png, .jpg)
Output: Console summary table + optional CSV report
```

**Processing steps:**
1. Scan folder for image files
2. For each image: extract embedding → match against gallery
3. Collect all results
4. Print summary table (filename, predicted identity, score)
5. Optional: save results to CSV

---

## 5. Data Flow Diagrams

### 5.1 Current Live Pipeline

```
┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────────┐
│ Webcam  │───►│ TrackingStep │───►│SilhouetteStep │───►│   LiveGEI     │
│ (cam 0) │    │ (YOLO+ByteT) │    │(Otsu on crop) │    │ (15-frame buf)│
└─────────┘    └──────────────┘    └───────────────┘    └──────┬────────┘
                                                               │ build()
                                                               ▼
┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│cv2.imshow│◄───│SecurityEngine│◄───│ MatchingStep  │◄───│ByGaitLight  │
│(display) │    │  (alerts)    │    │(cosine match) │    │(embedding)  │
└──────────┘    └──────────────┘    └───────────────┘    └─────────────┘
```

### 5.2 Proposed Video File Pipeline

```
┌───────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────────┐
│Video File │───►│ TrackingStep │───►│SilhouetteStep │───►│   LiveGEI     │
│(.mp4/avi) │    │ (YOLO+ByteT) │    │(Otsu on crop) │    │ (15-frame buf)│
└───────────┘    └──────────────┘    └───────────────┘    └──────┬────────┘
                                                                 │ build()
                                                                 ▼
 ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
 │  Report  │◄───│SecurityEngine│◄───│ MatchingStep  │◄───│ByGaitLight  │
 │(console/ │    │  (evaluate)  │    │(cosine match) │    │(embedding)  │
 │  CSV)    │    └──────────────┘    └───────────────┘    └─────────────┘
 └──────────┘
       ▲
       │ optional
 ┌──────────┐
 │cv2.imshow│
 │(--show)  │
 └──────────┘
```

### 5.3 Proposed Image Folder Pipeline

```
┌────────────┐    ┌────────────────────┐    ┌───────────────┐    ┌──────────┐
│GEI Folder  │───►│FeatureExtraction   │───►│ MatchingStep  │───►│  Report  │
│(*.png/jpg) │    │Step (ByGaitLight)  │    │(cosine match) │    │(console/ │
└────────────┘    └────────────────────┘    └───────────────┘    │  CSV)    │
                                                                 └──────────┘
```

### 5.4 Complete Architecture After Implementation

```
                            ┌─────────────────────────────────────┐
                            │           ARGUS CLI (cli.py)        │
                            │  --mode live | video | folder | ... │
                            └──────────┬──────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
          ┌─────────────────┐ ┌────────────────┐ ┌────────────────┐
          │LiveRecognition  │ │VideoRecognition│ │FolderRecognition│
          │Pipeline         │ │Pipeline        │ │Pipeline        │
          │(webcam)         │ │(video file)    │ │(GEI folder)    │
          └────────┬────────┘ └───────┬────────┘ └───────┬────────┘
                   │                  │                  │
        ┌──────────┴──────────────────┘                  │
        │    Shared Steps                                │
        ▼                                                ▼
  ┌──────────────┐  ┌───────────────┐           ┌────────────────────┐
  │TrackingStep  │  │SilhouetteStep │           │FeatureExtraction   │
  │(YOLO+ByteT) │  │(Otsu thresh)  │           │Step (ByGaitLight)  │
  └──────────────┘  └───────────────┘           └────────────────────┘
        │                  │                             │
        ▼                  ▼                             │
  ┌──────────────┐  ┌───────────────┐                   │
  │   LiveGEI    │  │  ByGaitLight  │◄──────────────────┘
  │(frame buffer)│  │  (embedding)  │
  └──────────────┘  └───────────────┘
                           │
                           ▼
                    ┌───────────────┐
                    │ MatchingStep  │◄─── VectorStore (gallery)
                    │(cosine sim)   │
                    └───────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌────────────┐ ┌──────────┐ ┌───────────┐
       │EventLogger │ │Security  │ │  Report   │
       │            │ │Engine    │ │(console/  │
       └────────────┘ └──────────┘ │  CSV)     │
                                   └───────────┘
```

---

## 6. Recommended New Files

### 6.1 Pipeline Modules

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `pipeline/video_recognition.py` | `VideoRecognitionPipeline` class — video file → full recognition with per-track GEI, smoothing, security, and report output | ~200 |
| `pipeline/folder_recognition.py` | `FolderRecognitionPipeline` class — GEI folder → batch inference with aggregated results | ~100 |

### 6.2 Script Entry Points

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `scripts/run_video_recognition.py` | Argparse wrapper: `--video`, `--threshold`, `--show`, `--output` | ~35 |
| `scripts/run_folder_recognition.py` | Argparse wrapper: `--folder`, `--threshold`, `--output` | ~30 |

### 6.3 Files to Update (Append Only)

| File | Change |
|------|--------|
| `cli.py` | Add `recognize-video` and `recognize-folder` modes (~30 lines) |
| `pipeline/pipeline_factory.py` | Register `"video"` and `"folder"` modes (~6 lines) |
| `configs/mode_config.yaml` | Add `video` and `folder` to `available_modes` (~2 lines) |

---

## 7. Recommended CLI Commands

### 7.1 Video File Recognition

```bash
# Basic usage
python cli.py --mode recognize-video --video path/to/walking_clip.mp4

# With options
python cli.py --mode recognize-video --video path/to/clip.mp4 --threshold 0.80 --show --output results.csv
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | — | Path to video file |
| `--threshold` | No | `0.75` | Match confidence threshold |
| `--show` | No | `False` | Display annotated frames (cv2.imshow) |
| `--output` | No | — | Save results to CSV file |
| `--gei-frames` | No | `15` | Number of frames for GEI construction |

### 7.2 Image Folder Recognition

```bash
# Basic usage
python cli.py --mode recognize-folder --folder path/to/gei_images/

# With options
python cli.py --mode recognize-folder --folder path/to/gei_images/ --threshold 0.80 --output results.csv
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--folder` | Yes | — | Path to folder with GEI images |
| `--threshold` | No | `0.75` | Match confidence threshold |
| `--output` | No | — | Save results to CSV file |

---

## 8. Implementation Path

### Minimal-Risk Implementation Order

```
Step 1: FolderRecognitionPipeline     (SAFEST — no new dependencies)
Step 2: run_folder_recognition.py     (script entry point)
Step 3: VideoRecognitionPipeline      (MODERATE — reuses many existing steps)
Step 4: run_video_recognition.py      (script entry point)
Step 5: CLI integration               (register both new modes)
Step 6: PipelineFactory update        (register for programmatic access)
Step 7: Config update                 (add to available_modes)
```

**Why this order?**
1. **Folder recognition is simpler** — it wraps `InferencePipeline.predict()` with iteration. No detection, tracking, or GEI building needed. If anything breaks, the blast radius is minimal.
2. **Video recognition is more complex** — it instantiates `TrackingStep`, `SilhouetteStep`, `LiveGEI`, model loading. Building it second means we've already validated the folder path works.
3. **CLI and factory updates come last** — they're trivial append-only changes with no regression risk.

### Implementation Principles

1. **Add, never modify** — all new functionality in new files
2. **Same step objects** — reuse `TrackingStep`, `SilhouetteStep`, `LiveGEI`, `MatchingStep` as-is
3. **Same model loading pattern** — follow `_load_model()` from `LiveRecognitionPipeline`
4. **Same gallery loading** — `VectorStore().load()` pattern
5. **No import changes in existing files** — new files import existing modules, not vice versa
6. **Headless by default** — video file processing should not require a display; `--show` is opt-in

---

## 9. Risk Assessment

### 9.1 Regression Risk Matrix

| Existing Feature | Risk Level | Reason |
|-----------------|------------|--------|
| Live webcam recognition | **NONE** | `live_recognition.py` is never modified |
| Single-image inference | **NONE** | `inference_pipeline.py` is never modified |
| Gallery building | **NONE** | `build_gallery.py` is never modified |
| Training pipeline | **NONE** | Completely unrelated |
| Enrollment system | **NONE** | Completely unrelated |
| API server | **NONE** | No route changes |
| All existing CLI modes | **NONE** | Existing mode map is only appended to |

### 9.2 New Feature Risk

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Video codec not supported by OpenCV | Medium | Low | Validate file can be opened in `__init__`; fail fast with clear error |
| Very long video exhausts memory | Medium | Low | Cap per-track buffer sizes; same `max_frames` pattern as `LiveGEI` |
| ByteTrack assigns unstable IDs for short clips | Low | Medium | Prediction smoothing handles this; same as live pipeline |
| Empty folder or no valid images | Low | Medium | Validate folder contents; fail fast with clear message |
| Model checkpoint not found | Medium | Low | Same `_load_model()` with `FileNotFoundError`; already proven pattern |

---

## 10. Final Architecture

### 10.1 File Tree (New and Modified)

```
e:\ARGUS_AI\
├── pipeline/
│   ├── live_recognition.py          # UNCHANGED
│   ├── inference_pipeline.py        # UNCHANGED
│   ├── video_recognition.py         # NEW — VideoRecognitionPipeline
│   ├── folder_recognition.py        # NEW — FolderRecognitionPipeline
│   ├── pipeline_factory.py          # APPEND — register "video", "folder"
│   └── steps/                       # ALL UNCHANGED
│       ├── tracking.py
│       ├── silhouette_step.py
│       ├── live_gei.py
│       ├── matching_step.py
│       ├── feature_extraction.py
│       └── detection.py
├── scripts/
│   ├── run_video_recognition.py     # NEW — video CLI entry point
│   └── run_folder_recognition.py    # NEW — folder CLI entry point
├── cli.py                           # APPEND — two new modes
├── configs/
│   └── mode_config.yaml             # APPEND — two new modes
└── docs/
    └── file_recognition_design.md   # THIS DOCUMENT
```

### 10.2 Class Dependency Map

```
VideoRecognitionPipeline
├── uses TrackingStep          (from pipeline/steps/tracking.py)
├── uses SilhouetteStep        (from pipeline/steps/silhouette_step.py)
├── uses LiveGEI               (from pipeline/steps/live_gei.py)
├── uses MatchingStep          (from pipeline/steps/matching_step.py)
├── uses ByGaitLight           (from models/architectures/bygait_light.py)
├── uses VectorStore           (from storage/vector_store.py)
├── uses PredictionSmoother    (from utils/prediction_smoother.py)
├── uses EventLogger           (from utils/event_logger.py)
└── uses SecurityEngine        (from security_layer/security_engine.py)

FolderRecognitionPipeline
├── uses InferencePipeline     (from pipeline/inference_pipeline.py)
│   ├── uses FeatureExtractionStep
│   ├── uses MatchingStep
│   └── uses VectorStore
└── (optional) uses EventLogger
```

### 10.3 Module Responsibility Summary

| Module | Responsibility |
|--------|---------------|
| `VideoRecognitionPipeline` | Open video file, iterate frames, run detection → tracking → silhouette → GEI → embedding → match → smooth → log. Print final report. Optional display. |
| `FolderRecognitionPipeline` | Scan folder for GEI images, run InferencePipeline.predict() per image, aggregate results, print summary table. Optional CSV export. |
| `run_video_recognition.py` | Parse CLI args (`--video`, `--threshold`, `--show`, `--output`), instantiate `VideoRecognitionPipeline`, call `run()`. |
| `run_folder_recognition.py` | Parse CLI args (`--folder`, `--threshold`, `--output`), instantiate `FolderRecognitionPipeline`, call `run()`. |

---

## Appendix: StreamEngine Compatibility Note

[`StreamEngine`](file:///e:/ARGUS_AI/streaming/stream_engine.py) wraps `cv2.VideoCapture(source)`. Its constructor already accepts any `source` parameter — integer for camera, string for file path. However, it also calls `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)`, which is only meaningful for cameras.

**Decision:** The `VideoRecognitionPipeline` should NOT use `StreamEngine`. Instead, it should directly use `cv2.VideoCapture(video_path)` to avoid the unnecessary resolution-setting calls and to keep a clean separation between live-stream and file-based processing. This also avoids any temptation to modify `StreamEngine` and risk breaking the live pipeline.

---

*End of Analysis — Step 15A Complete*
*No code has been modified. No files have been edited except this report.*
