# ARGUS AI Phase 2: Real CCTV Intelligence Pipeline Implementation Plan

## Current Pipeline Flow (Phase 1 Baseline)

```
CameraService (RTSP / USB / File)
  → ArgusService (Background Orchestration)
    → LiveRecognitionPipeline (YOLOv8 + ByteTrack inside)
      → Bounding Box Stabilizer
      → Silhouette Extraction from Crop
      → GEI Builder (LiveGEI rolling buffer)
      → ByGaitLight Neural Network (256-d Embedding)
      → Cosine Similarity Matcher (VectorStore)
      → Detection Logger & Alert Manager
```

## Phase 2 Architectural Objectives

Decouple and modularize the real-time CCTV vision pipeline into dedicated, thread-safe, config-driven pipeline components (`detection`, `tracking`, `silhouette`, `gei`) while maintaining full backward compatibility with the existing `ByGaitLight` CNN model, `ArcFace` metric learning, and `ArgusService` background orchestrator.

## Integration Points

1. **Detection Interface (`pipeline/detection/person_detector.py`)**:
   - Consumes raw video frames from `CameraService`.
   - Uses `configs/detection.yaml` for weights, confidence thresholds, and device settings.
   - Emits standardized list: `[{"bbox": [x1, y1, x2, y2], "confidence": score, "class_id": 0}]`.

2. **Tracking Engine (`pipeline/tracking/tracker.py`)**:
   - Consumes detection outputs per frame.
   - Integrates `ByteTrack` / `supervision` tracking state per camera stream.
   - Maintains `track_id`, `bbox`, `timestamp`, and cleans up inactive tracks.

3. **Silhouette Extractor (`pipeline/silhouette/extractor.py`)**:
   - Crops detected person bounding box.
   - Applies Otsu thresholding, Gaussian blur, morphological filtering, aspect ratio verification, and height normalization.
   - Emits normalized $64 \times 128$ binary silhouette mask array.

4. **Stream GEI Builder (`pipeline/gei/stream_gei_builder.py`)**:
   - Manages sliding window silhouette queues per active `track_id`.
   - Uses `configs/gei.yaml` for frame bounds ($N=15$).
   - Synthesizes 2D GEI image array $(64, 128, \text{uint8})$.

5. **ARGUS Service Orchestration (`services/argus_service.py`)**:
   - Connects `CameraService` -> `PersonDetector` -> `PersonTracker` -> `SilhouetteExtractor` -> `StreamGEIBuilder` -> `LiveRecognitionPipeline` / `VectorStore`.

---

## Files to Create

| File | Purpose |
|------|---------|
| `configs/detection.yaml` | Detection thresholds, model weights, execution device |
| `configs/gei.yaml` | GEI sliding window frame count, resolution, aspect bounds |
| `pipeline/detection/__init__.py` | Detection module package init |
| `pipeline/detection/person_detector.py` | Standalone thread-safe Person Detector |
| `pipeline/tracking/__init__.py` | Tracking module package init |
| `pipeline/tracking/tracker.py` | Multi-person tracker with ID persistence |
| `pipeline/silhouette/__init__.py` | Silhouette module package init |
| `pipeline/silhouette/extractor.py` | Crop to binary silhouette extractor |
| `pipeline/gei/__init__.py` | GEI module package init |
| `pipeline/gei/stream_gei_builder.py` | Multi-track sliding window GEI builder |
| `tests/test_detector.py` | Unit tests for PersonDetector |
| `tests/test_tracker.py` | Unit tests for PersonTracker |
| `tests/test_silhouette.py` | Unit tests for SilhouetteExtractor |
| `tests/test_gei_stream.py` | Unit tests for StreamGEIBuilder |
| `docs/PHASE2_DEPLOYMENT.md` | Phase 2 deployment & pipeline guide |

## Files to Modify

| File | Purpose |
|------|---------|
| `services/argus_service.py` | Integrate new modular pipeline into background recognition worker |

---

## Pipeline Execution Diagram (Phase 2 Target)

```
                          CameraService
                                |
                                v
                          PersonDetector (pipeline/detection)
                                |
                                v
                          PersonTracker (pipeline/tracking)
                                |
                                v
                        SilhouetteExtractor (pipeline/silhouette)
                                |
                                v
                        StreamGEIBuilder (pipeline/gei)
                                |
                                v
                      Existing ByGaitLight Gait Model
                                |
                                v
                      Matching Engine & Vector Store
```
