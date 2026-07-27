# ARGUS AI Phase 2: Real CCTV Intelligence Pipeline Deployment Guide

## Architecture

```
CCTV Stream (RTSP / USB / File)
        |
        v
  CameraService (services/camera_service.py)
        |
        v
  PersonDetector (pipeline/detection/person_detector.py)
        |  YOLOv8n person detection (class 0)
        |  Output: [{bbox, confidence}]
        v
  PersonTracker (pipeline/tracking/tracker.py)
        |  ByteTrack multi-person tracker
        |  Output: [{track_id, bbox, timestamp}]
        v
  SilhouetteExtractor (pipeline/silhouette/extractor.py)
        |  Otsu threshold + morphology + aspect ratio filter
        |  Output: 64x128 binary mask
        v
  StreamGEIBuilder (pipeline/gei/stream_gei_builder.py)
        |  Sliding window (15 frames) per track_id
        |  Output: 64x128 GEI uint8 image
        v
  ByGaitLight CNN (models/architectures/bygait_light.py)
        |  256-d embedding
        v
  VectorStore Cosine Matching (storage/vector_store.py)
        |
        v
  Identity Result + Alert System
```

## Configuration

### Detection (`configs/detection.yaml`)

```yaml
model_path: "models/weights/yolov8n.pt"
confidence: 0.4
classes: [0]
```

### GEI (`configs/gei.yaml`)

```yaml
max_frames: 15
min_frames: 10
target_size: [64, 128]
idle_timeout_seconds: 10.0
```

### System (`configs/system.yaml`)

Camera, logging, watchdog, recognition, and service settings remain unchanged from Phase 1.

---

## CCTV to Gait Recognition Process

1. **CameraService** captures frames from RTSP/USB/video source in a background thread.
2. **PersonDetector** runs YOLOv8n inference on each frame, returning bounding boxes for detected persons.
3. **PersonTracker** assigns persistent `track_id` values via ByteTrack, maintaining identity across frames.
4. **SilhouetteExtractor** crops each tracked person, applies Otsu thresholding, morphological cleaning, and normalizes to a 64×128 binary silhouette.
5. **StreamGEIBuilder** accumulates silhouettes per `track_id` in a sliding window. When the minimum frame threshold (10) is reached, it synthesizes a Gait Energy Image.
6. The existing **ByGaitLight** model generates a 256-dimensional embedding from the GEI.
7. The existing **MatchingStep** compares the embedding against the enrolled gallery via cosine similarity.
8. Results pass through the existing **PredictionSmoother**, **AlertManager**, and **SecurityEngine**.

---

## Testing

```bash
venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Test Results (30/30 Passed)

| Test Suite | Tests | Status |
|:-----------|:-----:|:------:|
| test_camera_service | 3 | PASS |
| test_detector | 3 | PASS |
| test_gei_stream | 3 | PASS |
| test_logging | 2 | PASS |
| test_silhouette | 3 | PASS |
| test_tracker | 4 | PASS |
| test_watchdog | 4 | PASS |
| unit.test_evaluation | 2 | PASS |
| unit.test_pipeline | 2 | PASS |
| unit.test_preprocessing | 2 | PASS |
| unit.test_vector_store | 2 | PASS |
