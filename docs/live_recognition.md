# Live Camera Recognition

This document details the real-time live webcam surveillance pipeline, tracking components, and visualization overlays.

---

## 1. Pipeline Architecture

The live recognition pipeline, implemented in [pipeline/live_recognition.py](file:///e:/ARGUS_AI/pipeline/live_recognition.py), processes frames captured from camera devices through sequential steps:

```
+---------------+      +-------------+      +---------------+      +-------------+
| Camera Stream | ---> |   YOLOv8    | ---> |   ByteTrack   | ---> | Silhouette  |
| (StreamEngine)|      | (Detection) |      | (Association) |      | Segmenter   |
+---------------+      +-------------+      +---------------+      +-------------+
                                                                          │
+---------------+      +-------------+      +---------------+             │
| Real-time UI  | <--- | Prediction  | <--- | ByGaitLight   | <--- [Rolling GEI  |
|  & CSV Log    |      |  Smoother   |      |  CNN Model    |      |   Buffer]   |
+---------------+      +-------------+      +---------------+      +-------------+
```

1. **Ingestion:** [StreamEngine](file:///e:/ARGUS_AI/streaming/stream_engine.py) grabs raw frames in a separate thread to maintain ingestion frame rates.
2. **Person Detection:** YOLOv8 runs detection on class `0` (person) on each frame.
3. **Association Tracking:** ByteTrack maps detected bounding boxes across frames to assign a unique, persistent `Track ID` to each individual.
4. **Silhouette Segmentation:** The bounding box crop is converted to grayscale, blurred, segmented via Otsu's adaptive thresholding, and morphological operations are applied to output a binary mask.
5. **Rolling GEI Buffer:** [LiveGEI](file:///e:/ARGUS_AI/pipeline/steps/live_gei.py) maintains a queue of up to 15 silhouette frames per tracked ID. When full, it averages the frames to build a dynamic Gait Energy Image.
6. **CNN Inference & Lookup:** The GEI is passed through the `ByGaitLight` CNN to produce an embedding, which is compared to the gallery.
7. **Prediction Smoothing:** [PredictionSmoother](file:///e:/ARGUS_AI/utils/prediction_smoother.py) stores prediction history for each track to filter out noise, using a majority voting scheme to prevent label flickering.

---

## 2. Thresholds & CCTV Overlay Status Policy

Match results are evaluated against configurations in [configs/inference.yaml](file:///e:/ARGUS_AI/configs/inference.yaml) and mapped to professional CCTV status tiers:
- **DETECTION:** Person detected, but not enough gait evidence accumulated yet. (Overlay color: Red `[0, 0, 255]`)
- **TRACKING:** Stable/predicted tracked person, recognition is in-progress/not finalized. (Overlay color: Orange `[0, 165, 255]`)
- **UNKNOWN:** Recognition done, no valid biometric gallery match. (Overlay color: Green `[0, 255, 0]`)
- **CONFIRMED:** Known enrolled identity successfully matched. (Overlay color: Green `[0, 255, 0]`)

Label format on overlays:
```
[camera_id] T{track_id} | STATUS | name_or_identity | score
```

---

## 3. Running Live Recognition (System Mode)

To run the system with health checks, auto-enrollment watcher, and live camera recognition, execute:
```bash
python cli.py --mode system
```
*Alternative (Makefile):*
```bash
make live
```

- Display viewports will open showing the video stream with professional CCTV analytics overlays.
- Bounding boxes will outline tracked subjects according to the 4-tier status system.
- **Controls:** Press `Q` inside the viewport window to terminate execution.
- Auto-detection reports are written in CSV and JSONL formats under `outputs/detection_reports/` for configured statuses.
- Full security events are logged under `outputs/events/recognition_log.csv`.

