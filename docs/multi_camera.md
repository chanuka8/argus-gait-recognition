# Multi-Camera Mode

This document details the multi-camera parallel processing architecture, thread scheduling, configuration profiles, and execution details.

---

## 1. Parallel Architecture Overview (Option B)

In surveillance setups (e.g. airport gates or border crossings), tracking targets across different viewpoints requires processing multiple feeds in parallel. To prevent thread cross-contamination and track collision, ARGUS implements the **Option B Multi-Camera Architecture** in [pipeline/multi_camera_recognition.py](file:///e:/ARGUS_AI/pipeline/multi_camera_recognition.py):

```
                       +-------------------------+
                       |  MultiStreamEngine      | (Ingests multiple sources)
                       +------------┬------------+
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼ (Camera Feed 1)        ▼ (Camera Feed 2)        ▼ (Camera Feed 3)
     +-----------+            +-----------+            +-----------+
     |  Worker   |            |  Worker   |            |  Worker   | (Parallel Thread Loops)
     +-----------+            +-----------+            +-----------+
     | Isolated: |            | Isolated: |            | Isolated: |
     | - Tracker |            | - Tracker |            | - Tracker |
     | - Silh.   |            | - Silh.   |            | - Silh.   |
     | - GEI Buf |            | - GEI Buf |            | - GEI Buf |
     | - Smoother|            | - Smoother|            | - Smoother|
     +-----------+            +-----------+            +-----------+
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │ (Queries shared read-only resources)
                                    ▼
                      +---------------------------+
                      | Shared Resources:         |
                      | - ByGaitLight Model (eval)|
                      | - VectorStore Gallery     |
                      +---------------------------+
```

### Isolated vs. Shared Components
- **Isolated per-camera thread state:** Each camera worker gets its own `YOLOv8` tracker, `ByteTrack` context, `SilhouetteStep` segmenter, rolling `LiveGEI` queues, and a dedicated `PredictionSmoother`. This isolates state and prevents track IDs collisions across cameras.
- **Shared read-only resources:** All worker threads query the same global `ByGaitLight` PyTorch weights (running in evaluation mode `model.eval()`) and load the same database memory mapping in `VectorStore`. This maintains high memory efficiency.

---

## 2. Ingestion & Thread Safety
- The `MultiStreamEngine` grabs frames inside background worker threads, placing them into thread-safe queues.
- Recognition alerts, events logging, and incident files operations use thread-safe lock objects (`threading.Lock()`) to prevent file corruption.

---

## 3. Configuration settings: `cameras.yaml`

Settings are configured inside [configs/cameras.yaml](file:///e:/ARGUS_AI/configs/cameras.yaml):
- `cameras`: List of camera entries defining:
  - `id`: Unique camera identifier.
  - `source`: Index or RTSP URI.
  - `width` / `height`: Resolution.
  - `enabled`: Process this source if true.
  - `location`: Optional location name (e.g. "Main Entrance") used for reporting. Defaults to "Unknown Location".
- `orchestrator`: Set global execution behaviors:
  - `show_gui`: Open display window overlays.
  - `max_cameras`: Maximum channels allowed in parallel.
  - `queue_max_size`: Thread-queue buffer sizes.


---

## 4. Multi-Camera Commands

Start the parallel multi-camera pipeline:
```bash
python cli.py --mode multi-camera
```
*Alternative (Makefile):*
```bash
make multi-camera
```

During execution:
- Display viewports open for each active camera (e.g. `ARGUS - cam_01`).
- Press `Q` inside any display viewport to stop.
- On exit, performance statistics (frames read, frames dropped, capture errors) are printed to the console.
