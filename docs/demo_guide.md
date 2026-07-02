# Demonstration and Evaluation Guide

This guide describes how to run and configure the runtime modes of the ARGUS AI prototype.

---

## 1. Prerequisites and Setup

Ensure you have initialized the virtual environment and installed all dependencies as outlined in the [README setup instructions](file:///e:/ARGUS_AI/README.md#4-setup-and-installation):

```bash
# Verify environment health and check file configurations
python cli.py --mode production-test
```

---

## 2. Interactive and Real-time Modes

### A. Pre-recorded Video Recognition
Processes an input video file, extracts silhouettes, computes GEIs, maps identities, and displays the tracking overlay window.
```bash
python cli.py --mode recognize-video --video "path/to/video.mp4" --threshold 0.75 --show
```
- `--video`: Path to the target MP4/AVI file.
- `--threshold`: Similarity threshold for gallery confirmation (0.0 to 1.0).
- `--show`: Display the processing visual overlay window.

### B. Live Webcam Recognition
Streams from a local camera/webcam, runs detection, computes dynamic GEIs over a rolling window, and shows identity matching.
```bash
python cli.py --mode live
```
Press `Q` in the display window to exit the stream loop.

### C. Multi-Camera Simulation
Processes multiple parallel camera streams using thread-isolated pipeline configurations sharing a single model instance.
```bash
python cli.py --mode multi-camera
```
- Configured via [configs/cameras.yaml](file:///e:/ARGUS_AI/configs/cameras.yaml).
- Each camera operates in its own thread and appends detections to the global security logs.

---

## 3. Automation and Evaluation Modes

### A. Folder Watch Auto-Enrollment
Monitors an input folder for new person silhouette or video captures and auto-registers them into the active biometric gallery database.
```bash
python cli.py --mode auto-enroll
```

### B. Research Evaluation Sweep
Runs a cross-evaluation sweep across view angles and database partitions.
```bash
python cli.py --mode research-eval
```
This is useful for examining accuracy benchmarks under controlled research environments (like CASIA-B testing partitions).

---

## 4. Understanding Outputs

All runtime modes dump telemetry, log files, and visual crops inside the [outputs/](file:///e:/ARGUS_AI/outputs/) directory:

- **Logs:** Security events are appended in a tabular format at `outputs/detection_reports/detections.csv`.
- **JSON Telemetry:** Detailed frame metadata is stored in `outputs/detection_reports/detections.jsonl`.
- **Visual Snapshots:** If a subject is identified or classified as unknown, cropped person frames are saved to `outputs/detection_reports/snapshots/` for record-keeping.
