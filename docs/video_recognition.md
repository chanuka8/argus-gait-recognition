# Video Recognition Pipeline

This document details the video-file processing pipeline, command configurations, and CSV execution outputs.

---

## 1. Pipeline Concept

The video recognition pipeline, implemented in [pipeline/video_recognition.py](file:///e:/ARGUS_AI/pipeline/video_recognition.py), is structurally identical to the live camera pipeline, but reads frames from a pre-recorded file (e.g., MP4 or AVI) rather than a live camera sensor.

This is useful for:
- Forensics investigations (identifying suspects in security footages).
- Offline performance verification and testing.
- Demonstrations and academic evaluations.

---

## 2. Command Configurations

Run the video recognition script using the CLI router:
```bash
python cli.py --mode recognize-video --video data/test_video.mp4 --threshold 0.80 --show --output outputs/reports/video_report.csv
```
Parameters details:
- `--video`: Path to the input video file (required).
- `--threshold`: Recognition matching threshold floor (default: 0.75).
- `--show`: Add this flag to open a real-time playback window with bounding box overlays.
- `--output`: Filepath to write the frame-level matching report (e.g., `outputs/reports/video_report.csv`).

---

## 3. Execution Logs & CSV Report Structure

The pipeline processes the video frame-by-frame. When a tracked individual's rolling GEI is compiled and matched, the details are written to the output CSV file:

```csv
frame,timestamp,track_id,identity,score,decision,camera_id
120,4.00,1,034,0.9412,CONFIRMED_MATCH,video_file
130,4.33,1,034,0.9481,CONFIRMED_MATCH,video_file
135,4.50,2,UNKNOWN,0.5122,UNKNOWN_PERSON,video_file
```

Columns mapping:
- `frame`: Video frame index.
- `timestamp`: Playback position in seconds.
- `track_id`: Persistent track ID from ByteTrack.
- `identity`: Closest matched subject ID or `UNKNOWN`.
- `score`: Cosine similarity score.
- `decision`: Security assessment category (`CONFIRMED_MATCH`, `REVIEW_REQUIRED`, `UNKNOWN_PERSON`).
- `camera_id`: Hardcoded to `video_file`.
