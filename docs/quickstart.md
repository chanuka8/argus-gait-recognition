# Quickstart Guide

This guide provides a step-by-step sequence to verify your installation, set up a sample gallery, enroll a subject, and run recognition pipelines in under 5 minutes.

---

## Step 1: Verify Installation Health

Ensure your virtual environment is activated, then run the boot diagnostic:
```bash
python cli.py --mode health
```
Ensure all directories, configurations, and imports report `[OK]`.

---

## Step 2: Download YOLOv8 Weights

The system will automatically attempt to download the YOLOv8 person detection weights (`yolov8n.pt`) when a tracking pipeline is launched. If your environment is offline or has limited connectivity, download the weight file manually:
1. Download `yolov8n.pt` from the Ultralytics release page.
2. Place it in the root folder `E:/ARGUS_AI/` or at `E:/ARGUS_AI/models/weights/yolov8n.pt`.

---

## Step 3: Compile the Biometric Gallery

To perform gait recognition, the system needs a gallery database of enrolled identities. For verification purposes, you can build a gallery using preprocessed validation classes:
```bash
python cli.py --mode gallery
```
This script reads GEIs under `data/casia_processed/gei/` and compiles feature representations into the primary database directory `models/gallery/`.

---

## Step 4: Run Offline Folder Recognition

Perform a batch scan of GEI files in a target directory against the gallery:
```bash
python cli.py --mode recognize-folder --folder data/casia_processed/gei/034 --threshold 0.75
```
Enter an output CSV path when prompted (or press enter to skip). The system will scan each image, query the vector database, and print the match scores.

---

## Step 5: Run Pre-recorded Video Recognition

Process a video file to track and recognize walking individuals:
```bash
python cli.py --mode recognize-video --video data/test_video.mp4 --threshold 0.80 --show
```
- Bounding boxes will wrap tracked targets.
- Recognition outcomes and matching states will display above and below the boxes in real-time.
- Press `Q` in the window to stop.

---

## Step 6: Launch Live Camera Surveillance

Start real-time surveillance processing with parallel folder watch enrollment enabled:
```bash
python cli.py --mode live
```
- Connects to the local default webcam (`camera_index: 0` in `configs/inference.yaml`).
- Watches the `data/new_input/` folder in the background. If a new folder containing GEIs/videos is added there, the watcher automatically extracts features and enrolls the new subject without stopping the webcam loop.
- Press `Q` in the camera viewport window to exit.

---

## Step 7: Run High-Level Orchestration Commands

Rather than running multiple scripts individually, you can use the unified high-level command modes:

- **Run full health, tests, and speed checks:**
  ```bash
  python cli.py --mode full-check
  ```
- **Prepare entire dataset and model pipelines (preprocess, train, build gallery, evaluate, benchmark):**
  ```bash
  python cli.py --mode prepare
  ```
  *(Prompts for training confirmation before executing the model training step)*
- **Run comprehensive academic research evaluations (evaluate, threshold sweep, open-set, cross-view):**
  ```bash
  python cli.py --mode research-eval
  ```
- **Run production staging diagnostic tests:**
  ```bash
  python cli.py --mode production-test
  ```
- **Validate all project documents presence:**
  ```bash
  python cli.py --mode docs-check
  ```
