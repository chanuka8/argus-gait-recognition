# Troubleshooting Guide

This document lists common errors, deployment warnings, and diagnostic paths for resolving issues with the ARGUS system.

---

## 1. Environment & Import Errors

### `ModuleNotFoundError: No module named 'cv2'` or `'torch'`
- **Cause:** The active Python shell is executing outside the local virtual environment, or dependencies were not installed successfully.
- **Resolution:**
  1. Confirm the virtual environment is activated:
     - PowerShell: `.\venv\Scripts\Activate.ps1`
     - Bash: `source venv/bin/activate`
  2. Verify installed packages:
     - Run `pip list` and check that `torch` and `opencv-python` are present. If not, re-run `pip install -r requirements.txt`.

---

## 2. Model & Weights Configuration Issues

### `FileNotFoundError: Model checkpoint not found: runs/exp_001/best_model.pth`
- **Cause:** You are running inference, evaluation, or live camera pipelines before a model has been trained or checkpoint weights have been copied.
- **Resolution:**
  1. Train a model to output checkpoints:
     ```bash
     python cli.py --mode train
     ```
  2. Alternatively, copy pretrained network weights to the destination folder: `runs/exp_001/best_model.pth`.

### YOLO Weight Download Failure
- **Cause:** The tracking pipeline cannot connect to the internet to download the default YOLOv8 weights (`yolov8n.pt`).
- **Resolution:**
  1. Download the weight file manually from the Ultralytics release page.
  2. Save it to the root folder `E:/ARGUS_AI/` or at `E:/ARGUS_AI/models/weights/yolov8n.pt`.

---

## 3. Database & Matching Errors

### Target matching returns `UNKNOWN` for all subjects
- **Cause:** The local vector gallery is empty, or the matching threshold is set too high.
- **Resolution:**
  1. Build the gallery from the preprocessed dataset:
     ```bash
     python cli.py --mode gallery
     ```
  2. Adjust matching thresholds in [configs/inference.yaml](file:///e:/ARGUS_AI/configs/inference.yaml):
     - Try lowering `live_threshold` or `evaluation_threshold` (e.g. from `0.85` to `0.70`).

---

## 4. Hardware & Camera Lag

### Camera feed window freezes or lags behind real-time action
- **Cause:** The system is processing frames slower than the camera's capture rate, causing frame buffers to pile up in memory.
- **Resolution:**
  1. Enable GPU acceleration for PyTorch (ensure CUDA drivers are installed).
  2. Adjust camera resolution and FPS settings in [configs/cameras.yaml](file:///e:/ARGUS_AI/configs/cameras.yaml) (e.g. set `target_fps: 5` and resolution to `640x480`).
  3. Verify that the async streaming queue is enabled to drop intermediate frames when latency spikes occur.
