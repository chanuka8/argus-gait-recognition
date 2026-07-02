# ARGUS AI: Contactless Long-Range Gait Biometric Surveillance System

ARGUS is a professional, high-performance intelligent video surveillance and biometric identification system designed for real-time person detection, multi-target tracking, and gait-based recognition. Optimized for security, defense, and airport/surveillance terminal environments, ARGUS enables contactless, non-cooperative, and distance-resilient identification of individuals based solely on their walking signature (gait pattern).

By utilizing high-accuracy deep learning, temporal feature aggregation via Gait Energy Images (GEI), and highly optimized vector similarity lookups, the system delivers real-time identity recognition and alert management.

---

## 1. Project Overview
Unlike face recognition or fingerprinting, which require high-resolution inputs, active cooperation, or close proximity, gait recognition functions at a distance, under low-resolution settings, and remains resilient to facial occlusions, masks, and illumination shifts.

ARGUS implements a modular, decoupled processing pipeline:
1. **Detects & Tracks** multiple individuals across video frames using YOLOv8 and ByteTrack.
2. **Extracts Silhouettes** from target crops using adaptive background subtraction.
3. **Accumulates Silhouettes** into a temporal Gait Energy Image (GEI) representing the subject's gait cycle.
4. **Extracts Deep Embeddings** from the GEI using a custom convolutional network architecture (`ByGaitLight`).
5. **Performs Vector Matching** against a local database using cosine distance metrics.
6. **Applies Risk & Threat Policies** based on matching confidence, logging occurrences, and generating real-time visual overlays.

---

## 2. Features
- **Real-Time Multi-Person Tracking:** Combines YOLOv8 target detection with ByteTrack for persistent ID tracking across camera streams.
- **Dynamic GEI Synthesis:** Generates rolling Gait Energy Images (GEI) over a sliding temporal window of 15 frames.
- **Fast Local Vector Matching:** Uses flat NumPy array indexing (`.npy` files) for vector matching in under 10 milliseconds.
- **Adaptive Matching Decisions:** Features a hybrid decision matrix combining flat matching thresholds with a centroid-margin top-K candidate match sweep.
- **Security Audit Logging:** Classifies matched subjects and logs severe threat events (`SECURITY_ALERT`, `REVIEW_REQUIRED`, `ALLOW`) to security logs in CSV.
- **Professional CCTV Display & Auto-Reporting:** Professional visual feedback with a 4-tier status overlay (DETECTION, TRACKING, UNKNOWN, CONFIRMED) and thread-safe reporting to CSV/JSONL files with customizable cooldown.
- **FastAPI API Gateway:** Integrates standard REST endpoints for third-party client verification and automated person registration.
- **Multi-Camera Operations:** Supports parallel, thread-safe camera feeds with isolated tracker contexts sharing a read-only CNN model, with custom per-camera location tags.


---

## 3. Architecture Summary
ARGUS is designed as a decoupled, layered pipeline processing framework.

```
       +-----------------------------------------------------------+
       |                     INTERFACE LAYER                       |
       |  FastAPI Server (api/) | Command CLI (cli.py) | Overlays  |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |                    INTELLIGENCE & RISK                    |
       |   ConfidenceScorer | SecurityEngine | CSV SecurityLogger  |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |                INFERENCE & MATCHING LAYER                 |
       |   ByGaitLight CNN | Cosine Similarity | VectorStore (npy) |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |                     PROCESSING LAYER                      |
       |   YOLOv8 Detect | ByteTrack | Silhouette Step | LiveGEI   |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |                     INGESTION LAYER                       |
       |    StreamEngine (OpenCV) | BufferQueue | FrameDropper     |
       +-----------------------------------------------------------+
```

### Module Boundary Layout
- **Ingestion & Streaming:** [StreamEngine](file:///e:/ARGUS_AI/streaming/stream_engine.py) provides raw video parsing.
- **Processing steps:** Custom steps for [Detection](file:///e:/ARGUS_AI/pipeline/steps/detection.py), [Tracking](file:///e:/ARGUS_AI/pipeline/steps/tracking.py), [Silhouette Segmentation](file:///e:/ARGUS_AI/pipeline/steps/silhouette_step.py), and [Live GEI Accumulation](file:///e:/ARGUS_AI/pipeline/steps/live_gei.py).
- **Inference Models:** Uses the custom [ByGaitLight](file:///e:/ARGUS_AI/models/architectures/bygait_light.py) CNN architecture.
- **Vector DB operations:** Managed via [VectorStore](file:///e:/ARGUS_AI/storage/vector_store.py).
- **Risk Assessment:** Evaluated by [SecurityEngine](file:///e:/ARGUS_AI/security_layer/security_engine.py) and logged in CSV by [SecurityLogger](file:///e:/ARGUS_AI/security_layer/security_logger.py).

---

## 4. Setup and Installation

### Prerequisites
- Python 3.10 or 3.11 (Python 3.11.9 recommended)
- C++ Build Tools (required for compiling select dependencies)

### Setup Steps
1. **Clone the project repository:**
   ```bash
   git clone <repository_url>
   cd ARGUS_AI
   ```

2. **Initialize a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. **Verify setup health:**
   ```bash
   python cli.py --mode health
   ```

---

## 5. CLI Commands Reference

ARGUS provides a unified CLI gateway via [cli.py](file:///e:/ARGUS_AI/cli.py). Commands can be run using the CLI dispatcher or through the `Makefile`:

| Target Command | CLI Invocation | Description |
| :--- | :--- | :--- |
| `make install` | `pip install -r requirements.txt` | Installs runtime dependencies |
| `make install-dev` | `pip install -r requirements-dev.txt` | Installs dev/test tools |
| `make health` | `python cli.py --mode health` | Diagnoses system environment |
| `make test` | `python cli.py --mode tests` | Runs all unit tests via CLI wrapper |
| `make pytest` | `pytest tests -v` | Runs unit/integration tests with pytest |
| `make train` | `python cli.py --mode train` | Trains custom ByGaitLight CNN model |
| `make gallery` | `python cli.py --mode gallery` | Builds model validation gallery |
| `make evaluate` | `python cli.py --mode evaluate` | Runs performance evaluation sweep |
| `make benchmark` | `python cli.py --mode benchmark` | Runs embedding extraction speed test |
| `make live` | `python cli.py --mode live` | Starts real-time webcam surveillance |
| `make system` | `python cli.py --mode system` | Boots health checks + camera stream + folder watcher |
| `make multi-camera`| `python cli.py --mode multi-camera`| Runs multi-camera stream processing |
| `make auto-enroll` | `python cli.py --mode auto-enroll` | Performs watch-based folder enrollment |
| `make clean` | `make clean` | Cleans temporary cache and pyc files |
| `make format-check`| `make format-check` | Verifies style guidelines (ruff/black) |
| - | `python cli.py --mode full-check` | Orchestrates health, tests, pytest, benchmark, and evaluate |
| - | `python cli.py --mode prepare` | Runs preprocess, train (confirm), gallery, evaluate, and benchmark |
| - | `python cli.py --mode research-eval`| Sweeps evaluate, threshold sweep, open-set, and cross-view |
| - | `python cli.py --mode production-test`| Verifies health, benchmark, tests, and configs/cameras.yaml presence |
| - | `python cli.py --mode docs-check` | Validates README.md, requirements, Makefile, and docs/ folder |

---

## 6. Project Workflows

### A. Dataset Preparation
ARGUS is designed to train and validate on the **CASIA-B** gait dataset.
1. Download the CASIA-B silhouettes zip file (`GaitDatasetB-silh.zip`).
2. Place the file inside `data/`.
3. Extract and preprocess the dataset to compile GEIs:
   ```bash
   python cli.py --mode preprocess
   ```
This extracts silhouettes, computes average gait cycle contours, and outputs 128x64 pixel GEI frames to `data/casia_processed/gei/`.

### B. Training Workflow
To train the custom `ByGaitLight` CNN model:
```bash
python cli.py --mode train --epochs 20 --batch-size 32
```
Model configurations, metrics, and network checkpoints (`best_model.pth`) are output to `runs/exp_001/`.

### C. Gallery Building
Prior to evaluations or live matching, compile the biometric gallery from the processed dataset:
```bash
python cli.py --mode gallery
```
This processes subject directories and saves feature vectors to the primary training gallery database `models/gallery/`.

### D. System Evaluation
Assess the accuracy of the trained model on validation partitions:
```bash
python cli.py --mode evaluate
```
Generates accuracy indices (Rank-1, Rank-5, Rank-10) and false match rates, plotting performance curves to `outputs/eval_reports/`.

### E. Speed Benchmark
Run execution speed diagnostics:
```bash
python cli.py --mode benchmark
```
Outputs loading latencies and single-frame inference speeds to `outputs/reports/benchmark_report.json`.

### F. Live Recognition Camera
Start camera surveillance:
```bash
python cli.py --mode live
```
Runs a real-time tracking camera window showing bounding boxes overlayed with similarity matching decisions. Press `Q` in the stream feed window to close.

### G. Video File Recognition
Recognize subjects in a pre-recorded video file:
```bash
python cli.py --mode recognize-video --video path/to/video.mp4 --threshold 0.85 --show
```

### H. Multi-Camera Operations
Process multiple parallel camera feeds with thread-safe resource usage:
```bash
python cli.py --mode multi-camera
```
Loads camera settings from `configs/cameras.yaml` and launches isolated tracking loops.

### I. CCTV Status Mapping & Reporting
Every recognition outcome maps to a professional CCTV visual status and reporting behavior:
- **DETECTION:** Person detected, awaiting gait cycle accumulation. (Red Overlay)
- **TRACKING:** Stabilized/predicted tracking in progress. (Orange Overlay)
- **UNKNOWN:** Recognition done, no biometric match in gallery. (Green Overlay)
- **CONFIRMED:** Enrolled biometric profile match confirmed. (Green Overlay)

Auto-detection events (configured by default to report `UNKNOWN` and `CONFIRMED` without per-frame spam) are saved to:
- CSV: `outputs/detection_reports/detections.csv`
- JSONL: `outputs/detection_reports/detections.jsonl`
- Snapshots: `outputs/detection_reports/snapshots/` (person crops)


---

## 7. Verified Performance Metrics
The following metrics were verified on this system during standard validation checks:
- **Gallery Size:** 13,544 embeddings representing 124 distinct subjects.
- **Gallery Loading Time:** **0.018 seconds** (18.4 milliseconds).
- **Single Inference Time (CPU):** **0.113 seconds** (113.4 milliseconds).
- **Rank-1 Accuracy:** **72.6%**
- **Rank-5 Accuracy:** **90.0%**
- **Rank-10 Accuracy:** **94.2%**
- **EER (Equal Error Rate):** **37.99%**
- **ROC AUC:** **67.61%**

---

## 8. System Limitations
1. **Background Constraint:** Silhouette extraction relies on background subtraction. Environments with dynamic lighting, heavy shadows, or moving background elements can distort the GEI contours.
2. **Occlusions:** Overlapping targets or camera occlusions break ByteTrack tracking chains, which restarts GEI accumulation.
3. **Database Scaling:** Flat NumPy vector lookup time scales linearly ($O(N)$) with the gallery size, which could degrade real-time performance at enterprise scales (e.g. $> 1,000,000$ profiles).

---

## 9. Future Roadmap
1. **Keypoint-Based Integration:** Complete `preprocessing/skeleton_extractor.py` to incorporate skeleton/pose keypoint estimation alongside GEI outlines.
2. **Distributed Indexing:** Replace flat NumPy scans with Faiss or Milvus indexes to maintain sub-millisecond similarity scans at large scale.
3. **AutoML retraining loops:** Connect retraining triggers to automate training scripts when a critical number of new subjects are enrolled.
4. **Advanced GPU Management:** Implement hardware performance profiles in `monitoring/gpu_tuner.py` to balance batch-size and GPU memory overhead.

---

## 10. License
This project is licensed under the MIT License. Details are provided in the `LICENSE` file (if present).
