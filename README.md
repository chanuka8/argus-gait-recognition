# ARGUS AI

![ARGUS AI Gait Recognition Banner](assets/github/Gitrepo_profilepic.png)

### Real-Time Gait Recognition & Cross-Camera Identity Tracking

ARGUS AI is a modular spatial-temporal gait recognition, multi-object tracking, and multi-camera surveillance intelligence framework built on PyTorch and OpenCV. It extracts gait biometric signatures from silhouette sequences to perform open-set subject identification, multi-camera tracking, and forensic trajectory reconstruction.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows / Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.17%2B-blue.svg)](https://onnxruntime.ai/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Pytest: 393 Passed](https://img.shields.io/badge/pytest-393%20passed-brightgreen.svg)](tests)
[![Status: Research Grade](https://img.shields.io/badge/status-RESEARCH__GRADE__PLATFORM-blue.svg)](docs/ARGUS_CURRENT_TECHNICAL_STATUS_REPORT.md)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)

---

## Project Status Matrix

ARGUS AI separates implementation into distinct functional components. System maturity is audited against empirical test suites and physical execution evidence:

| Component / Subsystem | Implementation Status | Validation Level | Primary Reference Source |
| :--- | :---: | :---: | :--- |
| **Person Detection (YOLOv8)** | **IMPLEMENTED** | Validated (PyTest / CUDA) | [pipeline/detection/person_detector.py](pipeline/detection/person_detector.py) |
| **Multi-Object Tracking (ByteTrack)** | **IMPLEMENTED** | Validated (Integration Suite) | [pipeline/steps/tracking.py](pipeline/steps/tracking.py) |
| **Bounding Box Stabilization (EMA)** | **IMPLEMENTED** | Validated (Unit Tests) | [utils/box_stabilizer.py](utils/box_stabilizer.py) |
| **Silhouette Extraction (Contract/Fallback)** | **IMPLEMENTED** | Validated (Otsu Strategy Active) | [pipeline/silhouette/extractor.py](pipeline/silhouette/extractor.py) |
| **Cycle-Aware GEI Generation** | **IMPLEMENTED** | Validated (Unit Tests) | [pipeline/steps/live_gei.py](pipeline/steps/live_gei.py) |
| **Gait Feature Encoder (ByGaitLight)** | **IMPLEMENTED** | Validated (HPP / 256-d L2) | [models/architectures/bygait_light.py](models/architectures/bygait_light.py) |
| **ArcFace & Triplet Training Engine** | **IMPLEMENTED** | Validated (Ablation Matrix) | [models/architectures/losses.py](models/architectures/losses.py) |
| **Hardened Gallery Storage (VectorStore)** | **IMPLEMENTED** | Validated (`allow_pickle=False`) | [storage/vector_store.py](storage/vector_store.py) |
| **Open-Set Recognizer (3-State Logic)** | **IMPLEMENTED** | Validated (Min-EER Thresholds) | [intelligence/open_set_recognizer.py](intelligence/open_set_recognizer.py) |
| **Centralized Threshold Manager** | **IMPLEMENTED** | Validated (Config & Calibration) | [core/threshold_manager.py](core/threshold_manager.py) |
| **Dual-Modal ReID Score Fusion** | **IMPLEMENTED** | Validated (OSNet Fallback) | [intelligence/dual_modal_fusion.py](intelligence/dual_modal_fusion.py) |
| **Temporal Gait Verification** | **IMPLEMENTED** | Validated (Sliding Window) | [pipeline/steps/temporal_gait_verifier.py](pipeline/steps/temporal_gait_verifier.py) |
| **Multi-Camera Evidence Fusion** | **IMPLEMENTED** | Validated (Multi-Stream) | [intelligence/multi_camera_evidence_fusion.py](intelligence/multi_camera_evidence_fusion.py) |
| **Cross-Camera Tracker & Topology Model** | **IMPLEMENTED** | Validated (Transition Graph) | [intelligence/cross_camera_tracker.py](intelligence/cross_camera_tracker.py) |
| **Inference Backends (PyTorch / ONNX)** | **IMPLEMENTED** | Validated (Parity Reached) | [models/inference/backend.py](models/inference/backend.py) |
| **TensorRT Execution Engine** | **EXPERIMENTAL** | Framework Implemented (HW Deferred) | [models/inference/tensorrt_backend.py](models/inference/tensorrt_backend.py) |
| **Learned UNet Silhouette Model Asset** | **PLANNED** | Asset Missing (Otsu Active) | [pipeline/silhouette/extractor.py](pipeline/silhouette/extractor.py) |

---

## Key Capabilities

### Core Recognition & Pipeline Processing
* **YOLOv8 Bounding Box Localization**: Deep learning person detection configured via externalized YAML ([configs/detection.yaml](configs/detection.yaml)).
* **ByteTrack & IoU Multi-Object Tracking**: Track ID assignment with Exponential Moving Average (EMA) box coordinate smoothing ([pipeline/steps/tracking.py](pipeline/steps/tracking.py)).
* **Unified Silhouette Contract**: Morphological cleaning, component filtering, aspect-ratio constraints, height normalization ($128 \times 64$, 85% centered height), and learned segmenter fallback ([pipeline/silhouette/extractor.py](pipeline/silhouette/extractor.py)).
* **Cycle-Aware Gait Energy Image (GEI)**: Dynamic sequence aggregation of silhouette sequences into 2D GEI signatures with duplicate frame rejection ([pipeline/steps/live_gei.py](pipeline/steps/live_gei.py)).
* **ByGaitLight CNN Gait Encoder**: Lightweight convolutional neural network supporting Horizontal Part Pooling (HPP, `part_bins=4`) and 256-dimensional L2-normalized embedding extraction ([models/architectures/bygait_light.py](models/architectures/bygait_light.py)).

### Recognition Intelligence & Multi-Camera Tracking
* **3-State Open-Set Identification**: Classifies tracks into `KNOWN`, `UNKNOWN`, or `UNCERTAIN` based on validation-calibrated cosine similarity thresholds and top-1/top-2 margin constraints ([intelligence/open_set_recognizer.py](intelligence/open_set_recognizer.py)).
* **Centralized Threshold Management**: Unified threshold loading and strict semantic validation (`unknown_threshold < known_threshold`) driven by [core/threshold_manager.py](core/threshold_manager.py).
* **Cross-Camera Topology Tracking**: Directed camera transition modeling with travel-time windows $[T_{min}, T_{max}]$, transition probabilities, entry/exit zones, and accumulated score decay ([intelligence/cross_camera_tracker.py](intelligence/cross_camera_tracker.py)).
* **Dual-Modal Gait + ReID Fusion**: Optional score-level fusion combining gait embeddings with OSNet appearance features ([intelligence/dual_modal_fusion.py](intelligence/dual_modal_fusion.py)).
* **Watchlist & Event Reconstruction**: Dynamic target registration, threat priority routing, and atomic timeline trajectory export to JSON, CSV, and Markdown ([intelligence/event_timeline_reconstructor.py](intelligence/event_timeline_reconstructor.py)).

---

## System Architecture

```mermaid
graph TD
    Input[Camera Streams / RTSP / Video] --> DET[Person Detector - YOLOv8]
    DET --> TRK[Multi-Object Tracker - ByteTrack]
    TRK --> EMA[Box Coordinate Stabilizer - EMA]
    EMA --> SIL[Silhouette Step - Unified Contract & Otsu/ONNX Strategy]
    SIL --> GEI[Live GEI Accumulator - Cycle-Aware Sequence Builder]
    GEI --> QUA[Quality Estimator - Area / Aspect Ratio / Sharpness]
    QUA --> FEA[ByGaitLight Encoder - HPP part_bins=4]
    FEA --> EMB[256-d L2-Normalized Embedding]
    EMB --> VEC[Gallery Search - Hardened VectorStore]
    VEC --> THR[Central Threshold Manager - calibration.json]
    THR --> OSR[Open-Set Recognizer - KNOWN / UNKNOWN / UNCERTAIN]
    OSR --> TPV[Temporal Gait Verifier - Sliding Window Smoother]
    TPV --> CCT[Cross-Camera Tracker - Spatial-Temporal Transition Graph]
    CCT --> OUT[Recognition Output / Watchlist Alert / Timeline Export]
```

---

## Current Gait Model & Benchmark Evidence

### Active Checkpoint vs. Top Candidate Model

ARGUS AI strictly separates the **active deployment checkpoint** from **experimental candidate models**:

* **Active Checkpoint (`runs/exp_001/best_model.pth`)**: Global average pooling `ByGaitLight` (`part_bins=1`), trained without margin-based classification. Preserved as the reference baseline.
* **Top Candidate Model (`EXP-003E` / `models/candidates/exp_003e_hpp_arcface_triplet025_best.pth`)**: `ByGaitLight` with Horizontal Part Pooling (`part_bins=4`), 256-d L2 embeddings, ArcMarginProduct (ArcFace margin=0.50, scale=30.0), and Batch-Hard Triplet loss (weight=0.25).

> **Evaluation Protocol Notice**: Historical benchmarks (Exp-001) were evaluated under non-subject-disjoint splits where test identities overlapped during training supervision. All recent benchmarks (**EXP-003A..E**) use a strict subject-disjoint CASIA-B partition: Train `001–062` (6,779 samples), Validation `063–074` (1,299 samples), Test `075–124` (5,466 samples).

### Controlled Ablation Study Matrix (EXP-003A .. EXP-003E)

To determine the individual impacts of HPP, ArcFace, and Triplet loss, a controlled 5-run ablation study was executed under identical training hyperparameters (25 epochs, Adam lr=0.0001, seed 42):

| Experiment | Pooling | Loss Mode | Triplet Weight | Rank-1 | Rank-5 | NM (Normal) | BG (Bag) | CL (Clothing) | ROC-AUC | EER | Open-Set FAR | Calibration Threshold | Impostor Score Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp-001** (Legacy)* | Global (1) | Standard CE | ~0.50 | 86.89%* | 93.96%* | 96.82%* | 91.23%* | 72.64%* | 0.9150 | 16.88% | 36.75% | 0.9913 | Severe saturation near 1.0 |
| **EXP-003A** (Disjoint Base) | Global (1) | Standard CE | 0.50 | 52.78% | 67.10% | 85.82% | 53.15% | 19.36% | 0.7499 | 31.95% | 70.49% | 0.7064 | Compressed `[0.208, 0.984]` |
| **EXP-003B** (HPP Alone) | HPP (4) | Standard CE | 0.50 | 61.43% | 75.63% | 91.55% | 60.55% | 32.18% | 0.8327 | 24.86% | 57.06% | 0.7942 | Compressed `[0.450, 0.995]` |
| **EXP-003C** (ArcFace Alone)| Global (1) | ArcFace | 0.50 | 59.58% | 73.78% | 91.00% | 61.55% | 26.18% | 0.8314 | 25.64% | 47.20% | 0.9927 | Severe saturation near 1.0 |
| **EXP-003D** (HPP+ArcFace) | HPP (4) | ArcFace | 0.00 | 69.71% | 80.91% | 96.73% | 72.79% | 39.64% | 0.8470 | 23.49% | 60.84% | 0.5287 | Wide range `[0.211, 0.965]` |
| **EXP-003E** (Top Candidate) | HPP (4) | ArcFace | **0.25** | **72.63%** | **82.76%** | **97.00%** | **78.26%** | **42.64%** | **0.8776** | **20.46%** | **62.26%** | **0.4906** | **Desaturated `[-0.60, 0.97]`** |

*\*Note: Exp-001 metrics reflect legacy non-subject-disjoint evaluation. Under strict subject-disjoint split, the global CE baseline achieves 52.78% Rank-1 (EXP-003A). EXP-003E improves Rank-1 by +19.85% absolute over the true baseline.*

### Key Research Findings
1. **HPP Architectural Impact**: HPP (`part_bins=4`) alone improves Rank-1 accuracy by **+8.65%** (52.78% $\rightarrow$ 61.43%) and ROC-AUC by **+0.0828** over global average pooling under identical loss formulations.
2. **ArcFace & Score Saturation**: Combining ArcFace margin loss with HPP completely eliminates cosine score saturation near 1.0, shifting average impostor similarity down to 0.0537 and expanding cosine dynamics across `[-0.60, 0.97]`.
3. **Triplet Loss Balance**: Reducing triplet weight from 0.50 to 0.25 (**EXP-003E**) achieves the highest Rank-1 identification accuracy (**72.63%**) and ROC-AUC (**0.8776**) across all subject-disjoint models.

---

## Deployment Health & Security Infrastructure

### Pre-Flight Deployment Checker (`scripts/doctor.py`)
`scripts/doctor.py` provides a non-destructive CLI diagnostic tool verifying:
* **Configuration Validity**: Scans YAML configurations via `ConfigValidator`.
* **Model Checkpoints**: Verifies PyTorch checkpoint (`runs/exp_001/best_model.pth`) and ONNX engine (`models/engines/bygait_light.onnx`).
* **Gallery Security**: Runs VectorStore checks (`allow_pickle=False`, numeric dtypes, 2D feature shapes, non-finite rejection).
* **Inference Backend Smoke Test**: Validates active PyTorch / ONNX Runtime backend execution.
* **Sanitized Reports**: Outputs machine-readable JSON (`outputs/reports/health_report.json`) and Markdown (`outputs/reports/health_report.md`).

### Security Controls
* **Vector Store Hardening**: All biometric gallery loads in [storage/vector_store.py](storage/vector_store.py) strictly enforce `allow_pickle=False` and reject object-type arrays (`dtype == object`), preventing arbitrary code execution exploits.
* **Encrypted RTSP Credentials**: Per-camera credentials are resolved via environment variables (`ARGUS_CAMERA_<ID>_PASSWORD`) or Fernet-encrypted stores (`configs/credentials.enc`), with automatic log masking (`rtsp://***:***@host:port`).

---

## Inference Backends & Selection Policy

Backend execution is managed by `get_inference_backend()` in [models/inference/backend.py](models/inference/backend.py):

* `requested=pytorch`: Directly executes PyTorch reference backend.
* `requested=onnxruntime`: Executes ONNX Runtime optimized backend. Falls back to PyTorch if session creation fails and `allow_fallback=true`.
* `requested=auto`: Attempts ONNX Runtime first, automatically falling back to PyTorch if ONNX Runtime or `.onnx` weights are unavailable.

Every initialized backend metadata dictionary reports `requested_backend`, `active_backend`, `execution_provider`, `fallback_used`, and `fallback_reason`.

---

## Installation & Setup

### Prerequisites
* **Python**: 3.11+
* **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)
* **GPU**: CUDA-compatible GPU (RTX 3050 6GB Laptop GPU tested with PyTorch CUDA 12.8)

### Setup Steps (Windows PowerShell / Linux Bash)

```powershell
# 1. Clone repository and initialize environment
git clone https://github.com/chanuka8/argus-gait-recognition.git
cd argus-gait-recognition

# 2. Create and activate virtual environment
python -m venv venv
& .\venv\Scripts\Activate.ps1   # On Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Git documentation pre-commit hooks
python scripts/install_git_hooks.py
```

---

## Quick Start & CLI Workflows

All primary operational modes are exposed through [cli.py](cli.py) and specialized scripts under `scripts/`:

```bash
# 1. Run deployment pre-flight health diagnostic
python scripts/doctor.py

# 2. Run multi-camera recognition stream
python cli.py --mode multi-camera

# 3. Process video file recognition
python cli.py --mode recognize-video --video "path/to/sample.mp4"

# 4. Export ONNX engine and validate numerical parity
python scripts/export_bygait_onnx.py --output-path models/engines/bygait_light.onnx

# 5. Execute controlled gait model ablation study (EXP-003A..E)
python scripts/run_ablation_study.py --epochs 25 --batch-size 16

# 6. Verify documentation alignment across package folders
python scripts/sync_folder_readmes.py --check
```

---

## Repository Testing & Verification

Repository integrity is validated using automated compilation, linting, documentation synchronization, and pytest suites:

```bash
# 1. Verify compilation across modules
python -m compileall models training pipeline core intelligence monitoring scripts tests

# 2. Run Ruff static code analysis
python -m ruff check .

# 3. Check documentation README synchronization
python scripts/sync_folder_readmes.py --check

# 4. Execute full automated test suite
python -m pytest tests -q
```

### Verified Test Status
* **Compilation**: `compileall` passed with 0 errors.
* **Linter**: `ruff check .` passed with 0 errors.
* **Documentation Sync**: `scripts/sync_folder_readmes.py --check` clean across all 19 package folders.
* **PyTest Suite**: **393 passed**, 3 skipped, 0 failed across 396 test items.

---

## Externalized Configuration

System parameters are controlled via external YAML configuration files in [configs/](configs/):

* **[configs/system.yaml](configs/system.yaml)**: Thread pool limits, logging levels, storage directories, and REST API bindings.
* **[configs/inference.yaml](configs/inference.yaml)**: Backend selection policy (`auto`, `pytorch`, `onnxruntime`), matching policy, quality bounds, and watchlist settings.
* **[configs/detection.yaml](configs/detection.yaml)**: YOLOv8 detector confidence (`0.4`), IoU threshold (`0.45`), target classes (`[0]`), execution device, and input resolution.
* **[configs/cameras.yaml](configs/cameras.yaml)**: Camera stream endpoints, worker pool limits, and ONVIF discovery options.
* **[configs/subject_split.json](configs/subject_split.json)**: Subject-disjoint partition manifest (Train `001–062`, Val `063–074`, Test `075–124`).

---

## Known Limitations & Research Targets

1. **Open-Set False Accept Rate (FAR = 62.26%)**: While score saturation near 1.0 was completely eliminated by ArcFace + HPP, the open-set FAR on unseen test subjects at validation-calibrated thresholds remains 62.26%, making open-set threshold calibration an active research target.
2. **Clothing Change Covariate Degradation (`CL` = 42.64%)**: Rank-1 accuracy under clothing changes (`CL`) drops from 97.00% (Normal Walking) to 42.64%, reflecting sensitivity of horizontal silhouette slicing to outer garments (coats/jackets).
3. **Learned Silhouette Model Asset Availability**: `models/engines/silhouette_segmenter.onnx` is not included in the repository by default; the system operates via Otsu background subtraction fallback.

---

## Prioritized Development Roadmap

* **P0 — Open-Set FAR Reduction**: Optimize ArcFace scale/margin parameters and explore adaptive thresholding to suppress open-set false acceptances.
* **P1 — Clothing-Change (`CL`) Robustness**: Implement part-aware spatial attention mechanisms to improve gait feature extraction invariant to outer apparel.
* **P2 — Learned Silhouette Model Validation & Batched Processing**: Validate lightweight UNet ONNX segmentation models and implement batched multi-person feature extraction.
* **P3 — Cross-Camera Benchmarking & Exporter Modernization**: Conduct quantitative multi-camera tracking evaluation and transition ONNX export to PyTorch `torch.export`.

---

## Cross References & Package Documentation

* **Technical Status Report**: [docs/ARGUS_CURRENT_TECHNICAL_STATUS_REPORT.md](docs/ARGUS_CURRENT_TECHNICAL_STATUS_REPORT.md)
* **Deployment Readiness Specification**: [docs/DEPLOYMENT_READINESS.md](docs/DEPLOYMENT_READINESS.md)
* **Documentation Index**: [docs/README_INDEX.md](docs/README_INDEX.md)
* **Package READMEs**:
  * [models/README.md](models/README.md) | [training/README.md](training/README.md) | [evaluation/README.md](evaluation/README.md)
  * [pipeline/README.md](pipeline/README.md) | [intelligence/README.md](intelligence/README.md) | [scripts/README.md](scripts/README.md)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Maintainer

**Chanuka Sandun**  
Undergraduate in Cybersecurity  
Developer of the ARGUS AI Gait Recognition Framework  
* GitHub: [github.com/chanuka8](https://github.com/chanuka8)  
* LinkedIn: [linkedin.com/in/chanukasandun](https://www.linkedin.com/in/chanukasandun/)
