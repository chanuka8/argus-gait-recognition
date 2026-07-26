# Phase 9 — Performance and Evaluation Metrics

## 10.1 Closed-Set Subject-Disjoint Evaluation

**Source:** `runs/exp_001/evaluation_subject_disjoint/closed_set_eval_report.json`

| Metric | Value | Notes |
|---|---|---|
| **Evaluation Type** | Subject-Disjoint Closed-Set Identification | |
| **Checkpoint** | `runs/exp_001/best_model.pth` | Trained on ALL 124 subjects |
| **Test Subjects** | 50 (subjects 075-124) | |
| **Gallery Samples** | 2,171 (nm-01 to nm-04, all views) | |
| **Probe Samples** | 3,295 (nm-05/06, bg-01/02, cl-01/02) | |
| **Operating Threshold** | 0.9913 | Calibrated on val set |
| **Rank-1 Accuracy** | **86.89%** | |
| **Rank-5 Accuracy** | **93.96%** | |
| **Rank-10 Accuracy** | **95.75%** | |
| **Avg Inference Latency** | 11.20 ms | Per probe (embedding + search) |
| **Inference FPS** | 89.32 | CPU |

### Condition-Wise Rank-1 Accuracy

| Condition | Correct | Total | Rank-1 Accuracy |
|---|---|---|---|
| **NM** (Normal Walking) | 1,065 | 1,100 | **96.82%** |
| **BG** (Bag Carrying) | 999 | 1,095 | **91.23%** |
| **CL** (Coat Wearing) | 799 | 1,100 | **72.64%** |

### Biometric Rates at θ=0.9913

| Metric | Value |
|---|---|
| FAR (False Accept Rate) | 0.6991 |
| FRR (False Reject Rate) | 0.0489 |
| TAR (True Accept Rate) | 0.9511 |
| TNR (True Negative Rate) | 0.3009 |
| Precision | 0.9002 |
| Recall | 0.9511 |
| F1-Score | 0.9249 |
| TP | 2,723 |
| FP | 302 |
| TN | 130 |
| FN | 140 |

> [!WARNING]
> **Validity Caveat:** The high FAR (69.91%) suggests the operating threshold (0.9913) is too low for the current embedding space. Additionally, the model was trained on all 124 subjects, including the 50 test subjects. This means the backbone has seen test subjects' GEIs during training, creating indirect leakage. Results should be reported as **"preliminary baseline with known indirect leakage"**, NOT as final thesis results.

## 10.2 Open-Set Subject-Disjoint Evaluation

**Source:** `runs/exp_001/evaluation_subject_disjoint/open_set_report.json`

| Metric | Value |
|---|---|
| **Evaluation Type** | Subject-Disjoint Open-Set Identification |
| **Known Test Subjects** | 25 (subjects 075-099) |
| **Unknown Test Subjects** | 25 (subjects 100-124) |
| **Gallery Samples** | 1,082 |
| **Total Probes** | 3,295 |
| **Known Probes** | 1,645 |
| **Unknown Probes** | 1,650 |
| **ROC-AUC** | **0.915** |
| **EER** | **0.1688** |
| **EER Threshold** | 0.9929 |

### Open-Set Operating Metrics at θ=0.9913

| Metric | Value |
|---|---|
| FAR | 0.3675 |
| FRR | 0.0627 |
| TAR | 0.9373 |
| TNR | 0.6325 |
| Precision | 0.6761 |
| Recall | 0.9373 |
| F1-Score | 0.7855 |

## 10.3 Cross-View Evaluation

**Source:** `runs/exp_001/evaluation_subject_disjoint/cross_view_report.md`

| Metric | Value |
|---|---|
| **Cross-View Average (excl. same view)** | **71.17%** |
| **Same-View Average** | **86.53%** |
| **Overall Average** | **72.57%** |

### Cross-View Rank-1 Accuracy Matrix (Selected)

| Gallery\Probe | 000° | 054° | 090° | 126° | 180° |
|---|---|---|---|---|---|
| **000°** | 83.7% | 66.1% | 63.0% | 66.7% | 68.0% |
| **054°** | 68.7% | 93.0% | 87.0% | 74.3% | 55.3% |
| **090°** | 61.3% | 87.6% | 90.7% | 80.7% | 50.3% |
| **126°** | 63.7% | 76.5% | 75.0% | 83.0% | 59.7% |
| **180°** | 73.7% | 60.1% | 54.7% | 65.7% | 79.7% |

**Best pair:** 054° gallery → 036° probe = **90.3%**
**Worst pair:** 180° gallery → 090° probe = **54.7%**

## 10.4 Inference Benchmark

**Source:** `runs/exp_001/evaluation_subject_disjoint/inference_benchmark.json`

| Metric | Value |
|---|---|
| **Device** | CPU |
| **Iterations** | 50 |
| **Avg Latency (embedding only)** | **0.78 ms** |
| **FPS (embedding only)** | **1,277** |

## 10.5 Training History Summary

**Source:** `runs/exp_001/metrics.json`

| Metric | Value |
|---|---|
| **Epochs** | 50 |
| **Best Validation Accuracy** | 80.14% |
| **Final Train Accuracy** | 88.87% |
| **Final Train Loss** | 9.718 |
| **Loss Mode** | ce_arcface (ArcFace s=64, m=0.35) |
| **Device** | CPU |

## 10.6 Metrics Validity Classification

| Metric | Value | Validity | Limitation | Thesis Usage |
|---|---|---|---|---|
| Rank-1 (closed-set) | 86.89% | **Preliminary** | Model trained on all 124 subjects (indirect leakage) | Report as preliminary baseline only |
| Rank-5 (closed-set) | 93.96% | **Preliminary** | Same leakage concern | Report as preliminary baseline only |
| Rank-10 (closed-set) | 95.75% | **Preliminary** | Same leakage concern | Report as preliminary baseline only |
| NM Rank-1 | 96.82% | **Preliminary** | Same leakage concern | Report as condition breakdown |
| BG Rank-1 | 91.23% | **Preliminary** | Same leakage concern | Report as condition breakdown |
| CL Rank-1 | 72.64% | **Preliminary** | Same leakage concern | Report as condition breakdown |
| ROC-AUC (open-set) | 0.915 | **Preliminary** | Same leakage concern | Report as preliminary |
| EER | 0.1688 | **Preliminary** | Same leakage concern | Report as preliminary |
| Cross-view avg | 71.17% | **Preliminary** | Same leakage concern | Report as preliminary |
| Embedding latency | 0.78 ms | **Valid** | Hardware-specific (CPU) | Report with hardware context |
| Inference FPS | 1,277 | **Valid** | Embedding only, no detection/tracking | Report as embedding-only |
| End-to-end latency | 11.20 ms | **Valid** | Per-probe (embedding + gallery search) | Report with context |
| Val accuracy (training) | 80.14% | **Invalid for thesis** | Includes test subjects in training | Do NOT report as test performance |
| F1-Score (closed-set) | 0.9249 | **Preliminary** | Same leakage concern | Report as preliminary |

> [!IMPORTANT]
> **Currently unavailable:** A clean subject-disjoint checkpoint and reproducible evaluation run are required to produce final thesis-valid metrics. The evaluation infrastructure (evaluator, leakage validators, split configs) is fully operational and ready to produce valid results once a clean checkpoint is trained on subjects 001-074 only.

## 10.7 Reproduction Commands

```powershell
# 1. Train model on subjects 001-074 only (requires full dependencies)
python scripts/train_model.py --data-dir data/casia_processed/gei --max-classes 74

# 2. Run subject-disjoint closed-set evaluation
python scripts/evaluate_subject_disjoint.py

# 3. Run cross-view evaluation
python scripts/evaluate_cross_view.py

# 4. Run open-set evaluation
python scripts/evaluate_open_set.py

# 5. Run threshold sweep
python scripts/evaluate_threshold_sweep.py

# 6. Run inference benchmark
python scripts/benchmark.py
```
