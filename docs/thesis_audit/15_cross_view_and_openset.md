# Phase 14 — Cross-View, Open-Set, and Condition Analysis

## 15.1 Cross-View Rank-1 Accuracy Matrix (Full)

**Source:** `runs/exp_001/evaluation_subject_disjoint/cross_view_report.md`

| Gal \ Probe | 000° | 018° | 036° | 054° | 072° | 090° | 108° | 126° | 144° | 162° | 180° |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **000°** | 83.7% | 76.6% | 67.9% | 66.1% | 61.3% | 63.0% | 67.3% | 66.7% | 68.3% | 67.2% | 68.0% |
| **018°** | 74.7% | 85.3% | 86.3% | 77.8% | 68.3% | 67.3% | 73.7% | 71.7% | 69.0% | 60.2% | 66.7% |
| **036°** | 69.3% | 82.9% | 92.6% | 86.2% | 81.0% | 78.3% | 79.0% | 74.7% | 68.0% | 55.5% | 58.0% |
| **054°** | 68.7% | 79.6% | 90.3% | 93.0% | 88.3% | 87.0% | 84.3% | 74.3% | 67.0% | 57.2% | 55.3% |
| **072°** | 63.0% | 73.2% | 85.6% | 88.9% | 91.7% | 88.0% | 85.0% | 74.7% | 67.3% | 62.9% | 58.3% |
| **090°** | 61.3% | 70.6% | 85.3% | 87.6% | 90.0% | 90.7% | 89.3% | 80.7% | 68.7% | 58.9% | 50.3% |
| **108°** | 60.7% | 71.6% | 85.0% | 84.9% | 83.0% | 88.3% | 89.7% | 84.7% | 74.7% | 62.5% | 61.0% |
| **126°** | 63.7% | 73.2% | 80.9% | 76.5% | 73.7% | 75.0% | 84.7% | 83.0% | 79.3% | 65.9% | 59.7% |
| **144°** | 65.0% | 72.6% | 69.9% | 65.8% | 61.7% | 63.3% | 71.0% | 78.3% | 81.7% | 72.9% | 65.3% |
| **162°** | 64.0% | 65.2% | 64.9% | 61.1% | 59.7% | 59.7% | 67.0% | 65.0% | 69.7% | 80.9% | 62.3% |
| **180°** | 73.7% | 67.2% | 68.2% | 60.1% | 57.0% | 54.7% | 61.7% | 65.7% | 72.3% | 72.2% | 79.7% |

### Key Observations

1. **Same-view accuracy is highest:** Diagonal values average 86.53%
2. **Adjacent views perform well:** ±18° views generally maintain >75% accuracy
3. **Frontal-lateral gap:** 000° gallery → 090° probe = 63.0% (significant drop)
4. **Lateral-frontal gap:** 090° gallery → 180° probe = 50.3% (worst pair)
5. **Best cross-view pair:** 054° gallery → 036° probe = 90.3%
6. **Worst cross-view pair:** 180° gallery → 090° probe = 54.7%
7. **Side views (036°-108°) form a strong cluster** with high mutual accuracy
8. **Extreme views (000°, 162°, 180°) are the most challenging** for cross-view matching

## 15.2 Condition-Wise Analysis

| Condition | Rank-1 | Correct/Total | Analysis |
|---|---|---|---|
| **NM** (Normal) | 96.82% | 1,065/1,100 | High accuracy; minimal covariates |
| **BG** (Bag) | 91.23% | 999/1,095 | ~5.6% drop from NM; bag alters silhouette shape |
| **CL** (Coat) | 72.64% | 799/1,100 | ~24.2% drop from NM; coat significantly changes body outline |

### Interpretation

- The NM→BG drop (5.6%) is modest, suggesting the model partially tolerates carrying-condition changes
- The NM→CL drop (24.2%) is substantial, confirming that clothing changes are the primary challenge for silhouette-based gait recognition
- This pattern is consistent with published literature on CASIA-B evaluation

## 15.3 Open-Set Analysis

**Source:** `runs/exp_001/evaluation_subject_disjoint/open_set_report.json`

### Protocol

| Property | Value |
|---|---|
| Known subjects (gallery enrolled) | 25 (subjects 075-099) |
| Unknown subjects (not enrolled) | 25 (subjects 100-124) |
| Gallery samples | 1,082 (nm-01..04 from known subjects) |
| Known probes | 1,645 (nm-05/06, bg-01/02, cl-01/02 from known subjects) |
| Unknown probes | 1,650 (all sequences from unknown subjects) |
| Operating threshold | 0.9913 |

### Open-Set Metrics

| Metric | Value | Interpretation |
|---|---|---|
| ROC-AUC | 0.915 | Good discrimination between genuine and impostor |
| EER | 0.1688 | ~16.9% equal error rate |
| TAR @ θ=0.9913 | 0.9373 | 93.7% of known subjects correctly accepted |
| FAR @ θ=0.9913 | 0.3675 | 36.8% of unknown subjects falsely accepted |
| Precision | 0.6761 | 67.6% of accepted identities are correct |
| F1-Score | 0.7855 | Balanced measure |

### Interpretation

- ROC-AUC of 0.915 indicates good overall separability
- However, FAR of 36.8% at the operating threshold is very high for a security application
- The EER of 16.88% suggests the system requires a higher threshold or more sophisticated rejection mechanism
- The high FAR is partially due to the indirect leakage (model trained on unknown subjects)

## 15.4 CMC Curve Data

**Source:** `runs/exp_001/evaluation_subject_disjoint/closed_set_eval_report.json`

| Rank | Cumulative Accuracy |
|---|---|
| 1 | 86.89% |
| 2 | 90.80% |
| 3 | 92.56% |
| 4 | 93.41% |
| 5 | 93.96% |
| 6 | 94.45% |
| 7 | 94.84% |
| 8 | 95.11% |
| 9 | 95.57% |
| 10 | 95.75% |
| 15 | 96.54% |
| 20 | 96.81% |

### Interpretation

- Steep initial climb: Rank-1 to Rank-5 adds 7.07%
- Diminishing returns after Rank-5
- By Rank-20, accuracy plateaus at ~96.81%
- ~3.19% of probes are never correctly matched in the top-20

## 15.5 Threshold Calibration Analysis

**Source:** `runs/exp_001/evaluation_subject_disjoint/threshold_calibration.json`

| Property | Value |
|---|---|
| Calibration data | Val subjects only (063-074) |
| Criterion | Minimize EER |
| Selected threshold | 0.9913 |
| Score range | 0.975 – 0.9989 |

### Observation

The very narrow score range (0.975 to 0.9989) and high threshold (0.9913) suggest:
1. All embeddings are highly similar (cosine ≈ 0.98+)
2. The embedding space is not well-separated for a model trained on all subjects
3. After clean retraining, the score distribution is expected to widen and the optimal threshold will likely decrease
