# Model Evaluation

This document outlines the performance evaluation suite, metrics formulations, and steps to execute validation checks.

---

## 1. Offline Validation Methodology

Model validation measures identification performance by dividing the evaluation dataset into two partitions:
1. **Gallery Partition:** Enrolled reference embeddings of known subjects.
2. **Probe Partition:** Testing query embeddings matched against the Gallery partition to test identification metrics.

By default, the evaluator performs a **50% split** (e.g., normal walking sequences `nm-01` to `nm-04` populate the gallery, while `nm-05` and `nm-06` serve as probe inputs) to test recognition across different walking sequences of the same subjects.

---

## 2. Calculated Biometric Metrics

The [evaluation/metrics.py](file:///e:/ARGUS_AI/evaluation/metrics.py) module calculates the following metrics:

### Rank-1, Rank-5, and Rank-K Identification Accuracy
Measures the percentage of query probe samples where the correct identity is found in the top $K$ database matches.
- **Rank-1 Accuracy:** The correct identity is the #1 closest match.
- **Rank-5 Accuracy:** The correct identity is within the 5 closest matches.

### False Match Rate (FMR)
The probability that the system incorrectly matches a query embedding to a different enrolled identity:
$$\text{FMR} = \frac{\text{False Matches}}{\text{Total Non-Match Attempts}}$$

### False Non-Match Rate (FNMR)
The probability that the system fails to match a query embedding of an enrolled subject to their correct gallery identity:
$$\text{FNMR} = \frac{\text{False Non-Matches}}{\text{Total Match Attempts}}$$

### Equal Error Rate (EER)
The operational threshold where the False Match Rate (FMR) equals the False Non-Match Rate (FNMR):
$$\text{FMR}(\theta) = \text{FNMR}(\theta)$$
A lower EER indicates better classification performance.

### ROC Area Under the Curve (ROC AUC)
Summarizes the model's ability to distinguish between matching and non-matching identities across all possible thresholds.

---

## 3. Visualizations Outputs

The evaluation process generates plots in `outputs/eval_reports/`:
- **ROC Curve (`roc_curve.png`):** Plots the True Positive Rate against the False Positive Rate across similarity thresholds.
- **Cumulative Match Characteristic Curve (`cmc_curve.png`):** Plots the identification accuracy as a function of the Rank parameter $K$.

---

## 4. Evaluation Commands

### Run standard evaluation sweep:
```bash
python cli.py --mode evaluate
```
*Alternative (Makefile):*
```bash
make evaluate
```
This runs a verification sweep, writes a summary log to `outputs/eval_reports/split_eval_report.json`, and saves the ROC/CMC charts.

### Run threshold sweep:
To sweep threshold parameters from 0.50 to 0.90 to find the optimal operational balance:
```bash
python scripts/evaluate_threshold_sweep.py
```
This writes progress records to `outputs/eval_reports/threshold_sweep.csv`.
