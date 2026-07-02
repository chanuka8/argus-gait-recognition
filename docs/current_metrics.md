# Current Verified Metrics

This document lists the performance, accuracy, and latency metrics verified during system diagnostics on the target dataset.

---

## 1. Biometric Accuracy Metrics

Compiled from standard validation sweeps (results logged in [outputs/eval_reports/split_eval_report.json](file:///e:/ARGUS_AI/outputs/eval_reports/split_eval_report.json)) matching gallery-probe partitions at a decision threshold of $0.75$:

- **Rank-1 Identification Accuracy:** **72.60%** (the correct subject is the closest match).
- **Rank-5 Identification Accuracy:** **90.00%** (the correct subject is within the top 5 matches).
- **Rank-10 Identification Accuracy:** **94.20%** (the correct subject is within the top 10 matches).
- **False Match Rate (FMR):** **27.40%**
- **False Non-Match Rate (FNMR):** **0.00%**
- **Equal Error Rate (EER):** **37.99%**
- **ROC Area Under the Curve (ROC AUC):** **67.61%**

---

## 2. System Latency & Speed Metrics

Compiled from speed diagnostic runs (results logged in [outputs/reports/benchmark_report.json](file:///e:/ARGUS_AI/outputs/reports/benchmark_report.json)):

- **Gallery Size:** 13,544 embeddings representing 124 subjects.
- **Gallery Loading Speed:** **0.0185 seconds** (18.47 milliseconds).
- **Pipeline Initialization Latency:** **0.0880 seconds** (88.04 milliseconds).
- **Single Target Inference Speed (CPU):** **0.1135 seconds** (113.50 milliseconds).
- **Total Diagnostic Duration:** **0.2227 seconds**.

---

## 3. Cross-View Identification Accuracy

Accuracy measured per camera view angle (reference sequences compared against probe viewpoints at threshold $0.75$, logged in [outputs/eval_reports/cross_view_report.json](file:///e:/ARGUS_AI/outputs/eval_reports/cross_view_report.json)):

| View Angle | Probe Count | Correct Matches | Rank-1 Accuracy |
| :---: | :---: | :---: | :---: |
| **$0^\circ$** | 46 | 31 | **67.39%** |
| **$18^\circ$** | 44 | 34 | **77.27%** |
| **$36^\circ$** | 48 | 28 | **58.33%** |
| **$54^\circ$** | 41 | 29 | **70.73%** |
| **$72^\circ$** | 48 | 28 | **58.33%** |
| **$90^\circ$** | 43 | 33 | **76.74%** |
| **$108^\circ$** | 50 | 31 | **62.00%** |
| **$126^\circ$** | 41 | 22 | **53.66%** |
| **$144^\circ$** | 45 | 23 | **51.11%** |
| **$162^\circ$** | 43 | 31 | **72.09%** |
| **$180^\circ$** | 51 | 29 | **56.86%** |

*Peak viewpoint performance:* Achieved at profile angle ($90^\circ$ at $76.74\%$) and near-profile angle ($18^\circ$ at $77.27\%$).
