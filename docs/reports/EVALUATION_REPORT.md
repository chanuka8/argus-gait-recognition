# ARGUS AI — Evaluation Audit Report

**Report Generated:** 2026-07-30T10:37:00+05:30  
**Repository Version / Commit Hash:** `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`  
**Working Tree Status:** Dirty (Uncommitted documentation/report suite additions)  
**Audit Policy:** Zero False Positive Evidence-Based Reporting Policy  
**Target Path:** `docs/reports/EVALUATION_REPORT.md`

---

## Executive Summary & Protocol Warnings

> [!IMPORTANT]
> **Evaluation Context & Field Performance Warning:**
> 1. **Closed-Set vs Open-Set:** Closed-set FAR represents forced-choice assignment across known identities. **Open-set metrics** are the authoritative evidence for rejecting unknown individuals.
> 2. **Dataset Domain:** All recognition metrics were evaluated on the **CASIA-B Gait Dataset**. Laboratory dataset metrics do **NOT** prove real-world outdoor CCTV performance without field validation.

---

## 1. Subject-Disjoint Dataset Partitioning

The evaluation strictly follows a **subject-disjoint protocol**, ensuring that zero subjects in the validation or test splits were present during model training.

| Partition | Subject IDs | Subject Count | Purpose | Evidence Source |
| :--- | :--- | :---: | :--- | :--- |
| **Train Set** | `001` .. `062` | **62** | ArcFace Model Training | [subject_split.json](../../configs/subject_split.json) |
| **Validation Set** | `063` .. `074` | **12** | Threshold Calibration (`min_eer`) | [subject_split.json](../../configs/subject_split.json) |
| **Test Set** | `075` .. `124` | **50** | Unseen Model Evaluation | [subject_split.json](../../configs/subject_split.json) |

---

## 2. Threshold Calibration

The operating threshold was calibrated exclusively on the **12 validation subjects** to prevent data leakage into test set evaluation.

| Calibration Parameter | Value | Details / Source | Verification Status |
| :--- | :--- | :--- | :--- |
| **Calibration Criterion** | `min_eer` | Equal Error Rate Minimization on Validation Split | **Verified** |
| **Selected Threshold** | **0.9913** | Operating cosine similarity threshold | **Verified** |
| **Validation Probes** | 792 probes | Score range: $[0.9750, 0.9989]$ | **Verified** |

---

## 3. Closed-Set Identification Performance

> **Test Partition:** 50 Unseen Subjects (`075` to `124`)  
> **Samples:** 2,171 Gallery Items, 3,295 Probe Items  
> **Evidence Source:** `runs/exp_001/evaluation_subject_disjoint/closed_set_eval_report.json`

| Metric | Measured Value | Type | Protocol / Details | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Rank-1 Accuracy** | **86.89%** (`0.86889`) | Measured | Top-1 Cumulative Match Characteristic | **Verified** |
| **Rank-5 Accuracy** | **93.96%** (`0.93961`) | Measured | Top-5 Cumulative Match Characteristic | **Verified** |
| **Rank-10 Accuracy** | **95.75%** (`0.95751`) | Measured | Top-10 Cumulative Match Characteristic | **Verified** |
| **Precision** | **90.02%** (`0.90017`) | Measured | Operating Threshold = `0.9913` | **Verified** |
| **Recall / TAR** | **95.11%** (`0.95110`) | Measured | Operating Threshold = `0.9913` | **Verified** |
| **F1-Score** | **92.49%** (`0.92493`) | Measured | Operating Threshold = `0.9913` | **Verified** |
| **False Acceptance Rate (FAR)** | **69.91%** | Measured | Closed-set forced choice (threshold=0.9913) | **Verified** |
| **False Rejection Rate (FRR)** | **4.89%** | Measured | Closed-set forced choice (threshold=0.9913) | **Verified** |

### Condition-Wise Rank-1 Breakdown

| Walking Condition | Description | Rank-1 Accuracy | Sample Count (Correct / Total) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **NM (Normal Walking)** | Standard gait without carrying or clothing changes | **96.82%** | 1,065 / 1,100 | **Verified** |
| **BG (Bag Carrying)** | Gait carrying a backpack or handbag | **91.23%** | 999 / 1,095 | **Verified** |
| **CL (Coat Wearing)** | Gait wearing a heavy coat or jacket | **72.64%** | 799 / 1,100 | **Verified** |

---

## 4. Open-Set Verification & Rejection (Known vs Unknown)

> **Known Subjects:** `075` to `099` (25 subjects, 1,082 gallery items, 1,645 probes)  
> **Unknown Subjects:** `100` to `124` (25 subjects, 1,650 probes)  
> **Evidence Source:** `runs/exp_001/evaluation_subject_disjoint/open_set_report.json`

| Metric | Measured Value | Type | Protocol / Details | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **ROC AUC** | **0.9150** | Measured | Area Under the ROC Curve | **Verified** |
| **Equal Error Rate (EER)** | **16.88%** (`0.1688`) | Measured | EER Threshold = `0.9929` | **Verified** |
| **FAR at Operating Threshold** | **36.75%** | Measured | Known vs Unknown at threshold `0.9913` | **Verified** |
| **FRR at Operating Threshold** | **6.27%** | Measured | Known vs Unknown at threshold `0.9913` | **Verified** |
| **True Acceptance Rate (TAR)** | **93.73%** | Measured | Known probes accepted at `0.9913` | **Verified** |
| **True Rejection Rate (TNR)** | **63.25%** | Measured | Unknown probes rejected at `0.9913` | **Verified** |
| **Open-Set Precision** | **67.61%** | Measured | At operating threshold `0.9913` | **Verified** |
| **Open-Set F1-Score** | **78.55%** | Measured | At operating threshold `0.9913` | **Verified** |

---

## 5. Cross-View Matrix Evaluation

Evaluation across **11 camera viewing angles** ($000^\circ, 018^\circ, 036^\circ, 054^\circ, 072^\circ, 090^\circ, 108^\circ, 126^\circ, 144^\circ, 162^\circ, 180^\circ$).

| Cross-View Metric | Measured Value | Type | Source File | Verification Status |
| :--- | :---: | :--- | :--- | :--- |
| **Same-View Average** | **86.53%** | Measured | `cross_view_report.json` | **Verified** |
| **Cross-View Average** | **71.17%** | Measured | `cross_view_report.json` (Excl. same view) | **Verified** |
| **Overall Matrix Average** | **72.57%** | Measured | `cross_view_report.json` (Full 11x11 matrix) | **Verified** |
| **Similarity Metric** | Cosine Similarity | Configured | Dot product of L2 normalized 128-d vectors | **Verified** |

---

## 6. Evaluation Limitations

1. **Clothing Changes (CL):** Coat wearing causes a performance drop from 96.82% (NM) to 72.64% (CL) due to silhouette deformation.
2. **Open-Set FAR (36.75%):** Unseen subjects have a 36.75% rate of matching a gallery identity at threshold `0.9913`. Operating thresholds must be adjusted for higher security requirements.
3. **Controlled Environment:** CASIA-B is captured under controlled lighting and background subtraction conditions. Real-world CCTV evaluation is required before deployment in uncontrolled environments.

---
**Status:** `VERIFIED - EVALUATION AUDITED`
