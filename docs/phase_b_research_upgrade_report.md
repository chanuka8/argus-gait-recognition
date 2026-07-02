# Phase B Research Upgrade Report

This report summarizes the design, implementations, and experimental setups introduced in Phase B.

---

## 1. Open-Set Evaluation (B1)
- **Goal**: Evaluate model performance in a realistic environment where unseen/unregistered subjects query the system, requiring the system to reject them as `"UNKNOWN"`.
- **Methodology**:
  - CASIA-B subjects are split using a `known_ratio` (default: `0.6`, representing 60% known and 40% unknown).
  - The gallery is constructed exclusively from known subjects using a specified `gallery_ratio` (default: `0.5`).
  - Test queries are collected from both known subjects (remaining images) and unknown subjects (all images).
  - Matches are evaluated using similarity scores. If a match falls below the classification threshold, it is rejected as `"UNKNOWN"`.
- **Metrics Collected**:
  - **TP (True Positive)**: Known class query matched correctly.
  - **FP (False Positive)**: Unknown class query accepted as a known ID, OR known class query matched to a wrong known ID.
  - **TN (True Negative)**: Unknown class query correctly rejected as `"UNKNOWN"`.
  - **FN (False Negative)**: Known class query incorrectly rejected as `"UNKNOWN"`.
  - **known_accuracy**: Accuracy on known queries ($\text{TP} / \text{total\_known}$).
  - **unknown_rejection_rate**: Rejection rate of unknown queries ($\text{TN} / \text{total\_unknown}$).
  - **false_accept_rate (FAR)**: Rate of unknown queries accepted as any known class ($\text{FP\_from\_unknown} / \text{total\_unknown}$).
  - **false_reject_rate (FRR)**: Rate of known queries rejected as `"UNKNOWN"` ($\text{FN} / \text{total\_known}$).
- **Reports Exported**:
  - `outputs/eval_reports/open_set_report.json`
  - `outputs/eval_reports/open_set_report.csv`

---

## 2. Cross-View Evaluation (B2)
- **Goal**: Measure the gait recognition accuracy under various query angles relative to the gallery views.
- **Methodology**:
  - The query view angle is parsed directly from the CASIA-B image filenames (e.g. `034_nm-01_126.png` corresponds to view angle `126`).
  - Matching accuracy is tabulated and reported separately for each unique view angle found in the test set.
  - **Fallback Logic**: If the filename format cannot be parsed (e.g. customized or processed files lacking the standard CASIA-B view suffixes), query images are grouped under the category `"Fallback-View"` and evaluated together.
- **Reports Exported**:
  - `outputs/eval_reports/cross_view_report.json`
  - `outputs/eval_reports/cross_view_report.csv`

---

## 3. ArcFace Support (B3)
- **Goal**: Support angular margin cross-entropy loss experiments to maximize inter-class gait feature separation.
- **Implementation**:
  - Added the `ArcMarginProduct` class to `models/architectures/losses.py`, which normalizes features and weights, computes the angular cosine, adds the margin penalty $m$, and scales by $s$.
  - Updated `GaitClassifier` in `training/trainer.py` to instantiate `ArcMarginProduct` when `--loss-mode ce_arcface` is configured.
  - Passes `labels` during the forward pass in training and validation mode to calculate the margin logits.
  - **CLI Arguments**:
    - `--loss-mode ce` (Default: Stable Cross-Entropy)
    - `--loss-mode ce_arcface` (ArcFace Cosine Margin Cross Entropy)
    - `--arcface-scale` (Default: `30.0`)
    - `--arcface-margin` (Default: `0.50`)
  - **Backward Compatibility**: Kept the default linear classification head and `nn.CrossEntropyLoss` intact when loading checkpoints or running standard inference/evaluation pipelines. Validation and active pipeline loaders filter weights with prefix `backbone.`, allowing existing checkpoints to load safely without error.

---

## 4. Triplet Loss Retuning (B4)
- **Goal**: Tune and adjust Batch-Hard Triplet Loss weight to prevent training degradation.
- **Degradation Documentation**:
  - In previous short experimental training runs, forcing batch-hard triplet loss on small batches alongside cross-entropy degraded classification accuracy due to high variance gradients and collapsing representations under restricted batch sizes.
- **Tuning**:
  - Set default `--triplet-weight` parameter to `0.0` to ensure training stability under normal configurations.
  - Keep triplet loss optional and configurable via:
    - `--triplet-margin` (Default: `0.3`)
    - `--triplet-weight` (Default: `0.0`)
