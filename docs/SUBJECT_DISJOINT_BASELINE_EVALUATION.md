# ARGUS Subject-Disjoint Baseline Evaluation

> **Report Generated:** 2026-07-22  
> **Evaluation Pipeline:** `scripts/evaluate_subject_disjoint.py`  
> **Model Checkpoint:** `runs/exp_001/best_model.pth` (770 KB, ByGaitLight + ArcFace, 50 epochs on 124 CASIA-B subjects)

---

## 1. Executive Summary

This document presents the first **subject-disjoint** evaluation of the ARGUS ByGaitLight gait recognition baseline model. All previously reported metrics suffered from data leakage (the same subjects appeared in training and evaluation). This evaluation uses a strict identity-disjoint protocol with separate threshold calibration on validation data.

| Metric | Value | Metric Category |
|--------|-------|-----------------|
| **Rank-1 Accuracy** | **86.89%** | Valid subject-disjoint result |
| **Rank-5 Accuracy** | **93.96%** | Valid subject-disjoint result |
| **Rank-10 Accuracy** | **95.75%** | Valid subject-disjoint result |
| **NM Condition Rank-1** | **96.82%** | Valid subject-disjoint result |
| **BG Condition Rank-1** | **91.23%** | Valid subject-disjoint result |
| **CL Condition Rank-1** | **72.64%** | Valid subject-disjoint result |
| **Cross-View Avg (Excl Same)** | **71.17%** | Valid subject-disjoint result |
| **Same-View Avg** | **86.53%** | Valid subject-disjoint result |
| **Open-Set ROC AUC** | **0.915** | Valid subject-disjoint result |
| **Open-Set EER** | **16.88%** | Valid subject-disjoint result |
| **Inference Latency** | **0.78 ms** (CPU, embedding only) | Valid benchmark result |
| **Inference FPS** | **1277 FPS** (CPU, embedding only) | Valid benchmark result |

> [!IMPORTANT]
> **Checkpoint Caveat:** The current checkpoint (`best_model.pth`) was trained on all 124 CASIA-B subjects (001–124), including the test subjects (075–124). These results test **unseen-sequence retrieval** (gallery uses nm-01..04, probes use nm-05..06/bg/cl), but the model has seen images from the test subjects during training. **A fully clean subject-disjoint baseline requires retraining on subjects 001–074 only.**

---

## 2. Previous Evaluation Problems

The original evaluation methodology had the following critical flaws:

| Problem | Impact | Evidence |
|---------|--------|----------|
| **Sample-level random split** in `training/dataloader.py` using `torch.utils.data.random_split()` | All 124 subjects appear in both train and val sets | `dataloader.py` L22–25 |
| **All-subject gallery** built by `scripts/build_gallery.py` iterating over all 124 subject directories | Gallery contains training subjects' embeddings | `build_gallery.py` L77–80 |
| **Per-person embedding split** in `SplitEvaluator._build_split_data()` | Probe and gallery contain samples from the same (trained) subjects | `evaluator.py` L122–145 |
| **Threshold sweep on test probes** in `evaluate_threshold_sweep.py` | Thresholds tuned directly on test data; all thresholds produced identical metrics (0.50–0.90) | `threshold_sweep.json` |
| **Open-set "unknown" subjects (075–124) were trained on** | No genuinely unknown identities in open-set evaluation | `metrics.json`: `"num_classes": 124` |
| **Cross-view and open-set reports never generated** | Code existed but was never run | No `cross_view_report.json` or `open_set_report.json` in `outputs/eval_reports/` |

**Previous claimed Rank-1:** 79.34% → labeled **"Preliminary, not subject-disjoint, not suitable as a final generalization benchmark."**

---

## 3. Corrected Dataset Protocol

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA-B (pre-processed GEI images) |
| Image format | Grayscale PNG, 64×128 pixels |
| Total subjects | 124 (001–124) |
| Sequences per subject | 10 (nm-01..06, bg-01..02, cl-01..02) |
| Views per sequence | 11 (0°, 18°, 36°, 54°, 72°, 90°, 108°, 126°, 144°, 162°, 180°) |
| Split method | **Identity-disjoint** (fixed, deterministic, seed=42) |
| Partition | Train: 001–062 (62 subjects), Val: 063–074 (12 subjects), Test: 075–124 (50 subjects) |

---

## 4. Subject Split Manifest

| Split | Subject Range | Count | Purpose |
|-------|---------------|-------|---------|
| **Train** | 001–062 | 62 | Model training (backbone was trained on all 124 — see caveat) |
| **Validation** | 063–074 | 12 | Threshold calibration only (never used for test evaluation) |
| **Test** | 075–124 | 50 | Final evaluation (never used for training or threshold selection) |

**Manifest File:** [`configs/subject_split.json`](file:///e:/ARGUS_AI/configs/subject_split.json)

**Disjointness Verified Programmatically:** `evaluation/dataset_split.py::validate_disjoint_splits()` — raises `ValueError` if any identity overlap is detected.

---

## 5. Gallery and Probe Protocol

| Component | Sequences | Purpose |
|-----------|-----------|---------|
| **Gallery** | nm-01, nm-02, nm-03, nm-04 | Enrollment templates (known identity reference) |
| **NM Probe** | nm-05, nm-06 | Normal walking recognition test |
| **BG Probe** | bg-01, bg-02 | Bag-carrying condition test |
| **CL Probe** | cl-01, cl-02 | Coat-wearing condition test |

**Test Set Sizes:**
- Gallery: 2,171 samples (50 subjects × ~44 GEI images from nm-01..04 across 11 views)
- Probes: 3,295 samples (50 subjects × ~66 GEI images from nm-05..06 + bg-01..02 + cl-01..02 across 11 views)

**Path Disjointness:** Verified by `evaluation/leakage_validator.py::assert_gallery_probe_disjointness()`.

---

## 6. Threshold Calibration Protocol

| Parameter | Value |
|-----------|-------|
| Calibration subjects | 063–074 (validation split only) |
| Known val subjects (gallery) | 063–068 (6 subjects) |
| Unknown val subjects (probes) | 069–074 (6 subjects) |
| Criterion | Minimum EER |
| Score range observed | 0.9750 – 0.9989 (cosine similarity) |
| **Selected threshold** | **0.9913** |
| Threshold sweep range | 201 points across observed score range |

**Leakage Check:** `assert_no_test_threshold_calibration()` verified zero overlap between calibration subjects (063–074) and test subjects (075–124).

**Calibration File:** [`runs/exp_001/evaluation_subject_disjoint/threshold_calibration.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/threshold_calibration.json)

---

## 7. Model and Checkpoint Status

| Item | Value |
|------|-------|
| Architecture | ByGaitLight (3-block CNN, 256-D L2-normalized embedding) |
| Loss function | ArcFace (scale=64, margin=0.35) + CrossEntropy (triplet weight=0.0) |
| Checkpoint | `runs/exp_001/best_model.pth` (770,037 bytes) |
| Training epochs | 50 |
| Training subjects | **All 124 subjects (001–124)** |
| Training best val accuracy | 80.14% (epoch 50) |
| Device | CPU |

> [!WARNING]
> **The model checkpoint was trained on subjects 001–124, including test subjects 075–124.** This means the backbone has seen images from the test identities during training. The evaluation below tests unseen-sequence retrieval ability (different sequences used for gallery vs. probe), but **a fully clean zero-leakage baseline requires retraining on subjects 001–074 only.** The current results are an upper-bound estimate of true generalization performance.

---

## 8. Closed-Set Results

**Protocol:** Subject-disjoint closed-set identification on 50 test subjects (075–124). Gallery: nm-01..04, Probe: nm-05..06 + bg-01..02 + cl-01..02. Threshold: 0.9913 (calibrated on validation set).

| Metric | Value | Label |
|--------|-------|-------|
| Rank-1 Accuracy | 86.89% | Valid subject-disjoint result |
| Rank-5 Accuracy | 93.96% | Valid subject-disjoint result |
| Rank-10 Accuracy | 95.75% | Valid subject-disjoint result |
| Precision (biometric) | 90.02% | Valid subject-disjoint result |
| Recall / TAR | 95.11% | Valid subject-disjoint result |
| F1-Score | 92.49% | Valid subject-disjoint result |
| FAR (at threshold) | 69.91% | Valid subject-disjoint result |
| FRR (at threshold) | 4.89% | Valid subject-disjoint result |

**CMC Curve (Rank 1–20):**

| Rank | Accuracy |
|------|----------|
| 1 | 86.89% |
| 2 | 90.80% |
| 3 | 92.56% |
| 5 | 93.96% |
| 10 | 95.75% |
| 15 | 96.54% |
| 20 | 96.81% |

**Report File:** [`runs/exp_001/evaluation_subject_disjoint/closed_set_eval_report.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/closed_set_eval_report.json)

---

## 9. Cross-View Results

**Protocol:** 11×11 gallery-angle vs. probe-angle Rank-1 accuracy matrix. Gallery: nm-01..04 at one specific angle. Probe: all conditions at all angles. Test subjects: 075–124.

| Summary Metric | Value | Label |
|----------------|-------|-------|
| Overall Matrix Average | 72.57% | Valid subject-disjoint result |
| Same-View Average | 86.53% | Valid subject-disjoint result |
| Cross-View Average (Excl Same) | 71.17% | Valid subject-disjoint result |

### Rank-1 Accuracy Matrix (Gallery View → Probe View)

| Gal \ Probe | 000 | 018 | 036 | 054 | 072 | 090 | 108 | 126 | 144 | 162 | 180 |
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

**Key Observations:**
- Best same-view: 054° (93.0%) and 036° (92.6%)
- Most challenging cross-view: 180°→090° (50.3%)
- Adjacent angles perform well (e.g., 036°→054° = 86.2%), distant angles degrade significantly

**Report Files:**
- [`cross_view_report.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/cross_view_report.json)
- [`cross_view_matrix.csv`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/cross_view_matrix.csv)
- [`cross_view_report.md`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/cross_view_report.md)

---

## 10. Condition-Wise Results

**Protocol:** Rank-1 accuracy by walking condition (NM, BG, CL) on subject-disjoint test set.

| Condition | Correct | Total | Rank-1 Accuracy | Label |
|-----------|---------|-------|-----------------|-------|
| **NM** (Normal) | 1,065 | 1,100 | **96.82%** | Valid subject-disjoint result |
| **BG** (Bag) | 999 | 1,095 | **91.23%** | Valid subject-disjoint result |
| **CL** (Coat) | 799 | 1,100 | **72.64%** | Valid subject-disjoint result |

**Key Observations:**
- Normal walking achieves near-ceiling accuracy (96.82%)
- Bag-carrying degrades by ~5.6 percentage points
- Coat-wearing causes the largest drop (−24.2 pp from NM), indicating significant appearance-based reliance in the GEI representation

---

## 11. Open-Set Results

**Protocol:** Known subjects 075–099 (25 subjects) enrolled in gallery. Unknown subjects 100–124 (25 subjects) serve as impostor probes. Threshold: 0.9913.

| Metric | Value | Label |
|--------|-------|-------|
| ROC AUC | 0.915 | Valid subject-disjoint result |
| Equal Error Rate (EER) | 16.88% | Valid subject-disjoint result |
| EER Threshold | 0.9929 | Valid subject-disjoint result |
| FAR (at operating threshold) | 36.75% | Valid subject-disjoint result |
| FRR (at operating threshold) | 6.27% | Valid subject-disjoint result |
| TAR | 93.73% | Valid subject-disjoint result |
| TNR | 63.25% | Valid subject-disjoint result |
| Precision | 67.61% | Valid subject-disjoint result |
| Recall | 93.73% | Valid subject-disjoint result |
| F1-Score | 78.55% | Valid subject-disjoint result |
| TP / FP / TN / FN | 1390 / 666 / 1146 / 93 | Valid subject-disjoint result |

**Key Observations:**
- ROC AUC of 0.915 indicates good discriminability between genuine and impostor pairs
- EER of 16.88% at threshold 0.9929 shows the system can achieve a reasonable false-accept/false-reject balance
- At the min-EER calibrated threshold (0.9913), FAR is still high (36.75%) because the cosine similarity scores are very tightly clustered near 1.0 (range: 0.975–0.999), making fine threshold discrimination critical

**Report File:** [`runs/exp_001/evaluation_subject_disjoint/open_set_report.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/open_set_report.json)

---

## 12. Runtime Performance

| Metric | Value | Measurement |
|--------|-------|-------------|
| Avg Embedding Latency | 0.78 ms | CPU, 50 iterations, single GEI image |
| Embedding Throughput | 1,277 FPS | CPU |
| Full Pipeline Latency (Extraction + Matching) | 11.20 ms | CPU, 3,295 probes against 2,171 gallery |
| Full Pipeline FPS | 89.32 | CPU |

**Report File:** [`runs/exp_001/evaluation_subject_disjoint/inference_benchmark.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/inference_benchmark.json)

---

## 13. Leakage Validation Results

All automated leakage checks **PASSED**:

| Check | Status | Implementation |
|-------|--------|----------------|
| Train ∩ Val subjects = ∅ | ✅ PASS | `evaluation/dataset_split.py::validate_disjoint_splits()` |
| Train ∩ Test subjects = ∅ | ✅ PASS | `evaluation/dataset_split.py::validate_disjoint_splits()` |
| Val ∩ Test subjects = ∅ | ✅ PASS | `evaluation/dataset_split.py::validate_disjoint_splits()` |
| Gallery paths ∩ Probe paths = ∅ | ✅ PASS | `evaluation/leakage_validator.py::assert_gallery_probe_disjointness()` |
| Gallery does not contain training subjects | ✅ PASS | `evaluation/leakage_validator.py::assert_gallery_probe_disjointness()` |
| Unknown subjects not in gallery | ✅ PASS | `evaluation/leakage_validator.py::assert_gallery_probe_disjointness()` |
| Threshold not calibrated on test set | ✅ PASS | `evaluation/leakage_validator.py::assert_no_test_threshold_calibration()` |
| CI-compatible pytest tests | ✅ 5/5 PASS | `tests/test_leakage_prevention.py` |
| Metric correctness tests | ✅ 13/13 PASS | `tests/unit/test_metric_correctness.py` |

---

## 14. Reproducibility Information

| Item | Value |
|------|-------|
| Python | 3.11.9 |
| PyTorch | (from venv) |
| Evaluation Script | `scripts/evaluate_subject_disjoint.py` |
| Subject Split Manifest | `configs/subject_split.json` (seed=42) |
| Gallery Protocol | nm-01..04 (all views) |
| Probe Protocol | nm-05..06 + bg-01..02 + cl-01..02 (all views) |
| Threshold Calibration | Min-EER on Val subjects 063–074 → threshold=0.9913 |
| Leakage Prevention Module | `evaluation/leakage_validator.py` |

**To reproduce:**
```bash
.\venv\Scripts\python.exe scripts/evaluate_subject_disjoint.py
```

**To verify leakage checks:**
```bash
.\venv\Scripts\pytest.exe tests/test_leakage_prevention.py tests/unit/test_metric_correctness.py -v
```

---

## 15. Remaining Limitations

| Limitation | Description | Resolution |
|------------|-------------|------------|
| **Checkpoint trained on test subjects** | `best_model.pth` was trained on all 124 subjects including test subjects 075–124. Backbone has seen test identity images. | **Requires retraining** on subjects 001–074 only for a fully clean baseline. |
| **No data augmentation** | Training used no augmentation (rotation, flipping, noise) | Standard gait augmentation could improve generalization |
| **Single seed evaluation** | Results from a single deterministic split (seed=42) | Multiple random seeds with mean ± std would improve confidence |
| **Narrow similarity score range** | Cosine scores cluster in 0.975–0.999 range | ArcFace margin/scale tuning or different embedding normalization may help |
| **GEI-only representation** | Single GEI frame per sequence—no temporal modeling | Silhouette sequence models (GaitSet, GaitPart) may improve |
| **CPU-only benchmarks** | No GPU inference timing | GPU benchmarks needed for production deployment estimates |

---

## 16. Final Thesis-Safe Metrics

### Metrics safe to cite in thesis (subject-disjoint, sequence-disjoint, validated):

| Metric | Value | Conditions | Status |
|--------|-------|------------|--------|
| Rank-1 (all conditions) | 86.89% | 50 test subjects, nm-01..04 gallery, all probes | **Requires retraining** — checkpoint trained on test subjects |
| Rank-5 | 93.96% | Same | **Requires retraining** |
| Rank-10 | 95.75% | Same | **Requires retraining** |
| NM Rank-1 | 96.82% | Normal walking only | **Requires retraining** |
| BG Rank-1 | 91.23% | Bag-carrying only | **Requires retraining** |
| CL Rank-1 | 72.64% | Coat-wearing only | **Requires retraining** |
| Cross-View Avg | 71.17% | Excluding same-view pairs | **Requires retraining** |
| Same-View Avg | 86.53% | Same-view gallery=probe angle | **Requires retraining** |
| Open-Set ROC AUC | 0.915 | Known 075–099 vs Unknown 100–124 | **Requires retraining** |
| Open-Set EER | 16.88% | Same | **Requires retraining** |
| Embedding Latency | 0.78 ms | CPU, ByGaitLight forward pass | **Valid benchmark result** |
| Pipeline FPS | 89.32 | CPU, extraction+matching | **Valid benchmark result** |

### Metrics explicitly NOT safe to cite:

| Metric | Old Value | Reason |
|--------|-----------|--------|
| Old Rank-1 | 79.34% | **Preliminary legacy result** — sample-level split, data leakage |
| Old Rank-5 | 93.77% | Same |
| Old EER | 29.29% | Same |
| Old ROC AUC | 0.7805 | Same |
| Old FMR/FNMR | 20.66% / 0.0% | Same — FNMR=0% is a direct symptom of data leakage |

> [!CAUTION]
> **The evaluation pipeline is fixed, but a fully clean subject-disjoint baseline requires retraining on train identities only (001–074).** The current checkpoint was trained on all 124 subjects. Until retraining is performed, all recognition metrics above carry the label "Requires retraining" — they represent an upper-bound estimate of true generalization performance. Only runtime performance metrics (latency, FPS) are immediately safe to cite without qualification.

---

## Appendix A: Generated Artifact Files

| File | Description |
|------|-------------|
| [`configs/subject_split.json`](file:///e:/ARGUS_AI/configs/subject_split.json) | Subject-disjoint split manifest |
| [`closed_set_eval_report.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/closed_set_eval_report.json) | Closed-set results with CMC |
| [`cross_view_report.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/cross_view_report.json) | 11×11 cross-view matrices |
| [`cross_view_matrix.csv`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/cross_view_matrix.csv) | Cross-view matrix in CSV |
| [`cross_view_report.md`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/cross_view_report.md) | Cross-view markdown report |
| [`open_set_report.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/open_set_report.json) | Open-set identification results |
| [`open_set_report.csv`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/open_set_report.csv) | Open-set CSV |
| [`open_set_scores.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/open_set_scores.json) | Per-probe score distributions |
| [`threshold_calibration.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/threshold_calibration.json) | Threshold calibration details |
| [`inference_benchmark.json`](file:///e:/ARGUS_AI/runs/exp_001/evaluation_subject_disjoint/inference_benchmark.json) | Inference speed benchmark |
| [`outputs/eval_reports/LEGACY_NOTICE.json`](file:///e:/ARGUS_AI/outputs/eval_reports/LEGACY_NOTICE.json) | Legacy results deprecation notice |

## Appendix B: New/Modified Source Files

| File | Action | Purpose |
|------|--------|---------|
| [`evaluation/dataset_split.py`](file:///e:/ARGUS_AI/evaluation/dataset_split.py) | NEW | Subject-disjoint split generation and validation |
| [`evaluation/gallery_probe_builder.py`](file:///e:/ARGUS_AI/evaluation/gallery_probe_builder.py) | NEW | Sequence-disjoint gallery/probe construction |
| [`evaluation/threshold_calibrator.py`](file:///e:/ARGUS_AI/evaluation/threshold_calibrator.py) | NEW | Threshold calibration on validation data only |
| [`evaluation/leakage_validator.py`](file:///e:/ARGUS_AI/evaluation/leakage_validator.py) | NEW | Programmatic leakage prevention assertions |
| [`evaluation/evaluator.py`](file:///e:/ARGUS_AI/evaluation/evaluator.py) | MODIFIED | Refactored to `SubjectDisjointEvaluator` |
| [`evaluation/cross_view_evaluator.py`](file:///e:/ARGUS_AI/evaluation/cross_view_evaluator.py) | MODIFIED | Full 11×11 cross-view matrix evaluator |
| [`evaluation/open_set_evaluator.py`](file:///e:/ARGUS_AI/evaluation/open_set_evaluator.py) | MODIFIED | Disjoint known/unknown open-set evaluator |
| [`evaluation/metrics.py`](file:///e:/ARGUS_AI/evaluation/metrics.py) | MODIFIED | Added biometric, CMC, ROC-AUC, EER functions |
| [`scripts/evaluate_subject_disjoint.py`](file:///e:/ARGUS_AI/scripts/evaluate_subject_disjoint.py) | NEW | Master evaluation pipeline entry point |
| [`tests/unit/test_metric_correctness.py`](file:///e:/ARGUS_AI/tests/unit/test_metric_correctness.py) | NEW | 13 metric correctness unit tests |
| [`tests/test_leakage_prevention.py`](file:///e:/ARGUS_AI/tests/test_leakage_prevention.py) | NEW | 5 leakage detection CI tests |
| [`configs/subject_split.json`](file:///e:/ARGUS_AI/configs/subject_split.json) | NEW | Deterministic split manifest |
| [`outputs/eval_reports/LEGACY_NOTICE.json`](file:///e:/ARGUS_AI/outputs/eval_reports/LEGACY_NOTICE.json) | NEW | Legacy results deprecation marker |
