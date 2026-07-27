# Evaluation

The `evaluation` package implements scientific benchmarks, metrics evaluation, subject-disjoint protocol split testing, data leakage validation, threshold calibration, and evaluation visualizers for ARGUS AI.

## Responsibilities

- Computing scientific biometric metrics: Rank-1/Rank-5 accuracy, EER (Equal Error Rate), ROC-AUC, FAR, and FRR.
- Enforcing strict subject-disjoint partitioning between training, gallery, and open-set probe identity sets.
- Generating evaluation report files and visual charts in `outputs/reports/evaluation/`.
- Boundaries: Does not train neural networks or execute real-time RTSP video capture streams.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [cross_view_evaluator.py](file:///E:/ARGUS_AI/evaluation/cross_view_evaluator.py) | Evaluates cross-camera view angle invariance across CASIA-B viewing angles |
| [dataset_split.py](file:///E:/ARGUS_AI/evaluation/dataset_split.py) | Partitions dataset subjects into subject-disjoint train, gallery, and probe splits |
| [evaluator.py](file:///E:/ARGUS_AI/evaluation/evaluator.py) | Core evaluation harness running identification evaluation protocols |
| [gallery_probe_builder.py](file:///E:/ARGUS_AI/evaluation/gallery_probe_builder.py) | Constructs gallery feature matrices and query probe vectors for test evaluation |
| [leakage_validator.py](file:///E:/ARGUS_AI/evaluation/leakage_validator.py) | Asserts zero identity overlap between gallery and probe datasets |
| [metrics.py](file:///E:/ARGUS_AI/evaluation/metrics.py) | Mathematical functions for Rank-k accuracy, EER, ROC AUC, FAR, and FRR |
| [open_set_evaluator.py](file:///E:/ARGUS_AI/evaluation/open_set_evaluator.py) | Evaluates open-set rejection and unknown subject detection performance |
| [roc.py](file:///E:/ARGUS_AI/evaluation/roc.py) | Computes Receiver Operating Characteristic (ROC) curve metrics |
| [threshold_calibrator.py](file:///E:/ARGUS_AI/evaluation/threshold_calibrator.py) | Calibrates verification decision thresholds to target FAR/FRR metrics |
| [visualizer.py](file:///E:/ARGUS_AI/evaluation/visualizer.py) | Generates evaluation plots and confusion matrices in `outputs/reports/evaluation/charts` |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Dataset GEIs / Models → `evaluation/evaluator.py` → `evaluation/metrics.py` → `evaluation/visualizer.py` → `outputs/reports/evaluation/`.

## Configuration

- [configs/subject_split.json](file:///e:/ARGUS_AI/configs/subject_split.json): subject-disjoint train/test split manifest
- [configs/gallery_probe_manifest.json](file:///e:/ARGUS_AI/configs/gallery_probe_manifest.json): gallery/probe set configuration

## Public Interfaces

- `SubjectDisjointEvaluator`: Benchmark evaluator in [evaluation/evaluator.py](file:///e:/ARGUS_AI/evaluation/evaluator.py).
- `SubjectDisjointOpenSetEvaluator`: Open-set evaluator in [evaluation/open_set_evaluator.py](file:///e:/ARGUS_AI/evaluation/open_set_evaluator.py).
- `EvaluationVisualizer`: Chart renderer in [evaluation/visualizer.py](file:///e:/ARGUS_AI/evaluation/visualizer.py).
- `compute_biometric_rates()`: Metrics calculation in [evaluation/metrics.py](file:///e:/ARGUS_AI/evaluation/metrics.py).

## Tests

- [tests/test_leakage_prevention.py](file:///e:/ARGUS_AI/tests/test_leakage_prevention.py)
- [scripts/evaluate_model.py](file:///e:/ARGUS_AI/scripts/evaluate_model.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Training Documentation](file:///e:/ARGUS_AI/training/README.md)
