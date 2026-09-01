# Evaluation

The `evaluation` package implements scientific benchmarks, metrics evaluation, subject-disjoint protocol split testing, data leakage validation, threshold calibration, and evaluation visualizers for ARGUS AI.

## Responsibilities

- Computing scientific biometric metrics: Rank-1/Rank-5 accuracy, EER (Equal Error Rate), ROC-AUC, FAR, and FRR.
- Enforcing strict subject-disjoint partitioning between training, gallery, and open-set probe identity sets.
- Housing full-scale biometric benchmarks in `evaluation/benchmarks/`.
- Housing evaluation diagnostic and audit scripts in `evaluation/scripts/`.
- Persisting evaluation reports, metric records, and generated charts in `evaluation/results/`.
- Boundaries: Does not train neural networks or execute real-time RTSP video capture streams.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [cross_view_evaluator.py](cross_view_evaluator.py) | Evaluates cross-camera view angle invariance across CASIA-B viewing angles |
| [dataset_split.py](dataset_split.py) | Partitions dataset subjects into subject-disjoint train, gallery, and probe splits |
| [evaluator.py](evaluator.py) | Core evaluation harness running identification evaluation protocols |
| [evaluator_3d.py](evaluator_3d.py) | Module/resource file evaluator_3d.py |
| [gallery_probe_builder.py](gallery_probe_builder.py) | Constructs gallery feature matrices and query probe vectors for test evaluation |
| [leakage_validator.py](leakage_validator.py) | Asserts zero identity overlap between gallery and probe datasets |
| [metrics.py](metrics.py) | Mathematical functions for Rank-k accuracy, EER, ROC AUC, FAR, and FRR |
| [open_set_evaluator.py](open_set_evaluator.py) | Evaluates open-set rejection and unknown subject detection performance |
| [roc.py](roc.py) | Computes Receiver Operating Characteristic (ROC) curve metrics |
| [threshold_calibrator.py](threshold_calibrator.py) | Calibrates verification decision thresholds to target FAR/FRR metrics |
| [visualizer.py](visualizer.py) | Generates evaluation plots and confusion matrices in `outputs/reports/evaluation/charts` |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Dataset GEIs / Models → `evaluation/evaluator.py` → `evaluation/metrics.py` → `evaluation/visualizer.py` → `outputs/reports/evaluation/`.

## Configuration

- [configs/subject_split.json](../configs/subject_split.json): subject-disjoint train/test split manifest
- [configs/gallery_probe_manifest.json](../configs/gallery_probe_manifest.json): gallery/probe set configuration

## Public Interfaces

- `SubjectDisjointEvaluator`: Benchmark evaluator in [evaluation/evaluator.py](evaluator.py).
- `SubjectDisjointOpenSetEvaluator`: Open-set evaluator in [evaluation/open_set_evaluator.py](open_set_evaluator.py).
- `EvaluationVisualizer`: Chart renderer in [evaluation/visualizer.py](visualizer.py).
- `compute_biometric_rates()`: Metrics calculation in [evaluation/metrics.py](metrics.py).

## Tests

- [tests/test_leakage_prevention.py](../tests/test_leakage_prevention.py)
- [scripts/evaluate_model.py](../scripts/evaluate_model.py)

## Related Documentation

- [Root README](../README.md)
- [Training Documentation](../training/README.md)
