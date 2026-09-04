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
| [analyze_cl_part_similarity.py](analyze_cl_part_similarity.py) | Module/resource file analyze_cl_part_similarity.py |
| [analyze_open_set_and_cl.py](analyze_open_set_and_cl.py) | Module/resource file analyze_open_set_and_cl.py |
| [cross_view_evaluator.py](cross_view_evaluator.py) | Evaluates cross-camera view angle invariance across CASIA-B viewing angles |
| [dataset_split.py](dataset_split.py) | Partitions dataset subjects into subject-disjoint train, gallery, and probe splits |
| [diagnose_f1_score.py](diagnose_f1_score.py) | Module/resource file diagnose_f1_score.py |
| [evaluate_appearance_recognition.py](evaluate_appearance_recognition.py) | Module/resource file evaluate_appearance_recognition.py |
| [evaluate_cross_view.py](evaluate_cross_view.py) | Module/resource file evaluate_cross_view.py |
| [evaluate_dual_modal_recognition.py](evaluate_dual_modal_recognition.py) | Module/resource file evaluate_dual_modal_recognition.py |
| [evaluate_exp004.py](evaluate_exp004.py) | Module/resource file evaluate_exp004.py |
| [evaluate_model.py](evaluate_model.py) | Module/resource file evaluate_model.py |
| [evaluate_open_set.py](evaluate_open_set.py) | Module/resource file evaluate_open_set.py |
| [evaluate_open_set_threshold_sweep.py](evaluate_open_set_threshold_sweep.py) | Module/resource file evaluate_open_set_threshold_sweep.py |
| [evaluate_subject_disjoint.py](evaluate_subject_disjoint.py) | Module/resource file evaluate_subject_disjoint.py |
| [evaluate_threshold_sweep.py](evaluate_threshold_sweep.py) | Module/resource file evaluate_threshold_sweep.py |
| [evaluator.py](evaluator.py) | Core evaluation harness running identification evaluation protocols |
| [evaluator_3d.py](evaluator_3d.py) | Module/resource file evaluator_3d.py |
| [f1_calibration_validation.py](f1_calibration_validation.py) | Module/resource file f1_calibration_validation.py |
| [gallery_probe_builder.py](gallery_probe_builder.py) | Constructs gallery feature matrices and query probe vectors for test evaluation |
| [generate_visualizer_charts.py](generate_visualizer_charts.py) | Module/resource file generate_visualizer_charts.py |
| [leakage_validator.py](leakage_validator.py) | Asserts zero identity overlap between gallery and probe datasets |
| [metrics.py](metrics.py) | Mathematical functions for Rank-k accuracy, EER, ROC AUC, FAR, and FRR |
| [open_set_evaluator.py](open_set_evaluator.py) | Evaluates open-set rejection and unknown subject detection performance |
| [roc.py](roc.py) | Computes Receiver Operating Characteristic (ROC) curve metrics |
| [run_ablation_study.py](run_ablation_study.py) | Module/resource file run_ablation_study.py |
| [run_exp004_ablations.py](run_exp004_ablations.py) | Module/resource file run_exp004_ablations.py |
| [run_exp006_3d.py](run_exp006_3d.py) | Module/resource file run_exp006_3d.py |
| [run_exp006_full.py](run_exp006_full.py) | Module/resource file run_exp006_full.py |
| [run_exp007_ablations.py](run_exp007_ablations.py) | Module/resource file run_exp007_ablations.py |
| [simulate_date_aware_learning.py](simulate_date_aware_learning.py) | Module/resource file simulate_date_aware_learning.py |
| [sweep_fine_thresholds.py](sweep_fine_thresholds.py) | Module/resource file sweep_fine_thresholds.py |
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
