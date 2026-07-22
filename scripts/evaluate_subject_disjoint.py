import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset_split import load_or_create_subject_split
from evaluation.threshold_calibrator import ThresholdCalibrator
from evaluation.evaluator import SubjectDisjointEvaluator
from evaluation.cross_view_evaluator import SubjectDisjointCrossViewEvaluator
from evaluation.open_set_evaluator import SubjectDisjointOpenSetEvaluator
from evaluation.leakage_validator import assert_no_test_threshold_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full ARGUS Subject-Disjoint Baseline Evaluation Pipeline")
    parser.add_argument("--model-path", type=str, default="runs/exp_001/best_model.pth")
    parser.add_argument("--gei-root", type=str, default="data/casia_processed/gei")
    parser.add_argument("--split-config", type=str, default="configs/subject_split.json")
    parser.add_argument("--output-dir", type=str, default="runs/exp_001/evaluation_subject_disjoint")
    parser.add_argument("--calibration-criterion", type=str, choices=["min_eer", "max_f1", "target_far"], default="min_eer")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("      ARGUS SUBJECT-DISJOINT EVALUATION PIPELINE       ")
    print("=======================================================")

    # 1. Load Subject Split Manifest
    split_manifest = load_or_create_subject_split(config_path=args.split_config, data_dir=args.gei_root)
    train_subs = split_manifest["train_subjects"]
    val_subs = split_manifest["val_subjects"]
    test_subs = split_manifest["test_subjects"]

    print("\n[1/6] Subject Partitioning Verified:")
    print(f"  - Train Subjects ({len(train_subs)}): {train_subs[0]} .. {train_subs[-1]}")
    print(f"  - Val Subjects   ({len(val_subs)}): {val_subs[0]} .. {val_subs[-1]}")
    print(f"  - Test Subjects  ({len(test_subs)}): {test_subs[0]} .. {test_subs[-1]}")

    # 2. Threshold Calibration on Validation Set ONLY
    print(f"\n[2/6] Calibrating Operating Threshold on Validation Set ({len(val_subs)} subjects)...")
    evaluator_base = SubjectDisjointEvaluator(
        gei_root=args.gei_root,
        model_path=args.model_path,
        split_config_path=args.split_config,
        report_dir=str(out_dir),
    )

    calibrator = ThresholdCalibrator(
        val_subjects=val_subs,
        feature_extractor_fn=evaluator_base.image_to_embedding,
    )
    calib_res = calibrator.calibrate(
        criterion=args.calibration_criterion,
        gei_root=args.gei_root,
        output_dir=str(out_dir),
    )

    calibrated_threshold = calib_res["selected_threshold"]
    print(f"  -> Calibration Criterion: {args.calibration_criterion}")
    print(f"  -> Selected Operating Threshold: {calibrated_threshold}")

    # Assert no test subjects were used in calibration
    assert_no_test_threshold_calibration(val_subs, test_subs)

    # 3. Closed-Set Identification on Test Set
    print(f"\n[3/6] Running Subject-Disjoint Closed-Set Evaluation on Test Set ({len(test_subs)} subjects)...")
    evaluator = SubjectDisjointEvaluator(
        gei_root=args.gei_root,
        model_path=args.model_path,
        split_config_path=args.split_config,
        threshold=calibrated_threshold,
        report_dir=str(out_dir),
    )
    closed_set_res = evaluator.evaluate()

    print(f"  -> Rank-1 Accuracy:  {closed_set_res['rank1_accuracy']*100:.2f}%")
    print(f"  -> Rank-5 Accuracy:  {closed_set_res['rank5_accuracy']*100:.2f}%")
    print(f"  -> Rank-10 Accuracy: {closed_set_res['rank10_accuracy']*100:.2f}%")

    # 4. Cross-View Matrix Evaluation
    print("\n[4/6] Running 11 x 11 Cross-View Matrix Evaluation...")
    cv_evaluator = SubjectDisjointCrossViewEvaluator(
        gei_root=args.gei_root,
        model_path=args.model_path,
        split_config_path=args.split_config,
        threshold=calibrated_threshold,
        report_dir=str(out_dir),
    )
    cv_res = cv_evaluator.evaluate_cross_view_matrices()
    print(f"  -> Cross-View Avg (Excl Same View): {cv_res['cross_view_average_rank1']*100:.2f}%")
    print(f"  -> Same-View Avg:                  {cv_res['same_view_average_rank1']*100:.2f}%")
    print(f"  -> Overall Matrix Avg:            {cv_res['overall_average_rank1']*100:.2f}%")

    # 5. Open-Set Unknown Rejection Evaluation
    print("\n[5/6] Running Open-Set Evaluation (Known 075-099 vs Unknown 100-124)...")
    open_set_evaluator = SubjectDisjointOpenSetEvaluator(
        gei_root=args.gei_root,
        model_path=args.model_path,
        split_config_path=args.split_config,
        threshold=calibrated_threshold,
        known_ratio=0.5,
        report_dir=str(out_dir),
    )
    open_set_res = open_set_evaluator.evaluate_open_set_protocol()
    print(f"  -> ROC AUC:        {open_set_res['ROC_AUC']}")
    print(f"  -> Equal Error Rate (EER): {open_set_res['EER']*100:.2f}%")
    print(f"  -> FAR at threshold:       {open_set_res['operating_metrics']['FAR']*100:.2f}%")
    print(f"  -> FRR at threshold:       {open_set_res['operating_metrics']['FRR']*100:.2f}%")

    # 6. CPU Inference Speed Benchmark
    print("\n[6/6] Running Inference Speed Benchmark...")
    test_sample = list(Path(args.gei_root).glob("*/*.png"))[0]
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        evaluator.image_to_embedding(test_sample)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    avg_lat_ms = (sum(latencies) / len(latencies)) * 1000.0
    fps = len(latencies) / sum(latencies)

    bench_res = {
        "device": "CPU",
        "sample_image": str(test_sample),
        "iterations": len(latencies),
        "avg_latency_ms": round(avg_lat_ms, 4),
        "fps": round(fps, 2),
    }
    with open(out_dir / "inference_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(bench_res, f, indent=4)

    print(f"  -> Avg Latency: {avg_lat_ms:.2f} ms | Throughput: {fps:.2f} FPS")

    print("\n=======================================================")
    print(f"   ALL ARTIFACTS GENERATED IN: {args.output_dir}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
