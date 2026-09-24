import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_platform.evaluation.evaluator import SubjectDisjointEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARGUS thresholds via sweep evaluation")
    parser.add_argument("--gei-root", type=str, default="data/datasets/casia_processed/gei")
    parser.add_argument("--model-path", type=str, default="runs/exp_001/best_model.pth")
    parser.add_argument("--split-config", type=str, default="configs/subject_split.json")
    parser.add_argument("--output-dir", type=str, default="runs/exp_001/evaluation_threshold_sweep")

    args = parser.parse_args()

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    sweep_results = []

    print(f"Starting subject-disjoint threshold sweep evaluation over: {thresholds}")

    for t in thresholds:
        print(f"Evaluating threshold: {t:.2f}...")

        evaluator = SubjectDisjointEvaluator(
            gei_root=args.gei_root,
            model_path=args.model_path,
            split_config_path=args.split_config,
            threshold=t,
            report_dir=str(Path(args.output_dir) / f"threshold_{t:.2f}"),
        )
        results = evaluator.evaluate()
        rates = results["biometric_rates"]

        sweep_results.append(
            {
                "threshold": t,
                "rank1_accuracy": results["rank1_accuracy"],
                "rank5_accuracy": results["rank5_accuracy"],
                "rank10_accuracy": results["rank10_accuracy"],
                "precision": rates["precision"],
                "recall": rates["recall"],
                "f1_score": rates["f1_score"],
                "FAR": rates["FAR"],
                "FRR": rates["FRR"],
                "avg_inference_latency_ms": results["avg_inference_latency_ms"],
                "fps": results["inference_fps"],
            }
        )

    sorted_results = sorted(
        sweep_results,
        key=lambda x: x["rank1_accuracy"],
        reverse=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "threshold_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=4)

    csv_path = output_dir / "threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "rank1_accuracy",
                "rank5_accuracy",
                "rank10_accuracy",
                "precision",
                "recall",
                "f1_score",
                "FAR",
                "FRR",
                "avg_inference_latency_ms",
                "fps",
            ],
        )
        writer.writeheader()
        for row in sweep_results:
            writer.writerow(row)

    print("\n=== THRESHOLD SWEEP SUMMARY TABLE (Sorted by Rank-1 Accuracy) ===")
    print(
        f"{'Threshold':<10} | {'Rank-1 Acc':<12} | {'Rank-5 Acc':<12} | {'Rank-10 Acc':<12} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1':<10} | "
        f"{'FAR':<8} | {'FRR':<8} | "
        f"{'Avg ms':<10} | {'FPS':<8}"
    )
    print("-" * 150)
    for res in sorted_results:
        print(
            f"{res['threshold']:<10.2f} | "
            f"{res['rank1_accuracy']:<12.4f} | "
            f"{res['rank5_accuracy']:<12.4f} | "
            f"{res['rank10_accuracy']:<12.4f} | "
            f"{res['precision']:<10.4f} | "
            f"{res['recall']:<10.4f} | "
            f"{res['f1_score']:<10.4f} | "
            f"{res['FAR']:<8.4f} | "
            f"{res['FRR']:<8.4f} | "
            f"{res['avg_inference_latency_ms']:<10.4f} | "
            f"{res['fps']:<8.2f}"
        )

    print(f"\nSaved JSON report -> {json_path}")
    print(f"Saved CSV report -> {csv_path}")


if __name__ == "__main__":
    main()
