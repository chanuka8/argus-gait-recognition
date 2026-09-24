import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml_platform.evaluation.cross_view_evaluator import SubjectDisjointCrossViewEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARGUS Cross-View Gait Recognition Metrics")

    parser.add_argument("--gei-root", type=str, default="data/datasets/casia_processed/gei")
    parser.add_argument("--model-path", type=str, default="runs/exp_001/best_model.pth")
    parser.add_argument("--split-config", type=str, default="configs/subject_split.json")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output-dir", type=str, default="runs/exp_001/evaluation_cross_view")

    args = parser.parse_args()

    print("\n=== STARTING CROSS-VIEW GAIT EVALUATION ===")
    print(f"GEI Root: {args.gei_root}")
    print(f"Model Path: {args.model_path}")
    print(f"Threshold: {args.threshold:.2f}")

    evaluator = SubjectDisjointCrossViewEvaluator(
        gei_root=args.gei_root,
        model_path=args.model_path,
        split_config_path=args.split_config,
        threshold=args.threshold,
        report_dir=args.output_dir,
    )

    results = evaluator.evaluate_cross_view_matrices()

    print("\n=== CROSS-VIEW EVALUATION RESULTS ===")
    print(f"{'Gallery Angle':<15} | {'Avg Rank-1 Accuracy':<20}")
    print("-" * 40)
    for gallery_angle in results["angles_evaluated"]:
        per_probe_angle = results["matrix_rank1"].get(gallery_angle, {})
        avg_accuracy = sum(per_probe_angle.values()) / len(per_probe_angle) if per_probe_angle else 0.0
        print(f"{gallery_angle:<15} | {avg_accuracy * 100:.2f}%")
    print("-" * 40)
    print(f"Same-View Avg:      {results['same_view_average_rank1'] * 100:.2f}%")
    print(f"Cross-View Avg:     {results['cross_view_average_rank1'] * 100:.2f}%")
    print(f"Overall Matrix Avg: {results['overall_average_rank1'] * 100:.2f}%")
    print("-" * 40)
    print(f"Saved JSON Report -> {evaluator.report_dir / 'cross_view_report.json'}")
    print(f"Saved CSV Report -> {evaluator.report_dir / 'cross_view_matrix.csv'}")
    print(f"Saved Markdown Report -> {evaluator.report_dir / 'cross_view_report.md'}")


if __name__ == "__main__":
    main()
