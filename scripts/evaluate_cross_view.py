import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.cross_view_evaluator import CrossViewEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ARGUS Cross-View Gait Recognition Metrics"
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Max images to evaluate. Default: 500.",
    )

    parser.add_argument(
        "--gallery-ratio",
        type=float,
        default=0.5,
        help="Ratio of features to keep in gallery. Default: 0.5.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Recognition threshold. Default: 0.75.",
    )

    args = parser.parse_args()

    print("\n=== STARTING CROSS-VIEW GAIT EVALUATION ===")
    print(f"Gallery Ratio: {args.gallery_ratio:.2f}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Max Images to test: {args.max_images}")

    evaluator = CrossViewEvaluator(
        gallery_ratio=args.gallery_ratio,
        threshold=args.threshold,
    )

    results = evaluator.evaluate_cross_view(max_test_images=args.max_images)

    print("\n=== CROSS-VIEW EVALUATION RESULTS ===")
    print(f"{'View Angle':<12} | {'Correct':<8} | {'Total':<8} | {'Accuracy':<10}")
    print("-" * 45)
    for angle, metrics in sorted(results["per_view_metrics"].items()):
        print(f"{angle:<12} | {metrics['correct']:<8} | {metrics['total']:<8} | {metrics['accuracy'] * 100:.2f}%")
    print("-" * 45)
    print(f"Fallback view parsing used: {results['fallback_used']}")
    print("Saved JSON Report -> outputs/eval_reports/cross_view_report.json")
    print("Saved CSV Report -> outputs/eval_reports/cross_view_report.csv")


if __name__ == "__main__":
    main()
