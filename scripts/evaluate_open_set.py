import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.open_set_evaluator import OpenSetEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ARGUS Open-Set Gait Recognition Metrics"
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
        help="Ratio of features to keep in gallery for known subjects. Default: 0.5.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Rejection threshold. Default: 0.85.",
    )

    parser.add_argument(
        "--known-ratio",
        type=float,
        default=0.6,
        help="Ratio of subjects to treat as known. Default: 0.6.",
    )

    parser.add_argument(
        "--matching-mode",
        type=str,
        choices=["flat", "centroid", "centroid_margin", "centroid_margin_topk"],
        default="flat",
        help="Matching step algorithm to evaluate. Default: flat.",
    )

    args = parser.parse_args()

    print("\n=== STARTING OPEN-SET GAIT EVALUATION ===")
    print(f"Matching Mode: {args.matching_mode}")
    print(f"Known Ratio: {args.known_ratio:.2f}")
    print(f"Gallery Ratio: {args.gallery_ratio:.2f}")
    print(f"Rejection Threshold: {args.threshold:.2f}")
    print(f"Max Images to test: {args.max_images}")

    evaluator = OpenSetEvaluator(
        gallery_ratio=args.gallery_ratio,
        threshold=args.threshold,
        known_ratio=args.known_ratio,
    )

    results = evaluator.evaluate_open_set(
        max_test_images=args.max_images,
        matching_mode=args.matching_mode,
    )

    print("\n=== OPEN-SET EVALUATION RESULTS ===")
    print(f"Total Tested: {results['total_tested']}")
    print(f"Known Subjects Queries: {results['total_known']}")
    print(f"Unknown Subjects Queries: {results['total_unknown']}")
    print("-" * 40)
    print(f"True Positives (TP): {results['TP']}")
    print(f"False Positives (FP): {results['FP']}")
    print(f"True Negatives (TN): {results['TN']}")
    print(f"False Negatives (FN): {results['FN']}")
    print("-" * 40)
    print(f"Known Subject Acc: {results['known_accuracy'] * 100:.2f}%")
    print(f"Unknown Rejection Rate: {results['unknown_rejection_rate'] * 100:.2f}%")
    print(f"False Accept Rate (FAR): {results['false_accept_rate'] * 100:.2f}%")
    print(f"False Reject Rate (FRR): {results['false_reject_rate'] * 100:.2f}%")
    print("-" * 40)
    print("Saved JSON Report -> outputs/eval_reports/open_set_report.json")
    print("Saved CSV Report -> outputs/eval_reports/open_set_report.csv")


if __name__ == "__main__":
    main()
