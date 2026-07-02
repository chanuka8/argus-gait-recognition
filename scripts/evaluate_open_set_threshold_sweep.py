import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.open_set_evaluator import OpenSetEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ARGUS Open-Set Threshold and Matching Mode Sweep"
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Max images to evaluate per configuration. Default: 500.",
    )

    parser.add_argument(
        "--gallery-ratio",
        type=float,
        default=0.5,
        help="Ratio of features to keep in gallery. Default: 0.5.",
    )

    parser.add_argument(
        "--known-ratio",
        type=float,
        default=0.6,
        help="Ratio of subjects to treat as known. Default: 0.6.",
    )

    args = parser.parse_args()

    thresholds = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    modes = ["flat", "centroid", "centroid_margin", "centroid_margin_topk"]

    print("\n=== RUNNING OPEN-SET THRESHOLD & MATCHING MODE SWEEP ===")
    print(f"Known Ratio: {args.known_ratio:.2f}")
    print(f"Gallery Ratio: {args.gallery_ratio:.2f}")
    print(f"Max Images to test per run: {args.max_images}")

    sweep_results = []

    for mode in modes:
        for t in thresholds:
            print(f"Running mode={mode:<22} threshold={t:.2f}...")
            evaluator = OpenSetEvaluator(
                gallery_ratio=args.gallery_ratio,
                threshold=t,
                known_ratio=args.known_ratio,
            )
            res = evaluator.evaluate_open_set(
                max_test_images=args.max_images,
                matching_mode=mode,
            )
            sweep_results.append({
                "matching_mode": mode,
                "threshold": t,
                "TP": res["TP"],
                "FP": res["FP"],
                "TN": res["TN"],
                "FN": res["FN"],
                "known_accuracy": res["known_accuracy"],
                "unknown_rejection_rate": res["unknown_rejection_rate"],
                "false_accept_rate": res["false_accept_rate"],
                "false_reject_rate": res["false_reject_rate"],
            })

    output_dir = Path("outputs/eval_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "open_set_threshold_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=4)

    csv_path = output_dir / "open_set_threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "matching_mode",
                "threshold",
                "TP",
                "FP",
                "TN",
                "FN",
                "known_accuracy",
                "unknown_rejection_rate",
                "false_accept_rate",
                "false_reject_rate",
            ]
        )
        writer.writeheader()
        for r in sweep_results:
            writer.writerow(r)

    print("\n" + "=" * 110)
    print("OPEN-SET MATCHING SWEEP SUMMARY TABLE")
    print("=" * 110)
    print(
        f"{'Matching Mode':<22} | {'Thresh':<6} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | "
        f"{'Known Acc':<9} | {'Unk Rej':<9} | {'FAR':<8} | {'FRR':<8}"
    )
    print("-" * 110)
    for r in sweep_results:
        print(
            f"{r['matching_mode']:<22} | {r['threshold']:<6.2f} | "
            f"{r['TP']:<4} | {r['FP']:<4} | {r['TN']:<4} | {r['FN']:<4} | "
            f"{r['known_accuracy'] * 100:<8.1f}% | {r['unknown_rejection_rate'] * 100:<8.1f}% | "
            f"{r['false_accept_rate'] * 100:<7.1f}% | {r['false_reject_rate'] * 100:<7.1f}%"
        )
    print("=" * 110)

    print(f"\nSaved JSON report -> {json_path}")
    print(f"Saved CSV report -> {csv_path}")


if __name__ == "__main__":
    main()
