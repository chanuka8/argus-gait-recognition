import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml_platform.evaluation.open_set_evaluator import SubjectDisjointOpenSetEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARGUS Open-Set Threshold Sweep")

    parser.add_argument("--gei-root", type=str, default="data/datasets/casia_processed/gei")
    parser.add_argument("--model-path", type=str, default="runs/exp_001/best_model.pth")
    parser.add_argument("--split-config", type=str, default="configs/subject_split.json")
    parser.add_argument("--known-ratio", type=float, default=0.5, help="Ratio of test subjects treated as known.")
    parser.add_argument("--output-dir", type=str, default="runs/exp_001/evaluation_open_set/threshold_sweep")

    args = parser.parse_args()

    thresholds = [0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]

    print("\n=== RUNNING OPEN-SET THRESHOLD SWEEP ===")
    print(f"Known Ratio: {args.known_ratio:.2f}")
    print(f"Thresholds: {thresholds}")

    sweep_results = []

    for t in thresholds:
        print(f"Running threshold={t:.2f}...")
        evaluator = SubjectDisjointOpenSetEvaluator(
            gei_root=args.gei_root,
            model_path=args.model_path,
            split_config_path=args.split_config,
            threshold=t,
            known_ratio=args.known_ratio,
            report_dir=str(Path(args.output_dir) / f"threshold_{t:.2f}"),
        )
        res = evaluator.evaluate_open_set_protocol()
        operating = res["operating_metrics"]

        sweep_results.append(
            {
                "threshold": t,
                "ROC_AUC": res["ROC_AUC"],
                "EER": res["EER"],
                "FAR": operating["FAR"],
                "FRR": operating["FRR"],
                "TAR": operating["TAR"],
                "TNR": operating["TNR"],
                "precision": operating["precision"],
                "recall": operating["recall"],
                "f1_score": operating["f1_score"],
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "open_set_threshold_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=4)

    csv_path = output_dir / "open_set_threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "ROC_AUC",
                "EER",
                "FAR",
                "FRR",
                "TAR",
                "TNR",
                "precision",
                "recall",
                "f1_score",
            ],
        )
        writer.writeheader()
        for r in sweep_results:
            writer.writerow(r)

    print("\n" + "=" * 110)
    print("OPEN-SET THRESHOLD SWEEP SUMMARY TABLE")
    print("=" * 110)
    print(
        f"{'Thresh':<8} | {'ROC AUC':<9} | {'EER':<8} | {'FAR':<8} | {'FRR':<8} | "
        f"{'TAR':<8} | {'TNR':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8}"
    )
    print("-" * 110)
    for r in sweep_results:
        print(
            f"{r['threshold']:<8.2f} | {r['ROC_AUC']:<9.4f} | {r['EER']:<8.4f} | "
            f"{r['FAR']:<8.4f} | {r['FRR']:<8.4f} | {r['TAR']:<8.4f} | {r['TNR']:<8.4f} | "
            f"{r['precision']:<8.4f} | {r['recall']:<8.4f} | {r['f1_score']:<8.4f}"
        )
    print("=" * 110)

    print(f"\nSaved JSON report -> {json_path}")
    print(f"Saved CSV report -> {csv_path}")


if __name__ == "__main__":
    main()
