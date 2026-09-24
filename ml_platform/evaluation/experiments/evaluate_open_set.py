import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml_platform.evaluation.open_set_evaluator import SubjectDisjointOpenSetEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARGUS Open-Set Gait Recognition Metrics")

    parser.add_argument("--gei-root", type=str, default="data/datasets/casia_processed/gei")
    parser.add_argument("--model-path", type=str, default="runs/exp_001/best_model.pth")
    parser.add_argument("--split-config", type=str, default="configs/subject_split.json")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--known-ratio", type=float, default=0.5, help="Ratio of test subjects treated as known.")
    parser.add_argument("--output-dir", type=str, default="runs/exp_001/evaluation_open_set")

    args = parser.parse_args()

    print("\n=== STARTING OPEN-SET GAIT EVALUATION ===")
    print(f"Known Ratio: {args.known_ratio:.2f}")
    print(f"Threshold: {args.threshold:.2f}")

    evaluator = SubjectDisjointOpenSetEvaluator(
        gei_root=args.gei_root,
        model_path=args.model_path,
        split_config_path=args.split_config,
        threshold=args.threshold,
        known_ratio=args.known_ratio,
        report_dir=args.output_dir,
    )

    results = evaluator.evaluate_open_set_protocol()

    operating = results["operating_metrics"]
    state_counts = results["open_set_state_counts"]

    print("\n=== OPEN-SET EVALUATION RESULTS ===")
    print(f"Gallery Samples: {results['gallery_samples_count']}")
    print(f"Total Probes: {results['total_probe_count']}")
    print(f"Known Probes: {results['known_probe_count']}")
    print(f"Unknown Probes: {results['unknown_probe_count']}")
    print("-" * 40)
    print(f"Open-Set State Counts: KNOWN={state_counts['KNOWN']} UNKNOWN={state_counts['UNKNOWN']} UNCERTAIN={state_counts['UNCERTAIN']}")
    print("-" * 40)
    print(f"ROC AUC: {results['ROC_AUC']}")
    print(f"Equal Error Rate (EER): {results['EER'] * 100:.2f}%")
    print(f"False Accept Rate (FAR): {operating['FAR'] * 100:.2f}%")
    print(f"False Reject Rate (FRR): {operating['FRR'] * 100:.2f}%")
    print(f"True Accept Rate (TAR): {operating['TAR'] * 100:.2f}%")
    print(f"True Negative Rate (TNR): {operating['TNR'] * 100:.2f}%")
    print(f"Precision: {operating['precision']:.4f}  Recall: {operating['recall']:.4f}  F1: {operating['f1_score']:.4f}")
    print("-" * 40)
    print(f"Saved JSON Report -> {evaluator.report_dir / 'open_set_report.json'}")
    print(f"Saved CSV Report -> {evaluator.report_dir / 'open_set_report.csv'}")


if __name__ == "__main__":
    main()
