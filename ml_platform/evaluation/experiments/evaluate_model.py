import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml_platform.evaluation.evaluator import SubjectDisjointEvaluator


def main() -> None:

    parser = argparse.ArgumentParser(description="Evaluate ARGUS gait recognition model")

    parser.add_argument("--gei-root", type=str, default="data/datasets/casia_processed/gei")
    parser.add_argument("--model-path", type=str, default="runs/exp_001/best_model.pth")
    parser.add_argument("--split-config", type=str, default="configs/subject_split.json")
    parser.add_argument("--threshold", type=float, default=0.85)

    args = parser.parse_args()

    evaluator = SubjectDisjointEvaluator(
        gei_root=args.gei_root,
        model_path=args.model_path,
        split_config_path=args.split_config,
        threshold=args.threshold,
    )

    results = evaluator.evaluate()

    print("\n=== ARGUS SPLIT EVALUATION ===")

    for key, value in results.items():
        print(f"{key}: {value}")

    report_dir = Path("outputs/reports/evaluation")

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = report_dir / "split_eval_report.json"

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(f"\nReport saved -> {report_file}")


if __name__ == "__main__":
    main()
