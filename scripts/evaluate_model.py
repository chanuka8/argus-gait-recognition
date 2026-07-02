import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import SplitEvaluator


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Evaluate ARGUS gait recognition model"
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--gallery-ratio",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    evaluator = SplitEvaluator(
        gallery_ratio=args.gallery_ratio,
    )

    results = evaluator.evaluate(
        max_test_images=args.max_images,
    )

    print("\n=== ARGUS SPLIT EVALUATION ===")

    for key, value in results.items():
        print(f"{key}: {value}")

    report_dir = Path(
        "outputs/eval_reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        report_dir /
        "split_eval_report.json"
    )

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

    print(
        f"\nReport saved -> {report_file}"
    )


if __name__ == "__main__":
    main()