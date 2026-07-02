import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.confidence_scorer import ConfidenceScorer


def main():
    scorer = ConfidenceScorer()

    scores = [
        0.91,
        0.84,
        0.77,
        0.65,
        0.42,
    ]

    for score in scores:
        print(
            scorer.evaluate(score)
        )


if __name__ == "__main__":
    main()