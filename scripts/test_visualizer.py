import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.visualizer import EvaluationVisualizer


def main():

    visualizer = EvaluationVisualizer()

    visualizer.plot_accuracy(
        0.528
    )

    visualizer.plot_correct_vs_incorrect(
        correct=264,
        incorrect=236,
    )

    visualizer.plot_confidence_scores(
        [
            0.81,
            0.83,
            0.84,
            0.85,
            0.88,
            0.90,
            0.92,
            0.86,
            0.87,
            0.89,
        ]
    )

    print("Evaluation charts generated.")


if __name__ == "__main__":
    main()