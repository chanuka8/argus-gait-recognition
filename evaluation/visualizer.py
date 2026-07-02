from pathlib import Path

import matplotlib.pyplot as plt


class EvaluationVisualizer:
    def __init__(
        self,
        output_dir="outputs/evaluation_charts",
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot_accuracy(
        self,
        accuracy,
        filename="accuracy.png",
    ):
        plt.figure(figsize=(6, 4))

        plt.bar(
            ["Rank-1 Accuracy"],
            [accuracy],
        )

        plt.ylim(0, 1)

        plt.ylabel("Accuracy")

        plt.title("ARGUS Rank-1 Accuracy")

        output = self.output_dir / filename

        plt.savefig(
            output,
            bbox_inches="tight",
        )

        plt.close()

        return output

    def plot_correct_vs_incorrect(
        self,
        correct,
        incorrect,
        filename="correct_vs_incorrect.png",
    ):
        plt.figure(figsize=(6, 4))

        plt.bar(
            ["Correct", "Incorrect"],
            [correct, incorrect],
        )

        plt.title(
            "Recognition Results"
        )

        output = self.output_dir / filename

        plt.savefig(
            output,
            bbox_inches="tight",
        )

        plt.close()

        return output

    def plot_confidence_scores(
        self,
        scores,
        filename="confidence_distribution.png",
    ):
        plt.figure(figsize=(7, 4))

        plt.hist(
            scores,
            bins=20,
        )

        plt.xlabel("Confidence")

        plt.ylabel("Frequency")

        plt.title(
            "Confidence Distribution"
        )

        output = self.output_dir / filename

        plt.savefig(
            output,
            bbox_inches="tight",
        )

        plt.close()

        return output