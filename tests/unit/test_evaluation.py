import unittest
from pathlib import Path

from evaluation.visualizer import EvaluationVisualizer


class TestEvaluation(unittest.TestCase):

    def test_visualizer_creation(self):
        visualizer = EvaluationVisualizer()

        self.assertIsNotNone(
            visualizer
        )

    def test_accuracy_plot(self):
        visualizer = EvaluationVisualizer()

        output = visualizer.plot_accuracy(
            0.90,
            "unit_accuracy.png",
        )

        self.assertTrue(
            Path(output).exists()
        )


if __name__ == "__main__":
    unittest.main()