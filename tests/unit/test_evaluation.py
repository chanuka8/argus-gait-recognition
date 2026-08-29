import unittest
from pathlib import Path

from evaluation.benchmarks.evaluate_enrollment_safeguards import EnrollmentSafeguardEvaluator
from evaluation.benchmarks.evaluate_temporal_aggregation import TemporalTrackEvaluator
from evaluation.visualizer import EvaluationVisualizer


class TestEvaluation(unittest.TestCase):
    def test_visualizer_creation(self):
        visualizer = EvaluationVisualizer()

        self.assertIsNotNone(visualizer)

    def test_accuracy_plot(self):
        visualizer = EvaluationVisualizer()

        output = visualizer.plot_accuracy(
            0.90,
            "unit_accuracy.png",
        )

        self.assertTrue(Path(output).exists())

    def test_enrollment_safeguard_evaluator_creation(self):
        evaluator = EnrollmentSafeguardEvaluator()
        self.assertIsNotNone(evaluator)
        self.assertIsNotNone(evaluator.quality_gate)

    def test_temporal_track_evaluator_synthetic(self):
        evaluator = TemporalTrackEvaluator(window_size=4, consensus_threshold=0.60, confirm_threshold=0.70)
        results = evaluator.evaluate_synthetic_tracks(
            known_subjects=["subject_a", "subject_b"],
            num_tracks=10,
            track_length=6,
            noise_level=0.0,
            seed=42,
        )
        self.assertEqual(results["num_tracks"], 10)
        self.assertGreaterEqual(results["confirmed_correct"], 0)
        self.assertIn("track_tar", results)
        self.assertIn("track_far", results)


if __name__ == "__main__":
    unittest.main()
