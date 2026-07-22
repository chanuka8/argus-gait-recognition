import unittest
import numpy as np

from pipeline.detection.person_detector import PersonDetector


class TestPersonDetector(unittest.TestCase):
    def test_detector_initialization(self) -> None:
        detector = PersonDetector(config_path="configs/detection.yaml")
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.confidence, 0.4)

    def test_detector_empty_and_zero_frame(self) -> None:
        detector = PersonDetector(config_path="configs/detection.yaml")
        results_none = detector.detect(None)
        self.assertEqual(results_none, [])

        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        results_empty = detector.detect(empty_frame)
        self.assertEqual(results_empty, [])

    def test_detector_format_on_synthetic_frame(self) -> None:
        detector = PersonDetector(config_path="configs/detection.yaml")
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(synthetic_frame)
        self.assertIsInstance(results, list)

        for item in results:
            self.assertIn("track_input", item)
            self.assertIn("bbox", item)
            self.assertIn("confidence", item)
            self.assertEqual(len(item["bbox"]), 4)


if __name__ == "__main__":
    unittest.main()
