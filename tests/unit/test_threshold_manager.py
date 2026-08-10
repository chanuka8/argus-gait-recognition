import unittest
from core.threshold_manager import ThresholdManager, RecognitionThresholds
from intelligence.open_set_recognizer import OpenSetRecognizer, OpenSetState


class TestThresholdManager(unittest.TestCase):
    def test_default_thresholds_loading(self) -> None:
        tm = ThresholdManager(config_path="configs/inference.yaml")
        thresholds = tm.load_thresholds()
        self.assertIsInstance(thresholds, RecognitionThresholds)
        self.assertGreater(thresholds.known_threshold, thresholds.unknown_threshold)
        self.assertGreaterEqual(thresholds.confirmed_threshold, thresholds.known_threshold)

    def test_ordering_validation(self) -> None:
        tm = ThresholdManager()
        invalid_cfg = {
            "open_set": {"known_threshold": 0.60, "unknown_threshold": 0.80}
        }
        with self.assertRaises(ValueError):
            tm.load_thresholds(config_override=invalid_cfg)

    def test_calibrated_threshold_fallback_safely(self) -> None:
        tm = ThresholdManager()
        override_cfg = {
            "matching_policy": {
                "use_calibrated_threshold": True,
                "calibration_file": "non_existent_file.json",
            }
        }
        thresholds = tm.load_thresholds(config_override=override_cfg)
        self.assertFalse(thresholds.calibrated)
        self.assertEqual(thresholds.known_threshold, 0.85)

    def test_open_set_recognizer_propagation_and_decisions(self) -> None:
        tm = ThresholdManager(config_path="configs/inference.yaml")
        thresholds = tm.load_thresholds()

        recognizer = OpenSetRecognizer(
            known_threshold=thresholds.known_threshold,
            unknown_threshold=thresholds.unknown_threshold,
            margin_threshold=thresholds.margin_threshold,
        )

        # 1. Score below unknown -> UNKNOWN
        res_unknown = recognizer.evaluate_open_set_decision([("target1", 0.50)])
        self.assertEqual(res_unknown.state, OpenSetState.UNKNOWN)
        self.assertEqual(res_unknown.identity, "UNKNOWN")

        # 2. Score in gray zone -> UNCERTAIN
        res_gray = recognizer.evaluate_open_set_decision([("target1", 0.78)])
        self.assertEqual(res_gray.state, OpenSetState.UNCERTAIN)

        # 3. High score + insufficient margin -> UNCERTAIN
        res_margin = recognizer.evaluate_open_set_decision([("target1", 0.90), ("target2", 0.88)])
        self.assertEqual(res_margin.state, OpenSetState.UNCERTAIN)

        # 4. High score + sufficient margin -> KNOWN
        res_known = recognizer.evaluate_open_set_decision([("target1", 0.90), ("target2", 0.70)])
        self.assertEqual(res_known.state, OpenSetState.KNOWN)
        self.assertEqual(res_known.identity, "target1")

    def test_open_set_recognizer_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            OpenSetRecognizer(known_threshold=0.70, unknown_threshold=0.85)


if __name__ == "__main__":
    unittest.main()
