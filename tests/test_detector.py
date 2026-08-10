import unittest
from unittest.mock import MagicMock
import numpy as np

from pipeline.detection.person_detector import PersonDetector
from pipeline.steps.tracking import TrackingStep


class TestPersonDetector(unittest.TestCase):
    def test_detector_initialization(self) -> None:
        detector = PersonDetector(config_path="configs/detection.yaml")
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.confidence, 0.4)
        self.assertEqual(detector.iou_threshold, 0.45)
        self.assertEqual(detector.classes, [0])

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

    def test_person_detector_config_passthrough(self) -> None:
        detector = PersonDetector(config_path="configs/detection.yaml")
        mock_yolo = MagicMock()
        mock_yolo.return_value = []
        detector.model = mock_yolo

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.detect(frame)

        mock_yolo.assert_called_once()
        _, kwargs = mock_yolo.call_args
        self.assertEqual(kwargs.get("conf"), 0.4)
        self.assertEqual(kwargs.get("iou"), 0.45)
        self.assertEqual(kwargs.get("classes"), [0])
        self.assertEqual(kwargs.get("device"), "cpu")
        self.assertEqual(kwargs.get("imgsz"), 640)

    def test_tracking_step_config_passthrough(self) -> None:
        step = TrackingStep(config_path="configs/detection.yaml")
        mock_yolo = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes.xyxy.cpu().numpy.return_value = np.empty((0, 4), dtype=np.float32)
        mock_result.boxes.conf.cpu().numpy.return_value = np.empty((0,), dtype=np.float32)
        mock_result.boxes.cls.cpu().numpy.return_value = np.empty((0,), dtype=np.float32)
        mock_result.boxes.id = None
        mock_result.obb = None
        mock_result.masks = None
        mock_result.orig_shape = (100, 100)
        mock_yolo.return_value = [mock_result]
        step.detector = mock_yolo

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        step.track(frame)

        mock_yolo.assert_called_once()
        _, kwargs = mock_yolo.call_args
        self.assertEqual(kwargs.get("conf"), 0.4)
        self.assertEqual(kwargs.get("iou"), 0.45)
        self.assertEqual(kwargs.get("classes"), [0])
        self.assertEqual(kwargs.get("device"), "cpu")
        self.assertEqual(kwargs.get("imgsz"), 640)

    def test_tracking_step_empty_frame_safety(self) -> None:
        step = TrackingStep(config_path="configs/detection.yaml")
        empty_res = step.track(None)
        self.assertEqual(len(empty_res), 0)


if __name__ == "__main__":
    unittest.main()

