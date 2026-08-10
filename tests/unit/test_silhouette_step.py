import unittest
from unittest.mock import MagicMock
import cv2
import numpy as np

from pipeline.steps.silhouette_step import (
    SilhouetteStep,
    LearnedSilhouetteSegmenter,
)
from pipeline.silhouette.extractor import SilhouetteExtractor


class TestSilhouetteStep(unittest.TestCase):
    def setUp(self) -> None:
        self.step = SilhouetteStep(target_size=(64, 128))

    def test_normal_person_crop_otsu_fallback(self) -> None:
        # Create a synthetic crop with a bright standing human-like rectangle
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (25, 20), (75, 180), (255, 255, 255), -1)

        mask = self.step.extract_from_crop(crop)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (128, 64))
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_empty_and_invalid_crop(self) -> None:
        self.assertIsNone(self.step.extract_from_crop(None))
        self.assertIsNone(self.step.extract_from_crop(np.zeros((0, 0, 3), dtype=np.uint8)))

    def test_extremely_small_crop(self) -> None:
        tiny_crop = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.rectangle(tiny_crop, (0, 0), (3, 3), (255, 255, 255), -1)
        self.assertIsNone(self.step.extract_from_crop(tiny_crop))

    def test_multiple_disconnected_components(self) -> None:
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        # Primary large body
        cv2.rectangle(crop, (25, 20), (75, 180), (255, 255, 255), -1)
        # Small noise dots
        cv2.circle(crop, (10, 10), 3, (255, 255, 255), -1)
        cv2.circle(crop, (90, 190), 4, (255, 255, 255), -1)

        mask = self.step.extract_from_crop(crop)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (128, 64))

    def test_learned_segmenter_unavailable_fallback(self) -> None:
        segmenter = LearnedSilhouetteSegmenter(model_path="non_existent_model.onnx")
        self.assertFalse(segmenter.is_available())
        self.assertIsNone(segmenter.segment(np.zeros((100, 50, 3), dtype=np.uint8)))

    def test_learned_segmenter_mocked_success(self) -> None:
        step = SilhouetteStep(target_size=(64, 128), method="learned")
        # Mock probability output for 100x50 crop
        raw_prob = np.zeros((1, 1, 256, 256), dtype=np.float32)
        raw_prob[0, 0, 32:224, 64:192] = 0.9  # High prob body
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.get_outputs.return_value = [MagicMock(name="output")]
        mock_session.run.return_value = [raw_prob]

        step.learned_segmenter.session = mock_session

        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        mask = step.extract_from_crop(crop)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (128, 64))

    def test_training_inference_contract_consistency(self) -> None:
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (25, 20), (75, 180), (255, 255, 255), -1)
        mask = self.step.extract_from_crop(crop)

        # Output shape must be 128x64 (H x W)
        self.assertEqual(mask.shape, (128, 64))
        # Data type must be uint8
        self.assertEqual(mask.dtype, np.uint8)
        # Values must be binary 0 or 255
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
        # No NaNs or Infs
        self.assertTrue(np.all(np.isfinite(mask)))

    def test_silhouette_extractor_wrapper(self) -> None:
        extractor = SilhouetteExtractor(target_size=(64, 128))
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (25, 20), (75, 180), (255, 255, 255), -1)

        mask = extractor.extract_from_crop(crop)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (128, 64))


if __name__ == "__main__":
    unittest.main()
