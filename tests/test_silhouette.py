import unittest
import cv2
import numpy as np

from pipeline.silhouette.extractor import SilhouetteExtractor


class TestSilhouetteExtractor(unittest.TestCase):
    def test_silhouette_initialization(self) -> None:
        extractor = SilhouetteExtractor(target_size=(64, 128))
        self.assertEqual(extractor.target_size, (64, 128))

    def test_silhouette_extract_invalid(self) -> None:
        extractor = SilhouetteExtractor()
        self.assertIsNone(extractor.extract_from_crop(None))

        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertIsNone(extractor.extract_from_crop(empty))

        self.assertIsNone(extractor.extract_from_frame(None, [0, 0, 10, 10]))

    def test_silhouette_extraction_output_shape(self) -> None:
        extractor = SilhouetteExtractor(target_size=(64, 128))

        synthetic_person_crop = np.zeros((200, 80, 3), dtype=np.uint8)
        cv2.ellipse(synthetic_person_crop, (40, 40), (20, 25), 0, 0, 360, (255, 255, 255), -1)
        cv2.rectangle(synthetic_person_crop, (20, 65), (60, 180), (255, 255, 255), -1)

        mask = extractor.extract_from_crop(synthetic_person_crop)

        if mask is not None:
            self.assertEqual(mask.shape, (128, 64))
            self.assertEqual(mask.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
