import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from models.architectures.silhouette_unet import SilhouetteUNet
from pipeline.steps.silhouette_step import LearnedSilhouetteSegmenter, SilhouetteStep
from scripts.export_silhouette_unet_onnx import export_and_validate_onnx
from training.silhouette_dataset import SilhouetteSegmentationDataset
from training.train_silhouette_unet import BCEDiceLoss, calculate_metrics


class TestSilhouetteUNetPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.onnx_weights_path = Path("models/weights/silhouette_segmenter.onnx")
        cls.onnx_engines_path = Path("models/engines/silhouette_segmenter.onnx")

    def test_unet_forward_shape_and_range(self) -> None:
        model = SilhouetteUNet()
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 256, dtype=torch.float32)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (2, 1, 256, 256))
        self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_bce_dice_loss(self) -> None:
        criterion = BCEDiceLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(pred, target)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_metrics_calculation(self) -> None:
        pred_binary = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        target_binary = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        metrics = calculate_metrics(pred_binary, target_binary)

        self.assertIn("dice", metrics)
        self.assertIn("iou", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertAlmostEqual(metrics["precision"], 0.5)

    def test_dataset_loader(self) -> None:
        ds = SilhouetteSegmentationDataset(
            zip_path="data/casia_b_raw.zip", subject_range=(1, 5), max_samples=10, seed=42
        )
        self.assertGreater(len(ds), 0)

        img_tensor, mask_tensor = ds[0]
        self.assertEqual(img_tensor.shape, (3, 256, 256))
        self.assertEqual(mask_tensor.shape, (1, 256, 256))
        self.assertEqual(img_tensor.dtype, torch.float32)
        self.assertEqual(mask_tensor.dtype, torch.float32)

    def test_onnx_export_and_validation(self) -> None:
        valid, msg = export_and_validate_onnx()
        self.assertTrue(valid, f"ONNX validation failed: {msg}")
        self.assertTrue(self.onnx_weights_path.exists())
        self.assertTrue(self.onnx_engines_path.exists())

    def test_learned_segmenter_success_path(self) -> None:
        segmenter = LearnedSilhouetteSegmenter(model_path=str(self.onnx_weights_path))
        self.assertTrue(segmenter.is_available())

        valid, msg = segmenter.validate_model()
        self.assertTrue(valid, f"Learned segmenter model validation failed: {msg}")

        dummy_crop = np.zeros((200, 100, 3), dtype=np.uint8)
        cv2.rectangle(dummy_crop, (25, 20), (75, 180), (200, 200, 200), -1)
        mask = segmenter.segment(dummy_crop)

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (200, 100))
        self.assertEqual(mask.dtype, np.uint8)

    def test_learned_segmenter_missing_fallback(self) -> None:
        segmenter = LearnedSilhouetteSegmenter(model_path="non_existent_path.onnx")
        self.assertFalse(segmenter.is_available())
        mask = segmenter.segment(np.zeros((100, 50, 3), dtype=np.uint8))
        self.assertIsNone(mask)

    def test_silhouette_step_end_to_end_learned(self) -> None:
        step = SilhouetteStep(target_size=(64, 128), method="learned")
        crop = np.zeros((200, 100, 3), dtype=np.uint8)
        cv2.rectangle(crop, (25, 20), (75, 180), (255, 255, 255), -1)

        mask = step.extract_from_crop(crop)
        self.assertIsNotNone(mask)
        # H x W format: (128, 64)
        self.assertEqual(mask.shape, (128, 64))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_gei_compatibility(self) -> None:
        from preprocessing.gei_builder import GEIBuilder

        step = SilhouetteStep(target_size=(64, 128), method="learned")
        gei_builder = GEIBuilder()

        for _ in range(15):
            crop = np.zeros((200, 100, 3), dtype=np.uint8)
            cv2.rectangle(crop, (25, 20), (75, 180), (255, 255, 255), -1)
            sil = step.extract_from_crop(crop)
            if sil is not None:
                gei_builder.add_frame(sil)

        gei = gei_builder.build()
        self.assertIsNotNone(gei)
        self.assertEqual(gei.shape, (128, 64))
        self.assertEqual(gei.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
