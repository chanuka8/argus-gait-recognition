import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from models.architectures.bygait_light import ByGaitLight
from models.architectures.losses import ArcMarginProduct


class TestModelArchitecture(unittest.TestCase):
    def test_forward_pass_default_hpp(self) -> None:
        model = ByGaitLight(embedding_dim=256, part_bins=4)
        dummy = torch.randn(4, 1, 128, 64)
        output = model(dummy)

        self.assertEqual(output.shape, (4, 256))
        self.assertTrue(torch.all(torch.isfinite(output)))

        norms = torch.norm(output, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_forward_pass_legacy_single_bin(self) -> None:
        model = ByGaitLight(embedding_dim=256, part_bins=1)
        dummy = torch.randn(2, 1, 128, 64)
        output = model(dummy)

        self.assertEqual(output.shape, (2, 256))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_gradient_flow(self) -> None:
        model = ByGaitLight(embedding_dim=256, part_bins=4)
        model.train()
        dummy = torch.randn(2, 1, 128, 64, requires_grad=True)
        output = model(dummy)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(dummy.grad)
        self.assertIsNotNone(model.embedding.weight.grad)

    def test_arcmargin_product(self) -> None:
        arcface = ArcMarginProduct(in_features=256, out_features=10, s=30.0, m=0.50)
        embeddings = torch.randn(4, 256)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        labels = torch.tensor([0, 1, 2, 3])

        output, _unscaled = arcface(embeddings, labels)
        self.assertEqual(output.shape, (4, 10))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_checkpoint_shape_mismatch_error(self) -> None:
        model_part4 = ByGaitLight(embedding_dim=256, part_bins=4)
        model_part1 = ByGaitLight(embedding_dim=256, part_bins=1)

        state_part1 = model_part1.state_dict()

        with self.assertRaises(ValueError) as ctx:
            model_part4.load_state_dict(state_part1)

        self.assertIn("embedding.weight expects in_features=512", str(ctx.exception))
        self.assertIn("Retraining required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
