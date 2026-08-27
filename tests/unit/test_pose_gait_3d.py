"""
Unit tests for 3D Pose Gait module, pipeline step, and evaluator.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator_3d import Evaluator3D
from models.architectures.pose_gait_3d import (
    PoseGait3DNet,
    PoseLifter3D,
    SkeletonNormalizer3D,
    TemporalPoseBuffer,
)
from pipeline.steps.gait_3d_step import Gait3DStep


class TestPoseGait3D(unittest.TestCase):
    def test_skeleton_normalizer_autograd_graph_and_shapes(self) -> None:
        normalizer = SkeletonNormalizer3D()
        dummy_joints = torch.randn(2, 30, 17, 3, requires_grad=True)
        normalized = normalizer(dummy_joints)

        self.assertEqual(normalized.shape, (2, 30, 17, 3))
        self.assertTrue(torch.all(torch.isfinite(normalized)))

        loss = normalized.sum()
        loss.backward()
        self.assertIsNotNone(dummy_joints.grad)
        self.assertTrue(torch.all(torch.isfinite(dummy_joints.grad)))

    def test_pose_lifter_forward_and_backward(self) -> None:
        lifter = PoseLifter3D()
        dummy_2d = torch.randn(2, 30, 17, 3, requires_grad=True)
        joints_3d = lifter(dummy_2d)

        self.assertEqual(joints_3d.shape, (2, 30, 17, 3))
        self.assertTrue(torch.all(torch.isfinite(joints_3d)))

        loss = joints_3d.sum()
        loss.backward()
        self.assertIsNotNone(dummy_2d.grad)

    def test_pose_gait_3d_net_embedding_norm_and_gradient(self) -> None:
        gait_net = PoseGait3DNet(embedding_dim=256)
        dummy_3d = torch.randn(4, 30, 17, 3, requires_grad=True)
        emb = gait_net(dummy_3d)

        self.assertEqual(emb.shape, (4, 256))
        self.assertTrue(torch.all(torch.isfinite(emb)))

        norms = torch.norm(emb, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

        loss = emb.sum()
        loss.backward()
        self.assertIsNotNone(dummy_3d.grad)

    def test_temporal_pose_buffer_interpolation(self) -> None:
        buffer = TemporalPoseBuffer(max_length=10, conf_threshold=0.30)
        track_id = "track_001"

        for i in range(10):
            kpts = np.random.randn(17, 3).astype(np.float32)
            if i == 3:
                kpts[0, 2] = 0.10
            else:
                kpts[0, 2] = 0.90
            buffer.add_keypoints(track_id, kpts)

        seq = buffer.get_sequence(track_id)
        self.assertIsNotNone(seq)
        self.assertEqual(seq.shape, (10, 17, 3))
        self.assertTrue(np.isfinite(seq[3, 0, 0]))

    def test_gait_3d_step_disabled_fallback(self) -> None:
        step = Gait3DStep(enabled=False)
        crop = np.zeros((128, 64, 3), dtype=np.uint8)
        emb = step.process_frame_crop("track_001", crop)
        self.assertIsNone(emb)

    def test_gait_3d_step_invalid_checkpoint_graceful_fallback(self) -> None:
        step = Gait3DStep(enabled=True, weights_path="invalid_path_to_ckpt.pth")
        self.assertFalse(step.enabled)

    def test_gait_3d_step_prune_stale_tracks(self) -> None:
        step = Gait3DStep(enabled=False)
        step.pose_buffer = TemporalPoseBuffer()
        step.pose_buffer.add_keypoints("track_1", np.zeros((17, 3)))
        step.pose_buffer.add_keypoints("track_2", np.zeros((17, 3)))

        self.assertIn("track_1", step.pose_buffer.buffers)
        self.assertIn("track_2", step.pose_buffer.buffers)

        step.prune_stale_tracks(active_track_ids=["track_1"])
        self.assertIn("track_1", step.pose_buffer.buffers)
        self.assertNotIn("track_2", step.pose_buffer.buffers)

    def test_evaluator_3d_disjoint_leakage_guard(self) -> None:
        evaluator = Evaluator3D()
        test_subs = ["075", "076"]
        gallery_items, probe_items = evaluator._build_gallery_and_probes(test_subs)

        gal_paths = {g["path"] for g in gallery_items}
        prb_paths = {p["path"] for p in probe_items}

        self.assertEqual(len(gal_paths.intersection(prb_paths)), 0)


if __name__ == "__main__":
    unittest.main()
