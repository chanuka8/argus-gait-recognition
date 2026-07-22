import unittest
import numpy as np

from pipeline.gei.stream_gei_builder import StreamGEIBuilder


class TestStreamGEIBuilder(unittest.TestCase):
    def test_gei_builder_initialization(self) -> None:
        builder = StreamGEIBuilder(config_path="configs/gei.yaml")
        self.assertEqual(builder.max_frames, 15)
        self.assertEqual(builder.min_frames, 10)
        self.assertEqual(builder.target_size, (64, 128))

    def test_gei_builder_add_and_build(self) -> None:
        builder = StreamGEIBuilder(config_path="configs/gei.yaml")
        track_id = 1

        self.assertFalse(builder.is_ready(track_id))
        self.assertIsNone(builder.build_gei(track_id))

        silhouette = np.zeros((128, 64), dtype=np.uint8)
        silhouette[20:100, 15:45] = 255

        for _ in range(12):
            builder.add_silhouette(track_id, silhouette)

        self.assertTrue(builder.is_ready(track_id))
        gei = builder.build_gei(track_id)

        self.assertIsNotNone(gei)
        self.assertEqual(gei.shape, (128, 64))
        self.assertEqual(gei.dtype, np.uint8)

    def test_gei_builder_cleanup_inactive(self) -> None:
        builder = StreamGEIBuilder(config_path="configs/gei.yaml")
        builder.last_updated[99] = 0.0
        builder.track_buffers[99] = [np.zeros((128, 64), dtype=np.float32)]

        removed = builder.cleanup_inactive(max_idle_seconds=1.0)
        self.assertIn(99, removed)
        self.assertNotIn(99, builder.track_buffers)


if __name__ == "__main__":
    unittest.main()
