import unittest
import numpy as np

from pipeline.steps.live_gei import LiveGEI


class TestPreprocessing(unittest.TestCase):

    def test_gei_buffer(self):

        gei = LiveGEI(
            max_frames=5,
        )

        frame = np.zeros(
            (128, 64),
            dtype=np.uint8,
        )

        for _ in range(5):
            gei.add(frame)

        self.assertTrue(
            gei.ready()
        )

    def test_gei_build(self):

        gei = LiveGEI(
            max_frames=5,
        )

        frame = np.zeros(
            (128, 64),
            dtype=np.uint8,
        )

        for _ in range(5):
            gei.add(frame)

        result = gei.build()

        self.assertIsNotNone(
            result
        )

    def test_legacy_mode_accepts_repeated_frames(self):
        gei = LiveGEI(
            max_frames=5,
            cycle_detection_enabled=False,
            duplicate_filter_enabled=False,
        )
        frame = np.zeros((128, 64), dtype=np.uint8)
        frame[20:100, 10:50] = 255
        for _ in range(5):
            gei.add(frame)

        self.assertEqual(gei.count(), 5)
        self.assertTrue(gei.ready())
        result = gei.build()
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (128, 64))

    def test_cycle_aware_mode_rejects_duplicates(self):
        gei = LiveGEI(
            max_frames=15,
            min_frames=5,
            cycle_detection_enabled=True,
            duplicate_filter_enabled=True,
        )
        frame = np.zeros((128, 64), dtype=np.uint8)
        frame[20:100, 10:50] = 255

        gei.add(frame)
        self.assertEqual(gei.count(), 1)

        gei.add(frame)
        self.assertEqual(gei.count(), 1)
        self.assertGreater(gei.duplicate_frames, 0)

    def test_ready_and_build_in_both_modes(self):
        legacy_gei = LiveGEI(
            max_frames=5,
            cycle_detection_enabled=False,
            duplicate_filter_enabled=False,
        )
        cycle_gei = LiveGEI(
            max_frames=15,
            min_frames=5,
            cycle_detection_enabled=True,
            duplicate_filter_enabled=True,
        )

        dummy_mask = np.zeros((128, 64), dtype=np.uint8)
        dummy_mask[30:90, 20:40] = 255

        self.assertFalse(legacy_gei.ready())
        self.assertFalse(cycle_gei.ready())

        for _ in range(5):
            legacy_gei.add(dummy_mask)

        self.assertTrue(legacy_gei.ready())
        self.assertIsNotNone(legacy_gei.build())

    def test_no_cycle_fallback(self):
        gei = LiveGEI(
            max_frames=15,
            min_frames=5,
            cycle_detection_enabled=True,
            duplicate_filter_enabled=False,
        )
        for i in range(8):
            mask = np.zeros((128, 64), dtype=np.uint8)
            w = 10 + (i % 3) * 5
            mask[20:100, 10:10 + w] = 255
            gei.add(mask)

        self.assertTrue(gei.ready())
        result = gei.build()
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (128, 64))


if __name__ == "__main__":
    unittest.main()

