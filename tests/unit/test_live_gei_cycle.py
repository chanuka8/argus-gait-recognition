import math
import unittest

import cv2
import numpy as np

from pipeline.steps.live_gei import LiveGEI


class TestLiveGEICycle(unittest.TestCase):
    def setUp(self) -> None:
        self.gei = LiveGEI(
            max_frames=30,
            min_frames=10,
            cycle_detection_enabled=True,
            min_cycle_frames=6,
            max_cycle_frames=16,
            cycle_confidence_threshold=0.30,
            duplicate_threshold=0.98,
        )

    def test_insufficient_frames(self) -> None:
        for _ in range(5):
            mask = np.zeros((128, 64), dtype=np.uint8)
            cv2.rectangle(mask, (10, 20), (50, 100), 255, -1)
            self.gei.add(mask)

        self.assertFalse(self.gei.ready())
        self.assertIsNone(self.gei.build())

    def test_duplicate_frame_rejection(self) -> None:
        mask = np.zeros((128, 64), dtype=np.uint8)
        cv2.rectangle(mask, (10, 20), (50, 100), 255, -1)

        self.gei.add(mask)
        initial_count = self.gei.count()
        self.assertEqual(initial_count, 1)

        # Add identical frame -> should be rejected as duplicate
        self.gei.add(mask)
        self.assertEqual(self.gei.count(), 1)
        self.assertGreater(self.gei.duplicate_frames, 0)

    def test_periodic_synthetic_gait_signal(self) -> None:
        period = 10
        for i in range(25):
            width = int(20 + 10 * math.sin(2 * math.pi * i / period))
            mask = np.zeros((128, 64), dtype=np.uint8)
            x1 = max(0, 32 - width // 2)
            x2 = min(64, 32 + width // 2)
            cv2.rectangle(mask, (x1, 10), (x2, 110), 255, -1)
            self.gei.add(mask)

        self.assertTrue(self.gei.ready())
        gei_img = self.gei.build()

        self.assertIsNotNone(gei_img)
        self.assertEqual(gei_img.shape, (128, 64))
        self.assertEqual(gei_img.dtype, np.uint8)
        self.assertIsNotNone(self.gei.last_cycle_detected)
        self.assertAlmostEqual(self.gei.last_cycle_detected, period, delta=2)

    def test_no_cycle_fallback(self) -> None:
        # Random non-periodic width masks
        np.random.seed(42)
        for i in range(12):
            w = 15 + (i % 3) * 5
            mask = np.zeros((128, 64), dtype=np.uint8)
            cv2.rectangle(mask, (10, 10), (10 + w, 110), 255, -1)
            self.gei.add(mask)

        self.assertTrue(self.gei.ready())
        gei_img = self.gei.build()
        self.assertIsNotNone(gei_img)
        self.assertEqual(gei_img.shape, (128, 64))

    def test_max_buffer_bound_and_reset(self) -> None:
        for i in range(40):
            mask = np.zeros((128, 64), dtype=np.uint8)
            w = 10 + (i % 7) * 2
            cv2.rectangle(mask, (5, 5), (5 + w, 100), 255, -1)
            self.gei.add(mask)

        self.assertLessEqual(self.gei.count(), 30)

        self.gei.clear()
        self.assertEqual(self.gei.count(), 0)
        self.assertFalse(self.gei.ready())


if __name__ == "__main__":
    unittest.main()
