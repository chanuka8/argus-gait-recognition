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


if __name__ == "__main__":
    unittest.main()