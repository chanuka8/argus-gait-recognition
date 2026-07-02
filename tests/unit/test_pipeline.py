import unittest

from pipeline.cache_engine import CacheEngine
from pipeline.speed_controller import SpeedController


class TestPipeline(unittest.TestCase):

    def test_cache_engine(self):

        cache = CacheEngine()

        cache.save_embedding(
            "person1",
            [1, 2, 3],
        )

        self.assertTrue(
            cache.has_embedding(
                "person1"
            )
        )

    def test_speed_controller(self):

        controller = SpeedController(
            target_fps=10,
        )

        self.assertIsNotNone(
            controller
        )


if __name__ == "__main__":
    unittest.main()