import unittest

from pipeline.tracking.tracker import PersonTracker


class TestPersonTracker(unittest.TestCase):
    def test_tracker_initialization(self) -> None:
        tracker = PersonTracker()
        self.assertIsNotNone(tracker.tracker)

    def test_tracker_update_empty(self) -> None:
        tracker = PersonTracker()
        results = tracker.update([], (480, 640, 3))
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_tracker_id_persistence_format(self) -> None:
        tracker = PersonTracker()
        detections = [
            {"bbox": [100, 100, 200, 300], "confidence": 0.9, "track_input": None},
        ]
        results = tracker.update(detections, (480, 640, 3))
        self.assertIsInstance(results, list)

        for item in results:
            self.assertIn("track_id", item)
            self.assertIn("bbox", item)
            self.assertIn("timestamp", item)
            self.assertEqual(len(item["bbox"]), 4)

    def test_tracker_cleanup_inactive(self) -> None:
        tracker = PersonTracker()
        tracker.last_seen[999] = 0.0
        removed = tracker.cleanup_inactive(max_idle_seconds=1.0)
        self.assertIn(999, removed)
        self.assertNotIn(999, tracker.last_seen)


if __name__ == "__main__":
    unittest.main()
