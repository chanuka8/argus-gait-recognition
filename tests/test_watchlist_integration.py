"""Unit and integration tests for Real-Time Watchlist Integration feature."""

import unittest
from unittest.mock import MagicMock, patch

from intelligence.missing_person_workflow import MissingPersonWorkflow, WatchlistEntry, WatchlistManager


class TestWatchlistEntry(unittest.TestCase):
    """Test WatchlistEntry dataclass model and serialization."""

    def test_default_watchlist_entry(self):
        entry = WatchlistEntry(identity_id="person_01")
        self.assertEqual(entry.identity_id, "person_01")
        self.assertEqual(entry.category, "MISSING_PERSON")
        self.assertEqual(entry.priority, "HIGH")
        self.assertTrue(entry.enabled)
        self.assertTrue(entry.alert_enabled)
        self.assertEqual(entry.notes, "")

    def test_custom_watchlist_entry(self):
        entry = WatchlistEntry(
            identity_id="person_02",
            category="PERSON_OF_INTEREST",
            priority="CRITICAL",
            enabled=False,
            alert_enabled=False,
            notes="Suspect on active list",
        )
        self.assertEqual(entry.identity_id, "person_02")
        self.assertEqual(entry.category, "PERSON_OF_INTEREST")
        self.assertEqual(entry.priority, "CRITICAL")
        self.assertFalse(entry.enabled)
        self.assertFalse(entry.alert_enabled)
        self.assertEqual(entry.notes, "Suspect on active list")

    def test_to_dict_from_dict(self):
        data = {
            "identity_id": "target_99",
            "category": "VIP",
            "priority": "MEDIUM",
            "enabled": True,
            "alert_enabled": True,
            "notes": "Escort required",
        }
        entry = WatchlistEntry.from_dict(data)
        self.assertEqual(entry.identity_id, "target_99")
        self.assertEqual(entry.category, "VIP")
        self.assertEqual(entry.to_dict()["notes"], "Escort required")


class TestWatchlistManager(unittest.TestCase):
    """Test WatchlistManager / MissingPersonWorkflow operational logic."""

    def setUp(self):
        self.manager = WatchlistManager(alert_threshold=0.85, cooldown_seconds=5.0)

    def test_register_and_get_entry(self):
        entry = self.manager.register_target(
            identity="subject_alpha",
            category="SECURITY_ALERT",
            priority="HIGH",
            notes="Do not approach",
        )
        self.assertEqual(entry.identity_id, "subject_alpha")
        retrieved = self.manager.get_entry("subject_alpha")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.category, "SECURITY_ALERT")
        self.assertIn("subject_alpha", self.manager.get_active_targets())

    def test_process_match_success(self):
        self.manager.register_target(
            identity="subject_beta",
            category="MISSING_PERSON",
            priority="HIGH",
        )
        match = self.manager.process_match(
            identity="subject_beta",
            confidence_score=0.90,
            camera_id="cam_east",
            track_id=42,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["event_type"], "MISSING_PERSON_MATCH")
        self.assertEqual(match["identity"], "subject_beta")
        self.assertEqual(match["confidence_score"], 0.90)
        self.assertEqual(match["track_id"], 42)

    def test_process_match_disabled_entry(self):
        self.manager.register_target(
            identity="subject_gamma",
            enabled=False,
        )
        match = self.manager.process_match(
            identity="subject_gamma",
            confidence_score=0.95,
            camera_id="cam_west",
        )
        self.assertIsNone(match)

    def test_process_match_low_score(self):
        self.manager.register_target(identity="subject_delta")
        match = self.manager.process_match(
            identity="subject_delta",
            confidence_score=0.80,  # Below threshold 0.85
            camera_id="cam_main",
        )
        self.assertIsNone(match)

    def test_process_match_cooldown(self):
        self.manager.register_target(identity="subject_epsilon")

        # First match succeeds
        m1 = self.manager.process_match("subject_epsilon", 0.92, "cam1")
        self.assertIsNotNone(m1)

        # Immediate second match is suppressed by cooldown
        m2 = self.manager.process_match("subject_epsilon", 0.92, "cam1")
        self.assertIsNone(m2)

    def test_unregister_target(self):
        self.manager.register_target(identity="subject_zeta")
        self.assertTrue(self.manager.unregister_target("subject_zeta"))
        self.assertNotIn("subject_zeta", self.manager.get_active_targets())
        self.assertFalse(self.manager.unregister_target("subject_zeta"))


class TestWatchlistPipelineIntegration(unittest.TestCase):
    """Test pipeline initialization and watchlist matching hooks."""

    @patch("pipeline.video_recognition.VectorStore")
    @patch("pipeline.video_recognition.ByGaitLight")
    def test_video_pipeline_watchlist_initialization(self, mock_model, mock_store):
        mock_store.return_value.load.return_value = (MagicMock(), MagicMock(), {})
        from pipeline.video_recognition import VideoRecognitionPipeline

        pipeline = VideoRecognitionPipeline()
        self.assertTrue(hasattr(pipeline, "watchlist_manager"))
        self.assertIsInstance(pipeline.watchlist_manager, MissingPersonWorkflow)


if __name__ == "__main__":
    unittest.main()
