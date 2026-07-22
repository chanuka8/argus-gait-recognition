"""Unit tests for Phase 6 intelligence, missing person workflow, and evidence manager."""

import os
import shutil
import tempfile
import unittest
import numpy as np

from intelligence.cross_camera_tracker import CrossCameraTracker
from intelligence.reid_cache import ReIDCache
from intelligence.identity_persistence import IdentityPersistence
from intelligence.missing_person_workflow import MissingPersonWorkflow
from storage.evidence_manager import EvidenceManager


class TestCrossCameraTracker(unittest.TestCase):
    """Test CrossCameraTracker."""

    def test_global_track_assignment(self):
        tracker = CrossCameraTracker(max_transition_time_seconds=60.0)
        gid1 = tracker.get_or_create_global_id("cam1", local_track_id=1, identity="person_A")
        gid2 = tracker.get_or_create_global_id("cam1", local_track_id=1, identity="person_A")
        self.assertEqual(gid1, gid2)

    def test_cross_camera_transition(self):
        tracker = CrossCameraTracker(max_transition_time_seconds=60.0)
        gid1 = tracker.get_or_create_global_id("cam1", local_track_id=1, identity="person_A")
        gid2 = tracker.get_or_create_global_id("cam2", local_track_id=10, identity="person_A")
        self.assertEqual(gid1, gid2)

        history = tracker.get_track_history(gid1)
        self.assertIsNotNone(history)
        self.assertEqual(len(history["transitions"]), 1)
        self.assertEqual(history["transitions"][0]["from"], "cam1")
        self.assertEqual(history["transitions"][0]["to"], "cam2")


class TestReIDCache(unittest.TestCase):
    """Test ReIDCache."""

    def test_cache_put_get(self):
        cache = ReIDCache(ttl_seconds=10.0, max_entries=5)
        emb = np.zeros((256,), dtype=np.float32)
        cache.put("key1", emb)
        retrieved = cache.get("key1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(cache.size(), 1)

    def test_cache_expiration(self):
        cache = ReIDCache(ttl_seconds=0.01, max_entries=5)
        cache.put("key1", [1, 2, 3])
        import time
        time.sleep(0.02)
        self.assertIsNone(cache.get("key1"))


class TestIdentityPersistence(unittest.TestCase):
    """Test IdentityPersistence."""

    def test_update_and_cooldown(self):
        persistence = IdentityPersistence(suppression_window_seconds=10.0)
        res = persistence.update_identity("subject_01", confidence_score=0.92, camera_id="cam1")
        self.assertEqual(res["identity"], "subject_01")
        self.assertFalse(persistence.should_suppress_alert("subject_01"))
        self.assertTrue(persistence.should_suppress_alert("subject_01"))


class TestMissingPersonWorkflow(unittest.TestCase):
    """Test MissingPersonWorkflow."""

    def test_missing_person_workflow(self):
        workflow = MissingPersonWorkflow(alert_threshold=0.85, cooldown_seconds=10.0)
        workflow.register_target("target_101", metadata={"name": "Jane Doe"})

        self.assertIn("target_101", workflow.get_active_targets())

        event = workflow.process_match("target_101", confidence_score=0.91, camera_id="cam_main")
        self.assertIsNotNone(event)
        self.assertEqual(event["event_type"], "MISSING_PERSON_MATCH")

        # Cooldown check
        event2 = workflow.process_match("target_101", confidence_score=0.91, camera_id="cam_main")
        self.assertIsNone(event2)


class TestEvidenceManager(unittest.TestCase):
    """Test EvidenceManager."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = EvidenceManager(base_evidence_dir=self.temp_dir, max_age_days=1)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_evidence(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        gei = np.ones((128, 64), dtype=np.uint8)
        saved = self.manager.save_evidence(
            target_id="target_01",
            camera_id="cam1",
            confidence=0.95,
            frame=frame,
            gei=gei,
        )
        self.assertIn("snapshot", saved)
        self.assertIn("gei", saved)
        self.assertIn("metadata", saved)
        self.assertTrue(os.path.exists(saved["snapshot"]))
        self.assertTrue(os.path.exists(saved["gei"]))
        self.assertTrue(os.path.exists(saved["metadata"]))


if __name__ == "__main__":
    unittest.main()
