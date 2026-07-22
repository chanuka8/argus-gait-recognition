"""Automated verification test suite for ARGUS AI surveillance system audit."""

import os
import shutil
import tempfile
import threading
import time
import unittest
import numpy as np

from intelligence.cross_camera_tracker import CrossCameraTracker
from intelligence.identity_persistence import IdentityPersistence
from intelligence.missing_person_workflow import MissingPersonWorkflow
from intelligence.reid_cache import ReIDCache
from storage.evidence_manager import EvidenceManager


class TestSystemAuditVerification(unittest.TestCase):
    """System Verification Test Suite covering all 12 audit test requirements."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Cross camera identity transfer simulation
    def test_cross_camera_identity_transfer_simulation(self):
        tracker = CrossCameraTracker(max_transition_time_seconds=10.0)
        gid_cam_a = tracker.get_or_create_global_id("camera_A", local_track_id=101, identity="subject_X")
        time.sleep(0.01)
        gid_cam_b = tracker.get_or_create_global_id("camera_B", local_track_id=202, identity="subject_X")

        self.assertEqual(gid_cam_a, gid_cam_b)
        history = tracker.get_track_history(gid_cam_a)
        self.assertEqual(len(history["transitions"]), 1)
        self.assertEqual(history["transitions"][0]["from"], "camera_A")
        self.assertEqual(history["transitions"][0]["to"], "camera_B")

    # 2. Track expiration and cleanup
    def test_track_expiration_and_cleanup(self):
        tracker = CrossCameraTracker(max_transition_time_seconds=0.01)
        gid = tracker.get_or_create_global_id("cam1", local_track_id=1, identity="subject_Y")
        time.sleep(0.05)
        removed = tracker.cleanup_stale_tracks(max_age_seconds=0.02)
        self.assertEqual(removed, 1)
        self.assertIsNone(tracker.get_track_history(gid))

    # 3. Cache TTL expiration
    def test_cache_ttl_expiration(self):
        cache = ReIDCache(ttl_seconds=0.02, max_entries=10)
        cache.put("item1", np.ones((128,)))
        self.assertIsNotNone(cache.get("item1"))
        time.sleep(0.04)
        self.assertIsNone(cache.get("item1"))

    # 4. Cache maximum size handling
    def test_cache_maximum_size_handling(self):
        cache = ReIDCache(ttl_seconds=60.0, max_entries=3)
        cache.put("k1", "val1")
        cache.put("k2", "val2")
        cache.put("k3", "val3")
        self.assertEqual(cache.size(), 3)
        cache.put("k4", "val4")  # Should trigger eviction of oldest
        self.assertEqual(cache.size(), 3)
        self.assertIsNone(cache.get("k1"))
        self.assertIsNotNone(cache.get("k4"))

    # 5. Concurrent cache access
    def test_concurrent_cache_access(self):
        cache = ReIDCache(ttl_seconds=10.0, max_entries=100)
        errors = []

        def worker(idx):
            try:
                for i in range(20):
                    cache.put(f"key_{idx}_{i}", f"val_{i}")
                    _ = cache.get(f"key_{idx}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

    # 6. Duplicate alert suppression
    def test_duplicate_alert_suppression(self):
        persistence = IdentityPersistence(suppression_window_seconds=5.0)
        self.assertFalse(persistence.should_suppress_alert("target_1"))
        self.assertTrue(persistence.should_suppress_alert("target_1"))
        self.assertTrue(persistence.should_suppress_alert("target_1"))

    # 7. Confidence accumulation consistency
    def test_confidence_accumulation_consistency(self):
        p1 = IdentityPersistence(score_accumulation_decay=0.9)
        p2 = IdentityPersistence(score_accumulation_decay=0.9)

        scores = [0.80, 0.85, 0.90, 0.95]
        for s in scores:
            res1 = p1.update_identity("person_1", s)
            res2 = p2.update_identity("person_1", s)

        self.assertAlmostEqual(res1["accumulated_score"], res2["accumulated_score"], places=6)
        self.assertEqual(res1["detections"], 4)

    # 8. Gallery scan non-blocking behavior
    def test_gallery_scan_non_blocking(self):
        workflow = MissingPersonWorkflow(alert_threshold=0.80, cooldown_seconds=1.0)
        workflow.register_target("target_A")

        start = time.monotonic()
        for _ in range(50):
            workflow.process_match("target_A", 0.85, "cam1")
        elapsed = time.monotonic() - start

        # 50 matches should process in under 0.1 seconds (non-blocking)
        self.assertLess(elapsed, 0.1)

    # 9. Event queue overflow handling
    def test_event_queue_overflow_handling(self):
        workflow = MissingPersonWorkflow(alert_threshold=0.80, cooldown_seconds=0.001)
        workflow.register_target("target_B")

        for i in range(200):
            time.sleep(0.002)
            workflow.process_match("target_B", 0.90, f"cam_{i}")

        events = workflow.get_recent_events()
        self.assertGreater(len(events), 0)

    # 10. Evidence filename collision
    def test_evidence_filename_collision(self):
        mgr = EvidenceManager(base_evidence_dir=self.temp_dir)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        saved1 = mgr.save_evidence("target_1", "cam1", 0.9, frame=frame)
        saved2 = mgr.save_evidence("target_1", "cam1", 0.9, frame=frame)

        self.assertNotEqual(saved1["snapshot"], saved2["snapshot"])
        self.assertTrue(os.path.exists(saved1["snapshot"]))
        self.assertTrue(os.path.exists(saved2["snapshot"]))

    # 11. Invalid snapshot handling
    def test_invalid_snapshot_handling(self):
        mgr = EvidenceManager(base_evidence_dir=self.temp_dir)
        invalid_frame = np.array([], dtype=np.uint8)  # Empty array

        saved = mgr.save_evidence("target_2", "cam1", 0.85, frame=invalid_frame)
        self.assertNotIn("snapshot", saved)
        self.assertIn("metadata", saved)
        self.assertTrue(os.path.exists(saved["metadata"]))

    # 12. Retention cleanup
    def test_retention_cleanup(self):
        mgr = EvidenceManager(base_evidence_dir=self.temp_dir, max_age_days=0)
        mgr.max_age_seconds = 0.01  # Immediate expiration for test

        mgr.save_evidence("target_old", "cam1", 0.9)
        time.sleep(0.03)

        deleted = mgr.enforce_retention_policy()
        self.assertEqual(deleted, 1)


if __name__ == "__main__":
    unittest.main()
