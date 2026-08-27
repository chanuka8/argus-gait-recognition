"""Unit tests for production-grade Camera Transition Model and CrossCameraTracker integration."""

import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from intelligence.camera_transition_model import CameraTransitionModel
from intelligence.cross_camera_tracker import CrossCameraTracker


class MockClock:
    """Injectable deterministic clock for unit testing."""

    def __init__(self, initial_time: float = 1000.0) -> None:
        self.current_time = initial_time

    def now(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class TestCameraTransitionModel(unittest.TestCase):
    """Focused tests for CameraTransitionModel topology, filtering, scoring, and safety."""

    def setUp(self) -> None:
        self.clock = MockClock(initial_time=1000.0)
        self.sample_config = {
            "camera_transitions": {
                "cam_a": {
                    "cam_b": {
                        "min_travel_seconds": 5.0,
                        "max_travel_seconds": 30.0,
                        "probability": 0.80,
                    }
                },
                "cam_b": {
                    "cam_c": {
                        "min_travel_seconds": 10.0,
                        "max_travel_seconds": 60.0,
                        "probability": 0.90,
                    }
                },
            },
            "weights": {
                "identity_similarity": 0.6,
                "transition_probability": 0.2,
                "travel_time_likelihood": 0.2,
            },
            "similarity_threshold": 0.40,
            "max_history_seconds": 300.0,
            "allow_same_camera": False,
        }

    def test_valid_a_to_b_transition(self) -> None:
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)
        tracker = CrossCameraTracker(transition_model=model, time_provider=self.clock.now)

        gid1 = tracker.get_or_create_global_id("cam_a", local_track_id=1, identity="person_alpha")
        self.clock.advance(15.0)

        gid2 = tracker.get_or_create_global_id("cam_b", local_track_id=10, identity="person_alpha")
        self.assertEqual(gid1, gid2)

        history = tracker.get_track_history(gid1)
        self.assertIsNotNone(history)
        self.assertEqual(len(history["transitions"]), 1)
        self.assertEqual(history["transitions"][0]["from"], "cam_a")
        self.assertEqual(history["transitions"][0]["to"], "cam_b")

    def test_invalid_a_to_c_transition(self) -> None:
        """Reject candidates where direct topology A -> C is not configured."""
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)
        tracker = CrossCameraTracker(transition_model=model, time_provider=self.clock.now)

        gid1 = tracker.get_or_create_global_id("cam_a", local_track_id=1, identity="person_alpha")
        self.clock.advance(15.0)

        gid2 = tracker.get_or_create_global_id("cam_c", local_track_id=5, identity=None)
        self.assertNotEqual(gid1, gid2)

    def test_too_early_arrival(self) -> None:
        """Reject candidate appearing before minimum travel time (e.g. delta_t = 3s < min 5s)."""
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)
        tracker = CrossCameraTracker(transition_model=model, time_provider=self.clock.now)

        gid1 = tracker.get_or_create_global_id("cam_a", local_track_id=1, identity=None)
        self.clock.advance(3.0)

        gid2 = tracker.get_or_create_global_id("cam_b", local_track_id=2, identity=None)
        self.assertNotEqual(gid1, gid2)

    def test_too_late_arrival(self) -> None:
        """Reject candidate appearing after maximum travel time (e.g. delta_t = 35s > max 30s)."""
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)
        tracker = CrossCameraTracker(transition_model=model, time_provider=self.clock.now)

        gid1 = tracker.get_or_create_global_id("cam_a", local_track_id=1, identity=None)
        self.clock.advance(35.0)

        gid2 = tracker.get_or_create_global_id("cam_b", local_track_id=2, identity=None)
        self.assertNotEqual(gid1, gid2)

    def test_exact_minimum_and_maximum_boundaries(self) -> None:
        """Exact min_travel_seconds (5.0s) and max_travel_seconds (30.0s) must be accepted."""
        clock1 = MockClock(1000.0)
        model1 = CameraTransitionModel(config=self.sample_config, time_provider=clock1.now)
        tracker1 = CrossCameraTracker(transition_model=model1, time_provider=clock1.now)

        gid1 = tracker1.get_or_create_global_id("cam_a", local_track_id=1, identity="subj_min")
        clock1.advance(5.0)
        gid2 = tracker1.get_or_create_global_id("cam_b", local_track_id=2, identity="subj_min")
        self.assertEqual(gid1, gid2)

        clock2 = MockClock(1000.0)
        model2 = CameraTransitionModel(config=self.sample_config, time_provider=clock2.now)
        tracker2 = CrossCameraTracker(transition_model=model2, time_provider=clock2.now)

        gid3 = tracker2.get_or_create_global_id("cam_a", local_track_id=1, identity="subj_max")
        clock2.advance(30.0)
        gid4 = tracker2.get_or_create_global_id("cam_b", local_track_id=2, identity="subj_max")
        self.assertEqual(gid3, gid4)

    def test_missing_transition_configuration(self) -> None:
        """When topology config is empty/missing, preserve existing fallback behavior."""
        empty_model = CameraTransitionModel(config={}, time_provider=self.clock.now)
        self.assertFalse(empty_model.is_enabled())

        tracker = CrossCameraTracker(transition_model=empty_model, time_provider=self.clock.now)
        gid1 = tracker.get_or_create_global_id("cam_x", local_track_id=1, identity="person_beta")
        self.clock.advance(10.0)
        gid2 = tracker.get_or_create_global_id("cam_y", local_track_id=1, identity="person_beta")
        self.assertEqual(gid1, gid2)

    def test_invalid_configuration_handling(self) -> None:
        """Configuration with invalid rules must fail safely with warnings."""
        invalid_config = {
            "camera_transitions": {
                "": {"cam_b": {"min_travel_seconds": 5}},
                "cam_a": {
                    "": {"min_travel_seconds": 5},
                    "cam_b": {"min_travel_seconds": -5},
                    "cam_c": {"min_travel_seconds": 20, "max_travel_seconds": 10},
                    "cam_d": {"min_travel_seconds": 5, "max_travel_seconds": 20, "probability": 1.5},
                    "cam_e": {"min_travel_seconds": 5, "max_travel_seconds": 20, "probability": 0.8},
                },
            }
        }
        model = CameraTransitionModel(config=invalid_config, time_provider=self.clock.now)
        self.assertTrue(model.is_enabled())
        with model._lock:
            self.assertIn("cam_a", model._topology)
            self.assertIn("cam_e", model._topology["cam_a"])
            self.assertNotIn("cam_b", model._topology["cam_a"])
            self.assertNotIn("cam_c", model._topology["cam_a"])
            self.assertNotIn("cam_d", model._topology["cam_a"])

    def test_same_local_track_id_on_different_cameras(self) -> None:
        """Ensure (cam_a, 42) and (cam_b, 42) are treated as distinct global keys."""
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)
        tracker = CrossCameraTracker(transition_model=model, time_provider=self.clock.now)

        gid1 = tracker.get_or_create_global_id("cam_a", local_track_id=42, identity="subject_1")
        self.clock.advance(15.0)
        gid2 = tracker.get_or_create_global_id("cam_b", local_track_id=42, identity="subject_1")

        self.assertEqual(gid1, gid2)
        self.assertIn(("cam_a", 42), tracker._local_to_global)
        self.assertIn(("cam_b", 42), tracker._local_to_global)

    def test_deterministic_tie_handling(self) -> None:
        """Resolve candidate ties deterministically."""
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)

        emb = np.ones((128,), dtype=np.float32)

        model.record_exit(
            camera_id="cam_a",
            local_track_id=1,
            global_id="GTRACK-FIRST",
            identity="candidate_1",
            feature_vector=emb,
            timestamp=1000.0,
        )
        model.record_exit(
            camera_id="cam_a",
            local_track_id=2,
            global_id="GTRACK-SECOND",
            identity="candidate_1",
            feature_vector=emb,
            timestamp=1005.0,
        )

        res = model.find_best_transition_candidate(
            dest_camera_id="cam_b",
            dest_local_track_id=99,
            identity="candidate_1",
            feature_vector=emb,
            timestamp=1020.0,
        )
        self.assertIsNotNone(res)
        exit_rec, _score = res
        self.assertEqual(exit_rec.global_id, "GTRACK-FIRST")

    def test_expired_state_cleanup(self) -> None:
        """Exit records older than max_history_seconds are automatically purged."""
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)
        model.record_exit("cam_a", local_track_id=1, global_id="GTRACK-OLD", timestamp=1000.0)

        self.clock.advance(350.0)
        cleaned = model.cleanup_stale_exits()
        self.assertEqual(cleaned, 1)
        self.assertEqual(len(model._exits), 0)

    def test_camera_reconnect_and_track_id_reuse(self) -> None:
        """Handle track ID reuse after long disconnection cleanly."""
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)
        tracker = CrossCameraTracker(transition_model=model, time_provider=self.clock.now)

        gid1 = tracker.get_or_create_global_id("cam_a", local_track_id=1, identity="user_x")
        self.clock.advance(400.0)

        tracker.cleanup_stale_tracks(max_age_seconds=300.0)

        gid2 = tracker.get_or_create_global_id("cam_a", local_track_id=1, identity="user_y")
        self.assertNotEqual(gid1, gid2)

    def test_concurrent_access(self) -> None:
        """Verify thread safety under heavy concurrent access."""
        model = CameraTransitionModel(config=self.sample_config, time_provider=self.clock.now)
        tracker = CrossCameraTracker(transition_model=model, time_provider=self.clock.now)

        def worker(thread_idx: int) -> None:
            for i in range(50):
                cam_id = "cam_a" if i % 2 == 0 else "cam_b"
                track_id = thread_idx * 1000 + i
                tracker.get_or_create_global_id(cam_id, local_track_id=track_id, identity=f"user_{thread_idx}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertGreater(len(tracker._global_tracks), 0)

    def test_backward_compatibility_when_disabled(self) -> None:
        """Verify CrossCameraTracker default instantiation operates identically to legacy tracker."""
        tracker = CrossCameraTracker(max_transition_time_seconds=60.0)
        self.assertIsNone(tracker.transition_model)

        gid1 = tracker.get_or_create_global_id("cam1", local_track_id=1, identity="legacy_user")
        gid2 = tracker.get_or_create_global_id("cam2", local_track_id=10, identity="legacy_user")
        self.assertEqual(gid1, gid2)


class TestMultiCameraPipelineTransitionIntegration(unittest.TestCase):
    """Integration test verifying full multi-camera recognition pipeline wiring with CameraTransitionModel."""

    @patch("pipeline.multi_camera_recognition.MultiCameraRecognitionPipeline._load_model")
    @patch("pipeline.multi_camera_recognition.VectorStore")
    @patch("pipeline.multi_camera_recognition.MultiStreamEngine")
    def test_pipeline_integration_wiring(self, mock_stream_engine, mock_vector_store, mock_load_model) -> None:
        """Verify MultiCameraRecognitionPipeline instantiates transition model and maps global IDs without external weights."""
        mock_load_model.return_value = MagicMock()
        mock_vector_store.return_value.load.return_value = None

        from pipeline.multi_camera_recognition import MultiCameraRecognitionPipeline

        pipeline = MultiCameraRecognitionPipeline(
            cameras_config_path="configs/cameras.yaml",
            gallery_dir="models/live_gallery",
        )

        self.assertIsNotNone(pipeline.transition_model)
        self.assertIsNotNone(pipeline.cross_camera_tracker)

        topology_config = {
            "camera_transitions": {
                "camera_01": {
                    "camera_02": {
                        "min_travel_seconds": 1.0,
                        "max_travel_seconds": 60.0,
                        "probability": 0.85,
                    }
                }
            },
            "similarity_threshold": 0.30,
        }
        pipeline.transition_model.load_config(topology_config)
        self.assertTrue(pipeline.transition_model.is_enabled())

        worker1 = pipeline.workers.get("camera_01")
        cam1_id = worker1.camera_id if worker1 else "camera_01"

        gid1 = pipeline.cross_camera_tracker.get_or_create_global_id(
            camera_id=cam1_id,
            local_track_id=100,
            identity="subject_alpha",
        )

        pipeline.cross_camera_tracker.record_track_exit(
            camera_id=cam1_id,
            local_track_id=100,
            identity="subject_alpha",
        )

        cam2_id = "camera_02"

        gid2 = pipeline.cross_camera_tracker.get_or_create_global_id(
            camera_id=cam2_id,
            local_track_id=200,
            identity="subject_alpha",
        )

        self.assertEqual(gid1, gid2)
