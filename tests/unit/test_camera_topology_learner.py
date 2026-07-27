"""
Unit tests for Stage 4: Automatic Camera Topology Learning.
"""

from pathlib import Path
from intelligence.camera_topology_learner import CameraTopologyLearner
from intelligence.camera_transition_model import CameraTransitionModel


def test_valid_observation_accepted_and_bounds():
    model = CameraTransitionModel()
    learner = CameraTopologyLearner(
        {
            "enabled": True,
            "shadow_mode": True,
            "minimum_samples": 2,
            "maximum_travel_seconds": 600.0,
            "sync_interval_seconds": 30.0,
        },
        transition_model=model,
    )
    learner.set_transition_model(model)

    # Record exit at cam_A at t=10.0
    learner.record_camera_exit("cam_A", "Person_10", reliability=0.90, occlusion=0.10, timestamp=10.0)

    # Observe entry at cam_B at t=25.0 (travel time 15s)
    accepted = learner.observe_transition(
        source_camera="cam_A",
        destination_camera="cam_B",
        identity="Person_10",
        reliability=0.90,
        occlusion=0.10,
        is_known_identity=True,
        is_temporally_confirmed=True,
        timestamp=25.0,
    )
    assert accepted is True
    edge = learner.learned_edges[("cam_A", "cam_B")]
    assert edge.transition_count == 1
    assert edge.mean_travel_time == 15.0


def test_low_reliability_and_occlusion_rejected():
    model = CameraTransitionModel()
    learner = CameraTopologyLearner(
        {
            "enabled": True,
            "shadow_mode": True,
            "minimum_samples": 1,
            "maximum_travel_seconds": 600.0,
            "sync_interval_seconds": 30.0,
        },
        transition_model=model,
    )
    learner.set_transition_model(model)

    learner.record_camera_exit("cam_A", "Person_11", reliability=0.90, occlusion=0.10, timestamp=10.0)

    # Low reliability observation at destination
    accepted = learner.observe_transition(
        source_camera="cam_A",
        destination_camera="cam_B",
        identity="Person_11",
        reliability=0.50,  # Low reliability
        occlusion=0.10,
        timestamp=25.0,
    )
    assert accepted is False


def test_invalid_travel_time_rejected():
    model = CameraTransitionModel()
    learner = CameraTopologyLearner(
        {
            "enabled": True,
            "shadow_mode": True,
            "minimum_samples": 1,
            "maximum_travel_seconds": 60.0,
            "sync_interval_seconds": 30.0,
        },
        transition_model=model,
    )
    learner.set_transition_model(model)

    learner.record_camera_exit("cam_A", "Person_12", reliability=0.90, occlusion=0.10, timestamp=10.0)

    # Travel time = 100s (> max 60s)
    accepted = learner.observe_transition(
        source_camera="cam_A",
        destination_camera="cam_B",
        identity="Person_12",
        reliability=0.90,
        occlusion=0.10,
        timestamp=110.0,
    )
    assert accepted is False


def test_minimum_samples_and_probability_normalization():
    model = CameraTransitionModel()
    learner = CameraTopologyLearner(
        {
            "enabled": True,
            "shadow_mode": True,
            "minimum_samples": 3,
            "maximum_travel_seconds": 600.0,
            "sync_interval_seconds": 30.0,
        },
        transition_model=model,
    )
    learner.set_transition_model(model)

    # Add 2 transitions cam_A -> cam_B
    for i in range(2):
        learner.record_camera_exit("cam_A", f"P_{i}", 0.90, 0.10, timestamp=10.0 + i)
        learner.observe_transition("cam_A", "cam_B", f"P_{i}", 0.90, 0.10, timestamp=20.0 + i)

    # Add 1 transition cam_A -> cam_C
    learner.record_camera_exit("cam_A", "P_C", 0.90, 0.10, timestamp=10.0)
    learner.observe_transition("cam_A", "cam_C", "P_C", 0.90, 0.10, timestamp=20.0)

    # Suggested topology should still be empty because minimum_samples = 3 (cam_A->cam_B has 2, cam_A->cam_C has 1)
    sug = learner.get_suggested_topology()
    assert len(sug) == 0

    # Add 3rd transition cam_A -> cam_B
    learner.record_camera_exit("cam_A", "P_3", 0.90, 0.10, timestamp=10.0)
    learner.observe_transition("cam_A", "cam_B", "P_3", 0.90, 0.10, timestamp=20.0)

    sug = learner.get_suggested_topology()
    assert "cam_A_to_cam_B" in sug
    assert sug["cam_A_to_cam_B"]["transition_count"] == 3
    # Probability = 3 / (3 + 1) = 0.75
    assert sug["cam_A_to_cam_B"]["learned_probability"] == 0.75


def test_shadow_mode_export_and_reset(tmp_path):
    export_file = tmp_path / "learned_topology.yaml"
    model = CameraTransitionModel()
    learner = CameraTopologyLearner(
        {
            "enabled": True,
            "shadow_mode": True,
            "minimum_samples": 1,
            "maximum_travel_seconds": 600.0,
            "sync_interval_seconds": 10.0,
            "export_path": str(export_file),
        },
        transition_model=model,
    )
    learner.set_transition_model(model)

    learner.record_camera_exit("cam_1", "User_X", 0.95, 0.05, timestamp=1.0)
    learner.observe_transition("cam_1", "cam_2", "User_X", 0.95, 0.05, timestamp=10.0)

    exported_path = learner.export_learned_topology()
    assert Path(exported_path).exists()

    # In shadow_mode=True, update_transition_model MUST NOT mutate active transition model
    count_shadow = learner.update_transition_model(model)
    assert count_shadow == 0
    assert not model.is_enabled()

    # Disabling shadow_mode allows update_transition_model to sync rules
    learner.shadow_mode = False
    count_active = learner.update_transition_model(model)
    assert count_active == 1
    assert model.is_enabled()

    # Reset
    learner.reset()
    assert len(learner.learned_edges) == 0
    assert len(learner.exit_events) == 0
    assert learner.last_sync_time == -float("inf")


def test_maybe_sync_transition_model_bounded_interval():
    model = CameraTransitionModel()
    learner = CameraTopologyLearner(
        {
            "enabled": True,
            "shadow_mode": False,
            "minimum_samples": 1,
            "maximum_travel_seconds": 600.0,
            "sync_interval_seconds": 10.0,
        },
        transition_model=model,
    )
    learner.set_transition_model(model)
    assert not model.is_enabled()

    learner.record_camera_exit("cam_A", "User_Z", 0.90, 0.05, timestamp=1.0)
    learner.observe_transition("cam_A", "cam_B", "User_Z", 0.90, 0.05, timestamp=5.0)

    # First sync at t=5.0 should succeed
    synced1 = learner.maybe_sync_transition_model(timestamp=5.0)
    assert synced1 == 1
    assert model.is_enabled()

    # Second sync at t=8.0 (< 10s interval) should skip sync
    synced2 = learner.maybe_sync_transition_model(timestamp=8.0)
    assert synced2 == 0

    # Third sync at t=16.0 (>= 10s interval) should succeed
    synced3 = learner.maybe_sync_transition_model(timestamp=16.0)
    assert synced3 == 1
