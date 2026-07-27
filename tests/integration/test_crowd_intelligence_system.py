"""
Integration tests for unified Crowd Intelligence System across Stages 1 to 4.
"""

from intelligence.crowd_intelligence_system import CrowdIntelligenceSystem
from intelligence.crowd_occlusion_analyzer import CrowdDensityLevel


def test_disabled_by_default_preserves_baseline():
    system = CrowdIntelligenceSystem()
    assert not system.is_enabled()

    # Frame processing
    detections = [{"track_id": 1, "bbox": [10, 10, 50, 100]}]
    res_frame = system.process_frame(detections, (1080, 1920), "cam_00", 1.0)
    assert res_frame.crowd_density_level == CrowdDensityLevel.LOW

    # Track evaluation in disabled mode
    eval_res = system.evaluate_track_recognition(
        camera_id="cam_00",
        track_id=1,
        identity_candidate="Person_X",
        similarity=0.90,
        quality=0.80,
        open_set_state="KNOWN",
        temporal_decision="MAJORITY_VOTE",
        reliability=0.85,
        occlusion_score=0.10,
        timestamp=1.0,
    )
    assert eval_res.recognition_state == "CONFIRMED"
    assert eval_res.recognition_deferred is False
    assert eval_res.should_alert is True


def test_full_crowd_intelligence_pipeline_enabled(tmp_path):
    config = {
        "enabled": True,
        "occlusion": {
            "enabled": True,
            "high_threshold": 0.50,
        },
        "recognition_deferral": {
            "enabled": True,
            "minimum_confirmations": 2,
            "minimum_reliability": 0.70,
        },
        "multi_camera_fusion": {
            "enabled": True,
            "minimum_cameras": 2,
            "minimum_fused_score": 0.85,
        },
        "topology_learning": {
            "enabled": True,
            "minimum_samples": 2,
            "export_path": str(tmp_path / "learned_topology.yaml"),
        },
    }
    system = CrowdIntelligenceSystem(config)
    assert system.is_enabled()

    # Frame 1 & 2 on cam_01
    detections_c1 = [{"track_id": 1, "bbox": [100, 100, 200, 300]}]
    system.process_frame(detections_c1, (1080, 1920), "cam_01", timestamp=10.0)

    # Eval 1 on cam_01 -> deferral (only 1 confirmation)
    eval1 = system.evaluate_track_recognition(
        camera_id="cam_01",
        track_id=1,
        identity_candidate="Subject_A",
        similarity=0.90,
        quality=0.80,
        open_set_state="KNOWN",
        temporal_decision="MAJORITY_VOTE",
        reliability=0.85,
        occlusion_score=0.10,
        global_track_id="global_subject_A",
        timestamp=10.0,
    )
    assert eval1.recognition_state == "DEFERRED_INSUFFICIENT_EVIDENCE"
    assert eval1.recognition_deferred is True

    # Eval 2 on cam_01 -> confirmation (2 confirmations)
    eval2 = system.evaluate_track_recognition(
        camera_id="cam_01",
        track_id=1,
        identity_candidate="Subject_A",
        similarity=0.90,
        quality=0.80,
        open_set_state="KNOWN",
        temporal_decision="MAJORITY_VOTE",
        reliability=0.85,
        occlusion_score=0.10,
        global_track_id="global_subject_A",
        timestamp=11.0,
    )
    assert eval2.recognition_state == "CONFIRMED"

    # Now add observations from cam_02 at t=25.0
    eval3 = system.evaluate_track_recognition(
        camera_id="cam_02",
        track_id=5,
        identity_candidate="Subject_A",
        similarity=0.92,
        quality=0.85,
        open_set_state="KNOWN",
        temporal_decision="MAJORITY_VOTE",
        reliability=0.90,
        occlusion_score=0.10,
        global_track_id="global_subject_A",
        source_camera="cam_01",
        timestamp=25.0,
    )
    assert eval3.fusion_state == "CONFIRMED"
    assert set(eval3.contributing_cameras) == {"cam_01", "cam_02"}

    # Topology learning observation accepted
    assert eval3.topology_observation_accepted is True


def test_runtime_topology_model_sync():
    from intelligence.camera_transition_model import CameraTransitionModel

    active_model = CameraTransitionModel()
    config = {
        "enabled": True,
        "occlusion": {
            "enabled": True,
        },
        "recognition_deferral": {
            "enabled": True,
            "minimum_confirmations": 1,
            "minimum_reliability": 0.70,
        },
        "multi_camera_fusion": {
            "enabled": True,
        },
        "topology_learning": {
            "enabled": True,
            "shadow_mode": False,  # Live mode enabled
            "minimum_samples": 1,
            "maximum_travel_seconds": 600.0,
            "sync_interval_seconds": 1.0,
        },
    }
    system = CrowdIntelligenceSystem(config, transition_model=active_model)
    system.set_transition_model(active_model)
    assert not active_model.is_enabled()

    # Record exit at cam_01
    system.topology_learner.record_camera_exit("cam_01", "User_X", 0.90, 0.05, timestamp=1.0)

    # Cross-camera observation at cam_02
    eval_res = system.evaluate_track_recognition(
        camera_id="cam_02",
        track_id=10,
        identity_candidate="User_X",
        similarity=0.90,
        quality=0.85,
        open_set_state="KNOWN",
        temporal_decision="MAJORITY_VOTE",
        reliability=0.85,
        source_camera="cam_01",
        timestamp=10.0,
    )

    assert eval_res.topology_observation_accepted is True
    # Active CameraTransitionModel now has rule cam_01 -> cam_02 applied online
    assert active_model.is_enabled() is True
