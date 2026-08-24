"""
Unit tests for Stage 3: Multi-Camera Evidence Fusion.
"""

from intelligence.multi_camera_evidence_fusion import (
    FusionState,
    MultiCameraEvidenceFusion,
)


def test_disabled_mode_single_camera_fallback():
    engine = MultiCameraEvidenceFusion({"enabled": False})
    res = engine.fuse_evidence("entity_1", fallback_identity="Person_A", fallback_score=0.90)

    assert res.fusion_state == FusionState.CONFIRMED
    assert res.fused_identity == "Person_A"
    assert res.fused_score == 0.90
    assert "disabled" in res.reason.lower()


def test_two_strong_cameras_confirm():
    engine = MultiCameraEvidenceFusion({"enabled": True, "minimum_cameras": 2, "minimum_fused_score": 0.85})

    engine.add_observation("cam_01", 1, "global_100", "Person_A", gait_similarity=0.92, timestamp=1.0)
    engine.add_observation("cam_02", 2, "global_100", "Person_A", gait_similarity=0.88, timestamp=1.0)

    res = engine.fuse_evidence("global_100", current_time=2.0)
    assert res.fusion_state == FusionState.CONFIRMED
    assert res.fused_identity == "Person_A"
    assert res.fused_score >= 0.85
    assert set(res.contributing_cameras) == {"cam_01", "cam_02"}


def test_single_camera_defers_when_minimum_cameras_two():
    engine = MultiCameraEvidenceFusion({"enabled": True, "minimum_cameras": 2, "minimum_fused_score": 0.85})
    engine.add_observation("cam_01", 1, "global_101", "Person_A", gait_similarity=0.95, timestamp=1.0)

    res = engine.fuse_evidence("global_101", current_time=2.0)
    assert res.fusion_state == FusionState.DEFERRED
    assert "minimum required" in res.reason


def test_conflicting_identities_and_deterministic_tie():
    engine = MultiCameraEvidenceFusion({"enabled": True, "minimum_cameras": 2, "minimum_fused_score": 0.85})

    engine.add_observation("cam_01", 1, "global_102", "Person_A", gait_similarity=0.90, timestamp=1.0)
    engine.add_observation("cam_02", 2, "global_102", "Person_B", gait_similarity=0.90, timestamp=1.0)

    res = engine.fuse_evidence("global_102", current_time=2.0)
    assert res.fusion_state == FusionState.DEFERRED
    assert res.fused_identity == "Person_A"


def test_expired_evidence_ignored():
    engine = MultiCameraEvidenceFusion({"enabled": True, "evidence_ttl_seconds": 10.0})

    engine.add_observation("cam_01", 1, "global_103", "Person_A", gait_similarity=0.90, timestamp=1.0)
    engine.add_observation("cam_02", 2, "global_103", "Person_A", gait_similarity=0.90, timestamp=20.0)

    res = engine.fuse_evidence("global_103", current_time=21.0)
    assert res.fusion_state == FusionState.DEFERRED
    assert len(res.contributing_cameras) == 1


def test_duplicate_suppression_and_from_config():
    engine = MultiCameraEvidenceFusion.from_config({"enabled": True, "minimum_cameras": 2})

    engine.add_observation("cam_01", 1, "global_104", "Person_A", gait_similarity=0.90, timestamp=1.0)
    engine.add_observation("cam_01", 1, "global_104", "Person_A", gait_similarity=0.90, timestamp=1.0)

    obs = engine.observations.get("global_104", [])
    assert len(obs) == 1
