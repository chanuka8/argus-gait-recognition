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

    # Add observations from cam_01 and cam_02 at timestamp 1.0
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

    # cam_01 reports Person_A with score 0.90
    engine.add_observation("cam_01", 1, "global_102", "Person_A", gait_similarity=0.90, timestamp=1.0)
    # cam_02 reports Person_B with score 0.90
    engine.add_observation("cam_02", 2, "global_102", "Person_B", gait_similarity=0.90, timestamp=1.0)

    res = engine.fuse_evidence("global_102", current_time=2.0)
    # Each identity only has 1 camera -> minimum 2 cameras per identity condition defers
    assert res.fusion_state == FusionState.DEFERRED
    # Deterministic tie-break picks Person_A (alphanumeric order)
    assert res.fused_identity == "Person_A"


def test_expired_evidence_ignored():
    engine = MultiCameraEvidenceFusion({"enabled": True, "evidence_ttl_seconds": 10.0})

    # Obs 1 at t=1.0
    engine.add_observation("cam_01", 1, "global_103", "Person_A", gait_similarity=0.90, timestamp=1.0)
    # Obs 2 at t=20.0 (> 10s later)
    engine.add_observation("cam_02", 2, "global_103", "Person_A", gait_similarity=0.90, timestamp=20.0)

    # Fuse at t=21.0 -> obs 1 is expired
    res = engine.fuse_evidence("global_103", current_time=21.0)
    assert res.fusion_state == FusionState.DEFERRED
    assert len(res.contributing_cameras) == 1
