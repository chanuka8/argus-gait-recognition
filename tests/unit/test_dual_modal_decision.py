import pytest

from intelligence.dual_modal_fusion import DualModalFusion
from services.recognition_worker import (
    RecognitionWorker,
)


@pytest.fixture
def fusion_engine():
    return DualModalFusion(
        default_gait_weight=0.70,
        default_reid_weight=0.30,
        enabled=True,
    )


def test_case_1_same_identity_fusion(fusion_engine):
    """Case 1: Both modalities identify the same person with scores passing thresholds."""
    res = fusion_engine.decide_identity(
        gait_identity="Person_001",
        gait_score=0.91,
        appearance_identity="Person_001",
        appearance_score=0.87,
        gait_threshold=0.85,
        appearance_threshold=0.60,
    )

    assert res["final_identity"] == "Person_001"
    assert res["status"] == "CONFIRMED"
    assert res["decision"] == "CONFIRMED"
    assert res["modality_state"] == "DUAL_MODAL_MATCH"
    assert res["conflict"] is False
    # Score should be fused: 0.70 * 0.91 + 0.30 * 0.87 = 0.637 + 0.261 = 0.898
    assert res["final_score"] == pytest.approx(0.898, abs=1e-3)


def test_case_2_gait_only_fallback(fusion_engine):
    """Case 2: Appearance is unavailable; fallback safely to gait-only."""
    # Subcase A: Appearance is None / unavailable
    res_none = fusion_engine.decide_identity(
        gait_identity="Person_001",
        gait_score=0.91,
        appearance_identity=None,
        appearance_score=None,
        gait_threshold=0.85,
        appearance_threshold=0.60,
    )
    assert res_none["final_identity"] == "Person_001"
    assert res_none["final_score"] == 0.91
    assert res_none["status"] == "CONFIRMED"
    assert res_none["modality_state"] == "GAIT_ONLY"
    assert res_none["conflict"] is False

    # Subcase B: Appearance is UNKNOWN_PERSON
    res_unknown = fusion_engine.decide_identity(
        gait_identity="Person_001",
        gait_score=0.91,
        appearance_identity="UNKNOWN_PERSON",
        appearance_score=0.45,
        gait_threshold=0.85,
        appearance_threshold=0.60,
    )
    assert res_unknown["final_identity"] == "Person_001"
    assert res_unknown["final_score"] == 0.91
    assert res_unknown["status"] == "CONFIRMED"
    assert res_unknown["modality_state"] == "GAIT_ONLY"


def test_case_3_appearance_only_fallback(fusion_engine):
    """Case 3: Gait is unavailable / unknown; use appearance-only decision."""
    res = fusion_engine.decide_identity(
        gait_identity="UNKNOWN",
        gait_score=0.0,
        appearance_identity="Person_001",
        appearance_score=0.87,
        gait_threshold=0.85,
        appearance_threshold=0.60,
    )

    assert res["final_identity"] == "Person_001"
    assert res["final_score"] == 0.87
    assert res["status"] == "CONFIRMED"
    assert res["modality_state"] == "APPEARANCE_ONLY"
    assert res["conflict"] is False


def test_case_4_both_unavailable(fusion_engine):
    """Case 4: Both modalities are unavailable / unknown."""
    res = fusion_engine.decide_identity(
        gait_identity="UNKNOWN",
        gait_score=0.0,
        appearance_identity="UNKNOWN_PERSON",
        appearance_score=0.0,
        gait_threshold=0.85,
        appearance_threshold=0.60,
    )

    assert res["final_identity"] == "UNKNOWN_PERSON"
    assert res["final_score"] == 0.0
    assert res["status"] == "UNKNOWN"
    assert res["decision"] == "UNKNOWN"
    assert res["conflict"] is False


def test_case_5_conflicting_identities_handling(fusion_engine):
    """Case 5: Both modalities pass thresholds but produce conflicting identities."""
    res = fusion_engine.decide_identity(
        gait_identity="Person_001",
        gait_score=0.91,
        appearance_identity="Person_007",
        appearance_score=0.89,
        gait_threshold=0.85,
        appearance_threshold=0.60,
    )

    # Must NOT blindly average scores across distinct candidates
    assert res["conflict"] is True
    assert res["decision"] == "REVIEW_REQUIRED"
    assert res["status"] == "REVIEW_REQUIRED"
    assert res["final_identity"] == "REVIEW_REQUIRED"
    assert res["modality_state"] == "CONFLICT"
    assert res["gait_candidate"] == "Person_001"
    assert res["appearance_candidate"] == "Person_007"


def test_case_6_scores_below_thresholds(fusion_engine):
    """Case 6: Candidate scores fail respective thresholds -> UNKNOWN_PERSON."""
    res = fusion_engine.decide_identity(
        gait_identity="Person_001",
        gait_score=0.72,  # Below gait threshold 0.85
        appearance_identity="Person_001",
        appearance_score=0.48,  # Below appearance threshold 0.60
        gait_threshold=0.85,
        appearance_threshold=0.60,
    )

    assert res["final_identity"] == "UNKNOWN_PERSON"
    assert res["status"] == "UNKNOWN"
    assert res["decision"] == "UNKNOWN"
    assert res["conflict"] is False


def test_custom_fusion_weights_configuration():
    """Requirement 8: Verify custom gait_weight and appearance_weight allocation."""
    custom_fusion = DualModalFusion(
        default_gait_weight=0.80,
        default_reid_weight=0.20,
        enabled=True,
    )

    res = custom_fusion.decide_identity(
        gait_identity="Subject_X",
        gait_score=0.90,
        appearance_identity="Subject_X",
        appearance_score=0.80,
        gait_threshold=0.85,
        appearance_threshold=0.60,
    )

    # 0.80 * 0.90 + 0.20 * 0.80 = 0.72 + 0.16 = 0.88
    assert res["final_score"] == pytest.approx(0.88, abs=1e-3)
    assert res["final_identity"] == "Subject_X"


def test_recognition_worker_fusion_enabled_vs_disabled():
    """Test RecognitionWorker decision behavior when fusion is explicitly enabled vs disabled."""
    # When fusion is enabled in config
    fusion_on = DualModalFusion(default_gait_weight=0.7, default_reid_weight=0.3, enabled=True)
    worker_on = RecognitionWorker(
        camera_id="cam_test_on",
        fusion_engine=fusion_on,
    )
    assert worker_on.fusion_engine.is_enabled() is True

    # When fusion is disabled (default)
    fusion_off = DualModalFusion(enabled=False)
    worker_off = RecognitionWorker(
        camera_id="cam_test_off",
        fusion_engine=fusion_off,
    )
    assert worker_off.fusion_engine.is_enabled() is False
