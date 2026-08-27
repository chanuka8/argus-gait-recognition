"""
Unit tests for Stage 2: Recognition Deferral Engine.
"""

from intelligence.recognition_deferral_engine import (
    RecognitionDeferralEngine,
    RecognitionState,
)


def test_disabled_mode_preserves_old_output():
    engine = RecognitionDeferralEngine({"enabled": False})
    res = engine.evaluate_and_accumulate(
        camera_id="cam_01",
        track_id=1,
        identity_candidate="Person_A",
        similarity=0.90,
        quality=0.80,
        open_set_state="KNOWN",
        temporal_decision="MAJORITY_VOTE",
        reliability=0.85,
        occlusion=0.10,
    )
    assert res.recognition_state == RecognitionState.CONFIRMED
    assert res.recognition_deferred is False
    assert res.identity == "Person_A"
    assert res.should_alert is True


def test_insufficient_evidence_defers():
    engine = RecognitionDeferralEngine({"enabled": True, "minimum_confirmations": 3})

    res = engine.evaluate_and_accumulate(
        camera_id="cam_01",
        track_id=1,
        identity_candidate="Person_A",
        similarity=0.90,
        quality=0.80,
        open_set_state="KNOWN",
        temporal_decision="MAJORITY_VOTE",
        reliability=0.85,
        occlusion=0.10,
    )
    assert res.recognition_state == RecognitionState.DEFERRED_INSUFFICIENT_EVIDENCE
    assert res.recognition_deferred is True
    assert res.should_alert is False
    assert "Confirmations count" in res.defer_reason


def test_sufficient_repeated_evidence_confirms():
    engine = RecognitionDeferralEngine({"enabled": True, "minimum_confirmations": 3})

    for t in range(3):
        res = engine.evaluate_and_accumulate(
            camera_id="cam_01",
            track_id=1,
            identity_candidate="Person_A",
            similarity=0.90,
            quality=0.80,
            open_set_state="KNOWN",
            temporal_decision="MAJORITY_VOTE",
            reliability=0.85,
            occlusion=0.10,
            timestamp=float(t),
        )

    assert res.recognition_state == RecognitionState.CONFIRMED
    assert res.recognition_deferred is False
    assert res.identity == "Person_A"
    assert res.should_alert is True


def test_unknown_never_confirms():
    engine = RecognitionDeferralEngine({"enabled": True})
    res = engine.evaluate_and_accumulate(
        camera_id="cam_01",
        track_id=1,
        identity_candidate="UNKNOWN",
        similarity=0.0,
        quality=0.80,
        open_set_state="UNKNOWN",
        temporal_decision="UNKNOWN_PERSON",
        reliability=0.30,
        occlusion=0.10,
    )
    assert res.recognition_state == RecognitionState.UNKNOWN
    assert res.identity == "UNKNOWN"
    assert res.should_alert is False


def test_uncertain_remains_deferred():
    engine = RecognitionDeferralEngine({"enabled": True, "minimum_confirmations": 1})
    res = engine.evaluate_and_accumulate(
        camera_id="cam_01",
        track_id=1,
        identity_candidate="Person_B",
        similarity=0.75,
        quality=0.80,
        open_set_state="UNCERTAIN",
        temporal_decision="SINGLE_MATCH",
        reliability=0.60,
        occlusion=0.10,
    )
    assert res.recognition_state == RecognitionState.DEFERRED_INSUFFICIENT_EVIDENCE
    assert res.should_alert is False


def test_ttl_expiry_and_cleanup():
    engine = RecognitionDeferralEngine({"enabled": True, "evidence_ttl_seconds": 5.0, "minimum_confirmations": 3})

    engine.evaluate_and_accumulate(
        "cam_01", 1, "Person_A", 0.90, 0.80, "KNOWN", "MAJORITY_VOTE", 0.85, 0.10, timestamp=1.0
    )
    engine.evaluate_and_accumulate(
        "cam_01", 1, "Person_A", 0.90, 0.80, "KNOWN", "MAJORITY_VOTE", 0.85, 0.10, timestamp=2.0
    )

    res = engine.evaluate_and_accumulate(
        "cam_01", 1, "Person_A", 0.90, 0.80, "KNOWN", "MAJORITY_VOTE", 0.85, 0.10, timestamp=10.0
    )
    assert res.recognition_state == RecognitionState.DEFERRED_INSUFFICIENT_EVIDENCE
    assert res.accumulated_evidence_count == 1
