from intelligence.track_identity_aggregator import TrackIdentityAggregator


def test_track_identity_aggregator_init() -> None:
    aggregator = TrackIdentityAggregator(
        window_size=8,
        consensus_threshold=0.60,
        confirm_threshold=0.72,
        near_miss_margin=0.05,
        min_frames_for_decision=3,
    )
    assert aggregator.window_size == 8
    assert aggregator.consensus_threshold == 0.60
    assert aggregator.confirm_threshold == 0.72
    assert aggregator.near_miss_margin == 0.05
    assert aggregator.min_frames_for_decision == 3


def test_track_identity_aggregator_confirmed_decision() -> None:
    aggregator = TrackIdentityAggregator(
        window_size=5,
        consensus_threshold=0.60,
        confirm_threshold=0.72,
        near_miss_margin=0.05,
        min_frames_for_decision=3,
        high_risk_confusion_groups=[["Devhan", "Isuru", "person01"]],
    )


    r1 = aggregator.update(track_id=1, identity="demo_person_001", score=0.75)
    assert r1["decision"] == "UNKNOWN"

    r2 = aggregator.update(track_id=1, identity="demo_person_001", score=0.74)
    assert r2["decision"] == "UNKNOWN"


    r3 = aggregator.update(track_id=1, identity="demo_person_001", score=0.76)
    assert r3["decision"] == "CONFIRMED"
    assert r3["status"] == "CONFIRMED"
    assert r3["identity"] == "demo_person_001"
    assert r3["confidence"] >= 0.72
    assert r3["consensus_fraction"] == 1.0


def test_track_identity_aggregator_confusion_safeguard() -> None:
    aggregator = TrackIdentityAggregator(
        window_size=5,
        consensus_threshold=0.60,
        confirm_threshold=0.72,
        near_miss_margin=0.05,
        min_frames_for_decision=3,
        high_risk_confusion_groups=[["Devhan", "Isuru", "person01"]],
    )

    for _ in range(3):
        r = aggregator.update(track_id=10, identity="Devhan", score=0.85)

    assert r["decision"] == "REVIEW_REQUIRED"
    assert r["status"] == "REVIEW_REQUIRED"
    assert r["identity"] == "Devhan"
    assert "High-risk confusion pair" in r["alert_reason"]


def test_track_identity_aggregator_near_miss_review() -> None:
    aggregator = TrackIdentityAggregator(
        window_size=5,
        consensus_threshold=0.60,
        confirm_threshold=0.72,
        near_miss_margin=0.05,
        min_frames_for_decision=3,
    )

    aggregator.update(track_id=2, identity="Isuru", score=0.68)
    aggregator.update(track_id=2, identity="Isuru", score=0.69)
    r3 = aggregator.update(track_id=2, identity="Isuru", score=0.68)

    assert r3["decision"] == "REVIEW_REQUIRED"
    assert r3["status"] == "REVIEW_REQUIRED"
    assert r3["identity"] == "Isuru"
    assert 0.67 <= r3["confidence"] < 0.72


def test_track_identity_aggregator_low_confidence() -> None:
    aggregator = TrackIdentityAggregator(
        window_size=5,
        consensus_threshold=0.60,
        confirm_threshold=0.72,
        near_miss_margin=0.05,
        min_frames_for_decision=3,
    )

    aggregator.update(track_id=3, identity="person01", score=0.55)
    aggregator.update(track_id=3, identity="person01", score=0.58)
    r3 = aggregator.update(track_id=3, identity="person01", score=0.56)

    assert r3["decision"] == "LOW_CONFIDENCE"
    assert r3["status"] == "UNKNOWN"
    assert r3["identity"] == "person01"


def test_track_identity_aggregator_track_lost_reset() -> None:
    aggregator = TrackIdentityAggregator(
        window_size=5,
        consensus_threshold=0.60,
        confirm_threshold=0.72,
        min_frames_for_decision=3,
    )

    for _ in range(4):
        aggregator.update(track_id=4, identity="demo_person_001", score=0.78)

    summary = aggregator.on_track_lost(track_id=4)
    assert summary is not None
    assert summary["outcome"] == "CONFIRMED"
    assert summary["final_candidate"] == "demo_person_001"
    assert summary["total_frames"] == 4


    r_new = aggregator.update(track_id=4, identity="demo_person_001", score=0.78)
    assert r_new["window_size"] == 1
    assert r_new["decision"] == "UNKNOWN"
