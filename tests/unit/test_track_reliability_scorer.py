"""
Unit tests for TrackReliabilityScorer and Pipeline Integration.
"""

from intelligence.track_reliability_scorer import TrackReliabilityScorer


def test_disabled_by_default_behavior():
    scorer = TrackReliabilityScorer()
    assert not scorer.is_enabled()
    assert not scorer.enabled


def test_unknown_open_set_mapping():
    scorer = TrackReliabilityScorer(enabled=True)
    subscore = scorer._compute_open_set_subscore("UNKNOWN")
    assert subscore == 0.0


def test_known_uncertain_unknown_ordering():
    scorer = TrackReliabilityScorer(enabled=True)

    score_known = scorer.compute_reliability(
        quality_score=1.0,
        temporal_decision="MAJORITY_VOTE",
        open_set_state="KNOWN",
        observation_count=15,
    )
    score_uncertain = scorer.compute_reliability(
        quality_score=1.0,
        temporal_decision="MAJORITY_VOTE",
        open_set_state="UNCERTAIN",
        observation_count=15,
    )
    score_unknown = scorer.compute_reliability(
        quality_score=1.0,
        temporal_decision="MAJORITY_VOTE",
        open_set_state="UNKNOWN",
        observation_count=15,
    )

    assert score_known > score_uncertain
    assert score_uncertain > score_unknown


def test_track_reliability_score_bounds():
    scorer = TrackReliabilityScorer(enabled=True)

    high_score = scorer.compute_reliability(
        quality_score=1.0,
        temporal_decision="MAJORITY_VOTE",
        open_set_state="KNOWN",
        observation_count=20,
        detection_confidence=0.95,
        persistence_score=0.9,
    )
    assert 0.0 <= high_score <= 1.0
    assert high_score > 0.8

    low_score = scorer.compute_reliability(
        quality_score=0.2,
        temporal_decision="UNCERTAIN",
        open_set_state="UNKNOWN",
        observation_count=1,
        detection_confidence=0.3,
    )
    assert 0.0 <= low_score <= 1.0
    assert low_score < 0.3


def test_track_reliability_evaluation_dict():
    scorer = TrackReliabilityScorer(enabled=True)
    res = scorer.evaluate_track(
        quality_score=0.8,
        temporal_decision="SINGLE_MATCH",
        open_set_state="KNOWN",
        observation_count=15,
        detection_confidence=0.9,
    )
    assert "reliability_score" in res
    assert "level" in res
    assert "is_reliable" in res
    assert "identity_confidence" in res
    assert "track_stability" in res
    assert "components" in res
    assert res["level"] in ("HIGH", "MEDIUM", "LOW")
    assert isinstance(res["reliability_score"], float)
    assert 0.0 <= res["reliability_score"] <= 1.0


def test_backward_compatibility_disabled_pipeline_output():
    from pipeline.video_recognition import _load_track_reliability_config

    cfg = _load_track_reliability_config()
    assert not cfg.get("enabled", False)
