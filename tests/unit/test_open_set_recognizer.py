"""Unit tests for OpenSetRecognizer 3-state decision model."""

from intelligence.open_set_recognizer import OpenSetDecisionResult, OpenSetRecognizer, OpenSetState


def test_open_set_recognizer_known():
    recognizer = OpenSetRecognizer(known_threshold=0.85, unknown_threshold=0.70, margin_threshold=0.05)

    top_matches = [("subject_001", 0.92), ("subject_002", 0.70)]
    res = recognizer.evaluate_open_set_decision(top_matches)

    assert isinstance(res, OpenSetDecisionResult)
    assert res.state == OpenSetState.KNOWN
    assert res.identity == "subject_001"
    assert res.score == 0.92


def test_open_set_recognizer_unknown():
    recognizer = OpenSetRecognizer(known_threshold=0.85, unknown_threshold=0.70, margin_threshold=0.05)

    top_matches = [("subject_001", 0.55), ("subject_002", 0.50)]
    res = recognizer.evaluate_open_set_decision(top_matches)

    assert res.state == OpenSetState.UNKNOWN
    assert res.identity == "UNKNOWN"
    assert res.score == 0.55

    res_empty = recognizer.evaluate_open_set_decision([])
    assert res_empty.state == OpenSetState.UNKNOWN
    assert res_empty.identity == "UNKNOWN"


def test_open_set_recognizer_uncertain():
    recognizer = OpenSetRecognizer(known_threshold=0.85, unknown_threshold=0.70, margin_threshold=0.05)

    top_matches_gray = [("subject_001", 0.78), ("subject_002", 0.60)]
    res_gray = recognizer.evaluate_open_set_decision(top_matches_gray)

    assert res_gray.state == OpenSetState.UNCERTAIN
    assert res_gray.identity == "subject_001"

    top_matches_tight = [("subject_001", 0.90), ("subject_002", 0.88)]
    res_tight = recognizer.evaluate_open_set_decision(top_matches_tight)

    assert res_tight.state == OpenSetState.UNCERTAIN
    assert res_tight.identity == "subject_001"

    top_matches_high = [("subject_001", 0.95)]
    res_qual = recognizer.evaluate_open_set_decision(top_matches_high, quality_score=0.40)
    assert res_qual.state == OpenSetState.UNCERTAIN


def test_matching_step_open_set_integration():
    import numpy as np

    from pipeline.steps.matching_step import MatchingStep

    matcher = MatchingStep(threshold=0.85)
    gal_feats = np.eye(4, dtype=np.float32)
    gal_labels = np.array(["sub_1", "sub_2", "sub_3", "sub_4"])
    metadata = {lbl: {"status": "ACTIVE", "enabled": True} for lbl in gal_labels}

    q1 = np.array([0.98, 0.01, 0.0, 0.0], dtype=np.float32)
    res1 = matcher.match_open_set(q1, gal_feats, gal_labels, metadata)
    assert res1.state == OpenSetState.KNOWN
    assert res1.identity == "sub_1"

    q_unk = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    res_unk = matcher.match_open_set(q_unk, gal_feats, gal_labels, metadata)
    assert res_unk.state == OpenSetState.UNKNOWN
    assert res_unk.identity == "UNKNOWN"
