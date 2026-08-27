"""Unit tests for QualityEstimator and TemporalGaitVerifier steps."""

import numpy as np

from pipeline.steps.quality_estimator import QualityEstimator
from pipeline.steps.temporal_gait_verifier import TemporalGaitVerifier


class TestQualityEstimator:
    def test_empty_or_none_gei(self):
        estimator = QualityEstimator(quality_threshold=0.6)
        res_none = estimator.evaluate(None)
        assert not res_none["accepted"]
        assert res_none["overall_quality"] == 0.0

        res_empty = estimator.evaluate(np.array([], dtype=np.uint8))
        assert not res_empty["accepted"]
        assert res_empty["overall_quality"] == 0.0

    def test_valid_gei_evaluation(self):
        estimator = QualityEstimator(quality_threshold=0.5)

        gei = np.zeros((128, 64), dtype=np.uint8)
        gei[10:30, 24:40] = 200
        gei[35:75, 20:44] = 230
        gei[80:115, 22:42] = 210

        res = estimator.evaluate(gei, box_aspect_ratio=2.0)
        assert "overall_quality" in res
        assert "metrics" in res
        assert 0.0 <= res["overall_quality"] <= 1.0
        assert res["metrics"]["blur"] >= 0.0
        assert res["metrics"]["completeness"] > 0.0
        assert res["metrics"]["stability"] == 1.0

    def test_quality_threshold_filtering(self):
        estimator = QualityEstimator(quality_threshold=0.99)
        gei = np.full((128, 64), 30, dtype=np.uint8)

        res = estimator.evaluate(gei)
        assert not res["accepted"]
        assert "below threshold" in res["reason"]


class TestTemporalGaitVerifier:
    def test_buffer_push_and_window_size(self):
        verifier = TemporalGaitVerifier(window_size=3)
        assert len(verifier.get_buffer(track_id=1)) == 0

        emb1 = np.ones(256, dtype=np.float32) * 0.1
        emb2 = np.ones(256, dtype=np.float32) * 0.2
        emb3 = np.ones(256, dtype=np.float32) * 0.3
        emb4 = np.ones(256, dtype=np.float32) * 0.4

        verifier.add_embedding(1, emb1)
        verifier.add_embedding(1, emb2)
        verifier.add_embedding(1, emb3)
        assert len(verifier.get_buffer(1)) == 3

        verifier.add_embedding(1, emb4)
        buf = verifier.get_buffer(1)
        assert len(buf) == 3
        assert np.allclose(buf[0], emb2)

    def test_majority_voting(self):
        verifier = TemporalGaitVerifier(window_size=3)

        def mock_matcher(emb, gf, gl, meta):
            val = emb[0]
            if val < 0.2:
                return [("Subject_A", 0.90)]
            elif val < 0.4:
                return [("Subject_A", 0.92)]
            else:
                return [("Subject_B", 0.88)]

        verifier.add_embedding(1, np.ones(256) * 0.1)
        verifier.add_embedding(1, np.ones(256) * 0.2)
        verifier.add_embedding(1, np.ones(256) * 0.5)

        identity, _score, decision = verifier.verify_identity(
            track_id=1,
            matcher_func=mock_matcher,
            gallery_features=None,
            gallery_labels=None,
        )

        assert identity == "Subject_A"
        assert decision == "MAJORITY_VOTE"

    def test_fallback_and_track_cleanup(self):
        verifier = TemporalGaitVerifier(window_size=3)

        def mock_matcher_split(emb, gf, gl, meta):
            val = emb[0]
            if val < 0.2:
                return [("Subject_A", 0.90)]
            elif val < 0.4:
                return [("Subject_B", 0.90)]
            else:
                return [("Subject_C", 0.90)]

        verifier.add_embedding(1, np.ones(256) * 0.1)
        verifier.add_embedding(1, np.ones(256) * 0.3)
        verifier.add_embedding(1, np.ones(256) * 0.5)

        identity, _score, decision = verifier.verify_identity(
            track_id=1,
            matcher_func=mock_matcher_split,
            gallery_features=None,
            gallery_labels=None,
        )

        assert identity == "UNKNOWN"
        assert decision == "UNCERTAIN"

        verifier.clear_track(1)
        assert len(verifier.get_buffer(1)) == 0
