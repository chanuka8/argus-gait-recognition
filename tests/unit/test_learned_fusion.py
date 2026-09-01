import numpy as np

from intelligence.learned_fusion import LearnedLogisticFusion
from intelligence.score_calibrator import PlattScoreCalibrator


def test_platt_calibrator_fitting() -> None:
    calib = PlattScoreCalibrator()
    scores = np.array([0.9, 0.85, 0.88, 0.2, 0.15, 0.3, 0.25], dtype=np.float32)
    labels = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int32)

    calib.fit(scores, labels)
    assert calib.is_fitted is True

    prob_high = calib.calibrate(0.95)
    prob_low = calib.calibrate(0.10)

    assert 0.0 <= prob_high <= 1.0
    assert 0.0 <= prob_low <= 1.0
    assert prob_high > prob_low


def test_learned_logistic_fusion_prediction() -> None:
    fusion = LearnedLogisticFusion()
    gait_scores = np.array([0.9, 0.85, 0.88, 0.2, 0.15, 0.3, 0.25], dtype=np.float32)
    app_scores = np.array([0.85, 0.80, 0.92, 0.3, 0.25, 0.15, 0.35], dtype=np.float32)
    labels = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int32)

    fusion.fit(gait_scores, app_scores, labels)
    assert fusion.is_fitted is True

    p_match = fusion.predict_probability(0.90, 0.85)
    p_impostor = fusion.predict_probability(0.20, 0.25)

    assert 0.0 <= p_match <= 1.0
    assert 0.0 <= p_impostor <= 1.0
    assert p_match > p_impostor


def test_learned_logistic_serialization() -> None:
    fusion = LearnedLogisticFusion(w_gait=3.0, w_app=4.5, w_inter=1.2, bias=-2.8)
    d = fusion.to_dict()
    assert d["w_gait"] == 3.0
    assert d["w_app"] == 4.5
    assert d["bias"] == -2.8

    restored = LearnedLogisticFusion.from_dict(d)
    assert restored.w_gait == 3.0
    assert restored.w_app == 4.5
