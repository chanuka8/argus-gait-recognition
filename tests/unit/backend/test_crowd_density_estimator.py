import pytest

from intelligence.crowd_density_estimator import (
    CrowdDensityEstimator,
    CrowdDensityLevel,
    CrowdDensityResult,
    compute_iou,
)


def test_compute_iou():
    box1 = [0, 0, 100, 100]
    box2 = [50, 0, 150, 100]
    iou = compute_iou(box1, box2)
    assert pytest.approx(iou, 0.01) == 0.3333

    box3 = [200, 200, 300, 300]
    assert compute_iou(box1, box3) == 0.0


def test_empty_detections():
    estimator = CrowdDensityEstimator()
    result = estimator.estimate_density([], (1080, 1920))

    assert isinstance(result, CrowdDensityResult)
    assert result.level == CrowdDensityLevel.LOW
    assert result.person_count == 0
    assert result.strongly_overlapping_count == 0
    assert result.avg_pairwise_overlap == 0.0


def test_low_density():
    estimator = CrowdDensityEstimator()
    detections = [
        {"bbox": [10, 10, 50, 100]},
        {"bbox": [200, 200, 250, 300]},
    ]
    result = estimator.estimate_density(detections, (1080, 1920))
    assert result.level == CrowdDensityLevel.LOW
    assert result.person_count == 2
    assert result.strongly_overlapping_count == 0


def test_moderate_density():
    estimator = CrowdDensityEstimator(moderate_count=5)
    detections = [{"bbox": [i * 50, 10, i * 50 + 40, 100]} for i in range(5)]
    result = estimator.estimate_density(detections, (1080, 1920))
    assert result.level in (CrowdDensityLevel.MODERATE, CrowdDensityLevel.HIGH, CrowdDensityLevel.SEVERE)
    assert result.person_count == 5


def test_severe_density_overlapping():
    estimator = CrowdDensityEstimator(severe_overlap_count=3, strong_overlap_threshold=0.20)
    detections = [
        {"bbox": [100, 100, 200, 300]},
        {"bbox": [120, 100, 220, 300]},
        {"bbox": [140, 100, 240, 300]},
        {"bbox": [160, 100, 260, 300]},
    ]
    result = estimator.estimate_density(detections, (1080, 1920))
    assert result.level == CrowdDensityLevel.SEVERE
    assert result.strongly_overlapping_count >= 3
