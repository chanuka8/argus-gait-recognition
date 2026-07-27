"""
Unit tests for Stage 1: Crowd-Aware Occlusion Analyzer.
"""

from intelligence.crowd_occlusion_analyzer import (
    CrowdDensityLevel,
    CrowdOcclusionAnalyzer,
)


def test_sparse_detections():
    analyzer = CrowdOcclusionAnalyzer({"enabled": True, "high_threshold": 0.60})
    detections = [
        {"track_id": 1, "bbox": [10, 10, 50, 100]},
        {"track_id": 2, "bbox": [500, 500, 550, 600]},
    ]
    res = analyzer.analyze_frame(detections, (1080, 1920), camera_id="cam_01")

    assert res.crowd_density_level == CrowdDensityLevel.LOW
    assert res.person_count == 2
    assert res.strongly_overlapping_count == 0
    assert res.track_occlusions[("cam_01", 1)] == 0.0
    assert res.track_occlusions[("cam_01", 2)] == 0.0
    assert res.silhouette_acceptance[("cam_01", 1)] is True
    assert res.silhouette_acceptance[("cam_01", 2)] is True


def test_overlapping_detections_and_score_bounds():
    analyzer = CrowdOcclusionAnalyzer({"enabled": True, "high_threshold": 0.60})
    detections = [
        {"track_id": 10, "bbox": [100, 100, 200, 300]},
        {"track_id": 11, "bbox": [110, 100, 210, 300]},  # ~0.70 IoU overlap
    ]
    res = analyzer.analyze_frame(detections, (1080, 1920), camera_id="cam_01")

    assert 0.0 <= res.crowd_density_score <= 1.0
    assert res.track_occlusions[("cam_01", 10)] > 0.50
    assert res.track_occlusions[("cam_01", 11)] > 0.50


def test_high_occlusion_silhouette_rejection():
    analyzer = CrowdOcclusionAnalyzer({"enabled": True, "high_threshold": 0.50})
    detections = [
        {"track_id": 100, "bbox": [100, 100, 200, 300]},
        {"track_id": 101, "bbox": [105, 100, 205, 300]},  # Severe overlap
    ]
    res = analyzer.analyze_frame(detections, (1080, 1920), camera_id="cam_01")

    assert res.silhouette_acceptance[("cam_01", 100)] is False
    assert res.silhouette_acceptance[("cam_01", 101)] is False


def test_one_bad_frame_does_not_clear_buffer():
    analyzer = CrowdOcclusionAnalyzer({"enabled": True, "high_threshold": 0.60, "smoothing_window": 3})

    # Frame 1 & 2 clean
    det_clean = [{"track_id": 5, "bbox": [10, 10, 50, 100]}]
    analyzer.analyze_frame(det_clean, (1080, 1920), camera_id="cam_01", timestamp=1.0)
    analyzer.analyze_frame(det_clean, (1080, 1920), camera_id="cam_01", timestamp=2.0)

    # Frame 3 heavily occluded
    det_occ = [
        {"track_id": 5, "bbox": [10, 10, 50, 100]},
        {"track_id": 6, "bbox": [12, 10, 52, 100]},
    ]
    analyzer.analyze_frame(det_occ, (1080, 1920), camera_id="cam_01", timestamp=3.0)

    # State still exists and clean history is preserved
    state = analyzer.track_states[("cam_01", 5)]
    assert len(state.clean_history) == 3
    assert state.clean_history[0] is True  # preserved history


def test_state_cleanup_and_camera_isolated_keys():
    analyzer = CrowdOcclusionAnalyzer({"enabled": True})
    analyzer.analyze_frame([{"track_id": 1, "bbox": [10, 10, 50, 100]}], (1080, 1920), camera_id="cam_A", timestamp=10.0)
    analyzer.analyze_frame([{"track_id": 1, "bbox": [10, 10, 50, 100]}], (1080, 1920), camera_id="cam_B", timestamp=10.0)

    assert ("cam_A", 1) in analyzer.track_states
    assert ("cam_B", 1) in analyzer.track_states
    assert len(analyzer.track_states) == 2

    # Cleanup cam_A at timestamp 25.0 (> 10s idle)
    purged = analyzer.cleanup_inactive(max_idle_seconds=10.0, current_time=25.0)
    assert ("cam_A", 1) in purged
    assert ("cam_B", 1) in purged
    assert len(analyzer.track_states) == 0
