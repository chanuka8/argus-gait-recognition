"""
Unit tests for CrowdRobustnessManager.
"""

from intelligence.crowd_density_estimator import CrowdDensityLevel
from intelligence.crowd_robustness_manager import CrowdRobustnessManager


def test_disabled_by_default():
    manager = CrowdRobustnessManager()
    assert not manager.is_enabled()

    # When disabled, occlusion identification returns empty
    tracked_objects = [
        {"track_id": 1, "bbox": [100, 100, 200, 300]},
        {"track_id": 2, "bbox": [110, 100, 210, 300]},
    ]
    occluded_ids, overlap_map = manager.identify_occluded_tracks(tracked_objects)
    assert occluded_ids == set()
    assert overlap_map == {}

    # Quality score remains unchanged
    assert manager.adapt_quality_score(0.85, CrowdDensityLevel.SEVERE, is_occluded=True) == 0.85

    # Margin remains unchanged
    assert manager.adapt_open_set_margin(0.05, CrowdDensityLevel.SEVERE) == 0.05


def test_enabled_occlusion_and_gating():
    config = {
        "enabled": True,
        "occlusion_overlap_threshold": 0.30,
        "adaptive_gating": {
            "high_density_quality_penalty": 0.10,
            "severe_density_margin_boost": 0.05,
        },
    }
    manager = CrowdRobustnessManager(config)
    assert manager.is_enabled()

    tracked_objects = [
        {"track_id": 10, "bbox": [100, 100, 200, 300]},
        {"track_id": 11, "bbox": [110, 100, 210, 300]},  # Heavy overlap with track 10
        {"track_id": 12, "bbox": [500, 500, 600, 700]},  # Isolated track
    ]

    occluded_ids, overlap_map = manager.identify_occluded_tracks(tracked_objects)
    assert 10 in occluded_ids
    assert 11 in occluded_ids
    assert 12 not in occluded_ids

    # Check adapted quality score
    base_q = 0.80
    adapted_severe = manager.adapt_quality_score(base_q, CrowdDensityLevel.SEVERE, is_occluded=True)
    assert adapted_severe < base_q

    # Check adapted margin
    base_margin = 0.05
    adapted_margin = manager.adapt_open_set_margin(base_margin, CrowdDensityLevel.SEVERE)
    assert adapted_margin == 0.10


def test_gei_contamination_prevention_and_clean_ratio():
    config = {"enabled": True, "occlusion_overlap_threshold": 0.35}
    manager = CrowdRobustnessManager(config)

    tracked_clean = [{"track_id": 5, "bbox": [10, 10, 50, 100]}]
    occluded_ids_clean, _ = manager.identify_occluded_tracks(tracked_clean)
    assert 5 not in occluded_ids_clean

    tracked_occluded = [
        {"track_id": 5, "bbox": [10, 10, 50, 100]},
        {"track_id": 6, "bbox": [12, 10, 52, 100]},
    ]
    occluded_ids_occ, _ = manager.identify_occluded_tracks(tracked_occluded)
    assert 5 in occluded_ids_occ
    assert 6 in occluded_ids_occ
