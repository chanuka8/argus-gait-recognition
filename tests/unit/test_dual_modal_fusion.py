import numpy as np
import pytest

from intelligence.dual_modal_fusion import DualModalFusion


def test_dual_modal_fusion_defaults_and_config():
    fusion = DualModalFusion(default_gait_weight=0.70, default_reid_weight=0.30, enabled=True)
    assert fusion.is_enabled() is True

    cfg = {
        "enabled": True,
        "gait_weight": 0.75,
        "appearance_weight": 0.25,
    }
    fusion_from_cfg = DualModalFusion.from_config(cfg)
    assert fusion_from_cfg.is_enabled() is True


def test_dual_modal_fusion_scoring_math():
    fusion = DualModalFusion(default_gait_weight=0.70, default_reid_weight=0.30, enabled=True)

    # Both scores present
    res = fusion.fuse(gait_score=0.80, reid_score=0.60)
    assert "final_score" in res
    assert "fusion_weight_gait" in res
    assert "fusion_weight_appearance" in res
    assert np.isclose(res["fusion_weight_gait"] + res["fusion_weight_appearance"], 1.0)
    assert 0.0 <= res["final_score"] <= 1.0


def test_dual_modal_fusion_fallback_gait_only():
    fusion = DualModalFusion(default_gait_weight=0.70, default_reid_weight=0.30, enabled=True)

    # Appearance score missing (None)
    res = fusion.fuse(gait_score=0.88, reid_score=None)
    assert res["final_score"] == pytest.approx(0.88)
    assert res["fusion_weight_gait"] == 1.0
    assert res["fusion_weight_appearance"] == 0.0


def test_dual_modal_fusion_adaptive_quality():
    fusion = DualModalFusion(default_gait_weight=0.70, default_reid_weight=0.30, enabled=True)

    # Heavy crowd & high occlusion should increase appearance weight relative to clean
    res_clean = fusion.fuse(
        gait_score=0.80,
        reid_score=0.80,
        crowd_density=0.0,
        occlusion_score=0.0,
        track_reliability=1.0,
    )
    res_crowd = fusion.fuse(
        gait_score=0.80,
        reid_score=0.80,
        crowd_density=0.9,
        occlusion_score=0.8,
        track_reliability=0.4,
    )

    assert res_crowd["fusion_weight_appearance"] > res_clean["fusion_weight_appearance"]
