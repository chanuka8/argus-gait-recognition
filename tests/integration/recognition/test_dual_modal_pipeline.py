import pytest

from intelligence.dual_modal_fusion import DualModalFusion
from utils.detection_reporter import DetectionReporter


def test_pipeline_zero_regression_when_disabled():
    fusion_disabled = DualModalFusion(enabled=False)
    assert fusion_disabled.is_enabled() is False

    res = fusion_disabled.fuse(gait_score=0.85, reid_score=None)
    assert res["final_score"] == pytest.approx(0.85)
    assert res["fusion_weight_gait"] == 1.0


def test_pipeline_reporter_export():
    reporter = DetectionReporter(source_mode="test")

    reported = reporter.report(
        camera_id="cam_01",
        location="Front Gate",
        track_id=42,
        identity="Person_A",
        status="CONFIRMED",
        score=0.91,
        bbox=[10, 20, 100, 200],
        gait_score=0.88,
        appearance_score=0.95,
        fusion_score=0.91,
        fusion_weight_gait=0.6,
        fusion_weight_appearance=0.4,
        appearance_quality=0.85,
    )
    assert reported is True
