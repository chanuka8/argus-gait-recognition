from unittest.mock import MagicMock, patch

from intelligence.crowd_density_estimator import CrowdDensityLevel
from intelligence.crowd_robustness_manager import CrowdRobustnessManager
from pipeline.live_recognition import LiveRecognitionPipeline
from pipeline.video_recognition import VideoRecognitionPipeline, _load_crowd_robustness_config


@patch("pipeline.live_recognition.StreamEngine")
@patch("pipeline.live_recognition.VectorStore")
@patch("pipeline.live_recognition.LiveRecognitionPipeline._load_model", return_value=MagicMock())
@patch("pipeline.video_recognition.VectorStore")
@patch("pipeline.video_recognition.VideoRecognitionPipeline._load_model", return_value=MagicMock())
def test_pipeline_crowd_robustness_initialization(
    mock_vid_model, mock_vid_store, mock_live_model, mock_live_store, mock_stream
):
    mock_vid_store.return_value.load.return_value = (MagicMock(), MagicMock(), {})
    mock_live_store.return_value.load.return_value = (MagicMock(), MagicMock(), {})

    cfg = _load_crowd_robustness_config()
    assert "enabled" in cfg
    assert cfg["enabled"] is False

    pipe = VideoRecognitionPipeline()
    assert hasattr(pipe, "crowd_robustness_manager")
    assert pipe.crowd_robustness_manager.is_enabled() is False

    live_pipe = LiveRecognitionPipeline()
    assert hasattr(live_pipe, "crowd_robustness_manager")
    assert live_pipe.crowd_robustness_manager.is_enabled() is False


def test_crowd_robustness_end_to_end_simulation():
    config = {
        "enabled": True,
        "strong_overlap_iou": 0.25,
        "occlusion_overlap_threshold": 0.30,
        "density_thresholds": {
            "moderate_count": 3,
            "high_count": 6,
            "severe_count": 10,
        },
    }
    mgr = CrowdRobustnessManager(config)
    assert mgr.is_enabled() is True

    detections = []
    for i in range(8):
        detections.append(
            {
                "track_id": i + 1,
                "bbox": [100 + (i % 3) * 15, 100 + (i % 3) * 15, 200 + (i % 3) * 15, 300 + (i % 3) * 15],
                "confidence": 0.85,
            }
        )

    density_res = mgr.process_frame_density(detections, (1080, 1920))
    assert density_res.person_count == 8
    assert density_res.level in (CrowdDensityLevel.HIGH, CrowdDensityLevel.SEVERE, CrowdDensityLevel.MODERATE)

    occluded_ids, _overlap_map = mgr.identify_occluded_tracks(detections)
    assert len(occluded_ids) > 0


def test_inference_skipping_when_evidence_insufficient():
    from intelligence.crowd_intelligence_system import CrowdIntelligenceSystem

    system = CrowdIntelligenceSystem(
        {"enabled": True, "recognition_deferral": {"enabled": True, "minimum_confirmations": 3}}
    )

    res = system.evaluate_track_recognition(
        camera_id="cam_00",
        track_id=1,
        identity_candidate="Subject_Test",
        similarity=0.90,
        quality=0.80,
        open_set_state="KNOWN",
        temporal_decision="MAJORITY_VOTE",
        reliability=0.85,
        occlusion_score=0.10,
        timestamp=1.0,
    )
    assert res.recognition_state == "DEFERRED_INSUFFICIENT_EVIDENCE"
    assert res.should_alert is False
