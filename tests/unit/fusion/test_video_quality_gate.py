import numpy as np

from preprocessing.video_quality_gate import DeterministicVideoQualityGate


def test_video_quality_gate_initialization() -> None:
    gate = DeterministicVideoQualityGate()
    assert gate.min_frames == 5
    assert gate.min_blur_var == 25.0
    assert gate.min_luminance == 30.0
    assert gate.max_luminance == 235.0


def test_video_quality_gate_assess_clean_clip() -> None:
    gate = DeterministicVideoQualityGate(min_frames=3)


    frames = [np.full((128, 64, 3), 120, dtype=np.uint8) for _ in range(5)]

    for f in frames:
        f[::4, ::4] = 255

    silhouettes = [np.zeros((128, 64), dtype=np.uint8) for _ in range(5)]
    for i, s in enumerate(silhouettes):

        s[10:120, (10 + i * 2):(50 + i * 2)] = 255

    res = gate.assess_video_clip(frames, silhouettes)
    assert res.passed is True
    assert len(res.issues) == 0
    assert res.usable_frames_count >= 3
    assert res.motion_dynamism_score > 0.0


def test_video_quality_gate_rejects_blurry_and_dark() -> None:
    gate = DeterministicVideoQualityGate(min_frames=3)


    dark_frames = [np.full((128, 64, 3), 5, dtype=np.uint8) for _ in range(5)]
    silhouettes = [np.zeros((128, 64), dtype=np.uint8) for _ in range(5)]
    for s in silhouettes:
        s[10:120, 20:40] = 255

    res = gate.assess_video_clip(dark_frames, silhouettes)
    assert res.passed is False
    assert any("underexposed" in issue.lower() or "blur" in issue.lower() for issue in res.issues)
    assert "Quality Gate Rejected" in res.actionable_guidance


def test_video_quality_gate_deterministic_enhancement() -> None:
    gate = DeterministicVideoQualityGate()
    dark_crop = np.full((100, 50, 3), 40, dtype=np.uint8)
    enhanced = gate.enhance_crop_deterministic(dark_crop)
    assert enhanced is not None
    assert enhanced.shape[0] >= 100
