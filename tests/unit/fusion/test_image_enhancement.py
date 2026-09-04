import cv2
import numpy as np

from preprocessing.image_enhancement import DeterministicImageEnhancer


def test_enhancer_initialization() -> None:
    enhancer = DeterministicImageEnhancer(
        target_height=256,
        target_width=128,
        min_height=40,
        min_width=20,
    )
    assert enhancer.target_height == 256
    assert enhancer.target_width == 128
    assert enhancer.min_height == 40
    assert enhancer.min_width == 20


def test_quality_gate_normal_photo() -> None:
    enhancer = DeterministicImageEnhancer()

    img = np.random.randint(50, 200, (200, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 180), (255, 255, 255), -1)
    cv2.circle(img, (50, 50), 20, (0, 0, 0), -1)

    result = enhancer.assess_quality(img)
    assert result.is_acceptable is True
    assert result.quality_score > 0.0
    assert result.rejection_reason is None


def test_quality_gate_underexposed_rejection() -> None:
    enhancer = DeterministicImageEnhancer(min_brightness=25.0)

    dark_img = np.full((256, 128, 3), 10, dtype=np.uint8)

    result = enhancer.assess_quality(dark_img)
    assert result.is_acceptable is False
    assert "underexposed" in result.rejection_reason.lower()
    assert result.recommendation is not None


def test_quality_gate_overexposed_rejection() -> None:
    enhancer = DeterministicImageEnhancer(max_brightness=240.0)

    bright_img = np.full((256, 128, 3), 250, dtype=np.uint8)

    result = enhancer.assess_quality(bright_img)
    assert result.is_acceptable is False
    assert "overexposed" in result.rejection_reason.lower()


def test_quality_gate_blur_rejection() -> None:
    enhancer = DeterministicImageEnhancer(min_blur_score=20.0)

    smooth_img = np.full((256, 128, 3), 128, dtype=np.uint8)

    result = enhancer.assess_quality(smooth_img)
    assert result.is_acceptable is False
    assert "blurry" in result.rejection_reason.lower()


def test_quality_gate_low_resolution_rejection() -> None:
    enhancer = DeterministicImageEnhancer(min_height=40, min_width=20)
    tiny_img = np.random.randint(50, 200, (15, 10, 3), dtype=np.uint8)

    result = enhancer.assess_quality(tiny_img)
    assert result.is_acceptable is False
    assert "resolution too low" in result.rejection_reason.lower()


def test_deterministic_enhancement_shape_and_range() -> None:
    enhancer = DeterministicImageEnhancer(target_height=256, target_width=128)
    img = np.random.randint(50, 200, (100, 50, 3), dtype=np.uint8)

    enhanced = enhancer.enhance(img)
    assert enhanced.shape[0] >= 256
    assert enhanced.shape[1] >= 128
    assert enhanced.dtype == np.uint8
    assert enhanced.min() >= 0
    assert enhanced.max() <= 255


def test_process_and_gate() -> None:
    enhancer = DeterministicImageEnhancer()

    img = np.random.randint(50, 200, (200, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 180), (255, 255, 255), -1)

    accepted, enh_img, _ = enhancer.process_and_gate(img)
    assert accepted is True
    assert enh_img is not None
    assert enh_img.shape[0] >= 200


    dark_img = np.full((256, 128, 3), 5, dtype=np.uint8)
    accepted_bad, enh_bad, assess_bad = enhancer.process_and_gate(dark_img)
    assert accepted_bad is False
    assert enh_bad is None
    assert assess_bad.rejection_reason is not None
