"""
Quality Assessment Module for Gait and ReID Input Stream.

Assesses signal quality without additional ML models by evaluating crop dimensions,
sharpness/blur (Laplacian variance), sequence completeness, and detection confidence.
"""

import cv2
import numpy as np


class QualityAssessment:
    """
    Computes quality scores in [0.0, 1.0] for Gait and ReID modalities.
    Uses classical image statistics and tracking metadata only.
    """

    def __init__(
        self,
        min_crop_height: int = 60,
        min_crop_width: int = 30,
        ideal_crop_height: int = 256,
        ideal_crop_width: int = 128,
        target_gei_frames: int = 30,
        blur_threshold: float = 100.0,
    ) -> None:
        self.min_crop_height = min_crop_height
        self.min_crop_width = min_crop_width
        self.ideal_crop_height = ideal_crop_height
        self.ideal_crop_width = ideal_crop_width
        self.target_gei_frames = target_gei_frames
        self.blur_threshold = blur_threshold

    def evaluate_reid_quality(
        self,
        crop: np.ndarray | None,
        confidence: float = 1.0,
    ) -> float:
        """
        Compute ReID crop quality score [0.0, 1.0].

        Evaluates crop dimensions, aspect ratio, sharpness (blur),
        and object detection confidence.
        """
        if crop is None or crop.size == 0 or len(crop.shape) != 3:
            return 0.0

        h, w, c = crop.shape
        if h < self.min_crop_height or w < self.min_crop_width:
            return 0.0

        ideal_area = self.ideal_crop_height * self.ideal_crop_width
        actual_area = h * w
        res_factor = min(1.0, actual_area / ideal_area)

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_factor = min(1.0, lap_var / max(1.0, self.blur_threshold))
        except Exception:
            blur_factor = 0.5

        conf_factor = max(0.0, min(1.0, float(confidence)))

        quality = 0.4 * res_factor + 0.4 * blur_factor + 0.2 * conf_factor
        return float(max(0.0, min(1.0, quality)))

    def evaluate_gait_quality(
        self,
        gei_frame_count: int = 0,
        gei: np.ndarray | None = None,
        confidence: float = 1.0,
    ) -> float:
        """
        Compute Gait quality score [0.0, 1.0].

        Evaluates sequence completeness (frame buffer count), silhouette
        density, and detection confidence.
        """
        if gei_frame_count <= 0:
            return 0.0

        seq_factor = min(1.0, gei_frame_count / max(1, self.target_gei_frames))

        density_factor = 0.5
        if gei is not None and gei.size > 0:
            non_zero_ratio = float(np.count_nonzero(gei)) / float(gei.size)
            if 0.05 <= non_zero_ratio <= 0.5:
                density_factor = 1.0
            elif non_zero_ratio > 0.0:
                density_factor = 0.5
            else:
                density_factor = 0.0

        conf_factor = max(0.0, min(1.0, float(confidence)))

        quality = 0.5 * seq_factor + 0.3 * density_factor + 0.2 * conf_factor
        return float(max(0.0, min(1.0, quality)))
