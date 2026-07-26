"""
Dual-Modal Biometric Fusion (Gait + ReID).

Combines Gait score and OSNet ReID score using quality-adaptive dynamic weighting.
Handles single-modality fallback automatically if one modality is unavailable.
"""

from typing import Any, Dict, Tuple

import numpy as np

from intelligence.fusion_weights import DynamicFusionWeights
from intelligence.quality_assessment import QualityAssessment
from intelligence.score_normalizer import ScoreNormalizer


class DualModalFusion:
    """
    Dual-Modal Fusion Engine (Gait + ReID).

    Performs score normalization, input quality assessment, dynamic weight
    allocation, and score fusion.
    """

    def __init__(
        self,
        default_gait_weight: float = 0.7,
        default_reid_weight: float = 0.3,
        gait_min_max: Tuple[float, float] = (0.0, 1.0),
        reid_min_max: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.normalizer = ScoreNormalizer(
            gait_min_max=gait_min_max,
            reid_min_max=reid_min_max,
        )
        self.quality_assessor = QualityAssessment()
        self.weight_allocator = DynamicFusionWeights(
            default_gait_weight=default_gait_weight,
            default_reid_weight=default_reid_weight,
        )

    def fuse(
        self,
        gait_score: float | None,
        reid_score: float | None,
        crop: np.ndarray | None = None,
        gei_frame_count: int = 0,
        gei: np.ndarray | None = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Perform dual-modal fusion of Gait and ReID scores.

        Args:
            gait_score: Raw Gait matching cosine score or None.
            reid_score: Raw ReID matching cosine score or None.
            crop: BGR person crop for ReID quality assessment.
            gei_frame_count: Number of frames in GEI buffer for Gait quality.
            gei: GEI numpy array for Gait quality assessment.
            confidence: Object detection confidence.

        Returns:
            Dict containing:
                - final_score: Fused similarity score [0.0, 1.0]
                - gait_score_norm: Normalized Gait score
                - reid_score_norm: Normalized ReID score
                - gait_weight: Applied Gait weight
                - reid_weight: Applied ReID weight
                - gait_quality: Evaluated Gait quality score
                - reid_quality: Evaluated ReID quality score
                - active_modalities: List of active modality names
        """
        # Normalize scores
        norm_gait = self.normalizer.normalize_gait(gait_score)
        norm_reid = self.normalizer.normalize_reid(reid_score)

        # Modality score presence
        g_present = norm_gait is not None
        r_present = norm_reid is not None

        # Assess modality quality
        gait_quality = (
            self.quality_assessor.evaluate_gait_quality(
                gei_frame_count=gei_frame_count if gei_frame_count > 0 else 30,
                gei=gei,
                confidence=confidence,
            )
            if g_present
            else 0.0
        )

        reid_quality = (
            self.quality_assessor.evaluate_reid_quality(
                crop=crop if crop is not None else np.ones((256, 128, 3), dtype=np.uint8),
                confidence=confidence,
            )
            if r_present
            else 0.0
        )

        if g_present and r_present:
            w_gait, w_reid = self.weight_allocator.compute_weights(
                gait_available=True,
                reid_available=True,
                gait_quality=gait_quality,
                reid_quality=reid_quality,
            )
            final_score = w_gait * norm_gait + w_reid * norm_reid
            active = ["gait", "reid"]
        elif g_present:
            final_score = norm_gait
            w_gait, w_reid = 1.0, 0.0
            active = ["gait"]
        elif r_present:
            final_score = norm_reid
            w_gait, w_reid = 0.0, 1.0
            active = ["reid"]
        else:
            final_score = 0.0
            w_gait, w_reid = self.weight_allocator.base_gait_weight, self.weight_allocator.base_reid_weight
            active = []

        return {
            "final_score": float(max(0.0, min(1.0, final_score))),
            "gait_score_norm": norm_gait,
            "reid_score_norm": norm_reid,
            "gait_weight": float(w_gait),
            "reid_weight": float(w_reid),
            "gait_quality": float(gait_quality),
            "reid_quality": float(reid_quality),
            "active_modalities": active,
        }

