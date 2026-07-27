"""
Dual-Modal Biometric Fusion (Gait + ReID).

Combines Gait score and OSNet ReID score using quality-adaptive dynamic weighting.
Handles single-modality fallback automatically if one modality is unavailable.
Reuses existing QualityEstimator, TrackReliabilityScorer, CrowdOcclusionAnalyzer,
CrowdDensityEstimator, and RecognitionDeferralEngine inputs.
"""

from typing import Any, Dict, Optional, Tuple
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
        enabled: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.normalizer = ScoreNormalizer(
            gait_min_max=gait_min_max,
            reid_min_max=reid_min_max,
        )
        self.quality_assessor = QualityAssessment()
        self.weight_allocator = DynamicFusionWeights(
            default_gait_weight=default_gait_weight,
            default_reid_weight=default_reid_weight,
        )

    def is_enabled(self) -> bool:
        return self.enabled

    @classmethod
    def from_config(cls, config: Dict[str, Any] | None = None) -> "DualModalFusion":
        cfg = config or {}
        g_min_max = tuple(cfg.get("gait_min_max", (0.0, 1.0)))
        r_min_max = tuple(cfg.get("reid_min_max", (-1.0, 1.0)))
        return cls(
            default_gait_weight=float(cfg.get("gait_weight", 0.70)),
            default_reid_weight=float(cfg.get("appearance_weight", cfg.get("reid_weight", 0.30))),
            gait_min_max=(float(g_min_max[0]), float(g_min_max[1])),
            reid_min_max=(float(r_min_max[0]), float(r_min_max[1])),
            enabled=bool(cfg.get("enabled", False)),
        )

    @staticmethod
    def compute_cosine_similarity(
        vec1: Optional[np.ndarray],
        vec2: Optional[np.ndarray],
    ) -> Optional[float]:
        """Compute cosine similarity between two feature embedding vectors."""
        if vec1 is None or vec2 is None:
            return None
        v1 = np.asarray(vec1, dtype=np.float32).ravel()
        v2 = np.asarray(vec2, dtype=np.float32).ravel()
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def fuse(
        self,
        gait_score: Optional[float] = None,
        reid_score: Optional[float] = None,
        crop: Optional[np.ndarray] = None,
        gei_frame_count: int = 0,
        gei: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        gait_embedding: Optional[np.ndarray] = None,
        gait_gallery_embedding: Optional[np.ndarray] = None,
        reid_embedding: Optional[np.ndarray] = None,
        reid_gallery_embedding: Optional[np.ndarray] = None,
        crowd_density: float = 0.0,
        occlusion_score: float = 0.0,
        track_reliability: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Perform dual-modal fusion of Gait and ReID scores or embeddings.

        Automatically adapts weights based on gait quality, crop quality,
        crowd density, occlusion ratio, and track reliability.
        Falls back to gait-only mode if appearance embedding or score is absent.
        """
        # Compute cosine similarity if raw scores are None but embeddings provided
        if gait_score is None and gait_embedding is not None and gait_gallery_embedding is not None:
            gait_score = self.compute_cosine_similarity(gait_embedding, gait_gallery_embedding)

        if reid_score is None and reid_embedding is not None and reid_gallery_embedding is not None:
            reid_score = self.compute_cosine_similarity(reid_embedding, reid_gallery_embedding)

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

        # Dynamic context adjustments:
        # High gait quality & high track reliability -> boost gait quality factor
        adjusted_gait_q = gait_quality * max(0.2, track_reliability)
        # Heavy crowd & high occlusion -> penalty on gait quality factor, boost appearance weight
        crowd_occlusion_factor = max(0.0, min(1.0, 0.5 * crowd_density + 0.5 * occlusion_score))
        if crowd_occlusion_factor > 0.3:
            adjusted_gait_q = adjusted_gait_q * (1.0 - 0.5 * crowd_occlusion_factor)
            reid_quality = min(1.0, reid_quality * (1.0 + 0.5 * crowd_occlusion_factor))

        if g_present and r_present:
            w_gait, w_reid = self.weight_allocator.compute_weights(
                gait_available=True,
                reid_available=True,
                gait_quality=adjusted_gait_q,
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

        gait_val = float(norm_gait) if norm_gait is not None else 0.0
        reid_val = float(norm_reid) if norm_reid is not None else 0.0
        final_val = float(max(0.0, min(1.0, final_score)))

        return {
            "final_score": final_val,
            "fusion_score": final_val,
            "gait_score": gait_val,
            "appearance_score": reid_val,
            "gait_score_norm": norm_gait,
            "reid_score_norm": norm_reid,
            "gait_weight": float(w_gait),
            "reid_weight": float(w_reid),
            "fusion_weight_gait": float(w_gait),
            "fusion_weight_appearance": float(w_reid),
            "gait_quality": float(gait_quality),
            "reid_quality": float(reid_quality),
            "appearance_quality": float(reid_quality),
            "active_modalities": active,
        }
