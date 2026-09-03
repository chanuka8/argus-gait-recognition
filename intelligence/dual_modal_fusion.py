from typing import Any

import numpy as np

from intelligence.fusion_weights import DynamicFusionWeights
from intelligence.quality_assessment import QualityAssessment
from intelligence.score_normalizer import ScoreNormalizer


class DualModalFusion:
    def __init__(
        self,
        default_gait_weight: float = 0.7,
        default_reid_weight: float = 0.3,
        gait_min_max: tuple[float, float] = (0.0, 1.0),
        reid_min_max: tuple[float, float] = (0.0, 1.0),
        adaptive_weighting: bool = True,
        enabled: bool = False,
        high_risk_confusion_groups: list[list[str]] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.adaptive_weighting = bool(adaptive_weighting)
        self.normalizer = ScoreNormalizer(
            gait_min_max=gait_min_max,
            reid_min_max=reid_min_max,
        )
        self.quality_assessor = QualityAssessment()
        self.weight_allocator = DynamicFusionWeights(
            default_gait_weight=default_gait_weight,
            default_reid_weight=default_reid_weight,
        )
        self.high_risk_confusion_groups = [
            set(g) for g in (high_risk_confusion_groups or [["Devhan", "Isuru", "person01"]])
        ]

    def is_enabled(self) -> bool:
        return self.enabled

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "DualModalFusion":
        cfg = config or {}
        g_min_max = tuple(cfg.get("gait_min_max", (0.0, 1.0)))
        r_min_max = tuple(cfg.get("reid_min_max", (0.0, 1.0)))
        confusion_groups = cfg.get("high_risk_confusion_groups", [["Devhan", "Isuru", "person01"]])
        return cls(
            default_gait_weight=float(cfg.get("gait_weight", 0.70)),
            default_reid_weight=float(cfg.get("appearance_weight", cfg.get("reid_weight", 0.30))),
            gait_min_max=(float(g_min_max[0]), float(g_min_max[1])),
            reid_min_max=(float(r_min_max[0]), float(r_min_max[1])),
            adaptive_weighting=bool(cfg.get("adaptive_weighting", True)),
            enabled=bool(cfg.get("enabled", False)),
            high_risk_confusion_groups=confusion_groups,
        )

    @staticmethod
    def compute_cosine_similarity(
        vec1: np.ndarray | None,
        vec2: np.ndarray | None,
    ) -> float | None:
        if vec1 is None or vec2 is None:
            return None
        v1 = np.asarray(vec1, dtype=np.float32).ravel()
        v2 = np.asarray(vec2, dtype=np.float32).ravel()
        d1 = float(np.dot(v1, v1))
        d2 = float(np.dot(v2, v2))
        if d1 == 0.0 or d2 == 0.0:
            return 0.0
        dot12 = float(np.dot(v1, v2))
        if abs(d1 - 1.0) < 1e-4 and abs(d2 - 1.0) < 1e-4:
            return dot12
        return float(dot12 / np.sqrt(d1 * d2))

    def fuse(
        self,
        gait_score: float | None = None,
        reid_score: float | None = None,
        crop: np.ndarray | None = None,
        gei_frame_count: int = 0,
        gei: np.ndarray | None = None,
        confidence: float = 1.0,
        gait_embedding: np.ndarray | None = None,
        gait_gallery_embedding: np.ndarray | None = None,
        reid_embedding: np.ndarray | None = None,
        reid_gallery_embedding: np.ndarray | None = None,
        crowd_density: float = 0.0,
        occlusion_score: float = 0.0,
        track_reliability: float = 1.0,
    ) -> dict[str, Any]:
        if gait_score is None and gait_embedding is not None and gait_gallery_embedding is not None:
            gait_score = self.compute_cosine_similarity(gait_embedding, gait_gallery_embedding)

        if reid_score is None and reid_embedding is not None and reid_gallery_embedding is not None:
            reid_score = self.compute_cosine_similarity(reid_embedding, reid_gallery_embedding)

        norm_gait = self.normalizer.normalize_gait(gait_score)
        norm_reid = self.normalizer.normalize_reid(reid_score)

        g_present = norm_gait is not None
        r_present = norm_reid is not None

        has_quality_context = (
            crop is not None
            or gei is not None
            or gei_frame_count > 0
            or crowd_density > 0.0
            or occlusion_score > 0.0
            or track_reliability < 1.0
        )

        gait_quality = (
            self.quality_assessor.evaluate_gait_quality(
                gei_frame_count=gei_frame_count if gei_frame_count > 0 else 30,
                gei=gei,
                confidence=confidence,
            )
            if (g_present and has_quality_context)
            else (1.0 if g_present else 0.0)
        )

        reid_quality = (
            self.quality_assessor.evaluate_reid_quality(
                crop=crop,
                confidence=confidence,
            )
            if (r_present and has_quality_context and crop is not None)
            else (1.0 if r_present else 0.0)
        )

        adjusted_gait_q = gait_quality * max(0.2, track_reliability)
        crowd_occlusion_factor = max(0.0, min(1.0, 0.5 * crowd_density + 0.5 * occlusion_score))
        if crowd_occlusion_factor > 0.3:
            adjusted_gait_q = adjusted_gait_q * (1.0 - 0.5 * crowd_occlusion_factor)
            reid_quality = min(1.0, reid_quality * (1.0 + 0.5 * crowd_occlusion_factor))

        if g_present and r_present:
            if self.adaptive_weighting and has_quality_context:
                w_gait, w_reid = self.weight_allocator.compute_weights(
                    gait_available=True,
                    reid_available=True,
                    gait_quality=adjusted_gait_q,
                    reid_quality=reid_quality,
                )
            else:
                w_gait = self.weight_allocator.base_gait_weight
                w_reid = self.weight_allocator.base_reid_weight
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

    @staticmethod
    def _is_valid_identity(identity: Any) -> bool:
        if identity is None:
            return False
        s = str(identity).strip().upper()
        return s not in ("", "UNKNOWN", "UNKNOWN_PERSON", "NONE", "UNAVAILABLE", "NULL")

    def decide_identity(
        self,
        gait_identity: str | None = None,
        gait_score: float | None = None,
        appearance_identity: str | None = None,
        appearance_score: float | None = None,
        gait_threshold: float = 0.85,
        appearance_threshold: float = 0.60,
        crop: np.ndarray | None = None,
        gei_frame_count: int = 0,
        gei: np.ndarray | None = None,
        confidence: float = 1.0,
        crowd_density: float = 0.0,
        occlusion_score: float = 0.0,
        track_reliability: float = 1.0,
        unknown_label: str = "UNKNOWN_PERSON",
    ) -> dict[str, Any]:
        raw_g_id = str(gait_identity).strip() if gait_identity is not None else ""
        raw_a_id = str(appearance_identity).strip() if appearance_identity is not None else ""

        g_valid = self._is_valid_identity(raw_g_id)
        a_valid = self._is_valid_identity(raw_a_id)

        g_score = float(gait_score) if (gait_score is not None and np.isfinite(gait_score)) else 0.0
        a_score = float(appearance_score) if (appearance_score is not None and np.isfinite(appearance_score)) else 0.0

        g_passes = g_valid and (g_score >= float(gait_threshold))
        a_passes = a_valid and (a_score >= float(appearance_threshold))

        fusion_res = self.fuse(
            gait_score=g_score if g_valid else None,
            reid_score=a_score if a_valid else None,
            crop=crop,
            gei_frame_count=gei_frame_count,
            gei=gei,
            confidence=confidence,
            crowd_density=crowd_density,
            occlusion_score=occlusion_score,
            track_reliability=track_reliability,
        )


        if g_passes and a_passes and (raw_g_id == raw_a_id):
            final_identity = raw_g_id
            final_score = fusion_res["final_score"]
            status = "CONFIRMED"
            decision = "CONFIRMED"
            modality_state = "DUAL_MODAL_MATCH"
            conflict = False


        elif g_passes and a_passes and (raw_g_id != raw_a_id):
            final_identity = "REVIEW_REQUIRED"
            final_score = max(g_score, a_score)
            status = "REVIEW_REQUIRED"
            decision = "REVIEW_REQUIRED"
            modality_state = "CONFLICT"
            conflict = True


        elif g_passes and not a_passes:
            final_identity = raw_g_id
            final_score = g_score
            status = "CONFIRMED"
            decision = "CONFIRMED"
            modality_state = "GAIT_ONLY"
            conflict = False


        elif a_passes and not g_passes:
            final_identity = raw_a_id
            final_score = a_score
            status = "CONFIRMED"
            decision = "CONFIRMED"
            modality_state = "APPEARANCE_ONLY"
            conflict = False


        else:
            final_identity = unknown_label
            final_score = max(g_score, a_score) if (g_score > 0 or a_score > 0) else 0.0
            status = "UNKNOWN"
            decision = "UNKNOWN"
            modality_state = "UNAVAILABLE" if (g_score == 0 and a_score == 0) else "BELOW_THRESHOLD"
            conflict = False


        if decision == "CONFIRMED" and self._is_valid_identity(final_identity):
            for group in self.high_risk_confusion_groups:
                if final_identity in group:
                    decision = "REVIEW_REQUIRED"
                    status = "REVIEW_REQUIRED"
                    modality_state = f"{modality_state}_CONFUSION_SAFEGUARD"
                    break

        return {
            "final_identity": final_identity,
            "final_score": round(float(final_score), 4),
            "status": status,
            "decision": decision,
            "modality_state": modality_state,
            "conflict": conflict,
            "gait_candidate": raw_g_id if g_valid else None,
            "gait_score": round(g_score, 4),
            "gait_status": "MATCH" if g_passes else "UNKNOWN",
            "appearance_candidate": raw_a_id if a_valid else None,
            "appearance_score": round(a_score, 4),
            "appearance_status": "MATCH" if a_passes else "UNKNOWN",
            "fusion": fusion_res,
        }
