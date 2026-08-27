"""
Crowd Robustness Manager Engine.

Coordinates crowd density estimation, occlusion-aware silhouette filtering,
and adaptive identity/reliability decision thresholds in crowded environments.
"""

from typing import Any

from intelligence.crowd_density_estimator import (
    CrowdDensityEstimator,
    CrowdDensityLevel,
    CrowdDensityResult,
    compute_iou,
)
from monitoring.logging_config import get_logger


class CrowdRobustnessManager:
    """
    Manager for crowd-robust detection, tracking, silhouette filtering, and decision gating.

    Disabled by default to ensure baseline behavior remains unaffected.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.logger = get_logger("crowd_robustness")
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))

        self.occlusion_overlap_threshold = float(cfg.get("occlusion_overlap_threshold", 0.35))
        density_cfg = cfg.get("density_thresholds", {})
        adaptive_cfg = cfg.get("adaptive_gating", {})

        self.estimator = CrowdDensityEstimator(
            strong_overlap_threshold=float(cfg.get("strong_overlap_iou", 0.25)),
            moderate_count=int(density_cfg.get("moderate_count", 5)),
            high_count=int(density_cfg.get("high_count", 12)),
            severe_count=int(density_cfg.get("severe_count", 20)),
            moderate_overlap_count=int(density_cfg.get("moderate_overlap_count", 1)),
            high_overlap_count=int(density_cfg.get("high_overlap_count", 4)),
            severe_overlap_count=int(density_cfg.get("severe_overlap_count", 8)),
            moderate_area_ratio=float(density_cfg.get("moderate_area_ratio", 0.15)),
            high_area_ratio=float(density_cfg.get("high_area_ratio", 0.30)),
            severe_area_ratio=float(density_cfg.get("severe_area_ratio", 0.50)),
        )

        self.high_density_quality_penalty = float(adaptive_cfg.get("high_density_quality_penalty", 0.10))
        self.severe_density_margin_boost = float(adaptive_cfg.get("severe_density_margin_boost", 0.05))

    def is_enabled(self) -> bool:
        """Return whether crowd robustness features are enabled."""
        return self.enabled

    def process_frame_density(
        self,
        detections: list[dict[str, Any]],
        frame_shape: tuple[int, int] = (1080, 1920),
    ) -> CrowdDensityResult:
        """Estimate crowd density for current frame."""
        return self.estimator.estimate_density(detections, frame_shape)

    def identify_occluded_tracks(
        self,
        tracked_objects: list[dict[str, Any]],
    ) -> tuple[set[int], dict[int, float]]:
        """
        Identify tracks whose bounding boxes severely overlap with other active tracks.

        Args:
            tracked_objects: List of dicts containing 'track_id' and 'bbox'.

        Returns:
            Tuple of (occluded_track_ids_set, max_overlap_iou_by_track_id_dict)
        """
        if not self.enabled or not tracked_objects:
            return set(), {}

        occluded_ids: set[int] = set()
        max_overlap_map: dict[int, float] = {}

        n = len(tracked_objects)
        for i in range(n):
            t_id_i = int(tracked_objects[i]["track_id"])
            box_i = tracked_objects[i]["bbox"]
            if t_id_i not in max_overlap_map:
                max_overlap_map[t_id_i] = 0.0

            for j in range(i + 1, n):
                t_id_j = int(tracked_objects[j]["track_id"])
                box_j = tracked_objects[j]["bbox"]
                if t_id_j not in max_overlap_map:
                    max_overlap_map[t_id_j] = 0.0

                iou = compute_iou(box_i, box_j)

                max_overlap_map[t_id_i] = max(max_overlap_map[t_id_i], iou)
                max_overlap_map[t_id_j] = max(max_overlap_map[t_id_j], iou)

                if iou >= self.occlusion_overlap_threshold:
                    occluded_ids.add(t_id_i)
                    occluded_ids.add(t_id_j)

        return occluded_ids, max_overlap_map

    def adapt_quality_score(
        self,
        base_quality: float,
        density_level: CrowdDensityLevel,
        is_occluded: bool = False,
    ) -> float:
        """Apply crowd density and occlusion adjustments to quality scores."""
        if not self.enabled:
            return base_quality

        quality = float(base_quality)
        if is_occluded:
            quality -= 0.15

        if density_level == CrowdDensityLevel.SEVERE:
            quality -= self.high_density_quality_penalty * 1.5
        elif density_level == CrowdDensityLevel.HIGH:
            quality -= self.high_density_quality_penalty

        return float(max(0.0, min(1.0, quality)))

    def adapt_open_set_margin(
        self,
        base_margin_threshold: float,
        density_level: CrowdDensityLevel,
    ) -> float:
        """Boost margin requirement in severe/high crowds to prevent false positives."""
        if not self.enabled:
            return base_margin_threshold

        margin = float(base_margin_threshold)
        if density_level == CrowdDensityLevel.SEVERE:
            margin += self.severe_density_margin_boost
        elif density_level == CrowdDensityLevel.HIGH:
            margin += self.severe_density_margin_boost * 0.5

        return margin
