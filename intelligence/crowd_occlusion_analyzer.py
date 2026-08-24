"""
Stage 1: Crowd-Aware Occlusion Analyzer Engine.

Provides lightweight crowd density analysis, per-track bounding-box mutual occlusion scoring,
moving-window smoothing, clean-frame ratio calculation, and silhouette acceptance decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CrowdDensityLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = float((x_right - x_left) * (y_bottom - y_top))
    area1 = float((x2_1 - x1_1) * (y2_1 - y1_1))
    area2 = float((x2_2 - x1_2) * (y2_2 - y1_2))
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


@dataclass
class TrackOcclusionState:
    camera_id: str
    track_id: Any
    history: List[float] = field(default_factory=list)
    clean_history: List[bool] = field(default_factory=list)
    last_seen: float = 0.0
    smooth_occlusion_score: float = 0.0
    clean_frame_ratio: float = 1.0
    clean_frame_count: int = 0


@dataclass
class FrameCrowdAnalysis:
    crowd_density_score: float
    crowd_density_level: CrowdDensityLevel
    person_count: int
    total_area_ratio: float
    avg_pairwise_overlap: float
    strongly_overlapping_count: int
    track_occlusions: Dict[Tuple[str, Any], float] = field(default_factory=dict)
    clean_frame_ratios: Dict[Tuple[str, Any], float] = field(default_factory=dict)
    silhouette_acceptance: Dict[Tuple[str, Any], bool] = field(default_factory=dict)


class CrowdOcclusionAnalyzer:
    """
    Analyzes frame crowd density and maintains per-track occlusion state and silhouette gating.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.smoothing_window = max(1, int(cfg.get("smoothing_window", 5)))
        self.moderate_threshold = float(cfg.get("moderate_threshold", 0.35))
        self.high_threshold = float(cfg.get("high_threshold", 0.60))
        self.severe_threshold = float(cfg.get("severe_threshold", 0.80))
        self.minimum_clean_frames = int(cfg.get("minimum_clean_frames", 18))
        self.minimum_clean_ratio = float(cfg.get("minimum_clean_ratio", 0.70))

        self.track_states: Dict[Tuple[str, Any], TrackOcclusionState] = {}

    def is_enabled(self) -> bool:
        return self.enabled

    def analyze_frame(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int] = (1080, 1920),
        camera_id: str = "cam_00",
        timestamp: Optional[float] = None,
    ) -> FrameCrowdAnalysis:
        """
        Analyze crowd density and compute per-track occlusion scores for a frame.

        Args:
            detections: List of detection dicts with 'track_id' (or index) and 'bbox'.
            frame_shape: (height, width) of image frame.
            camera_id: Camera identifier string.
            timestamp: Monotonic timestamp.

        Returns:
            FrameCrowdAnalysis dataclass.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        person_count = len(detections)

        if person_count == 0:
            return FrameCrowdAnalysis(
                crowd_density_score=0.0,
                crowd_density_level=CrowdDensityLevel.LOW,
                person_count=0,
                total_area_ratio=0.0,
                avg_pairwise_overlap=0.0,
                strongly_overlapping_count=0,
            )

        frame_h, frame_w = frame_shape[:2]
        frame_area = float(max(1, frame_h * frame_w))

        bboxes = []
        track_keys = []
        for idx, det in enumerate(detections):
            box = det.get("bbox")
            if box and len(box) == 4:
                bboxes.append(box)
                t_id = det.get("track_id", idx)
                track_keys.append((camera_id, t_id))

        n_boxes = len(bboxes)
        total_bbox_area = sum(
            float(max(0, b[2] - b[0]) * max(0, b[3] - b[1])) for b in bboxes
        )
        total_area_ratio = float(total_bbox_area / frame_area)

        raw_occlusions = [0.0] * n_boxes
        total_iou = 0.0
        pair_count = 0
        strongly_overlapping_count = 0

        for i in range(n_boxes):
            for j in range(i + 1, n_boxes):
                iou = compute_iou(bboxes[i], bboxes[j])
                total_iou += iou
                pair_count += 1
                if iou > raw_occlusions[i]:
                    raw_occlusions[i] = iou
                if iou > raw_occlusions[j]:
                    raw_occlusions[j] = iou
                if iou >= 0.25:
                    strongly_overlapping_count += 1

        avg_pairwise_overlap = float(total_iou / pair_count) if pair_count > 0 else 0.0

        density_score = float(
            min(1.0, 0.4 * (person_count / 20.0)
            + 0.3 * (total_area_ratio / 0.50)
            + 0.3 * avg_pairwise_overlap)
        )

        if density_score >= self.severe_threshold or person_count >= 20 or strongly_overlapping_count >= 8:
            density_level = CrowdDensityLevel.SEVERE
        elif density_score >= self.high_threshold or person_count >= 12 or strongly_overlapping_count >= 4:
            density_level = CrowdDensityLevel.HIGH
        elif density_score >= self.moderate_threshold or person_count >= 5 or strongly_overlapping_count >= 1:
            density_level = CrowdDensityLevel.MODERATE
        else:
            density_level = CrowdDensityLevel.LOW

        track_occlusions = {}
        clean_frame_ratios = {}
        silhouette_acceptance = {}

        for idx, key in enumerate(track_keys):
            raw_occ = raw_occlusions[idx]
            if key not in self.track_states:
                self.track_states[key] = TrackOcclusionState(
                    camera_id=camera_id,
                    track_id=key[1],
                )

            state = self.track_states[key]
            state.last_seen = now
            state.history.append(raw_occ)
            if len(state.history) > self.smoothing_window:
                state.history.pop(0)

            smooth_score = float(np.mean(state.history)) if state.history else raw_occ
            state.smooth_occlusion_score = smooth_score

            is_clean = raw_occ < self.moderate_threshold
            state.clean_history.append(is_clean)
            if len(state.clean_history) > max(30, self.smoothing_window * 6):
                state.clean_history.pop(0)

            clean_ratio = float(np.mean(state.clean_history)) if state.clean_history else 1.0
            state.clean_frame_ratio = clean_ratio
            state.clean_frame_count = sum(state.clean_history)

            accepted = smooth_score < self.high_threshold

            track_occlusions[key] = round(smooth_score, 4)
            clean_frame_ratios[key] = round(clean_ratio, 4)
            silhouette_acceptance[key] = accepted

        return FrameCrowdAnalysis(
            crowd_density_score=round(density_score, 4),
            crowd_density_level=density_level,
            person_count=person_count,
            total_area_ratio=round(total_area_ratio, 4),
            avg_pairwise_overlap=round(avg_pairwise_overlap, 4),
            strongly_overlapping_count=strongly_overlapping_count,
            track_occlusions=track_occlusions,
            clean_frame_ratios=clean_frame_ratios,
            silhouette_acceptance=silhouette_acceptance,
        )

    def cleanup_inactive(self, max_idle_seconds: float = 10.0, current_time: Optional[float] = None) -> List[Tuple[str, Any]]:
        """Purge inactive track states that have timed out."""
        now = current_time if current_time is not None else time.monotonic()
        purged = []
        for key, state in list(self.track_states.items()):
            if now - state.last_seen > max_idle_seconds:
                purged.append(key)
                del self.track_states[key]
        return purged
