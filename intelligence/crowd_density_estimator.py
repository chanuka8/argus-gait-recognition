from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CrowdDensityLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


@dataclass
class CrowdDensityResult:
    level: CrowdDensityLevel
    person_count: int
    total_area_ratio: float
    avg_pairwise_overlap: float
    strongly_overlapping_count: int
    metrics: dict[str, Any] = field(default_factory=dict)


def compute_iou(box1: list[float], box2: list[float]) -> float:
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


class CrowdDensityEstimator:
    def __init__(
        self,
        strong_overlap_threshold: float = 0.25,
        moderate_count: int = 5,
        high_count: int = 12,
        severe_count: int = 20,
        moderate_overlap_count: int = 1,
        high_overlap_count: int = 4,
        severe_overlap_count: int = 8,
        moderate_area_ratio: float = 0.15,
        high_area_ratio: float = 0.30,
        severe_area_ratio: float = 0.50,
    ) -> None:
        self.strong_overlap_threshold = strong_overlap_threshold
        self.moderate_count = moderate_count
        self.high_count = high_count
        self.severe_count = severe_count
        self.moderate_overlap_count = moderate_overlap_count
        self.high_overlap_count = high_overlap_count
        self.severe_overlap_count = severe_overlap_count
        self.moderate_area_ratio = moderate_area_ratio
        self.high_area_ratio = high_area_ratio
        self.severe_area_ratio = severe_area_ratio

    def estimate_density(
        self,
        detections: list[dict[str, Any]],
        frame_shape: tuple[int, int] = (1080, 1920),
    ) -> CrowdDensityResult:
        person_count = len(detections)
        if person_count == 0:
            return CrowdDensityResult(
                level=CrowdDensityLevel.LOW,
                person_count=0,
                total_area_ratio=0.0,
                avg_pairwise_overlap=0.0,
                strongly_overlapping_count=0,
                metrics={"density_score": 0.0},
            )

        frame_h, frame_w = frame_shape[:2]
        frame_area = float(max(1, frame_h * frame_w))

        bboxes = [d["bbox"] for d in detections if "bbox" in d]
        total_bbox_area = sum(float(max(0, b[2] - b[0]) * max(0, b[3] - b[1])) for b in bboxes)
        total_area_ratio = float(total_bbox_area / frame_area)

        total_iou = 0.0
        pair_count = 0
        strongly_overlapping_count = 0

        num_boxes = len(bboxes)
        for i in range(num_boxes):
            for j in range(i + 1, num_boxes):
                iou = compute_iou(bboxes[i], bboxes[j])
                total_iou += iou
                pair_count += 1
                if iou >= self.strong_overlap_threshold:
                    strongly_overlapping_count += 1

        avg_pairwise_overlap = float(total_iou / pair_count) if pair_count > 0 else 0.0

        if (
            person_count >= self.severe_count
            or strongly_overlapping_count >= self.severe_overlap_count
            or total_area_ratio >= self.severe_area_ratio
            or avg_pairwise_overlap >= 0.25
        ):
            level = CrowdDensityLevel.SEVERE
        elif (
            person_count >= self.high_count
            or strongly_overlapping_count >= self.high_overlap_count
            or total_area_ratio >= self.high_area_ratio
            or avg_pairwise_overlap >= 0.15
        ):
            level = CrowdDensityLevel.HIGH
        elif (
            person_count >= self.moderate_count
            or strongly_overlapping_count >= self.moderate_overlap_count
            or total_area_ratio >= self.moderate_area_ratio
            or avg_pairwise_overlap >= 0.05
        ):
            level = CrowdDensityLevel.MODERATE
        else:
            level = CrowdDensityLevel.LOW

        density_score = float(
            min(
                1.0,
                0.4 * (person_count / max(1, self.severe_count))
                + 0.3 * (total_area_ratio / max(0.01, self.severe_area_ratio))
                + 0.3 * avg_pairwise_overlap,
            )
        )

        return CrowdDensityResult(
            level=level,
            person_count=person_count,
            total_area_ratio=round(total_area_ratio, 4),
            avg_pairwise_overlap=round(avg_pairwise_overlap, 4),
            strongly_overlapping_count=strongly_overlapping_count,
            metrics={"density_score": round(density_score, 4)},
        )
