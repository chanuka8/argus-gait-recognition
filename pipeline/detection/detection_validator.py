from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectionMetadata:
    camera_id: str
    bbox: list[int]
    confidence: float
    is_valid: bool
    validity_reason: str = "OK"
    timestamp: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    track_id: int | None = None
    identity: str = "UNKNOWN_PERSON"
    similarity: float = 0.0
    decision: str = "UNKNOWN"
    status: str = "UNKNOWN"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DetectionValidator:
    def __init__(
        self,
        min_width: int | None = None,
        min_height: int | None = None,
        min_confidence: float | None = None,
        min_aspect_ratio: float | None = None,
        max_aspect_ratio: float | None = None,
        config_path: str = "configs/detection.yaml",
    ) -> None:
        self.min_width = 16
        self.min_height = 32
        self.min_confidence = 0.40
        self.min_aspect_ratio = 1.0
        self.max_aspect_ratio = 6.0

        self._load_from_config(config_path)

        if min_width is not None:
            self.min_width = int(min_width)
        if min_height is not None:
            self.min_height = int(min_height)
        if min_confidence is not None:
            self.min_confidence = float(min_confidence)
        if min_aspect_ratio is not None:
            self.min_aspect_ratio = float(min_aspect_ratio)
        if max_aspect_ratio is not None:
            self.max_aspect_ratio = float(max_aspect_ratio)

    def _load_from_config(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                if isinstance(cfg, dict):
                    validity_cfg = cfg.get("validity", {})
                    if isinstance(validity_cfg, dict):
                        self.min_width = int(validity_cfg.get("min_width", self.min_width))
                        self.min_height = int(validity_cfg.get("min_height", self.min_height))
                        self.min_aspect_ratio = float(validity_cfg.get("min_aspect_ratio", self.min_aspect_ratio))
                        self.max_aspect_ratio = float(validity_cfg.get("max_aspect_ratio", self.max_aspect_ratio))
                        self.min_confidence = float(validity_cfg.get("min_confidence", self.min_confidence))
                    elif "confidence" in cfg:
                        self.min_confidence = float(cfg.get("confidence", self.min_confidence))
        except (yaml.YAMLError, OSError, ValueError, KeyError):
            pass

    def validate_detection(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        confidence: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> tuple[bool, str]:
        if not bbox or len(bbox) < 4:
            return False, "INVALID_BBOX_COORDINATES"

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        width = x2 - x1
        height = y2 - y1


        if width <= 0 or height <= 0:
            return False, "DEGENERATE_BBOX_DIMENSIONS"

        if width < self.min_width or height < (self.min_height // 2):
            return False, f"SUB_MINIMUM_SIZE_{width}x{height}_LT_{self.min_width}x{self.min_height // 2}"


        if frame_shape is not None and len(frame_shape) >= 2:
            frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
            if (x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h) and (
                width < (self.min_width // 2) or height < (self.min_height // 4)
            ):
                return False, "OUT_OF_BOUNDS_TRUNCATED"


        if float(confidence) < max(0.20, self.min_confidence - 0.10):
            return False, f"LOW_CONFIDENCE_{confidence:.2f}_LT_{self.min_confidence:.2f}"

        return True, "VALID_DETECTION"

    def assess_detection(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        confidence: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> tuple[bool, str, bool, bool, str]:
        is_valid, valid_reason = self.validate_detection(bbox, confidence, frame_shape)
        if not is_valid:
            return False, "NON_STANDARD_GAIT", False, False, valid_reason

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = float(height) / float(width)


        if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            return True, "STANDARD_WALKING", True, True, "STANDARD_UPRIGHT_GAIT_ELIGIBLE"


        if 0.4 <= aspect_ratio < self.min_aspect_ratio:
            return True, "WHEELCHAIR", False, True, "WHEELCHAIR_SEATED_GAIT_INAPPLICABLE"


        return True, "NON_STANDARD_GAIT", False, False, f"NON_STANDARD_ASPECT_RATIO_{aspect_ratio:.2f}"

    def is_valid_detection(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        confidence: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> bool:
        is_valid, _ = self.validate_detection(bbox, confidence, frame_shape)
        return is_valid

    def is_gait_eligible(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        confidence: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> bool:
        _, _, gait_eligible, _, _ = self.assess_detection(bbox, confidence, frame_shape)
        return gait_eligible

    def is_appearance_eligible(
        self,
        bbox: list[int] | tuple[int, int, int, int],
        confidence: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> bool:
        _, _, _, appearance_eligible, _ = self.assess_detection(bbox, confidence, frame_shape)
        return appearance_eligible

    def tag_detections(
        self,
        detections: list[dict],
        frame_shape: tuple[int, ...] | None = None,
        camera_id: str = "camera_00",
        frame_id: int = 0,
    ) -> list[DetectionMetadata]:
        tagged_items: list[DetectionMetadata] = []
        now = time.monotonic()

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = float(det.get("confidence", 0.0))
            is_valid, reason = self.validate_detection(bbox, conf, frame_shape)

            tagged = DetectionMetadata(
                camera_id=camera_id,
                bbox=list(bbox),
                confidence=conf,
                is_valid=is_valid,
                validity_reason=reason,
                timestamp=now,
                frame_id=frame_id,
                track_id=det.get("track_id"),
                identity=str(det.get("identity", "UNKNOWN_PERSON")),
                similarity=float(det.get("similarity", 0.0)),
                decision=str(det.get("decision", "UNKNOWN")),
                status=str(det.get("status", "UNKNOWN")),
                details=dict(det.get("details", {})),
            )
            tagged_items.append(tagged)

        return tagged_items
