from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

DISPLAY_STATE_CONFIRMED = "CONFIRMED"
DISPLAY_STATE_UNCONFIRMED = "UNCONFIRMED"
DISPLAY_STATE_SPECIAL_ATTENTION = "SPECIAL_ATTENTION"


DISPLAY_STATE_ASSESSING = "ASSESSING"
DISPLAY_STATE_INAPPLICABLE = "BIOMETRIC_INAPPLICABLE"
DISPLAY_STATE_EVIDENCE_COLLECTING = "EVIDENCE_COLLECTING"
DISPLAY_STATE_PENDING = "PENDING"
DISPLAY_STATE_UNKNOWN = "UNKNOWN"


COLOR_RED_BGR = (0, 0, 255)
COLOR_GREEN_BGR = (0, 255, 0)
COLOR_YELLOW_BGR = (0, 255, 255)


def load_display_config() -> dict:
    config_path = Path("configs/inference.yaml")

    defaults: dict = {
        "enabled": True,
        "show_camera_id": True,
        "show_track_id": True,
        "show_score": True,
        "line_thickness": 2,
        "font_scale": 0.6,
        "colors": {
            "confirmed": [0, 0, 255],
            "unconfirmed": [0, 255, 0],
            "special_attention": [0, 255, 255],
            "assessing": [0, 255, 0],
            "inapplicable": [0, 255, 0],
            "tracking": [0, 255, 0],
            "unknown": [0, 255, 0],
            "uncertain": [0, 255, 0],
            "non_valid": [0, 255, 0],
            "detection": [0, 255, 0],
        },
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError, ValueError, KeyError):
        return defaults

    section = data.get("display", {})
    if not isinstance(section, dict):
        return defaults

    merged: dict = {}
    for key, default_value in defaults.items():
        if key == "colors":
            raw_colors = section.get("colors", {})
            if not isinstance(raw_colors, dict):
                merged["colors"] = default_value
            else:
                merged["colors"] = {ck: raw_colors.get(ck, cv) for ck, cv in default_value.items()}
        else:
            merged[key] = section.get(key, default_value)

    return merged


def map_to_display_state(
    status: str = "UNKNOWN",
    decision: str = "UNKNOWN",
    identity: str = "UNKNOWN",
    mobility_state: str = "STANDARD_WALKING",
    gait_eligible: bool = True,
    display_state: str | None = None,
    is_valid: bool | None = None,
    is_special_attention: bool = False,
) -> str:
    if display_state is not None:
        normalized = str(display_state).upper()
        if (normalized in ("CONFIRMED", "MATCH", "VERIFIED_MATCH", "CONFIRMED_MATCH") or normalized.startswith("CONFIRM")) and not normalized.startswith("UN"):
            return DISPLAY_STATE_CONFIRMED
        if "SPECIAL_ATTENTION" in normalized or normalized == "ATTENTION" or "SECURITY_ALERT" in normalized:
            return DISPLAY_STATE_SPECIAL_ATTENTION
        if "INAPPLICABLE" in normalized or "WHEELCHAIR" in normalized:
            return DISPLAY_STATE_INAPPLICABLE
        if "ASSESS" in normalized or "COLLECT" in normalized:
            return DISPLAY_STATE_ASSESSING
        return DISPLAY_STATE_UNCONFIRMED


    if is_special_attention:
        return DISPLAY_STATE_SPECIAL_ATTENTION


    if (
        status in ("CONFIRMED", "MATCH", "VERIFIED_MATCH")
        or decision in ("CONFIRMED", "CONFIRMED_MATCH", "MATCH", "VERIFIED_MATCH")
    ) and identity not in ("UNKNOWN", "UNKNOWN_PERSON", ""):
        return DISPLAY_STATE_CONFIRMED


    if mobility_state in ("WHEELCHAIR", "CRUTCHES_AID", "STATIONARY_SEATED", "NON_STANDARD_GAIT"):
        return DISPLAY_STATE_INAPPLICABLE

    if not gait_eligible and decision in ("BIOMETRIC_INAPPLICABLE", "GAIT_UNAVAILABLE", "INAPPLICABLE"):
        return DISPLAY_STATE_INAPPLICABLE


    return DISPLAY_STATE_UNCONFIRMED


class DetectionDisplayRenderer:
    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config if config is not None else load_display_config()

        colors_raw = self.cfg.get("colors", {})
        self._color_confirmed: tuple[int, int, int] = tuple(colors_raw.get("confirmed", [0, 0, 255]))
        self._color_unconfirmed: tuple[int, int, int] = tuple(colors_raw.get("unconfirmed", [0, 255, 0]))
        self._color_special_attention: tuple[int, int, int] = tuple(colors_raw.get("special_attention", [0, 255, 255]))

        self._thickness: int = int(self.cfg.get("line_thickness", 2))
        self._font_scale: float = float(self.cfg.get("font_scale", 0.6))
        self._show_cam: bool = bool(self.cfg.get("show_camera_id", True))
        self._show_tid: bool = bool(self.cfg.get("show_track_id", True))
        self._show_score: bool = bool(self.cfg.get("show_score", True))

    def get_color_for_state(self, state: str) -> tuple[int, int, int]:
        normalized = str(state).upper()
        if (normalized in ("CONFIRMED", "MATCH", "VERIFIED_MATCH", "CONFIRMED_MATCH") or normalized.startswith("CONFIRM")) and not normalized.startswith("UN"):
            return self._color_confirmed
        if (
            normalized in ("SPECIAL_ATTENTION", "ATTENTION", "SECURITY_ALERT", "OPERATIONAL_ATTENTION", "FLAGGED")
            or "SPECIAL_ATTENTION" in normalized
        ):
            return self._color_special_attention
        return self._color_unconfirmed

    def get_status(self, decision: str) -> str:
        return map_to_display_state(decision=decision)

    def get_color(self, status: str) -> tuple[int, int, int]:
        return self.get_color_for_state(status)

    def draw(
        self,
        frame: np.ndarray,
        box: Any,
        track_id: int | None = None,
        identity: str = "UNKNOWN",
        score: float = 0.0,
        decision: str = "UNKNOWN",
        camera_id: str = "cam_00",
        display_state: str | None = None,
        is_valid: bool | None = None,
        mobility_state: str = "STANDARD_WALKING",
        gait_eligible: bool = True,
        is_special_attention: bool = False,
    ) -> None:
        if not self.cfg.get("enabled", True):
            return

        if box is None or len(box) < 4:
            return

        x1, y1, x2, y2 = map(int, box[:4])


        state = map_to_display_state(
            status=decision,
            decision=decision,
            identity=identity,
            mobility_state=mobility_state,
            gait_eligible=gait_eligible,
            display_state=display_state,
            is_valid=is_valid,
            is_special_attention=is_special_attention,
        )

        box_color = self.get_color_for_state(state)


        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, self._thickness)

        label = self._build_label(
            camera_id=camera_id,
            track_id=track_id,
            display_state=state,
            identity=identity,
            score=score,
            decision=decision,
            mobility_state=mobility_state,
        )

        self._draw_label(frame, label, x1, y1, box_color)

    def _build_label(
        self,
        camera_id: str,
        track_id: int | None,
        display_state: str,
        identity: str,
        score: float,
        decision: str = "",
        mobility_state: str = "STANDARD_WALKING",
    ) -> str:
        parts: list[str] = []

        if self._show_cam and camera_id:
            parts.append(f"[{camera_id}]")

        if self._show_tid:
            if track_id is not None and track_id >= 0:
                parts.append(f"T{track_id}")
            else:
                parts.append("DET")


        parts.append(display_state)


        if mobility_state == "WHEELCHAIR":
            parts.append("WHEELCHAIR")
        elif mobility_state == "CRUTCHES_AID":
            parts.append("CRUTCHES")


        clean_id = identity if identity and identity not in ("UNKNOWN_PERSON", "UNKNOWN") else "UNKNOWN"
        parts.append(clean_id)


        if self._show_score and score > 0.0 and clean_id != "UNKNOWN":
            parts.append(f"{score:.2f}")

        return " | ".join(parts)

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._font_scale
        thick = 1

        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

        label_y = max(th + baseline + 4, y - 6)
        bg_x1 = max(0, x)
        bg_y1 = max(0, label_y - th - baseline - 2)
        bg_x2 = min(frame.shape[1], bg_x1 + tw + 6)
        bg_y2 = min(frame.shape[0], label_y + 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)


        text_color = color if color != (0, 0, 0) else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (bg_x1 + 3, label_y - baseline),
            font,
            scale,
            text_color,
            thick,
            cv2.LINE_AA,
        )
