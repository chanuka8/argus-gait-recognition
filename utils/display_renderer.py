"""
ARGUS AI — Detection Display & System Assessment Overlay Renderer.

Professional CCTV-style overlay engine implementing the certified production color semantics:
- RED BGR (0, 0, 255): Confirmed / successfully recognized identity (CONFIRMED / MATCH).
- GREEN BGR (0, 255, 0): Person detected and tracked, but identity is not confirmed (default for all
  normal unconfirmed states including UNKNOWN, PENDING, ASSESSING, EVIDENCE_COLLECTING,
  INSUFFICIENT_EVIDENCE, GAIT_UNAVAILABLE, APPEARANCE_UNAVAILABLE, BIOMETRIC_INAPPLICABLE,
  WHEELCHAIR, CRUTCHES, NON_STANDARD_GAIT, SEATED, STATIONARY, etc.).
- YELLOW BGR (0, 255, 255): Reserved special operational / attention state ONLY (e.g. SPECIAL_ATTENTION,
  SECURITY_ALERT, FLAGGED). Never used for ordinary unconfirmed or evidence-collecting persons.

Core Invariants:
1. Every detected person receives a visible bounding box regardless of identity, enrollment, or gait usability.
2. Bounding-box color represents the person's CONFIRMATION STATE (CONFIRMED = RED, NOT CONFIRMED = GREEN,
   SPECIAL ATTENTION = YELLOW).
3. UNKNOWN, ASSESSING, and EVIDENCE_COLLECTING persons ALWAYS render GREEN.
4. Text labels independently display Camera ID, Track ID, Assessment/Mobility State, Identity, and Score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# Assessment & Display State Constants
DISPLAY_STATE_CONFIRMED = "CONFIRMED"                      # RED (0, 0, 255)
DISPLAY_STATE_UNCONFIRMED = "UNCONFIRMED"                  # GREEN (0, 255, 0)
DISPLAY_STATE_SPECIAL_ATTENTION = "SPECIAL_ATTENTION"      # YELLOW (0, 255, 255) - Reserved Operational Attention

# Descriptive state aliases (all unconfirmed states map to GREEN)
DISPLAY_STATE_ASSESSING = "ASSESSING"                      # GREEN (0, 255, 0)
DISPLAY_STATE_INAPPLICABLE = "BIOMETRIC_INAPPLICABLE"      # GREEN (0, 255, 0)
DISPLAY_STATE_EVIDENCE_COLLECTING = "EVIDENCE_COLLECTING"  # GREEN (0, 255, 0)
DISPLAY_STATE_PENDING = "PENDING"                          # GREEN (0, 255, 0)
DISPLAY_STATE_UNKNOWN = "UNKNOWN"                          # GREEN (0, 255, 0)

# BGR Color Constants
COLOR_RED_BGR = (0, 0, 255)        # Confirmed Match / Recognized Person
COLOR_GREEN_BGR = (0, 255, 0)      # Detected/Tracked but Unconfirmed (Default for all non-confirmed)
COLOR_YELLOW_BGR = (0, 255, 255)    # Explicit Special Operational Attention Only (Reserved)


def load_display_config() -> dict:
    """Load the ``display`` section from ``configs/inference.yaml``.

    Returns safe defaults when the file or section is absent.
    """
    config_path = Path("configs/inference.yaml")

    defaults: dict = {
        "enabled": True,
        "show_camera_id": True,
        "show_track_id": True,
        "show_score": True,
        "line_thickness": 2,
        "font_scale": 0.6,
        "colors": {
            "confirmed": [0, 0, 255],            # Red (BGR) — Confirmed match
            "unconfirmed": [0, 255, 0],          # Green (BGR) — Default detected/tracked unconfirmed
            "special_attention": [0, 255, 255],  # Yellow (BGR) — Explicit operational attention only
            "assessing": [0, 255, 0],            # Green (BGR)
            "inapplicable": [0, 255, 0],         # Green (BGR)
            "tracking": [0, 255, 0],             # Green (BGR)
            "unknown": [0, 255, 0],              # Green (BGR)
            "uncertain": [0, 255, 0],            # Green (BGR)
            "non_valid": [0, 255, 0],            # Green (BGR)
            "detection": [0, 255, 0],            # Green (BGR)
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
    """Map internal pipeline state to one of the display states:

    1. CONFIRMED (RED) - Confirmed identity / successfully recognized
    2. SPECIAL_ATTENTION (YELLOW) - Reserved explicit operational attention only
    3. UNCONFIRMED / ASSESSING / INAPPLICABLE (GREEN) - Detected & tracked, but identity not confirmed (default)
    """
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

    # 1. Explicit Special Attention (YELLOW)
    if is_special_attention:
        return DISPLAY_STATE_SPECIAL_ATTENTION

    # 2. Confirmed Match (RED)
    if (
        status in ("CONFIRMED", "MATCH", "VERIFIED_MATCH")
        or decision in ("CONFIRMED", "CONFIRMED_MATCH", "MATCH", "VERIFIED_MATCH")
    ) and identity not in ("UNKNOWN", "UNKNOWN_PERSON", ""):
        return DISPLAY_STATE_CONFIRMED

    # 3. Specific descriptive non-confirmed states (all render GREEN)
    if mobility_state in ("WHEELCHAIR", "CRUTCHES_AID", "STATIONARY_SEATED", "NON_STANDARD_GAIT"):
        return DISPLAY_STATE_INAPPLICABLE

    if not gait_eligible and decision in ("BIOMETRIC_INAPPLICABLE", "GAIT_UNAVAILABLE", "INAPPLICABLE"):
        return DISPLAY_STATE_INAPPLICABLE

    # 4. Default: GREEN for all detected and tracked persons whose identity is not confirmed
    return DISPLAY_STATE_UNCONFIRMED


class DetectionDisplayRenderer:
    """Draws professional CCTV-style overlays on video frames.

    Implements production color semantics (RED = Confirmed, GREEN = Unconfirmed/Default, YELLOW = Special Attention).
    """

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
        """Return the BGR colour tuple for a given display state string.

        - RED (0, 0, 255): Confirmed identity
        - YELLOW (0, 255, 255): Explicit special operational attention only
        - GREEN (0, 255, 0): All other detected, tracked, unconfirmed, assessing, or inapplicable states
        """
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
        """Return display state string for a given ARGUS decision."""
        return map_to_display_state(decision=decision)

    def get_color(self, status: str) -> tuple[int, int, int]:
        """Return the BGR colour tuple for *status*."""
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
        """Render a single detection overlay on *frame* (mutates in-place).

        Parameters
        ----------
        frame : np.ndarray
            BGR image to draw on.
        box : array-like
            Bounding box as ``[x1, y1, x2, y2]``.
        track_id : int or None
            Tracker-assigned ID.
        identity : str
            Gallery identity string (or ``"UNKNOWN"``).
        score : float
            Match confidence or detector score.
        decision : str
            ARGUS internal decision string.
        camera_id : str
            Camera identifier for the label prefix.
        display_state : str or None
            Explicit display state ("CONFIRMED", "UNCONFIRMED", "SPECIAL_ATTENTION", etc.).
        is_valid : bool or None
            Detection validity flag.
        mobility_state : str
            Observed mobility classification ("STANDARD_WALKING", "WHEELCHAIR", etc.).
        gait_eligible : bool
            Gait biometric eligibility flag.
        is_special_attention : bool
            Explicit operational attention flag.
        """
        if not self.cfg.get("enabled", True):
            return

        if box is None or len(box) < 4:
            return

        x1, y1, x2, y2 = map(int, box[:4])

        # Derive visual assessment state
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

        # Draw bounding box
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
        """Assemble the label string following the spec:

        ``[camera_id] T{track_id} | STATE | identity | score``
        """
        parts: list[str] = []

        if self._show_cam and camera_id:
            parts.append(f"[{camera_id}]")

        if self._show_tid:
            if track_id is not None and track_id >= 0:
                parts.append(f"T{track_id}")
            else:
                parts.append("DET")

        # Display State
        parts.append(display_state)

        # Mobility annotation if non-standard
        if mobility_state == "WHEELCHAIR":
            parts.append("WHEELCHAIR")
        elif mobility_state == "CRUTCHES_AID":
            parts.append("CRUTCHES")

        # Identity
        clean_id = identity if identity and identity not in ("UNKNOWN_PERSON", "UNKNOWN") else "UNKNOWN"
        parts.append(clean_id)

        # Score
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
        """Draw a text label with background rectangle above *(x, y)*."""
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

        # Text color matches box color or white for high contrast
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
