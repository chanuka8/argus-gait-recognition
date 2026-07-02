"""
ARGUS Detection Display Renderer.

Professional CCTV-style overlay engine for all pipeline modes.
Maps internal ARGUS decision states to a 4-tier visual status system
(DETECTION / TRACKING / UNKNOWN / CONFIRMED) and renders stabilized
bounding boxes with filled label backgrounds.

No model weights, recognition thresholds, or matching math are
modified or referenced by this module.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_DETECTION = "DETECTION"
STATUS_TRACKING = "TRACKING"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_CONFIRMED = "CONFIRMED"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_display_config() -> dict:
    """Load the ``display`` section from ``configs/inference.yaml``.

    Returns safe defaults when the file or section is absent, so the
    renderer is always usable even without explicit configuration.
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
            "detection": [0, 0, 255],
            "tracking": [0, 165, 255],
            "unknown": [0, 255, 0],
            "confirmed": [0, 255, 0],
        },
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
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
                merged["colors"] = {
                    ck: raw_colors.get(ck, cv)
                    for ck, cv in default_value.items()
                }
        else:
            merged[key] = section.get(key, default_value)

    return merged


# ---------------------------------------------------------------------------
# Decision → CCTV status mapping
# ---------------------------------------------------------------------------

def _map_decision_to_status(decision: str) -> str:
    """Map an internal ARGUS decision string to one of 4 CCTV statuses.

    +---------------------------------+----------------+
    | ARGUS internal decision         | CCTV status    |
    +---------------------------------+----------------+
    | COLLECTING                      | DETECTION      |
    | TRACKING                        | TRACKING       |
    | REVIEW_REQUIRED, LOW_CONFIDENCE | TRACKING       |
    | UNKNOWN_PERSON                  | UNKNOWN        |
    | CONFIRMED_MATCH, VERIFIED_MATCH | CONFIRMED      |
    +---------------------------------+----------------+
    """
    if decision == "COLLECTING":
        return STATUS_DETECTION
    if decision == "TRACKING":
        return STATUS_TRACKING
    if decision in ("REVIEW_REQUIRED", "LOW_CONFIDENCE"):
        return STATUS_TRACKING
    if decision == "UNKNOWN_PERSON":
        return STATUS_UNKNOWN
    if decision in ("CONFIRMED_MATCH", "VERIFIED_MATCH"):
        return STATUS_CONFIRMED
    # Fallback: treat anything else as DETECTION
    return STATUS_DETECTION


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class DetectionDisplayRenderer:
    """Draws professional CCTV-style overlays on video frames.

    The renderer is **stateless** with respect to recognition logic;
    it receives pre-computed results and only handles drawing.

    Parameters
    ----------
    config : dict, optional
        Display configuration dictionary.  When *None*, ``load_display_config``
        is called automatically so the renderer always has safe defaults.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.cfg = config if config is not None else load_display_config()

        # Pre-resolve colour tuples (BGR ints) for each status tier.
        colors_raw = self.cfg.get("colors", {})
        self._colors: dict[str, tuple[int, int, int]] = {
            STATUS_DETECTION: tuple(colors_raw.get("detection", [0, 0, 255])),
            STATUS_TRACKING: tuple(colors_raw.get("tracking", [0, 165, 255])),
            STATUS_UNKNOWN: tuple(colors_raw.get("unknown", [0, 255, 0])),
            STATUS_CONFIRMED: tuple(colors_raw.get("confirmed", [0, 255, 0])),
        }

        self._thickness: int = int(self.cfg.get("line_thickness", 2))
        self._font_scale: float = float(self.cfg.get("font_scale", 0.6))
        self._show_cam: bool = bool(self.cfg.get("show_camera_id", True))
        self._show_tid: bool = bool(self.cfg.get("show_track_id", True))
        self._show_score: bool = bool(self.cfg.get("show_score", True))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_status(self, decision: str) -> str:
        """Return the CCTV status string for a given ARGUS decision."""
        return _map_decision_to_status(decision)

    def get_color(self, status: str) -> tuple[int, int, int]:
        """Return the BGR colour tuple for *status*."""
        return self._colors.get(status, (0, 255, 255))

    # ------------------------------------------------------------------
    # Main draw entry-point
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        box,
        track_id: int,
        identity: str,
        score: float,
        decision: str,
        camera_id: str = "cam_00",
    ) -> None:
        """Render a single detection overlay on *frame* (mutates in-place).

        Parameters
        ----------
        frame : np.ndarray
            BGR image to draw on.
        box : array-like
            Bounding box as ``[x1, y1, x2, y2]``.
        track_id : int
            Tracker-assigned ID.
        identity : str
            Gallery identity string (or ``"UNKNOWN"``).
        score : float
            Match confidence score.
        decision : str
            ARGUS internal decision string (e.g. ``"CONFIRMED_MATCH"``).
        camera_id : str
            Camera identifier for the label prefix.
        """
        if not self.cfg.get("enabled", True):
            return

        x1, y1, x2, y2 = map(int, box)
        status = self.get_status(decision)
        color = self.get_color(status)

        # --- Bounding box ---------------------------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._thickness)

        # --- Label assembly --------------------------------------------------
        label = self._build_label(
            camera_id=camera_id,
            track_id=track_id,
            status=status,
            identity=identity,
            score=score,
        )

        # --- Draw filled label background + text ----------------------------
        self._draw_label(frame, label, x1, y1, color)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_label(
        self,
        camera_id: str,
        track_id: int,
        status: str,
        identity: str,
        score: float,
    ) -> str:
        """Assemble the label string following the spec:

        ``[camera_id] T{track_id} | STATUS | identity | score``
        """
        parts: list[str] = []

        if self._show_cam:
            parts.append(f"[{camera_id}]")

        if self._show_tid:
            parts.append(f"T{track_id}")

        # Separator before status
        if parts:
            label_prefix = " ".join(parts) + " | "
        else:
            label_prefix = ""

        # Display identity: for DETECTION/TRACKING show the status itself
        if status in (STATUS_DETECTION, STATUS_TRACKING):
            display_name = status
        else:
            display_name = identity

        score_part = ""
        if self._show_score:
            score_part = f" | {score:.2f}"

        return f"{label_prefix}{status} | {display_name}{score_part}"

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw *text* with a filled background rectangle above the box."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self._font_scale
        thickness = max(1, self._thickness - 1)

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Position the label above the bounding box top-left
        label_y = max(th + baseline + 4, y - 6)
        bg_y1 = label_y - th - baseline - 2
        bg_y2 = label_y + 2
        bg_x2 = x + tw + 6

        # Filled background rectangle (semi-dark)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, bg_y1), (bg_x2, bg_y2), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Text
        cv2.putText(
            frame,
            text,
            (x + 3, label_y - baseline),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
