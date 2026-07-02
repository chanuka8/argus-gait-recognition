"""
ARGUS Detection Reporter.

Thread-safe automated detection report writer that produces JSONL and CSV
logs plus optional person-crop snapshots.  A configurable cooldown per
(camera_id, track_id, identity, status) key prevents per-frame report spam.

Only *UNKNOWN* and *CONFIRMED* events are reported by default; this is
fully configurable via ``configs/inference.yaml → reporting``.

No model weights, recognition thresholds, or matching math are
modified or referenced by this module.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import csv
import json
import threading
import time

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_reporting_config() -> dict:
    """Load the ``reporting`` section from ``configs/inference.yaml``.

    Returns safe defaults when the file or section is absent.
    """
    config_path = Path("configs/inference.yaml")

    defaults: dict = {
        "enabled": True,
        "output_dir": "outputs/detection_reports",
        "save_jsonl": True,
        "save_csv": True,
        "save_snapshots": True,
        "snapshot_dir": "outputs/detection_reports/snapshots",
        "cooldown_seconds": 10,
        "report_detection": False,
        "report_tracking": False,
        "report_unknown": True,
        "report_confirmed": True,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    section = data.get("reporting", {})
    if not isinstance(section, dict):
        return defaults

    merged: dict = {}
    for key, default_value in defaults.items():
        merged[key] = section.get(key, default_value)

    return merged


# ---------------------------------------------------------------------------
# CSV field order
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "timestamp",
    "date",
    "time",
    "camera_id",
    "location",
    "source_mode",
    "track_id",
    "person_name",
    "identity",
    "status",
    "score",
    "bbox",
    "snapshot_path",
]


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class DetectionReporter:
    """Thread-safe detection event reporter with cooldown deduplication.

    Parameters
    ----------
    config : dict, optional
        Reporting configuration.  When *None*, ``load_reporting_config``
        is called automatically.
    source_mode : str
        Pipeline mode identifier (``"live"``, ``"video"``, ``"multi-camera"``).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        source_mode: str = "live",
    ) -> None:
        self.cfg = config if config is not None else load_reporting_config()
        self.source_mode = source_mode

        self._enabled: bool = bool(self.cfg.get("enabled", True))
        self._save_jsonl: bool = bool(self.cfg.get("save_jsonl", True))
        self._save_csv: bool = bool(self.cfg.get("save_csv", True))
        self._save_snapshots: bool = bool(self.cfg.get("save_snapshots", True))
        self._cooldown: float = float(self.cfg.get("cooldown_seconds", 10))

        # Status reporting flags
        self._report_flags: dict[str, bool] = {
            "DETECTION": bool(self.cfg.get("report_detection", False)),
            "TRACKING": bool(self.cfg.get("report_tracking", False)),
            "UNKNOWN": bool(self.cfg.get("report_unknown", True)),
            "CONFIRMED": bool(self.cfg.get("report_confirmed", True)),
        }

        # Output paths
        output_dir = Path(self.cfg.get("output_dir", "outputs/detection_reports"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path: Path = output_dir / "detections.jsonl"
        self._csv_path: Path = output_dir / "detections.csv"
        self._snapshot_dir: Path = Path(
            self.cfg.get("snapshot_dir", "outputs/detection_reports/snapshots"),
        )

        if self._save_snapshots:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.Lock()

        # Cooldown tracker: key → last report unix timestamp
        self._cooldown_map: dict[str, float] = {}

        # Initialise CSV header if needed
        if self._save_csv and not self._csv_path.exists():
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                writer.writeheader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(
        self,
        camera_id: str,
        location: str,
        track_id: int,
        identity: str,
        status: str,
        score: float,
        bbox: list[int],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """Record a detection event if it passes status and cooldown filters.

        Parameters
        ----------
        camera_id : str
            Camera identifier.
        location : str
            Camera location string (from ``cameras.yaml``).
        track_id : int
            Tracker-assigned ID.
        identity : str
            Gallery identity string (or ``"UNKNOWN"``).
        status : str
            CCTV status: ``DETECTION``, ``TRACKING``, ``UNKNOWN``, ``CONFIRMED``.
        score : float
            Match confidence score.
        bbox : list[int]
            Bounding box ``[x1, y1, x2, y2]``.
        frame : np.ndarray, optional
            Current video frame for snapshot cropping.

        Returns
        -------
        bool
            ``True`` if the event was actually written; ``False`` if
            filtered or on cooldown.
        """
        if not self._enabled:
            return False

        # Status gate
        if not self._report_flags.get(status, False):
            return False

        # Cooldown gate
        cooldown_key = f"{camera_id}:{track_id}:{identity}:{status}"
        now = time.time()

        with self._lock:
            last_time = self._cooldown_map.get(cooldown_key, 0.0)
            if now - last_time < self._cooldown:
                return False
            self._cooldown_map[cooldown_key] = now

        # Build record
        ts = datetime.now()
        snapshot_path = ""

        if self._save_snapshots and frame is not None:
            snapshot_path = self._save_snapshot(
                frame, bbox, camera_id, track_id, ts,
            )

        record = {
            "timestamp": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "time": ts.strftime("%H:%M:%S"),
            "camera_id": camera_id,
            "location": location,
            "source_mode": self.source_mode,
            "track_id": track_id,
            "person_name": identity,
            "identity": identity,
            "status": status,
            "score": round(float(score), 4),
            "bbox": bbox,
            "snapshot_path": snapshot_path,
        }

        with self._lock:
            if self._save_jsonl:
                self._write_jsonl(record)
            if self._save_csv:
                self._write_csv(record)

        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: dict) -> None:
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_csv(self, record: dict) -> None:
        row = dict(record)
        # Serialise bbox list to string for CSV
        row["bbox"] = str(row["bbox"])
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writerow(row)

    def _save_snapshot(
        self,
        frame: np.ndarray,
        bbox: list[int],
        camera_id: str,
        track_id: int,
        ts: datetime,
    ) -> str:
        """Crop and save a JPEG snapshot of the detected person region."""
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                return ""

            crop = frame[y1:y2, x1:x2]
            filename = (
                f"{camera_id}_T{track_id}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            path = self._snapshot_dir / filename
            cv2.imwrite(str(path), crop)
            return str(path)
        except Exception:
            return ""
