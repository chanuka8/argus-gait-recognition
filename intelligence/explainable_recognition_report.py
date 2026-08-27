"""
Explainable Recognition Report Generator for ARGUS AI.

Generates concise, evidence-driven reports explaining identity recognition
decisions (confirmed, deferred, watchlist match, identity change, or manual export).
Supports JSON, CSV, and Markdown exports with atomic file writing and duplicate suppression.
Excludes secrets, RTSP credentials, and raw biometric embeddings.
"""

import csv
import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from security_layer.credentials import sanitize_rtsp_url


def load_explainable_reporting_config() -> dict:
    """Load explainable_reporting configuration section from configs/inference.yaml."""
    config_path = Path("configs/inference.yaml")
    defaults = {
        "enabled": False,
        "output_dir": "outputs/reports/explainable",
        "formats": ["json", "csv", "markdown"],
        "report_confirmed": True,
        "report_deferred": True,
        "report_watchlist": True,
        "duplicate_cooldown_seconds": 10.0,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError, ValueError, KeyError):
        return defaults

    section = data.get("explainable_reporting", {})
    if not isinstance(section, dict):
        return defaults

    merged = dict(defaults)
    for key in defaults:
        if key in section:
            merged[key] = section[key]

    return merged


@dataclass
class RecognitionEvidence:
    """Evidence fields collected during recognition decision."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    camera_id: str = "default"
    local_track_id: int | None = None
    global_track_id: str | None = None
    predicted_identity: str = "UNKNOWN"
    final_identity: str = "UNKNOWN"
    final_decision: str = "UNKNOWN"
    decision_reason: str = "Standard evaluation"
    open_set_state: str = "UNKNOWN"
    gait_similarity: float | None = None
    appearance_similarity: float | None = None
    fusion_score: float | None = None
    gait_weight: float | None = None
    appearance_weight: float | None = None
    quality_score: float | None = None
    temporal_decision: str | None = None
    temporal_vote_count: int | None = None
    track_reliability: float | None = None
    occlusion_score: float | None = None
    clean_frame_ratio: float | None = None
    crowd_density: str | None = None
    transition_score: float | None = None
    identity_persistence_score: float | None = None
    watchlist_matched: bool = False
    watchlist_category: str | None = None
    watchlist_priority: str | None = None
    recognition_deferred: bool = False
    defer_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert fields to dictionary, ensuring credential sanitization."""
        raw = asdict(self)
        sanitized = {}
        for k, v in raw.items():
            if isinstance(v, str):
                sanitized[k] = sanitize_rtsp_url(v)
            else:
                sanitized[k] = v
        return sanitized


class ExplainableRecognitionReporter:
    """Thread-safe evidence report generator for recognition decisions."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_explainable_reporting_config()
        self.enabled = bool(self.config.get("enabled", False))
        self.output_dir = Path(self.config.get("output_dir", "outputs/reports/explainable"))
        self.formats = [f.lower() for f in self.config.get("formats", ["json", "csv", "markdown"])]
        self.cooldown = float(self.config.get("duplicate_cooldown_seconds", 10.0))

        self.report_confirmed = bool(self.config.get("report_confirmed", True))
        self.report_deferred = bool(self.config.get("report_deferred", True))
        self.report_watchlist = bool(self.config.get("report_watchlist", True))

        self._lock = threading.Lock()
        self._last_report_times: dict[str, float] = {}

    def should_report(
        self,
        evidence: RecognitionEvidence,
        identity_changed: bool = False,
        force_export: bool = False,
    ) -> bool:
        """Determine if a report should be emitted based on trigger rules and cooldown."""
        if not self.enabled and not force_export:
            return False

        if force_export or identity_changed:
            return True

        if evidence.watchlist_matched and self.report_watchlist:
            return True

        if evidence.recognition_deferred and self.report_deferred:
            return True

        return self.report_confirmed and evidence.final_decision in ("KNOWN", "CONFIRMED")

    def _is_cooldown_active(self, key: str) -> bool:
        """Check if duplicate report cooldown is active."""
        now = time.monotonic()
        with self._lock:
            last_time = self._last_report_times.get(key, 0.0)
            if now - last_time < self.cooldown:
                return True
            self._last_report_times[key] = now
            return False

    def generate_report(
        self,
        evidence: RecognitionEvidence,
        identity_changed: bool = False,
        force_export: bool = False,
    ) -> dict[str, Path] | None:
        """Generate explainable recognition report files atomically."""
        if not self.should_report(evidence, identity_changed=identity_changed, force_export=force_export):
            return None

        cooldown_key = (
            f"{evidence.camera_id}_{evidence.local_track_id}_{evidence.final_identity}_{evidence.final_decision}"
        )
        if not force_export and self._is_cooldown_active(cooldown_key):
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        evidence_dict = evidence.to_dict()

        safe_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        track_str = str(evidence.local_track_id if evidence.local_track_id is not None else "0")
        cam_str = re.sub(r"\W+", "_", str(evidence.camera_id))
        base_name = f"recognition_{cam_str}_{track_str}_{safe_ts}"

        generated_files: dict[str, Path] = {}

        if "json" in self.formats:
            json_path = self.output_dir / f"{base_name}.json"
            self._write_atomic_json(json_path, evidence_dict)
            generated_files["json"] = json_path

        if "csv" in self.formats:
            csv_path = self.output_dir / f"{base_name}.csv"
            self._write_atomic_csv(csv_path, evidence_dict)
            generated_files["csv"] = csv_path

        if "markdown" in self.formats or "md" in self.formats:
            md_path = self.output_dir / f"{base_name}.md"
            self._write_atomic_markdown(md_path, evidence_dict)
            generated_files["markdown"] = md_path

        return generated_files

    def _write_atomic_json(self, target_path: Path, data: dict) -> None:
        """Write JSON atomically using temporary file rename."""
        temp_fd, temp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(temp_path, target_path)

    def _write_atomic_csv(self, target_path: Path, data: dict) -> None:
        """Write CSV evidence key-value rows atomically."""
        temp_fd, temp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
        with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Field", "Value"])
            for key, val in data.items():
                writer.writerow([key, "" if val is None else str(val)])
        os.replace(temp_path, target_path)

    def _write_atomic_markdown(self, target_path: Path, data: dict) -> None:
        """Write Markdown evidence document atomically."""
        temp_fd, temp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
        lines = [
            f"# Explainable Recognition Report — Event {data.get('event_id')}",
            "",
            f"**Timestamp**: {data.get('timestamp')}  ",
            f"**Camera ID**: `{data.get('camera_id')}`  ",
            f"**Local Track ID**: `{data.get('local_track_id')}`  ",
            f"**Global Track ID**: `{data.get('global_track_id')}`  ",
            "",
            "## Identity & Decision Summary",
            "",
            f"- **Predicted Identity**: `{data.get('predicted_identity')}`",
            f"- **Final Identity**: `{data.get('final_identity')}`",
            f"- **Final Decision**: `{data.get('final_decision')}`",
            f"- **Decision Reason**: {data.get('decision_reason')}",
            f"- **Open-Set State**: `{data.get('open_set_state')}`",
            "",
            "## Biometric Evidence & Scores",
            "",
            "| Metric / Parameter | Value |",
            "| :--- | :--- |",
            f"| Gait Similarity | {data.get('gait_similarity')} |",
            f"| Appearance Similarity | {data.get('appearance_similarity')} |",
            f"| Dual-Modal Fusion Score | {data.get('fusion_score')} |",
            f"| Gait Weight | {data.get('gait_weight')} |",
            f"| Appearance Weight | {data.get('appearance_weight')} |",
            f"| GEI Quality Score | {data.get('quality_score')} |",
            f"| Temporal Decision | {data.get('temporal_decision')} |",
            f"| Temporal Vote Count | {data.get('temporal_vote_count')} |",
            f"| Track Reliability | {data.get('track_reliability')} |",
            f"| Transition Topology Score | {data.get('transition_score')} |",
            f"| Identity Persistence Score | {data.get('identity_persistence_score')} |",
            "",
            "## Contextual & Operational Flags",
            "",
            f"- **Occlusion Score**: {data.get('occlusion_score')}",
            f"- **Clean Frame Ratio**: {data.get('clean_frame_ratio')}",
            f"- **Crowd Density**: {data.get('crowd_density')}",
            f"- **Watchlist Matched**: `{data.get('watchlist_matched')}` (Category: {data.get('watchlist_category')}, Priority: {data.get('watchlist_priority')})",
            f"- **Recognition Deferred**: `{data.get('recognition_deferred')}` (Reason: {data.get('defer_reason')})",
            "",
            "---",
            "*Report generated automatically by ARGUS AI Explainable Recognition Engine.*",
        ]
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(temp_path, target_path)
