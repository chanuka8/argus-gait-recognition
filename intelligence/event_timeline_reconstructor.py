"""
Event Timeline Reconstruction Module for ARGUS AI.

Accumulates spatial-temporal surveillance events across camera streams into
chronological trajectories per global track or identity.
Supports deduplication, event pruning, retention cleanup, thread-safe mutations,
and atomic export to JSON, CSV, and Markdown formats.
Excludes secrets, RTSP credentials, and raw biometric embeddings.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

import yaml

from security_layer.credentials import sanitize_rtsp_url


def load_event_timeline_config() -> dict:
    """Load event_timeline configuration section from configs/inference.yaml."""
    config_path = Path("configs/inference.yaml")
    defaults = {
        "enabled": False,
        "output_dir": "outputs/reports/timelines",
        "formats": ["json", "csv", "markdown"],
        "export_on_track_close": True,
        "export_on_watchlist": True,
        "retention_seconds": 3600.0,
        "maximum_events_per_track": 500,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    section = data.get("event_timeline", {})
    if not isinstance(section, dict):
        return defaults

    merged = dict(defaults)
    for key, val in defaults.items():
        if key in section:
            merged[key] = section[key]

    return merged


@dataclass
class TimelineEvent:
    """Individual event payload in a track timeline."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_type: str = "UNKNOWN"
    camera_id: str = "default"
    local_track_id: Optional[int] = None
    global_track_id: Optional[str] = None
    identity_id: str = "UNKNOWN"
    confidence: Optional[float] = None
    reliability: Optional[float] = None
    transition_score: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary with sanitized strings."""
        raw = asdict(self)
        sanitized = {}
        for k, v in raw.items():
            if isinstance(v, str):
                sanitized[k] = sanitize_rtsp_url(v)
            elif isinstance(v, dict):
                clean_meta = {}
                for mk, mv in v.items():
                    if isinstance(mv, str):
                        clean_meta[mk] = sanitize_rtsp_url(mv)
                    elif not isinstance(mv, (list, tuple, dict)):
                        clean_meta[mk] = mv
                sanitized[k] = clean_meta
            else:
                sanitized[k] = v
        return sanitized


class EventTimelineReconstructor:
    """Thread-safe event timeline accumulator and exporter."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_event_timeline_config()
        self.enabled = bool(self.config.get("enabled", False))
        self.output_dir = Path(self.config.get("output_dir", "outputs/reports/timelines"))
        self.formats = [f.lower() for f in self.config.get("formats", ["json", "csv", "markdown"])]
        self.export_on_track_close = bool(self.config.get("export_on_track_close", True))
        self.export_on_watchlist = bool(self.config.get("export_on_watchlist", True))
        self.retention_seconds = float(self.config.get("retention_seconds", 3600.0))
        self.max_events = int(self.config.get("maximum_events_per_track", 500))

        self._lock = threading.Lock()
        self._timelines: Dict[str, List[TimelineEvent]] = {}
        self._last_updated: Dict[str, float] = {}

    def _get_key(self, global_track_id: Optional[str], camera_id: str, local_track_id: Optional[int]) -> str:
        """Resolve a unique trajectory key."""
        if global_track_id:
            return f"global_{global_track_id}"
        loc_id = local_track_id if local_track_id is not None else 0
        safe_cam = sanitize_rtsp_url(camera_id)
        cam_clean = re.sub(r"\W+", "_", safe_cam)
        return f"track_{cam_clean}_{loc_id}"

    def record_event(
        self,
        event_type: str,
        camera_id: str = "default",
        local_track_id: Optional[int] = None,
        global_track_id: Optional[str] = None,
        identity_id: str = "UNKNOWN",
        confidence: Optional[float] = None,
        reliability: Optional[float] = None,
        transition_score: Optional[float] = None,
        reason: str = "",
        metadata: Optional[dict] = None,
    ) -> Optional[TimelineEvent]:
        """Record an event into the trajectory timeline, enforcing bounds and deduplication."""
        if not self.enabled:
            return None

        event = TimelineEvent(
            event_type=event_type,
            camera_id=camera_id,
            local_track_id=local_track_id,
            global_track_id=global_track_id,
            identity_id=identity_id,
            confidence=confidence,
            reliability=reliability,
            transition_score=transition_score,
            reason=reason,
            metadata=metadata or {},
        )

        key = self._get_key(global_track_id, camera_id, local_track_id)
        now = time.monotonic()

        should_export = False
        with self._lock:
            if key not in self._timelines:
                self._timelines[key] = []
            events = self._timelines[key]

            # Deduplicate identical consecutive event
            if events:
                last = events[-1]
                if (
                    last.event_type == event.event_type
                    and last.camera_id == event.camera_id
                    and last.identity_id == event.identity_id
                    and last.reason == event.reason
                ):
                    return None

            events.append(event)
            if len(events) > self.max_events:
                events.pop(0)

            self._last_updated[key] = now

            if event_type == "TRACK_CLOSED" and self.export_on_track_close:
                should_export = True
            elif event_type == "WATCHLIST_MATCH" and self.export_on_watchlist:
                should_export = True

        if should_export:
            self.export_timeline(key)

        return event

    def cleanup_expired_tracks(self) -> int:
        """Purge trajectories that have been idle longer than retention_seconds."""
        now = time.monotonic()
        expired_keys = []

        with self._lock:
            for key, last_ts in list(self._last_updated.items()):
                if now - last_ts > self.retention_seconds:
                    expired_keys.append(key)
                    self._timelines.pop(key, None)
                    self._last_updated.pop(key, None)

        return len(expired_keys)

    def get_timeline(self, key_or_track_id: str) -> List[TimelineEvent]:
        """Retrieve copy of event sequence for a given trajectory key."""
        with self._lock:
            return list(self._timelines.get(str(key_or_track_id), []))

    def export_timeline(self, key_or_track_id: str) -> Optional[Dict[str, Path]]:
        """Export timeline trajectory to configured file formats atomically."""
        key = str(key_or_track_id)
        events_copy = self.get_timeline(key)
        if not events_copy:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_key = re.sub(r"\W+", "_", key)
        safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"timeline_{safe_key}_{safe_ts}"

        event_dicts = [e.to_dict() for e in events_copy]
        generated_files: Dict[str, Path] = {}

        if "json" in self.formats:
            json_path = self.output_dir / f"{base_name}.json"
            self._write_atomic_json(json_path, key, event_dicts)
            generated_files["json"] = json_path

        if "csv" in self.formats:
            csv_path = self.output_dir / f"{base_name}.csv"
            self._write_atomic_csv(csv_path, event_dicts)
            generated_files["csv"] = csv_path

        if "markdown" in self.formats or "md" in self.formats:
            md_path = self.output_dir / f"{base_name}.md"
            self._write_atomic_markdown(md_path, key, event_dicts)
            generated_files["markdown"] = md_path

        return generated_files

    def _write_atomic_json(self, target_path: Path, track_key: str, events: List[dict]) -> None:
        """Write JSON trajectory file atomically."""
        payload = {
            "track_key": track_key,
            "event_count": len(events),
            "created_at": datetime.now().isoformat(),
            "events": events,
        }
        temp_fd, temp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(temp_path, target_path)

    def _write_atomic_csv(self, target_path: Path, events: List[dict]) -> None:
        """Write CSV trajectory file atomically."""
        temp_fd, temp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
        fields = [
            "event_id",
            "timestamp",
            "event_type",
            "camera_id",
            "local_track_id",
            "global_track_id",
            "identity_id",
            "confidence",
            "reliability",
            "transition_score",
            "reason",
        ]
        with os.fdopen(temp_fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for ev in events:
                writer.writerow([ev.get(col, "") for col in fields])
        os.replace(temp_path, target_path)

    def _write_atomic_markdown(self, target_path: Path, track_key: str, events: List[dict]) -> None:
        """Write Markdown timeline file atomically."""
        temp_fd, temp_path = tempfile.mkstemp(dir=str(target_path.parent), suffix=".tmp")
        lines = [
            f"# Event Timeline Reconstruction — Trajectory `{track_key}`",
            "",
            f"**Event Count**: {len(events)}  ",
            f"**Generated At**: {datetime.now().isoformat()}  ",
            "",
            "| Event # | Timestamp | Event Type | Camera | Track ID | Identity | Confidence | Reliability | Reason |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for idx, ev in enumerate(events, 1):
            conf_str = f"{ev.get('confidence'):.4f}" if isinstance(ev.get("confidence"), float) else "-"
            rel_str = f"{ev.get('reliability'):.4f}" if isinstance(ev.get("reliability"), float) else "-"
            lines.append(
                f"| {idx} | {ev.get('timestamp')} | `{ev.get('event_type')}` | `{ev.get('camera_id')}` | "
                f"`{ev.get('local_track_id')}` | `{ev.get('identity_id')}` | {conf_str} | {rel_str} | {ev.get('reason')} |"
            )

        lines.extend([
            "",
            "---",
            "*Timeline reconstruction generated automatically by ARGUS AI.*",
        ])

        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(temp_path, target_path)
