"""Missing person search, continuous monitoring, operational watchlist, and evidence triggering workflow."""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from monitoring.logging_config import get_logger


@dataclass
class WatchlistEntry:
    """Represents an operational watchlist entry for real-time recognition."""

    identity_id: str
    category: str = "MISSING_PERSON"
    priority: str = "HIGH"
    enabled: bool = True
    alert_enabled: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary representation."""
        return {
            "identity_id": self.identity_id,
            "category": self.category,
            "priority": self.priority,
            "enabled": self.enabled,
            "alert_enabled": self.alert_enabled,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchlistEntry":
        """Create a WatchlistEntry instance from dictionary."""
        return cls(
            identity_id=str(data.get("identity_id") or data.get("identity", "")),
            category=str(data.get("category", "MISSING_PERSON")),
            priority=str(data.get("priority", "HIGH")),
            enabled=bool(data.get("enabled", True)),
            alert_enabled=bool(data.get("alert_enabled", True)),
            notes=str(data.get("notes", "")),
        )


class MissingPersonWorkflow:
    """Automates missing person target monitoring, watchlist management, alert throttling, and evidence trigger generation."""

    def __init__(
        self,
        alert_threshold: float = 0.85,
        cooldown_seconds: float = 60.0,
        output_dir: str = "outputs/watchlist",
    ) -> None:
        from pathlib import Path

        self.alert_threshold = alert_threshold
        self.cooldown_seconds = cooldown_seconds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("missing_person_workflow")
        self._lock = Lock()

        self._watchlist_entries: dict[str, WatchlistEntry] = {}
        self._target_watchlist: dict[str, dict[str, Any]] = {}
        self._last_alerts: dict[str, float] = {}
        self._events: list[dict[str, Any]] = []

    def register_target(
        self,
        identity: str,
        metadata: dict[str, Any] | None = None,
        category: str = "MISSING_PERSON",
        priority: str = "HIGH",
        enabled: bool = True,
        alert_enabled: bool = True,
        notes: str = "",
    ) -> WatchlistEntry:
        """Add or update a target identity in the active watchlist."""
        with self._lock:
            meta = dict(metadata or {})
            cat = str(meta.get("category", category))
            prio = str(meta.get("priority", priority))
            en = bool(meta.get("enabled", enabled))
            al_en = bool(meta.get("alert_enabled", alert_enabled))
            nts = str(meta.get("notes", notes))

            entry = WatchlistEntry(
                identity_id=identity,
                category=cat,
                priority=prio,
                enabled=en,
                alert_enabled=al_en,
                notes=nts,
            )

            self._watchlist_entries[identity] = entry
            self._target_watchlist[identity] = {
                "identity": identity,
                "registered_at": time.monotonic(),
                "metadata": meta,
                "category": cat,
                "priority": prio,
                "enabled": en,
                "alert_enabled": al_en,
                "notes": nts,
                "watchlist_entry": entry.to_dict(),
            }
            self._logger.info(f"Registered missing person watchlist target: {identity}")
            return entry

    def unregister_target(self, identity: str) -> bool:
        """Remove a target identity from the active watchlist."""
        with self._lock:
            if identity in self._target_watchlist or identity in self._watchlist_entries:
                self._watchlist_entries.pop(identity, None)
                self._target_watchlist.pop(identity, None)
                self._last_alerts.pop(identity, None)
                self._logger.info(f"Unregistered target: {identity}")
                return True
            return False

    def get_entry(self, identity: str) -> WatchlistEntry | None:
        """Return WatchlistEntry object for identity if registered."""
        with self._lock:
            return self._watchlist_entries.get(identity)

    def process_match(
        self,
        identity: str,
        confidence_score: float,
        camera_id: str,
        gei_data: Any | None = None,
        frame_data: Any | None = None,
        track_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Evaluate a gait match against the active watchlist."""
        now = time.monotonic()
        with self._lock:
            if identity not in self._target_watchlist and identity not in self._watchlist_entries:
                return None

            entry = self._watchlist_entries.get(identity)
            if entry and not entry.enabled:
                return None

            if confidence_score < self.alert_threshold:
                return None

            last_alert = self._last_alerts.get(identity, 0.0)
            if (now - last_alert) < self.cooldown_seconds:
                return None

            self._last_alerts[identity] = now
            cat = entry.category if entry else "MISSING_PERSON"
            prio = entry.priority if entry else "HIGH"
            al_en = entry.alert_enabled if entry else True
            nts = entry.notes if entry else ""

            event_type = "MISSING_PERSON_MATCH" if cat == "MISSING_PERSON" else f"WATCHLIST_{cat}_MATCH"

            event = {
                "event_type": event_type,
                "identity": identity,
                "confidence_score": confidence_score,
                "camera_id": camera_id,
                "track_id": track_id,
                "timestamp": now,
                "category": cat,
                "priority": prio,
                "alert_enabled": al_en,
                "notes": nts,
                "target_info": self._target_watchlist.get(identity, {}),
                "trigger_evidence": True,
            }
            self._events.append(event)
            self._logger.warning(
                f"WATCHLIST MATCH DETECTED! Target={identity}, Category={cat}, Score={confidence_score:.4f}, Camera={camera_id}"
            )
            return event

    def get_active_targets(self) -> list[str]:
        """Return list of active watchlist target identities."""
        with self._lock:
            return [k for k, v in self._watchlist_entries.items() if v.enabled] or list(self._target_watchlist.keys())

    def get_recent_events(self) -> list[dict[str, Any]]:
        """Return all logged missing person match events."""
        with self._lock:
            return self._events.copy()

    def export_watchlist_events(self, filename: str = "watchlist_events.json"):
        """Export watchlist match events to output directory."""
        import json

        with self._lock:
            target_path = self.output_dir / filename
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(self._events, f, indent=2)
            return target_path


WatchlistManager = MissingPersonWorkflow
