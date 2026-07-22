"""Missing person search, continuous monitoring, and evidence triggering workflow."""

import time
from threading import Lock
from typing import Any, Dict, List, Optional

from monitoring.logging_config import get_logger


class MissingPersonWorkflow:
    """Automates missing person target monitoring, alert throttling, and evidence trigger generation."""

    def __init__(self, alert_threshold: float = 0.85, cooldown_seconds: float = 60.0) -> None:
        self.alert_threshold = alert_threshold
        self.cooldown_seconds = cooldown_seconds
        self._logger = get_logger("missing_person_workflow")
        self._lock = Lock()

        # target_identity -> dict of metadata
        self._target_watchlist: Dict[str, Dict[str, Any]] = {}
        # target_identity -> last alert timestamp
        self._last_alerts: Dict[str, float] = {}
        # list of generated event records
        self._events: List[Dict[str, Any]] = []

    def register_target(self, identity: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a missing person target identity to the active watchlist."""
        with self._lock:
            self._target_watchlist[identity] = {
                "identity": identity,
                "registered_at": time.monotonic(),
                "metadata": metadata or {},
            }
            self._logger.info(f"Registered missing person watchlist target: {identity}")

    def unregister_target(self, identity: str) -> bool:
        """Remove a target identity from the active watchlist."""
        with self._lock:
            if identity in self._target_watchlist:
                del self._target_watchlist[identity]
                self._last_alerts.pop(identity, None)
                self._logger.info(f"Unregistered target: {identity}")
                return True
            return False

    def process_match(self, identity: str, confidence_score: float, camera_id: str, gei_data: Optional[Any] = None, frame_data: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Evaluate a gait match against the missing person watchlist."""
        now = time.monotonic()
        with self._lock:
            if identity not in self._target_watchlist:
                return None

            if confidence_score < self.alert_threshold:
                return None

            last_alert = self._last_alerts.get(identity, 0.0)
            if (now - last_alert) < self.cooldown_seconds:
                return None

            self._last_alerts[identity] = now

            event = {
                "event_type": "MISSING_PERSON_MATCH",
                "identity": identity,
                "confidence_score": confidence_score,
                "camera_id": camera_id,
                "timestamp": now,
                "target_info": self._target_watchlist[identity],
                "trigger_evidence": True,
            }
            self._events.append(event)
            self._logger.warning(f"MISSING PERSON MATCH DETECTED! Target={identity}, Score={confidence_score:.4f}, Camera={camera_id}")
            return event

    def get_active_targets(self) -> List[str]:
        """Return list of active watchlist target identities."""
        with self._lock:
            return list(self._target_watchlist.keys())

    def get_recent_events(self) -> List[Dict[str, Any]]:
        """Return all logged missing person match events."""
        with self._lock:
            return self._events.copy()
