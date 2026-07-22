"""Identity persistence engine for tracking, confidence accumulation, and alert suppression."""

import time
from threading import Lock
from typing import Any, Dict, Optional

from monitoring.logging_config import get_logger


class IdentityPersistence:
    """Manages persistent identity state, confidence accumulation, and duplicate suppression."""

    def __init__(self, suppression_window_seconds: float = 30.0, score_accumulation_decay: float = 0.9) -> None:
        self.suppression_window = suppression_window_seconds
        self.decay = score_accumulation_decay
        self._logger = get_logger("identity_persistence")
        self._lock = Lock()

        # identity_id -> {"accumulated_score": float, "last_seen": float, "detections": int, "history": list}
        self._identities: Dict[str, Dict[str, Any]] = {}
        # identity_id -> last alert timestamp
        self._alert_cooldowns: Dict[str, float] = {}

    def update_identity(self, identity: str, confidence_score: float, camera_id: str = "") -> Dict[str, Any]:
        """Update confidence score and history for a recognized identity."""
        now = time.monotonic()
        with self._lock:
            if identity not in self._identities:
                self._identities[identity] = {
                    "identity": identity,
                    "accumulated_score": confidence_score,
                    "detections": 1,
                    "first_seen": now,
                    "last_seen": now,
                    "history": [],
                }
            else:
                data = self._identities[identity]
                data["accumulated_score"] = (data["accumulated_score"] * self.decay) + (confidence_score * (1.0 - self.decay))
                data["detections"] += 1
                data["last_seen"] = now

            entry = {
                "timestamp": now,
                "score": confidence_score,
                "camera_id": camera_id,
            }
            self._identities[identity]["history"].append(entry)
            if len(self._identities[identity]["history"]) > 100:
                self._identities[identity]["history"] = self._identities[identity]["history"][-100:]

            return self._identities[identity].copy()

    def should_suppress_alert(self, identity: str) -> bool:
        """Check if an alert for identity should be suppressed due to cooldown."""
        now = time.monotonic()
        with self._lock:
            last_alert = self._alert_cooldowns.get(identity, 0.0)
            if (now - last_alert) < self.suppression_window:
                return True
            self._alert_cooldowns[identity] = now
            return False

    def get_identity_state(self, identity: str) -> Optional[Dict[str, Any]]:
        """Get current accumulated state for an identity."""
        with self._lock:
            state = self._identities.get(identity)
            return state.copy() if state else None

    def get_all_identities(self) -> Dict[str, Dict[str, Any]]:
        """Get states for all active identities."""
        with self._lock:
            return {id_: data.copy() for id_, data in self._identities.items()}
