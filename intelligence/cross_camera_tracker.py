"""Cross-camera tracking and trajectory continuity across multiple camera feeds."""

import time
import uuid
from threading import Lock
from typing import Any, Dict, Optional

from monitoring.logging_config import get_logger


class CrossCameraTracker:
    """Manages global track IDs and cross-camera transitions."""

    def __init__(self, max_transition_time_seconds: float = 60.0) -> None:
        self.max_transition_time = max_transition_time_seconds
        self._logger = get_logger("cross_camera_tracker")
        self._lock = Lock()

        # global_track_id -> dict of track info
        self._global_tracks: Dict[str, Dict[str, Any]] = {}
        # (camera_id, local_track_id) -> global_track_id
        self._local_to_global: Dict[tuple, str] = {}

    def get_or_create_global_id(
        self,
        camera_id: str,
        local_track_id: int,
        identity: Optional[str] = None,
        feature_vector: Optional[Any] = None,
    ) -> str:
        """Assign or retrieve a global track ID for a camera stream track."""
        now = time.monotonic()
        key = (camera_id, local_track_id)

        with self._lock:
            if key in self._local_to_global:
                gid = self._local_to_global[key]
                self._global_tracks[gid]["last_seen"] = now
                self._global_tracks[gid]["last_camera"] = camera_id
                if identity:
                    self._global_tracks[gid]["identity"] = identity
                return gid

            # Check if identity was recently seen on another camera for continuity
            if identity:
                for gid, data in self._global_tracks.items():
                    if data.get("identity") == identity and (now - data["last_seen"]) <= self.max_transition_time:
                        old_cam = data["last_camera"]
                        data["last_seen"] = now
                        data["last_camera"] = camera_id
                        data["transitions"].append({"from": old_cam, "to": camera_id, "timestamp": now})
                        self._local_to_global[key] = gid
                        self._logger.info(f"Transition for {identity}: {old_cam} -> {camera_id} (Global ID: {gid})")
                        return gid

            # Create new global track ID
            gid = f"GTRACK-{uuid.uuid4().hex[:8].upper()}"
            self._global_tracks[gid] = {
                "global_id": gid,
                "identity": identity,
                "first_seen": now,
                "last_seen": now,
                "last_camera": camera_id,
                "transitions": [],
            }
            self._local_to_global[key] = gid
            return gid

    def get_track_history(self, global_track_id: str) -> Optional[Dict[str, Any]]:
        """Get history and transition log for a global track ID."""
        with self._lock:
            track = self._global_tracks.get(global_track_id)
            return track.copy() if track else None

    def cleanup_stale_tracks(self, max_age_seconds: float = 300.0) -> int:
        """Remove tracks inactive past max age."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            stale_gids = [
                gid for gid, data in self._global_tracks.items()
                if (now - data["last_seen"]) > max_age_seconds
            ]
            for gid in stale_gids:
                del self._global_tracks[gid]
                removed += 1

            stale_keys = [
                k for k, gid in self._local_to_global.items()
                if gid in stale_gids
            ]
            for k in stale_keys:
                del self._local_to_global[k]

        return removed
