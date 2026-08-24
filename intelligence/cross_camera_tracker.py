"""Cross-camera tracking and trajectory continuity across multiple camera feeds."""

import time
import uuid
from threading import Lock
from typing import Any, Callable, Dict, Optional

from intelligence.camera_transition_model import CameraTransitionModel
from monitoring.logging_config import get_logger


class CrossCameraTracker:
    """Manages global track IDs and cross-camera transitions."""

    def __init__(
        self,
        max_transition_time_seconds: float = 60.0,
        transition_model: Optional[CameraTransitionModel] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self.max_transition_time = max_transition_time_seconds
        self.transition_model = transition_model
        self._logger = get_logger("cross_camera_tracker")
        self._lock = Lock()
        self._time_provider = time_provider or time.monotonic

        self._global_tracks: Dict[str, Dict[str, Any]] = {}
        self._local_to_global: Dict[tuple, str] = {}

    def get_or_create_global_id(
        self,
        camera_id: str,
        local_track_id: int,
        identity: Optional[str] = None,
        feature_vector: Optional[Any] = None,
        quality: float = 1.0,
        direction: Optional[str] = None,
        entry_zone: Optional[str] = None,
    ) -> str:
        """Assign or retrieve a global track ID for a camera stream track."""
        now = self._time_provider()
        key = (camera_id, local_track_id)

        with self._lock:
            if key in self._local_to_global:
                gid = self._local_to_global[key]
                self._global_tracks[gid]["last_seen"] = now
                self._global_tracks[gid]["last_camera"] = camera_id
                if identity:
                    self._global_tracks[gid]["identity"] = identity

                if self.transition_model:
                    self.transition_model.record_exit(
                        camera_id=camera_id,
                        local_track_id=local_track_id,
                        global_id=gid,
                        identity=identity,
                        feature_vector=feature_vector,
                        quality=quality,
                        direction=direction,
                        timestamp=now,
                    )
                return gid

            if self.transition_model and self.transition_model.is_enabled():
                match_res = self.transition_model.find_best_transition_candidate(
                    dest_camera_id=camera_id,
                    dest_local_track_id=local_track_id,
                    identity=identity,
                    feature_vector=feature_vector,
                    entry_zone=entry_zone,
                    timestamp=now,
                )
                if match_res:
                    exit_rec, score = match_res
                    candidate_gid = exit_rec.global_id
                    if candidate_gid and candidate_gid in self._global_tracks:
                        gid = candidate_gid
                        old_cam = exit_rec.camera_id
                        data = self._global_tracks[gid]
                        data["last_seen"] = now
                        data["last_camera"] = camera_id
                        if identity:
                            data["identity"] = identity
                        data["transitions"].append({
                            "from": old_cam,
                            "to": camera_id,
                            "timestamp": now,
                            "score": score,
                        })
                        self._local_to_global[key] = gid
                        self._logger.info(
                            f"Transition model match for {identity or gid}: {old_cam} -> {camera_id} "
                            f"(Global ID: {gid}, score: {score:.3f})"
                        )
                        self.transition_model.record_exit(
                            camera_id=camera_id,
                            local_track_id=local_track_id,
                            global_id=gid,
                            identity=identity,
                            feature_vector=feature_vector,
                            quality=quality,
                            direction=direction,
                            timestamp=now,
                        )
                        return gid

            if identity:
                for gid, data in self._global_tracks.items():
                    if data.get("identity") == identity and (now - data["last_seen"]) <= self.max_transition_time:
                        old_cam = data["last_camera"]
                        data["last_seen"] = now
                        data["last_camera"] = camera_id
                        data["transitions"].append({"from": old_cam, "to": camera_id, "timestamp": now})
                        self._local_to_global[key] = gid
                        self._logger.info(f"Transition for {identity}: {old_cam} -> {camera_id} (Global ID: {gid})")

                        if self.transition_model:
                            self.transition_model.record_exit(
                                camera_id=camera_id,
                                local_track_id=local_track_id,
                                global_id=gid,
                                identity=identity,
                                feature_vector=feature_vector,
                                quality=quality,
                                direction=direction,
                                timestamp=now,
                            )
                        return gid

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

            if self.transition_model:
                self.transition_model.record_exit(
                    camera_id=camera_id,
                    local_track_id=local_track_id,
                    global_id=gid,
                    identity=identity,
                    feature_vector=feature_vector,
                    quality=quality,
                    direction=direction,
                    timestamp=now,
                )
            return gid

    def record_track_exit(
        self,
        camera_id: str,
        local_track_id: Any,
        identity: Optional[str] = None,
        feature_vector: Optional[Any] = None,
        quality: float = 1.0,
        exit_zone: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> None:
        """Explicitly record a track exit in the transition model."""
        now = self._time_provider()
        with self._lock:
            key = (camera_id, local_track_id)
            gid = self._local_to_global.get(key)
            if self.transition_model:
                self.transition_model.record_exit(
                    camera_id=camera_id,
                    local_track_id=local_track_id,
                    global_id=gid,
                    identity=identity,
                    feature_vector=feature_vector,
                    quality=quality,
                    exit_zone=exit_zone,
                    direction=direction,
                    timestamp=now,
                )

    def get_track_history(self, global_track_id: str) -> Optional[Dict[str, Any]]:
        """Get history and transition log for a global track ID."""
        with self._lock:
            track = self._global_tracks.get(global_track_id)
            return track.copy() if track else None

    def cleanup_stale_tracks(self, max_age_seconds: float = 300.0) -> int:
        """Remove tracks inactive past max age."""
        now = self._time_provider()
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

            if self.transition_model:
                self.transition_model.cleanup_stale_exits(max_age_seconds)

        return removed

