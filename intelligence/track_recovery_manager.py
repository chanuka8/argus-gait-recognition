"""
Stage 5: Deterministic Track Recovery Engine.

Maintains bounded recently-lost track states and performs deterministic ID-switch checks
and track recovery using spatial IoU, bounding box dimensions, and historical evidence.
"""

import time
from dataclasses import dataclass
from typing import Any

from intelligence.crowd_occlusion_analyzer import compute_iou
from monitoring.logging_config import get_logger


@dataclass
class LostTrackRecord:
    camera_id: str
    track_id: int
    last_bbox: list[int]
    last_seen_time: float
    identity: str = "UNKNOWN"
    feature_vector: Any | None = None
    quality: float = 1.0


class TrackRecoveryManager:
    """
    Deterministic track recovery manager for bridging temporary lost-track gaps in crowded scenes.
    """

    def __init__(
        self,
        max_lost_seconds: float = 3.0,
        max_buffered_tracks: int = 50,
        min_recovery_iou: float = 0.30,
    ) -> None:
        self.logger = get_logger("track_recovery")
        self.max_lost_seconds = max_lost_seconds
        self.max_buffered_tracks = max_buffered_tracks
        self.min_recovery_iou = min_recovery_iou

        self.lost_tracks: dict[tuple[str, int], LostTrackRecord] = {}

    def register_lost_track(
        self,
        camera_id: str,
        track_id: int,
        last_bbox: list[int],
        identity: str = "UNKNOWN",
        feature_vector: Any | None = None,
        quality: float = 1.0,
        timestamp: float | None = None,
    ) -> None:
        """Register a track that was lost by ByteTrack tracker."""
        now = timestamp if timestamp is not None else time.monotonic()
        key = (camera_id, int(track_id))

        record = LostTrackRecord(
            camera_id=camera_id,
            track_id=int(track_id),
            last_bbox=last_bbox,
            last_seen_time=now,
            identity=identity,
            feature_vector=feature_vector,
            quality=quality,
        )

        if len(self.lost_tracks) >= self.max_buffered_tracks:
            oldest_key = min(self.lost_tracks.keys(), key=lambda k: self.lost_tracks[k].last_seen_time)
            del self.lost_tracks[oldest_key]

        self.lost_tracks[key] = record

    def attempt_recovery(
        self,
        camera_id: str,
        new_track_id: int,
        new_bbox: list[int],
        timestamp: float | None = None,
    ) -> LostTrackRecord | None:
        """
        Attempt to match a newly initialized track ID to a recently lost track.

        Returns LostTrackRecord if matched, or None.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        key = (camera_id, int(new_track_id))

        if key in self.lost_tracks:
            return self.lost_tracks[key]

        self.cleanup_expired(now)

        best_match: LostTrackRecord | None = None
        best_score = 0.0

        w_new = max(1, new_bbox[2] - new_bbox[0])
        h_new = max(1, new_bbox[3] - new_bbox[1])
        aspect_new = float(w_new / h_new)

        for (cam, tid), record in self.lost_tracks.items():
            if cam != camera_id:
                continue

            dt = now - record.last_seen_time
            if dt > self.max_lost_seconds:
                continue

            iou = compute_iou(new_bbox, record.last_bbox)
            if iou < self.min_recovery_iou:
                continue

            w_old = max(1, record.last_bbox[2] - record.last_bbox[0])
            h_old = max(1, record.last_bbox[3] - record.last_bbox[1])
            aspect_old = float(w_old / h_old)

            aspect_sim = 1.0 - abs(aspect_new - aspect_old) / max(0.1, aspect_old)
            match_score = 0.7 * iou + 0.3 * max(0.0, aspect_sim)

            if match_score > best_score:
                best_score = match_score
                best_match = record

        if best_match:
            old_key = (best_match.camera_id, best_match.track_id)
            self.lost_tracks.pop(old_key, None)
            return best_match

        return None

    def cleanup_expired(self, current_time: float | None = None) -> None:
        """Purge expired lost tracks."""
        now = current_time if current_time is not None else time.monotonic()
        for key, record in list(self.lost_tracks.items()):
            if (now - record.last_seen_time) > self.max_lost_seconds:
                del self.lost_tracks[key]
