from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intelligence.crowd_occlusion_analyzer import compute_iou
from monitoring.logging_config import get_logger


class TrackLifecycleState(str, enum.Enum):
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    PROCESSING = "PROCESSING"
    IDENTIFIED = "IDENTIFIED"
    UNKNOWN = "UNKNOWN"
    TEMPORARILY_MISSING = "TEMPORARILY_MISSING"
    RECOVERED = "RECOVERED"
    EXPIRED = "EXPIRED"


class PersonAssessmentState(str, enum.Enum):
    CONFIRMED = "CONFIRMED"
    UNCONFIRMED = "UNCONFIRMED"
    SPECIAL_ATTENTION = "SPECIAL_ATTENTION"
    ASSESSING = "ASSESSING"
    BIOMETRIC_INAPPLICABLE = "BIOMETRIC_INAPPLICABLE"


class MobilityState(str, enum.Enum):
    STANDARD_WALKING = "STANDARD_WALKING"
    WHEELCHAIR = "WHEELCHAIR"
    CRUTCHES_AID = "CRUTCHES_AID"
    STATIONARY_SEATED = "STATIONARY_SEATED"
    NON_STANDARD_GAIT = "NON_STANDARD_GAIT"


@dataclass
class PersonTrackContext:
    camera_id: str
    track_id: int
    state: TrackLifecycleState = TrackLifecycleState.DETECTED
    assessment_state: PersonAssessmentState = PersonAssessmentState.UNCONFIRMED
    mobility_state: MobilityState = MobilityState.STANDARD_WALKING
    gait_eligible: bool = True
    appearance_eligible: bool = True
    gait_usability_reason: str = "ELIGIBLE"
    appearance_usability_reason: str = "ELIGIBLE"
    bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    detection_confidence: float = 0.0
    track_confidence: float = 0.0
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    frame_count: int = 0
    quality_score: float = 1.0
    occlusion_level: str = "LOW"
    consecutive_missing_frames: int = 0


    appearance_embedding: np.ndarray | None = None
    appearance_identity: str = "UNKNOWN_PERSON"
    appearance_score: float = 0.0
    appearance_last_frame: int = 0


    gait_embedding: np.ndarray | None = None
    gait_identity: str = "UNKNOWN_PERSON"
    gait_score: float = 0.0
    gait_last_frame: int = 0


    fused_identity: str = "UNKNOWN_PERSON"
    fused_score: float = 0.0
    decision: str = "UNKNOWN"
    status: str = "UNKNOWN"


    details: dict[str, Any] = field(default_factory=dict)
    identity_history: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.state in (
            TrackLifecycleState.DETECTED,
            TrackLifecycleState.TRACKING,
            TrackLifecycleState.PROCESSING,
            TrackLifecycleState.IDENTIFIED,
            TrackLifecycleState.UNKNOWN,
            TrackLifecycleState.RECOVERED,
        )

    def is_expired(self) -> bool:
        return self.state == TrackLifecycleState.EXPIRED

    def evaluate_display_state(self) -> str:
        if (
            self.status in ("CONFIRMED", "MATCH", "VERIFIED_MATCH")
            or self.decision in ("CONFIRMED", "CONFIRMED_MATCH", "MATCH", "VERIFIED_MATCH")
        ) and self.fused_identity not in ("UNKNOWN_PERSON", "UNKNOWN", ""):
            self.assessment_state = PersonAssessmentState.CONFIRMED
            return PersonAssessmentState.CONFIRMED.value

        if self.fused_identity not in ("UNKNOWN_PERSON", "UNKNOWN", "") and self.fused_score >= 0.85:
            self.assessment_state = PersonAssessmentState.CONFIRMED
            return PersonAssessmentState.CONFIRMED.value


        if self.details.get("special_attention", False) or self.details.get("security_alert", False):
            self.assessment_state = PersonAssessmentState.SPECIAL_ATTENTION
            return PersonAssessmentState.SPECIAL_ATTENTION.value


        if not self.gait_eligible and (
            self.mobility_state in (
                MobilityState.WHEELCHAIR,
                MobilityState.CRUTCHES_AID,
                MobilityState.STATIONARY_SEATED,
                MobilityState.NON_STANDARD_GAIT,
            )
            or not self.appearance_eligible
        ):
            self.assessment_state = PersonAssessmentState.BIOMETRIC_INAPPLICABLE
            return PersonAssessmentState.BIOMETRIC_INAPPLICABLE.value


        self.assessment_state = PersonAssessmentState.UNCONFIRMED
        return PersonAssessmentState.UNCONFIRMED.value

    def update_appearance(
        self,
        embedding: np.ndarray | None,
        identity: str,
        score: float,
        frame_index: int,
    ) -> None:
        if embedding is not None:
            self.appearance_embedding = embedding
        self.appearance_identity = identity
        self.appearance_score = float(score)
        self.appearance_last_frame = frame_index

    def update_gait(
        self,
        embedding: np.ndarray | None,
        identity: str,
        score: float,
        frame_index: int,
    ) -> None:
        if embedding is not None:
            self.gait_embedding = embedding
        self.gait_identity = identity
        self.gait_score = float(score)
        self.gait_last_frame = frame_index

    def update_fusion(
        self,
        identity: str,
        score: float,
        decision: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.fused_identity = identity
        self.fused_score = float(score)
        self.decision = decision
        self.status = status
        if details:
            self.details.update(details)
        if identity and identity not in ("UNKNOWN_PERSON", "UNKNOWN") and status == "CONFIRMED":
            self.state = TrackLifecycleState.IDENTIFIED
            if len(self.identity_history) < 16:
                self.identity_history.append(identity)
        elif self.state not in (TrackLifecycleState.IDENTIFIED, TrackLifecycleState.EXPIRED):
            self.state = TrackLifecycleState.TRACKING
        self.evaluate_display_state()


class ConcurrentTrackManager:
    def __init__(
        self,
        max_idle_seconds: float = 5.0,
        max_missing_frames: int = 30,
        recovery_iou_threshold: float = 0.30,
        recovery_time_window_seconds: float = 3.0,
    ) -> None:
        self.max_idle_seconds = float(max_idle_seconds)
        self.max_missing_frames = int(max_missing_frames)
        self.recovery_iou_threshold = float(recovery_iou_threshold)
        self.recovery_time_window_seconds = float(recovery_time_window_seconds)

        self._lock = threading.RLock()
        self._tracks: dict[tuple[str, int], PersonTrackContext] = {}
        self._recently_lost_tracks: dict[tuple[str, int], PersonTrackContext] = {}
        self._logger = get_logger("concurrent_track_manager")


        self._total_created_tracks = 0
        self._total_recovered_tracks = 0
        self._total_expired_tracks = 0

    def get_track(self, camera_id: str, track_id: int) -> PersonTrackContext | None:
        key = (camera_id, int(track_id))
        with self._lock:
            return self._tracks.get(key)

    def get_active_tracks(self, camera_id: str | None = None) -> list[PersonTrackContext]:
        with self._lock:
            if camera_id is not None:
                return [t for (cid, _), t in self._tracks.items() if cid == camera_id and t.is_active()]
            return [t for t in self._tracks.values() if t.is_active()]

    def get_all_tracks(self, camera_id: str | None = None) -> list[PersonTrackContext]:
        with self._lock:
            if camera_id is not None:
                return [t for (cid, _), t in self._tracks.items() if cid == camera_id]
            return list(self._tracks.values())

    def update_or_create_track(
        self,
        camera_id: str,
        track_id: int,
        bbox: list[int],
        confidence: float = 1.0,
        quality: float = 1.0,
        occlusion_level: str = "LOW",
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> PersonTrackContext:
        now = timestamp if timestamp is not None else time.monotonic()
        key = (camera_id, int(track_id))

        with self._lock:
            track = self._tracks.get(key)
            if track is None:

                recovered_ctx = self._attempt_recovery(camera_id, int(track_id), bbox, now)
                if recovered_ctx is not None:
                    track = recovered_ctx
                    track.track_id = int(track_id)
                    track.state = TrackLifecycleState.RECOVERED
                    track.last_seen = now
                    track.consecutive_missing_frames = 0
                    track.bbox = [int(b) for b in bbox]
                    track.detection_confidence = float(confidence)
                    track.track_confidence = float(quality)
                    track.frame_count += 1
                    self._tracks[key] = track
                    self._total_recovered_tracks += 1
                    self._logger.debug(
                        f"Track recovered: Cam '{camera_id}' Track {track_id} "
                        f"(Previous ID {recovered_ctx.track_id}, identity '{track.fused_identity}')"
                    )
                    return track


                track = PersonTrackContext(
                    camera_id=camera_id,
                    track_id=int(track_id),
                    state=TrackLifecycleState.TRACKING,
                    bbox=[int(b) for b in bbox],
                    detection_confidence=float(confidence),
                    track_confidence=float(quality),
                    first_seen=now,
                    last_seen=now,
                    frame_count=1,
                    quality_score=float(quality),
                    occlusion_level=occlusion_level,
                )
                self._tracks[key] = track
                self._total_created_tracks += 1
                return track


            track.bbox = [int(b) for b in bbox]
            track.detection_confidence = float(confidence)
            track.track_confidence = float(quality)
            track.quality_score = float(quality)
            track.occlusion_level = occlusion_level
            track.last_seen = now
            track.frame_count += 1
            track.consecutive_missing_frames = 0
            if track.state == TrackLifecycleState.TEMPORARILY_MISSING:
                track.state = (
                    TrackLifecycleState.IDENTIFIED
                    if track.fused_identity != "UNKNOWN_PERSON"
                    else TrackLifecycleState.TRACKING
                )
            return track

    def mark_missing_tracks(
        self,
        camera_id: str,
        current_active_track_ids: set[int],
        timestamp: float | None = None,
    ) -> list[PersonTrackContext]:
        missing_tracks: list[PersonTrackContext] = []

        with self._lock:
            for (cid, tid), track in list(self._tracks.items()):
                if cid != camera_id:
                    continue
                if tid not in current_active_track_ids and track.is_active():
                    track.consecutive_missing_frames += 1
                    track.state = TrackLifecycleState.TEMPORARILY_MISSING
                    missing_tracks.append(track)


                    self._recently_lost_tracks[(cid, tid)] = track

        return missing_tracks

    def _attempt_recovery(
        self,
        camera_id: str,
        new_track_id: int,
        new_bbox: list[int],
        now: float,
    ) -> PersonTrackContext | None:
        best_candidate_key = None
        best_iou = self.recovery_iou_threshold

        for (cid, tid), lost_track in list(self._recently_lost_tracks.items()):
            if cid != camera_id or tid == new_track_id:
                continue
            if (now - lost_track.last_seen) > self.recovery_time_window_seconds:
                continue

            iou = compute_iou(lost_track.bbox, new_bbox)
            if iou > best_iou:
                best_iou = iou
                best_candidate_key = (cid, tid)

        if best_candidate_key is not None:
            candidate = self._recently_lost_tracks.pop(best_candidate_key)

            self._tracks.pop(best_candidate_key, None)
            return candidate

        return None

    def cleanup_expired_tracks(
        self,
        max_idle_seconds: float | None = None,
        cleanup_callbacks: list[Callable[[str, int], None]] | None = None,
        timestamp: float | None = None,
    ) -> list[tuple[str, int]]:
        now = timestamp if timestamp is not None else time.monotonic()
        idle_threshold = max_idle_seconds if max_idle_seconds is not None else self.max_idle_seconds
        expired_keys: list[tuple[str, int]] = []

        with self._lock:

            for key, track in list(self._tracks.items()):
                if (now - track.last_seen) > idle_threshold or track.consecutive_missing_frames > self.max_missing_frames:
                    track.state = TrackLifecycleState.EXPIRED
                    expired_keys.append(key)
                    self._tracks.pop(key, None)
                    self._total_expired_tracks += 1


            for key, lost_track in list(self._recently_lost_tracks.items()):
                if (now - lost_track.last_seen) > self.recovery_time_window_seconds:
                    self._recently_lost_tracks.pop(key, None)


        if cleanup_callbacks and expired_keys:
            for cid, tid in expired_keys:
                for cb in cleanup_callbacks:
                    try:
                        cb(cid, tid)
                    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
                        self._logger.debug(f"Track cleanup callback error for ({cid}, {tid}): {exc}")

        return expired_keys

    def clear_camera(
        self,
        camera_id: str,
        cleanup_callbacks: list[Callable[[str, int], None]] | None = None,
    ) -> int:
        removed_count = 0
        with self._lock:
            keys_to_remove = [k for k in self._tracks if k[0] == camera_id]
            for k in keys_to_remove:
                self._tracks.pop(k, None)
                removed_count += 1

            lost_to_remove = [k for k in self._recently_lost_tracks if k[0] == camera_id]
            for k in lost_to_remove:
                self._recently_lost_tracks.pop(k, None)

        if cleanup_callbacks and keys_to_remove:
            for cid, tid in keys_to_remove:
                for cb in cleanup_callbacks:
                    try:
                        cb(cid, tid)
                    except (RuntimeError, ValueError, TypeError, OSError) as exc:
                        self._logger.debug(f"Camera clear callback error: {exc}")

        return removed_count

    def clear_all(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._recently_lost_tracks.clear()
            self._total_created_tracks = 0
            self._total_recovered_tracks = 0
            self._total_expired_tracks = 0

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            active = [t for t in self._tracks.values() if t.is_active()]
            missing = [t for t in self._tracks.values() if t.state == TrackLifecycleState.TEMPORARILY_MISSING]
            identified = [t for t in active if t.state == TrackLifecycleState.IDENTIFIED]

            cam_distribution: dict[str, int] = {}
            for t in active:
                cam_distribution[t.camera_id] = cam_distribution.get(t.camera_id, 0) + 1

            return {
                "active_tracks_count": len(active),
                "temporarily_missing_count": len(missing),
                "identified_tracks_count": len(identified),
                "total_created_tracks": self._total_created_tracks,
                "total_recovered_tracks": self._total_recovered_tracks,
                "total_expired_tracks": self._total_expired_tracks,
                "camera_track_distribution": cam_distribution,
                "recently_lost_buffered": len(self._recently_lost_tracks),
            }
