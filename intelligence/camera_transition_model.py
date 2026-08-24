"""Production-grade Camera Transition Model for spatial-temporal multi-camera tracking."""

from dataclasses import dataclass
from threading import Lock
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from monitoring.logging_config import get_logger


@dataclass
class CameraTransitionRule:
    """Directed transition parameters between source and destination cameras."""

    source_camera: str
    destination_camera: str
    min_travel_seconds: float
    max_travel_seconds: float
    probability: float
    entry_zone: Optional[str] = None
    exit_zone: Optional[str] = None


@dataclass
class ExitRecord:
    """Record of a track exiting or observed on a camera."""

    camera_id: str
    local_track_id: Any
    global_id: Optional[str]
    exit_timestamp: float
    identity: Optional[str] = None
    feature_vector: Optional[Any] = None
    quality: float = 1.0
    exit_zone: Optional[str] = None
    direction: Optional[str] = None


class CameraTransitionModel:
    """Validates camera topology, filters transition candidates, and scores transitions."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self._logger = get_logger("camera_transition_model")
        self._lock = Lock()
        self._time_provider = time_provider or time.monotonic

        self._topology: Dict[str, Dict[str, CameraTransitionRule]] = {}
        self._exits: Dict[Tuple[str, Any], ExitRecord] = {}

        self._weight_identity = 0.6
        self._weight_probability = 0.2
        self._weight_time = 0.2
        self._similarity_threshold = 0.50
        self._max_history_seconds = 300.0
        self._allow_same_camera = False

        if config:
            self.load_config(config)

    def load_config(self, config: Dict[str, Any]) -> None:
        """Parse and validate camera transition topology and scoring weights."""
        with self._lock:
            self._topology.clear()

            if not isinstance(config, dict):
                self._logger.warning("Invalid configuration object; expecting dict.")
                return

            weights = config.get("weights", {})
            if isinstance(weights, dict):
                try:
                    w_id = float(weights.get("identity_similarity", self._weight_identity))
                    w_prob = float(weights.get("transition_probability", self._weight_probability))
                    w_time = float(weights.get("travel_time_likelihood", self._weight_time))

                    total_w = w_id + w_prob + w_time
                    if total_w > 0:
                        self._weight_identity = w_id / total_w
                        self._weight_probability = w_prob / total_w
                        self._weight_time = w_time / total_w
                except (ValueError, TypeError) as e:
                    self._logger.warning(f"Error parsing scoring weights: {e}")

            try:
                self._similarity_threshold = float(config.get("similarity_threshold", self._similarity_threshold))
                self._max_history_seconds = float(config.get("max_history_seconds", self._max_history_seconds))
            except (ValueError, TypeError) as e:
                self._logger.warning(f"Error parsing threshold or history settings: {e}")

            self._allow_same_camera = bool(config.get("allow_same_camera", self._allow_same_camera))

            raw_transitions = config.get("camera_transitions", {})
            if not isinstance(raw_transitions, dict):
                self._logger.warning("Invalid 'camera_transitions' format; expecting dict.")
                return

            for src_cam, dests in raw_transitions.items():
                if src_cam is None or not isinstance(src_cam, str):
                    self._logger.warning(f"Invalid source camera ID: {src_cam}")
                    continue
                src_cam_str = str(src_cam).strip()
                if not src_cam_str:
                    self._logger.warning("Empty source camera ID encountered.")
                    continue

                if not isinstance(dests, dict):
                    self._logger.warning(f"Invalid destination mapping for source camera {src_cam_str}")
                    continue

                for dest_cam, rule_data in dests.items():
                    if dest_cam is None or not isinstance(dest_cam, str):
                        self._logger.warning(f"Invalid destination camera ID for source {src_cam_str}: {dest_cam}")
                        continue
                    dest_cam_str = str(dest_cam).strip()
                    if not dest_cam_str:
                        self._logger.warning(f"Empty destination camera ID for source {src_cam_str}.")
                        continue

                    if not isinstance(rule_data, dict):
                        self._logger.warning(f"Invalid rule data for {src_cam_str} -> {dest_cam_str}")
                        continue

                    try:
                        min_t = float(rule_data.get("min_travel_seconds", 0.0))
                        max_t = float(rule_data.get("max_travel_seconds", 60.0))
                        prob = float(rule_data.get("probability", 1.0))
                    except (ValueError, TypeError) as e:
                        self._logger.warning(f"Failed to parse transition rule numbers for {src_cam_str} -> {dest_cam_str}: {e}")
                        continue

                    if min_t < 0.0:
                        self._logger.warning(f"Invalid min_travel_seconds ({min_t}) for {src_cam_str} -> {dest_cam_str}; must be >= 0.")
                        continue
                    if max_t < min_t:
                        self._logger.warning(f"Invalid max_travel_seconds ({max_t} < {min_t}) for {src_cam_str} -> {dest_cam_str}.")
                        continue
                    if not (0.0 <= prob <= 1.0):
                        self._logger.warning(f"Invalid probability ({prob}) for {src_cam_str} -> {dest_cam_str}; must be in [0.0, 1.0].")
                        continue

                    entry_z = rule_data.get("entry_zone")
                    exit_z = rule_data.get("exit_zone")

                    rule = CameraTransitionRule(
                        source_camera=src_cam_str,
                        destination_camera=dest_cam_str,
                        min_travel_seconds=min_t,
                        max_travel_seconds=max_t,
                        probability=prob,
                        entry_zone=str(entry_z) if entry_z else None,
                        exit_zone=str(exit_z) if exit_z else None,
                    )

                    if src_cam_str not in self._topology:
                        self._topology[src_cam_str] = {}
                    self._topology[src_cam_str][dest_cam_str] = rule

    def is_enabled(self) -> bool:
        """Check if any valid transition topology is loaded."""
        with self._lock:
            return len(self._topology) > 0

    def add_or_update_rule(
        self,
        source_camera: str,
        destination_camera: str,
        min_travel_seconds: float,
        max_travel_seconds: float,
        probability: float,
        entry_zone: str | None = None,
        exit_zone: str | None = None,
    ) -> None:
        """Add or dynamically update a directed transition topology rule online."""
        if not source_camera or not destination_camera:
            return
        with self._lock:
            if source_camera not in self._topology:
                self._topology[source_camera] = {}
            rule = CameraTransitionRule(
                source_camera=str(source_camera),
                destination_camera=str(destination_camera),
                min_travel_seconds=float(max(0.0, min_travel_seconds)),
                max_travel_seconds=float(max(min_travel_seconds, max_travel_seconds)),
                probability=float(max(0.0, min(1.0, probability))),
                entry_zone=str(entry_zone) if entry_zone else None,
                exit_zone=str(exit_zone) if exit_zone else None,
            )
            self._topology[source_camera][destination_camera] = rule

    def record_exit(
        self,
        camera_id: str,
        local_track_id: Any,
        global_id: Optional[str] = None,
        identity: Optional[str] = None,
        feature_vector: Optional[Any] = None,
        quality: float = 1.0,
        exit_zone: Optional[str] = None,
        direction: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record an exit or track observation for a camera track."""
        if not camera_id:
            return

        now = timestamp if timestamp is not None else self._time_provider()
        key = (camera_id, local_track_id)

        record = ExitRecord(
            camera_id=camera_id,
            local_track_id=local_track_id,
            global_id=global_id,
            exit_timestamp=now,
            identity=identity,
            feature_vector=feature_vector,
            quality=quality,
            exit_zone=exit_zone,
            direction=direction,
        )

        with self._lock:
            self._exits[key] = record
            self._cleanup_expired_exits_locked(now)

    def find_best_transition_candidate(
        self,
        dest_camera_id: str,
        dest_local_track_id: Any,
        identity: Optional[str] = None,
        feature_vector: Optional[Any] = None,
        entry_zone: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[Tuple[ExitRecord, float]]:
        """Filter candidates by topology/time-window and score candidates deterministically."""
        now = timestamp if timestamp is not None else self._time_provider()

        with self._lock:
            self._cleanup_expired_exits_locked(now)

            if not self._topology:
                return None

            valid_candidates = []

            for key, exit_rec in self._exits.items():
                src_cam = exit_rec.camera_id

                if src_cam == dest_camera_id:
                    rule = self._topology.get(src_cam, {}).get(dest_camera_id)
                    if not rule or not self._allow_same_camera:
                        continue

                rule = self._topology.get(src_cam, {}).get(dest_camera_id)
                if not rule:
                    continue

                delta_t = now - exit_rec.exit_timestamp
                if delta_t < rule.min_travel_seconds:
                    continue
                if delta_t > rule.max_travel_seconds:
                    continue

                if rule.exit_zone and exit_rec.exit_zone and rule.exit_zone != exit_rec.exit_zone:
                    continue
                if rule.entry_zone and entry_zone and rule.entry_zone != entry_zone:
                    continue

                id_sim = self._calculate_identity_similarity(
                    identity, feature_vector, exit_rec.identity, exit_rec.feature_vector
                )

                time_like = self._calculate_travel_time_likelihood(delta_t, rule.min_travel_seconds, rule.max_travel_seconds)

                trans_prob = rule.probability

                final_score = (
                    self._weight_identity * id_sim
                    + self._weight_probability * trans_prob
                    + self._weight_time * time_like
                )
                final_score = max(0.0, min(1.0, final_score))

                if final_score >= self._similarity_threshold:
                    mid_t = (rule.min_travel_seconds + rule.max_travel_seconds) / 2.0
                    dist_from_mid = abs(delta_t - mid_t)
                    valid_candidates.append({
                        "record": exit_rec,
                        "score": final_score,
                        "dist_from_mid": dist_from_mid,
                        "delta_t": delta_t,
                        "src_key": key,
                    })

            if not valid_candidates:
                return None

            valid_candidates.sort(
                key=lambda c: (
                    -c["score"],
                    c["dist_from_mid"],
                    c["record"].exit_timestamp,
                    str(c["src_key"]),
                )
            )

            best = valid_candidates[0]
            return (best["record"], best["score"])

    def _calculate_identity_similarity(
        self,
        candidate_identity: Optional[str],
        candidate_feature: Optional[Any],
        exit_identity: Optional[str],
        exit_feature: Optional[Any],
    ) -> float:
        """Compute normalized similarity score in [0.0, 1.0]."""
        if candidate_feature is not None and exit_feature is not None:
            try:
                f1 = np.asarray(candidate_feature, dtype=np.float32).flatten()
                f2 = np.asarray(exit_feature, dtype=np.float32).flatten()
                norm1 = float(np.linalg.norm(f1))
                norm2 = float(np.linalg.norm(f2))
                if norm1 > 0 and norm2 > 0:
                    cos_sim = float(np.dot(f1, f2) / (norm1 * norm2))
                    return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))
            except Exception:
                pass

        if candidate_identity and exit_identity:
            if candidate_identity == exit_identity:
                return 1.0
            return 0.0

        return 0.5

    def _calculate_travel_time_likelihood(self, delta_t: float, min_t: float, max_t: float) -> float:
        """Compute triangular travel time likelihood centered in [min_t, max_t]."""
        if max_t <= min_t:
            return 1.0 if abs(delta_t - min_t) < 1e-5 else 0.0

        center = (min_t + max_t) / 2.0
        half_width = (max_t - min_t) / 2.0

        likelihood = 1.0 - (abs(delta_t - center) / half_width)
        return max(0.0, min(1.0, likelihood))

    def _cleanup_expired_exits_locked(self, now: float) -> int:
        """Purge exit records exceeding maximum history window."""
        expired_keys = [
            key for key, rec in self._exits.items()
            if (now - rec.exit_timestamp) > self._max_history_seconds
        ]
        for key in expired_keys:
            del self._exits[key]
        return len(expired_keys)

    def cleanup_stale_exits(self, max_age_seconds: Optional[float] = None) -> int:
        """Public method to purge stale exit records."""
        now = self._time_provider()
        with self._lock:
            cutoff = max_age_seconds if max_age_seconds is not None else self._max_history_seconds
            expired = [
                key for key, rec in self._exits.items()
                if (now - rec.exit_timestamp) > cutoff
            ]
            for key in expired:
                del self._exits[key]
            return len(expired)
