"""
Stage 3: Multi-Camera Evidence Fusion Engine.

Performs score-level evidence fusion across multiple cameras for a shared track identity,
incorporating gait similarity, appearance/ReID, open-set margin, temporal verification,
track reliability, camera transition probability, travel-time likelihood, and occlusion/quality weights.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple


class FusionState(str, Enum):
    CONFIRMED = "CONFIRMED"
    DEFERRED = "DEFERRED"
    UNKNOWN = "UNKNOWN"


@dataclass
class CameraObservationRecord:
    camera_id: str
    local_track_id: Any
    global_track_id: Optional[str]
    identity_candidate: str
    gait_similarity: float
    appearance_similarity: float
    open_set_margin: float
    temporal_consistency: float
    track_reliability: float
    transition_score: float
    travel_time_likelihood: float
    quality_score: float
    occlusion_score: float
    timestamp: float


@dataclass
class MultiCameraFusionResult:
    fused_identity: str
    fused_score: float
    fusion_state: FusionState
    component_scores: Dict[str, float]
    contributing_cameras: List[str]
    reason: str


class MultiCameraEvidenceFusion:
    """
    Score-level multi-camera evidence fusion engine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.evidence_ttl_seconds = float(cfg.get("evidence_ttl_seconds", 15.0))
        self.minimum_cameras = int(cfg.get("minimum_cameras", 2))
        self.minimum_fused_score = float(cfg.get("minimum_fused_score", 0.85))

        weights_cfg = cfg.get("weights", {})
        default_weights = {
            "gait": 0.30,
            "appearance": 0.15,
            "open_set": 0.15,
            "temporal": 0.15,
            "reliability": 0.15,
            "transition": 0.10,
        }
        if isinstance(weights_cfg, dict):
            default_weights.update(weights_cfg)

        tot = sum(default_weights.values())
        if tot > 0:
            self.weights = {k: float(v) / float(tot) for k, v in default_weights.items()}
        else:
            self.weights = default_weights

        # (global_track_id or entity_key) -> list of CameraObservationRecord
        self.observations: Dict[str, List[CameraObservationRecord]] = {}

    def is_enabled(self) -> bool:
        return self.enabled

    @classmethod
    def from_config(cls, config: Dict[str, Any] | None = None) -> "MultiCameraEvidenceFusion":
        """Factory method to instantiate from config dictionary."""
        return cls(config=config)

    def add_observation(
        self,
        camera_id: str,
        local_track_id: Any,
        global_track_id: Optional[str],
        identity_candidate: str,
        gait_similarity: float,
        appearance_similarity: float = 0.0,
        open_set_margin: float = 0.05,
        temporal_consistency: float = 0.85,
        track_reliability: float = 0.80,
        transition_score: float = 1.0,
        travel_time_likelihood: float = 1.0,
        quality_score: float = 0.80,
        occlusion_score: float = 0.10,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record an observation for multi-camera fusion."""
        now = timestamp if timestamp is not None else time.monotonic()
        entity_key = global_track_id or f"{camera_id}_{local_track_id}"

        record = CameraObservationRecord(
            camera_id=camera_id,
            local_track_id=local_track_id,
            global_track_id=global_track_id,
            identity_candidate=identity_candidate,
            gait_similarity=gait_similarity,
            appearance_similarity=appearance_similarity,
            open_set_margin=open_set_margin,
            temporal_consistency=temporal_consistency,
            track_reliability=track_reliability,
            transition_score=transition_score,
            travel_time_likelihood=travel_time_likelihood,
            quality_score=quality_score,
            occlusion_score=occlusion_score,
            timestamp=now,
        )

        if entity_key not in self.observations:
            self.observations[entity_key] = []

        buf = self.observations[entity_key]

        # Prevent duplicate evidence from exact same camera and timestamp
        dup = False
        for existing in buf:
            if existing.camera_id == camera_id and abs(existing.timestamp - now) < 1e-4:
                dup = True
                break

        if not dup:
            buf.append(record)

        # TTL eviction
        buf[:] = [r for r in buf if (now - r.timestamp) <= self.evidence_ttl_seconds]

    def fuse_evidence(
        self,
        entity_key: str,
        fallback_identity: str = "UNKNOWN",
        fallback_score: float = 0.0,
        current_time: Optional[float] = None,
    ) -> MultiCameraFusionResult:
        """
        Fuse multi-camera observations for entity_key.
        """
        now = current_time if current_time is not None else time.monotonic()

        if not self.enabled:
            is_confirmed = fallback_score >= 0.85 and fallback_identity != "UNKNOWN"
            return MultiCameraFusionResult(
                fused_identity=fallback_identity,
                fused_score=fallback_score,
                fusion_state=FusionState.CONFIRMED if is_confirmed else FusionState.UNKNOWN,
                component_scores={"single_camera_score": fallback_score},
                contributing_cameras=["cam_00"],
                reason="Multi-camera fusion disabled (single camera fallback)",
            )

        if entity_key not in self.observations:
            return MultiCameraFusionResult(
                fused_identity=fallback_identity,
                fused_score=fallback_score,
                fusion_state=FusionState.UNKNOWN,
                component_scores={},
                contributing_cameras=[],
                reason="No observations recorded for entity",
            )

        # Filter active unexpired observations
        obs_list = [r for r in self.observations[entity_key] if (now - r.timestamp) <= self.evidence_ttl_seconds]
        if not obs_list:
            return MultiCameraFusionResult(
                fused_identity="UNKNOWN",
                fused_score=0.0,
                fusion_state=FusionState.UNKNOWN,
                component_scores={},
                contributing_cameras=[],
                reason="All observations expired",
            )

        # Group by candidate identity (filtering out UNKNOWN)
        valid_obs = [r for r in obs_list if r.identity_candidate != "UNKNOWN"]
        if not valid_obs:
            return MultiCameraFusionResult(
                fused_identity="UNKNOWN",
                fused_score=0.0,
                fusion_state=FusionState.UNKNOWN,
                component_scores={},
                contributing_cameras=list({r.camera_id for r in obs_list}),
                reason="All observations are UNKNOWN",
            )

        candidates: Dict[str, List[CameraObservationRecord]] = {}
        for r in valid_obs:
            candidates.setdefault(r.identity_candidate, []).append(r)

        # Evaluate fused scores for each candidate identity
        identity_scores: Dict[str, Tuple[float, Dict[str, float], List[str]]] = {}

        for identity, records in candidates.items():
            cameras = list({r.camera_id for r in records})

            # Calculate weighted average for each component, weighted by observation quality & low-occlusion
            total_obs_weight = 0.0
            comp_sums = {
                "gait": 0.0,
                "appearance": 0.0,
                "open_set": 0.0,
                "temporal": 0.0,
                "reliability": 0.0,
                "transition": 0.0,
            }

            for r in records:
                # Observation quality weight: high quality + low occlusion
                w_obs = max(0.1, r.quality_score * (1.0 - 0.5 * r.occlusion_score))
                total_obs_weight += w_obs

                comp_sums["gait"] += w_obs * r.gait_similarity
                comp_sums["appearance"] += w_obs * (r.appearance_similarity if r.appearance_similarity > 0 else r.gait_similarity)
                comp_sums["open_set"] += w_obs * min(1.0, max(0.70, r.open_set_margin / 0.05 if r.open_set_margin > 0 else 0.85))
                comp_sums["temporal"] += w_obs * r.temporal_consistency
                comp_sums["reliability"] += w_obs * r.track_reliability
                comp_sums["transition"] += w_obs * (0.5 * r.transition_score + 0.5 * r.travel_time_likelihood)

            comp_scores = {k: float(v / total_obs_weight) for k, v in comp_sums.items()} if total_obs_weight > 0 else comp_sums

            # Combined score using configured feature weights
            fused_score = float(sum(self.weights[k] * comp_scores[k] for k in self.weights))

            identity_scores[identity] = (fused_score, comp_scores, cameras)

        # Select top identity (deterministic tie-breaking: highest score, then alphabetical identity)
        sorted_candidates = sorted(
            identity_scores.items(),
            key=lambda x: (x[1][0], [-ord(c) for c in x[0]]),
            reverse=True,
        )

        best_id, (best_score, best_comps, best_cams) = sorted_candidates[0]

        # Check multi-camera confirmation conditions
        unique_cam_count = len(best_cams)
        if unique_cam_count < self.minimum_cameras:
            return MultiCameraFusionResult(
                fused_identity=best_id,
                fused_score=round(best_score, 4),
                fusion_state=FusionState.DEFERRED,
                component_scores={k: round(v, 4) for k, v in best_comps.items()},
                contributing_cameras=best_cams,
                reason=f"Contributing camera count ({unique_cam_count}) < minimum required ({self.minimum_cameras})",
            )

        if best_score < self.minimum_fused_score:
            return MultiCameraFusionResult(
                fused_identity=best_id,
                fused_score=round(best_score, 4),
                fusion_state=FusionState.DEFERRED,
                component_scores={k: round(v, 4) for k, v in best_comps.items()},
                contributing_cameras=best_cams,
                reason=f"Fused score ({best_score:.4f}) < threshold ({self.minimum_fused_score:.4f})",
            )

        return MultiCameraFusionResult(
            fused_identity=best_id,
            fused_score=round(best_score, 4),
            fusion_state=FusionState.CONFIRMED,
            component_scores={k: round(v, 4) for k, v in best_comps.items()},
            contributing_cameras=best_cams,
            reason="Multi-camera evidence fusion confirmed identity",
        )

    def cleanup_inactive(self, max_idle_seconds: float = 20.0, current_time: Optional[float] = None) -> None:
        """Clean expired observations."""
        now = current_time if current_time is not None else time.monotonic()
        for key, obs in list(self.observations.items()):
            self.observations[key] = [r for r in obs if (now - r.timestamp) <= max_idle_seconds]
            if not self.observations[key]:
                del self.observations[key]
