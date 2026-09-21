from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


class TrackReliabilityScorer:
    def __init__(
        self,
        enabled: bool = False,
        weights: dict[str, float] | None = None,
        target_observation_frames: int = 15,
        min_reliability_threshold: float = 0.50,
        high_reliability_threshold: float = 0.80,
    ) -> None:
        self.enabled = enabled
        self.logger = get_logger("track_reliability_scorer")
        self.target_observation_frames = max(1, int(target_observation_frames))
        self.min_reliability_threshold = float(min_reliability_threshold)
        self.high_reliability_threshold = float(high_reliability_threshold)

        default_weights = {
            "quality": 0.25,
            "temporal": 0.25,
            "open_set": 0.25,
            "observation": 0.15,
            "detection": 0.10,
        }
        if weights:
            default_weights.update(weights)

        total_w = sum(default_weights.values())
        if total_w > 0:
            self.weights = {k: float(v) / float(total_w) for k, v in default_weights.items()}
        else:
            self.weights = default_weights

    def is_enabled(self) -> bool:
        return self.enabled

    def _compute_quality_subscore(
        self,
        quality_score: float | dict[str, Any] | None,
    ) -> float:
        if quality_score is None:
            return 1.0
        if isinstance(quality_score, dict):
            val = quality_score.get("overall_quality", quality_score.get("quality", 1.0))
            return float(np.clip(val, 0.0, 1.0))
        return float(np.clip(quality_score, 0.0, 1.0))

    def _compute_temporal_subscore(
        self,
        temporal_decision: str | None,
    ) -> float:
        if not temporal_decision:
            return 0.50

        decision_upper = str(temporal_decision).upper()
        if decision_upper in ("MAJORITY_VOTE", "CONFIRMED"):
            return 1.00
        elif decision_upper == "SINGLE_MATCH":
            return 0.85
        elif decision_upper == "PREVIOUS_IDENTITY":
            return 0.70
        elif decision_upper == "REVIEW_REQUIRED":
            return 0.50
        elif decision_upper in ("UNCERTAIN", "UNKNOWN_PERSON"):
            return 0.30

        return 0.50

    def _compute_open_set_subscore(
        self,
        open_set_state: str | None,
    ) -> float:
        if not open_set_state:
            return 0.00

        state_upper = str(open_set_state).upper()
        if state_upper == "KNOWN":
            return 1.00
        elif state_upper == "UNCERTAIN":
            return 0.30
        elif state_upper == "UNKNOWN":
            return 0.00

        return 0.00

    def _compute_observation_subscore(
        self,
        observation_count: int,
    ) -> float:
        count = max(0, int(observation_count))
        return float(min(1.0, count / float(self.target_observation_frames)))

    def _compute_detection_subscore(
        self,
        detection_confidence: float = 1.0,
        stability_score: float | None = None,
    ) -> float:
        conf = float(np.clip(detection_confidence, 0.0, 1.0))
        if stability_score is not None:
            stab = float(np.clip(stability_score, 0.0, 1.0))
            return 0.6 * conf + 0.4 * stab
        return conf

    def compute_reliability(
        self,
        quality_score: float | dict[str, Any] | None = 1.0,
        temporal_decision: str | None = None,
        open_set_state: str | None = None,
        observation_count: int = 1,
        detection_confidence: float = 1.0,
        persistence_score: float | None = None,
        transition_score: float | None = None,
        stability_score: float | None = None,
        occlusion_score: float | None = None,
        clean_frame_ratio: float | None = None,
    ) -> float:
        s_qual = self._compute_quality_subscore(quality_score)
        s_temp = self._compute_temporal_subscore(temporal_decision)
        s_open = self._compute_open_set_subscore(open_set_state)
        s_obs = self._compute_observation_subscore(observation_count)
        s_det = self._compute_detection_subscore(detection_confidence, stability_score)

        reliability = (
            s_qual * self.weights["quality"]
            + s_temp * self.weights["temporal"]
            + s_open * self.weights["open_set"]
            + s_obs * self.weights["observation"]
            + s_det * self.weights["detection"]
        )

        if persistence_score is not None:
            p_score = float(np.clip(persistence_score, 0.0, 1.0))
            reliability = 0.85 * reliability + 0.15 * p_score

        if transition_score is not None:
            t_score = float(np.clip(transition_score, 0.0, 1.0))
            reliability = 0.85 * reliability + 0.15 * t_score

        if occlusion_score is not None:
            occ_penalty = float(np.clip(occlusion_score, 0.0, 1.0))
            reliability *= 1.0 - 0.3 * occ_penalty

        if clean_frame_ratio is not None:
            clean_factor = float(np.clip(clean_frame_ratio, 0.0, 1.0))
            reliability *= 0.7 + 0.3 * clean_factor

        final_score = float(np.clip(reliability, 0.0, 1.0))
        return final_score

    def evaluate_track(
        self,
        quality_score: float | dict[str, Any] | None = 1.0,
        temporal_decision: str | None = None,
        open_set_state: str | None = None,
        observation_count: int = 1,
        detection_confidence: float = 1.0,
        persistence_score: float | None = None,
        transition_score: float | None = None,
        stability_score: float | None = None,
    ) -> dict[str, Any]:
        s_qual = self._compute_quality_subscore(quality_score)
        s_temp = self._compute_temporal_subscore(temporal_decision)
        s_open = self._compute_open_set_subscore(open_set_state)
        s_obs = self._compute_observation_subscore(observation_count)
        s_det = self._compute_detection_subscore(detection_confidence, stability_score)

        reliability = self.compute_reliability(
            quality_score=quality_score,
            temporal_decision=temporal_decision,
            open_set_state=open_set_state,
            observation_count=observation_count,
            detection_confidence=detection_confidence,
            persistence_score=persistence_score,
            transition_score=transition_score,
            stability_score=stability_score,
        )

        identity_confidence = 0.5 * s_temp + 0.5 * s_open
        track_stability = (
            s_qual * self.weights["quality"] + s_obs * self.weights["observation"] + s_det * self.weights["detection"]
        ) / max(1e-5, (self.weights["quality"] + self.weights["observation"] + self.weights["detection"]))

        if reliability >= self.high_reliability_threshold:
            level = "HIGH"
        elif reliability >= self.min_reliability_threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "reliability_score": round(reliability, 4),
            "level": level,
            "is_reliable": reliability >= self.min_reliability_threshold,
            "identity_confidence": round(identity_confidence, 4),
            "track_stability": round(track_stability, 4),
            "components": {
                "quality": round(s_qual, 4),
                "temporal": round(s_temp, 4),
                "open_set": round(s_open, 4),
                "observation": round(s_obs, 4),
                "detection": round(s_det, 4),
            },
        }
