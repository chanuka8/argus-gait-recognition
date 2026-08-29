"""
Operational Embedding Observation Collector for ARGUS AI.

Collects and categorizes CCTV inference observations without storing raw CCTV video:
- OBSERVED: Feature vector captured from live video stream.
- PREDICTED: Inference engine output (identity + confidence score).
- VERIFIED: Operator or high-consensus multi-factor ground truth (manually or securely verified).
- TRAINING_ELIGIBLE: Verified observation meeting quality and diversity gates for candidate calibration.

Safety Invariant:
Never automatically promotes PREDICTED -> VERIFIED to prevent unvalidated self-reinforcing model drift.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


class ObservationState(str, Enum):
    OBSERVED = "OBSERVED"
    PREDICTED = "PREDICTED"
    VERIFIED = "VERIFIED"
    TRAINING_ELIGIBLE = "TRAINING_ELIGIBLE"


@dataclass
class OperationalObservation:
    """Represents a single derived biometric observation from live surveillance."""

    observation_id: str
    camera_id: str
    track_id: int
    modality: str  # "gait", "appearance", "dual"
    embedding_dim: int
    vector: list[float]  # L2-normalized feature representation (NO raw video stored)
    predicted_identity: str
    confidence: float
    state: ObservationState = ObservationState.PREDICTED
    verified_identity: str | None = None
    verification_source: str | None = None
    quality_score: float = 1.0
    created_at: float = field(default_factory=time.time)
    observation_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observation_date:
            self.observation_date = time.strftime("%Y-%m-%d", time.gmtime(self.created_at))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["observation_date"] = self.observation_date
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperationalObservation":
        st = data.get("state", "PREDICTED")
        c_at = float(data.get("created_at", time.time()))
        obs_date = str(data.get("observation_date", "")) or time.strftime("%Y-%m-%d", time.gmtime(c_at))
        return cls(
            observation_id=str(data["observation_id"]),
            camera_id=str(data.get("camera_id", "cctv-unknown")),
            track_id=int(data.get("track_id", 0)),
            modality=str(data.get("modality", "gait")),
            embedding_dim=int(data.get("embedding_dim", len(data.get("vector", [])))),
            vector=[float(v) for v in data.get("vector", [])],
            predicted_identity=str(data.get("predicted_identity", "UNKNOWN")),
            confidence=float(data.get("confidence", 0.0)),
            state=ObservationState(st) if isinstance(st, str) else ObservationState.PREDICTED,
            verified_identity=data.get("verified_identity"),
            verification_source=data.get("verification_source"),
            quality_score=float(data.get("quality_score", 1.0)),
            created_at=c_at,
            observation_date=obs_date,
            metadata=dict(data.get("metadata", {})),
        )


class OperationalEmbeddingCollector:
    """
    Asynchronous, non-blocking collector for CCTV inference observations.
    Maintains a ring-buffer in memory and periodic JSON persistence.
    """

    def __init__(
        self,
        output_dir: str = "data/operational_observations",
        max_buffer_size: int = 1000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffer_size = max_buffer_size
        self._buffer: list[OperationalObservation] = []
        self._logger = get_logger("operational_collector")
        self._load_recent()

    def _load_recent(self) -> None:
        obs_file = self.output_dir / "recent_observations.json"
        if obs_file.exists():
            try:
                with open(obs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._buffer = [OperationalObservation.from_dict(d) for d in data]
            except (OSError, json.JSONDecodeError, ValueError) as err:
                self._logger.warning(f"Failed to load recent observations: {err}")

    def _flush(self) -> None:
        obs_file = self.output_dir / "recent_observations.json"
        tmp = obs_file.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump([o.to_dict() for o in self._buffer[-self.max_buffer_size :]], f, indent=2)
            tmp.replace(obs_file)
        except (OSError, ValueError) as err:
            self._logger.error(f"Failed to flush observations: {err}")

    def record_observation(
        self,
        camera_id: str,
        track_id: int,
        vector: np.ndarray | list[float],
        predicted_identity: str,
        confidence: float,
        modality: str = "gait",
        quality_score: float = 1.0,
        observation_date: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationalObservation:
        """
        Record a new observation from live inference. Always enters in PREDICTED state.
        """
        vec_arr = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(vec_arr))
        if norm > 0:
            vec_arr = vec_arr / norm

        now = time.time()
        obs_date = observation_date or time.strftime("%Y-%m-%d", time.gmtime(now))

        obs = OperationalObservation(
            observation_id=f"obs_{int(now)}_{uuid.uuid4().hex[:6]}",
            camera_id=camera_id,
            track_id=track_id,
            modality=modality,
            embedding_dim=len(vec_arr),
            vector=vec_arr.tolist(),
            predicted_identity=predicted_identity,
            confidence=confidence,
            state=ObservationState.PREDICTED,
            quality_score=quality_score,
            created_at=now,
            observation_date=obs_date,
            metadata=metadata or {},
        )

        self._buffer.append(obs)
        if len(self._buffer) > self.max_buffer_size:
            self._buffer.pop(0)

        if len(self._buffer) % 25 == 0:
            self._flush()

        return obs

    def verify_observation(
        self,
        observation_id: str,
        verified_identity: str,
        verification_source: str = "operator_confirmation",
    ) -> bool:
        """
        Explicitly verify an observation with ground truth.
        Only verified observations meeting quality, dimensional, and mathematical
        validity gates can become TRAINING_ELIGIBLE.
        """
        for obs in self._buffer:
            if obs.observation_id == observation_id:
                obs.verified_identity = verified_identity
                obs.verification_source = verification_source
                obs.state = ObservationState.VERIFIED

                vec_arr = np.asarray(obs.vector, dtype=np.float32)
                is_valid = (
                    obs.quality_score >= 0.70
                    and np.isfinite(vec_arr).all()
                    and vec_arr.size in (256, 512)
                    and float(np.linalg.norm(vec_arr)) > 0.0
                )
                if is_valid:
                    obs.state = ObservationState.TRAINING_ELIGIBLE

                self._flush()
                self._logger.info(
                    f"Observation '{observation_id}' marked as {obs.state.value} (ID: {verified_identity}, Date: {obs.observation_date})"
                )
                return True
        return False

    def get_training_eligible(self, modality: str | None = None) -> list[OperationalObservation]:
        """Return all observations eligible for candidate training/calibration."""
        eligible = [o for o in self._buffer if o.state == ObservationState.TRAINING_ELIGIBLE]
        if modality:
            eligible = [o for o in eligible if o.modality == modality]
        return eligible

    def get_distinct_eligible_dates(self) -> list[str]:
        """Return sorted unique observation dates that contain TRAINING_ELIGIBLE observations."""
        dates = {o.observation_date for o in self.get_training_eligible() if o.observation_date}
        return sorted(dates)

    def get_eligible_by_date(self, observation_date: str, modality: str | None = None) -> list[OperationalObservation]:
        """Return all TRAINING_ELIGIBLE observations for a specific observation date (YYYY-MM-DD)."""
        eligible = [
            o
            for o in self._buffer
            if o.state == ObservationState.TRAINING_ELIGIBLE and o.observation_date == observation_date
        ]
        if modality:
            eligible = [o for o in eligible if o.modality == modality]
        return eligible

    def get_recent_observations(self, limit: int = 100) -> list[OperationalObservation]:
        return self._buffer[-limit:]
