import json
import threading
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
    TRAINING_CONSUMED = "TRAINING_CONSUMED"
    REJECTED = "REJECTED"


@dataclass
class OperationalObservation:
    observation_id: str
    camera_id: str
    track_id: int
    modality: str
    embedding_dim: int
    vector: list[float]
    predicted_identity: str
    confidence: float
    state: ObservationState = ObservationState.PREDICTED
    verified_identity: str | None = None
    verification_source: str | None = None
    quality_score: float = 1.0
    model_name: str = ""
    model_version: str = "v1.0.0"
    identity_type: str = "LIVE_OPERATIONAL"
    source_type: str = "live_surveillance"
    training_consumed: bool = False
    consumed_in_job: str = ""
    consumed_by_model: str = ""
    created_at: float = field(default_factory=time.time)
    observation_date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    gei_image: Any | None = None
    crop_image: Any | None = None

    def __post_init__(self) -> None:
        if not self.observation_date:
            self.observation_date = time.strftime("%Y-%m-%d", time.gmtime(self.created_at))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["observation_date"] = self.observation_date
        d["model_name"] = self.model_name
        d["model_version"] = self.model_version
        d.pop("gei_image", None)
        d.pop("crop_image", None)
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
            state=ObservationState(st) if isinstance(st, str) and st in [e.value for e in ObservationState] else ObservationState.PREDICTED,
            verified_identity=data.get("verified_identity"),
            verification_source=data.get("verification_source"),
            quality_score=float(data.get("quality_score", 1.0)),
            model_name=str(data.get("model_name", "")),
            model_version=str(data.get("model_version", "v1.0.0")),
            identity_type=str(data.get("identity_type", "LIVE_OPERATIONAL")),
            source_type=str(data.get("source_type", "live_surveillance")),
            training_consumed=bool(data.get("training_consumed", False)),
            consumed_in_job=str(data.get("consumed_in_job", "")),
            consumed_by_model=str(data.get("consumed_by_model", "")),
            created_at=c_at,
            observation_date=obs_date,
            metadata=dict(data.get("metadata", {})),
        )


class OperationalEmbeddingCollector:
    def __init__(
        self,
        output_dir: str = "data/operational_observations",
        max_buffer_size: int = 1000,
        dedup_window_seconds: float = 1.0,
        dedup_similarity_threshold: float = 0.98,
        evidence_manager: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_buffer_size = max_buffer_size
        self.dedup_window_seconds = float(dedup_window_seconds)
        self.dedup_similarity_threshold = float(dedup_similarity_threshold)
        if evidence_manager is None:
            try:
                from intelligence.operational_evidence_manager import OperationalEvidenceManager

                self.evidence_manager = OperationalEvidenceManager()
            except (ImportError, RuntimeError, OSError):
                self.evidence_manager = None
        else:
            self.evidence_manager = evidence_manager
        self._buffer: list[OperationalObservation] = []
        self._logger = get_logger("operational_collector")
        self._lock = threading.RLock()
        self._load_recent()

    def _load_recent(self) -> None:
        with self._lock:
            obs_file = self.output_dir / "recent_observations.json"
            if obs_file.exists():
                try:
                    with open(obs_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._buffer = [OperationalObservation.from_dict(d) for d in data]
                except (OSError, json.JSONDecodeError, ValueError) as err:
                    self._logger.warning(f"Failed to load recent observations: {err}")

    def _flush(self) -> None:
        with self._lock:
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
        model_name: str = "",
        model_version: str = "v1.0.0",
        metadata: dict[str, Any] | None = None,
        media_array: np.ndarray | None = None,
    ) -> OperationalObservation:
        vec_arr = np.asarray(vector, dtype=np.float32).ravel() if vector is not None else np.array([], dtype=np.float32)

        is_finite = bool(np.isfinite(vec_arr).all()) if vec_arr.size > 0 else False
        is_valid_dim = bool(vec_arr.size in (256, 512))
        norm = float(np.linalg.norm(vec_arr)) if is_finite and vec_arr.size > 0 else 0.0

        if is_finite and is_valid_dim and norm > 1e-6:
            vec_arr = vec_arr / norm
        else:
            quality_score = 0.0

        now = time.time()
        obs_date = observation_date or time.strftime("%Y-%m-%d", time.gmtime(now))

        with self._lock:
            if quality_score > 0.0 and vec_arr.size > 0:
                for past_obs in reversed(self._buffer[-50:]):
                    if (
                        past_obs.camera_id == camera_id
                        and past_obs.track_id == track_id
                        and past_obs.modality == modality
                        and (now - past_obs.created_at) < self.dedup_window_seconds
                    ):
                        past_vec = np.asarray(past_obs.vector, dtype=np.float32)
                        if past_vec.size == vec_arr.size and np.isfinite(past_vec).all():
                            sim = float(np.dot(vec_arr, past_vec))
                            if sim >= self.dedup_similarity_threshold:
                                past_obs.created_at = now
                                if metadata:
                                    past_obs.metadata.update(metadata)
                                return past_obs

            meta_dict = dict(metadata or {})
            if "bbox" in meta_dict and isinstance(meta_dict["bbox"], (list, tuple)) and len(meta_dict["bbox"]) >= 4:
                b = meta_dict["bbox"]
                bw = max(1.0, float(b[2] - b[0]))
                bh = max(1.0, float(b[3] - b[1]))
                ar = round(bh / bw, 2)
                meta_dict["aspect_ratio"] = ar
                if ar >= 2.6:
                    meta_dict.setdefault("viewpoint_class", "frontal")
                elif ar <= 1.9:
                    meta_dict.setdefault("viewpoint_class", "profile")
                else:
                    meta_dict.setdefault("viewpoint_class", "oblique")

            if media_array is not None and isinstance(media_array, np.ndarray):
                try:
                    meta_dict.setdefault("mean_intensity", round(float(np.mean(media_array)), 2))
                    meta_dict.setdefault("media_shape", list(media_array.shape))
                except (TypeError, ValueError, AttributeError) as arr_err:
                    self._logger.debug(f"Could not compute media stats: {arr_err}")

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
                model_name=model_name,
                model_version=model_version,
                identity_type="LIVE_OPERATIONAL",
                source_type="live_surveillance",
                created_at=now,
                observation_date=obs_date,
                metadata=meta_dict,
                gei_image=media_array if modality == "gait" else None,
                crop_image=media_array if modality == "appearance" else None,
            )

            if media_array is not None and self.evidence_manager is not None:
                try:
                    self.evidence_manager.store_evidence(
                        observation_id=obs.observation_id,
                        camera_id=camera_id,
                        track_id=track_id,
                        person_id=predicted_identity,
                        modality=modality,
                        media_array=media_array,
                        session_id=str(meta_dict.get("session_id", "")),
                        condition_metadata=meta_dict,
                    )
                except (RuntimeError, ValueError, TypeError, OSError) as ev_err:
                    self._logger.debug(f"Failed to store evidence array: {ev_err}")

            self._buffer.append(obs)
            if len(self._buffer) > self.max_buffer_size:
                self._buffer.pop(0)

            if len(self._buffer) % 10 == 0 or len(self._buffer) <= 5:
                self._flush()

            return obs

    def verify_observation(
        self,
        observation_id: str,
        verified_identity: str,
        verification_source: str = "operator_confirmation",
        verification_metadata: dict[str, Any] | None = None,
    ) -> bool:
        with self._lock:
            for obs in self._buffer:
                if obs.observation_id == observation_id:
                    if obs.state == ObservationState.TRAINING_CONSUMED:
                        self._logger.warning(
                            f"[INVALID_STATE_TRANSITION] Cannot re-verify already consumed observation '{observation_id}'"
                        )
                        return False

                    obs.verified_identity = verified_identity
                    obs.verification_source = verification_source
                    obs.state = ObservationState.VERIFIED
                    if verification_metadata:
                        obs.metadata.update(verification_metadata)

                    vec_arr = np.asarray(obs.vector, dtype=np.float32)
                    is_valid = (
                        obs.quality_score >= 0.70
                        and np.isfinite(vec_arr).all()
                        and vec_arr.size in (256, 512)
                        and float(np.linalg.norm(vec_arr)) > 0.0
                        and obs.identity_type != "USER_REFERENCE"
                    )
                    if is_valid:
                        obs.state = ObservationState.TRAINING_ELIGIBLE

                    self._flush()
                    self._logger.info(
                        f"Observation '{observation_id}' transitioned to {obs.state.value} "
                        f"(ID: {verified_identity}, Date: {obs.observation_date})"
                    )
                    return True
            return False

    def mark_training_consumed(
        self,
        observation_ids: list[str] | set[str],
        training_job_id: str,
        candidate_version: str,
    ) -> int:
        consumed_count = 0
        with self._lock:
            id_set = set(observation_ids)
            for obs in self._buffer:
                if obs.observation_id in id_set:
                    obs.state = ObservationState.TRAINING_CONSUMED
                    obs.training_consumed = True
                    obs.consumed_in_job = training_job_id
                    obs.consumed_by_model = candidate_version
                    consumed_count += 1
            if consumed_count > 0:
                self._flush()
                self._logger.info(
                    f"[TRAINING_CONSUMPTION] Marked {consumed_count} observations as TRAINING_CONSUMED "
                    f"by job '{training_job_id}' (candidate: {candidate_version})"
                )
        return consumed_count

    def get_training_eligible(self, modality: str | None = None) -> list[OperationalObservation]:
        with self._lock:
            eligible = [
                o
                for o in self._buffer
                if o.state == ObservationState.TRAINING_ELIGIBLE
                and not o.training_consumed
                and o.identity_type != "USER_REFERENCE"
            ]
            if modality:
                eligible = [o for o in eligible if o.modality == modality]
            return eligible

    def get_distinct_eligible_dates(self) -> list[str]:
        with self._lock:
            dates = {o.observation_date for o in self.get_training_eligible() if o.observation_date}
            return sorted(dates)

    def get_eligible_by_date(
        self, observation_date: str, modality: str | None = None
    ) -> list[OperationalObservation]:
        with self._lock:
            eligible = [
                o
                for o in self._buffer
                if o.state == ObservationState.TRAINING_ELIGIBLE
                and not o.training_consumed
                and o.identity_type != "USER_REFERENCE"
                and o.observation_date == observation_date
            ]
            if modality:
                eligible = [o for o in eligible if o.modality == modality]
            return eligible

    def get_recent_observations(self, limit: int = 100) -> list[OperationalObservation]:
        with self._lock:
            return list(self._buffer[-limit:])

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            obs_file = self.output_dir / "recent_observations.json"
            if obs_file.exists():
                try:
                    obs_file.unlink()
                except OSError:
                    pass
