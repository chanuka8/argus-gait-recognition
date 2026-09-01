import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from intelligence.operational_embedding_collector import (
    ObservationState,
    OperationalEmbeddingCollector,
)
from intelligence.operational_evidence_manager import (
    OperationalEvidenceManager,
)
from monitoring.logging_config import get_logger
from storage.embedding_database import EmbeddingDatabase


@dataclass
class DatasetSampleRecord:
    sample_id: str
    person_id: str
    camera_id: str
    track_id: int
    timestamp: float
    observation_date: str
    modality: str
    vector: list[float]
    session_id: str = ""
    image_data: Any | None = None
    training_media_status: str = "AVAILABLE"
    quality_score: float = 1.0
    split_type: str = "train"
    condition_tags: dict[str, Any] = field(default_factory=dict)
    provenance: str = "operational_observation"

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"sess_{self.camera_id}_{self.track_id}_{int(self.timestamp // 3600)}"

    def to_dict(self, include_image: bool = False) -> dict[str, Any]:
        d = {
            "sample_id": self.sample_id,
            "person_id": self.person_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "observation_date": self.observation_date,
            "modality": self.modality,
            "vector_dim": len(self.vector),
            "training_media_status": self.training_media_status,
            "quality_score": self.quality_score,
            "split_type": self.split_type,
            "condition_tags": self.condition_tags,
            "provenance": self.provenance,
        }
        if include_image and self.image_data is not None and isinstance(self.image_data, np.ndarray):
            d["image_shape"] = list(self.image_data.shape)
            d["image_dtype"] = str(self.image_data.dtype)
        return d


@dataclass
class DatasetManifest:
    dataset_id: str
    created_at: float
    observation_date: str
    model_type: str
    total_samples: int
    train_count: int
    val_count: int
    independent_test_count: int
    historical_replay_count: int
    historical_test_count: int
    future_holdout_count: int
    identities: list[str]
    camera_ids: list[str]
    session_ids: list[str]
    manifest_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingDatasetBuilder:
    def __init__(
        self,
        collector: OperationalEmbeddingCollector | None = None,
        db: EmbeddingDatabase | None = None,
        evidence_manager: OperationalEvidenceManager | None = None,
        manifest_dir: str = "data/dataset_manifests",
        historical_replay_ratio: float = 0.50,
        test_split_ratio: float = 0.20,
        val_split_ratio: float = 0.15,
        min_samples_per_identity: int = 2,
    ) -> None:
        self._logger = get_logger("training_dataset_builder")
        self.collector = collector or OperationalEmbeddingCollector()
        self.db = db or EmbeddingDatabase()
        self.evidence_manager = evidence_manager
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.historical_replay_ratio = float(historical_replay_ratio)
        self.test_split_ratio = float(test_split_ratio)
        self.val_split_ratio = float(val_split_ratio)
        self.min_samples_per_identity = max(1, min_samples_per_identity)

    def build_dataset_for_date(
        self,
        training_date: str,
        model_type: str = "bygait_light",
        include_historical: bool = True,
        future_date: str | None = None,
    ) -> tuple[
        list[DatasetSampleRecord],
        list[DatasetSampleRecord],
        list[DatasetSampleRecord],
        list[DatasetSampleRecord],
        list[DatasetSampleRecord],
        list[DatasetSampleRecord],
        DatasetManifest,
    ]:
        modality = "gait" if model_type == "bygait_light" else "appearance"
        expected_dim = 256 if model_type == "bygait_light" else 512


        raw_obs = self.collector.get_eligible_by_date(training_date)
        eligible_samples: list[DatasetSampleRecord] = []

        for obs in raw_obs:
            if obs.modality != modality:
                continue
            if obs.state != ObservationState.TRAINING_ELIGIBLE:
                continue

            vec = np.asarray(obs.vector, dtype=np.float32)
            if vec.size != expected_dim or not np.isfinite(vec).all():
                continue

            ident = obs.verified_identity or obs.predicted_identity
            if not ident or ident in ("UNKNOWN", "UNKNOWN_PERSON"):
                continue


            img = getattr(obs, "gei_image", None) if modality == "gait" else getattr(obs, "crop_image", None)
            if img is None and self.evidence_manager is not None:

                for rec in self.evidence_manager._records.values():
                    if rec.observation_id == obs.observation_id and rec.modality == modality:
                        img = self.evidence_manager.load_evidence(rec.evidence_id)
                        break

            media_status = "AVAILABLE" if img is not None else "TRAINING_MEDIA_UNAVAILABLE"
            session_id = obs.metadata.get("session_id", f"sess_{obs.camera_id}_{obs.track_id}_{int(obs.created_at // 3600)}")

            sample = DatasetSampleRecord(
                sample_id=obs.observation_id,
                person_id=ident,
                camera_id=obs.camera_id,
                track_id=obs.track_id,
                session_id=session_id,
                timestamp=getattr(obs, "created_at", getattr(obs, "timestamp", time.time())),
                observation_date=obs.observation_date,
                modality=modality,
                vector=vec.tolist(),
                image_data=img,
                training_media_status=media_status,
                quality_score=obs.quality_score,
                condition_tags=dict(obs.metadata),
                provenance="operational_observation",
            )
            eligible_samples.append(sample)



        by_identity: dict[str, dict[str, list[DatasetSampleRecord]]] = {}
        for s in eligible_samples:
            track_key = f"{s.session_id}_{s.track_id}"
            by_identity.setdefault(s.person_id, {}).setdefault(track_key, []).append(s)

        train_samples: list[DatasetSampleRecord] = []
        val_samples: list[DatasetSampleRecord] = []
        independent_test_samples: list[DatasetSampleRecord] = []

        for tracks_dict in by_identity.values():
            track_items = list(tracks_dict.values())
            n_tracks = len(track_items)

            if n_tracks == 1:
                for s in track_items[0]:
                    s.split_type = "train"
                    train_samples.append(s)
            elif n_tracks == 2:
                for s in track_items[0]:
                    s.split_type = "train"
                    train_samples.append(s)
                for s in track_items[1]:
                    s.split_type = "independent_test"
                    independent_test_samples.append(s)
            elif n_tracks == 3:
                for s in track_items[0]:
                    s.split_type = "train"
                    train_samples.append(s)
                for s in track_items[1]:
                    s.split_type = "val"
                    val_samples.append(s)
                for s in track_items[2]:
                    s.split_type = "independent_test"
                    independent_test_samples.append(s)
            else:
                n_test_tracks = max(1, round(n_tracks * self.test_split_ratio))
                n_val_tracks = max(1, round(n_tracks * self.val_split_ratio))
                n_train_tracks = n_tracks - n_test_tracks - n_val_tracks
                if n_train_tracks < 1:
                    n_train_tracks = 1
                    n_val_tracks = max(0, n_tracks - n_test_tracks - n_train_tracks)

                train_tracks = track_items[:n_train_tracks]
                val_tracks = track_items[n_train_tracks : n_train_tracks + n_val_tracks]
                test_tracks = track_items[n_train_tracks + n_val_tracks :]

                for trk in train_tracks:
                    for s in trk:
                        s.split_type = "train"
                        train_samples.append(s)
                for trk in val_tracks:
                    for s in trk:
                        s.split_type = "val"
                        val_samples.append(s)
                for trk in test_tracks:
                    for s in trk:
                        s.split_type = "independent_test"
                        independent_test_samples.append(s)


        historical_replay_samples: list[DatasetSampleRecord] = []
        historical_test_samples: list[DatasetSampleRecord] = []

        if include_historical:
            all_persons = self.db.list_all_persons()
            for p in all_persons:
                if p.status != "ACTIVE":
                    continue
                embs = p.gait_embeddings if modality == "gait" else p.appearance_embeddings
                hist_records = [
                    e
                    for e in embs
                    if e.status == "ACTIVE"
                    and e.observation_date != training_date
                    and len(e.vector) == expected_dim
                    and np.isfinite(e.vector).all()
                ]

                if not hist_records:
                    continue

                for idx, e in enumerate(hist_records[:4]):
                    s_id = f"hist_{e.embedding_id}"
                    split = "historical_test" if idx == len(hist_records[:4]) - 1 and len(hist_records[:4]) > 1 else "historical_replay"
                    sample = DatasetSampleRecord(
                        sample_id=s_id,
                        person_id=p.person_id,
                        camera_id=getattr(e, "camera_id", "historical_camera"),
                        track_id=0,
                        session_id="hist_session",
                        timestamp=e.created_at,
                        observation_date=e.observation_date or "historical",
                        modality=modality,
                        vector=e.vector,
                        split_type=split,
                        provenance="historical_gallery",
                    )
                    if split == "historical_test":
                        historical_test_samples.append(sample)
                    else:
                        historical_replay_samples.append(sample)


        future_holdout_samples: list[DatasetSampleRecord] = []
        if future_date and future_date > training_date:
            raw_future = self.collector.get_eligible_by_date(future_date)
            for obs in raw_future:
                if obs.modality == modality and obs.state == ObservationState.TRAINING_ELIGIBLE:
                    vec = np.asarray(obs.vector, dtype=np.float32)
                    if vec.size == expected_dim and np.isfinite(vec).all():
                        fut_sample = DatasetSampleRecord(
                            sample_id=obs.observation_id,
                            person_id=obs.verified_identity or obs.predicted_identity,
                            camera_id=obs.camera_id,
                            track_id=obs.track_id,
                            session_id=obs.metadata.get("session_id", "future_session"),
                            timestamp=obs.created_at,
                            observation_date=obs.observation_date,
                            modality=modality,
                            vector=vec.tolist(),
                            split_type="future_holdout",
                            provenance="future_holdout",
                        )
                        future_holdout_samples.append(fut_sample)


        self._verify_zero_leakage(train_samples, val_samples, independent_test_samples, historical_replay_samples, historical_test_samples, future_holdout_samples)


        dataset_id = f"ds-{training_date.replace('-', '')}-{model_type[:4]}-{uuid.uuid4().hex[:6]}"
        manifest_dict = {
            "dataset_id": dataset_id,
            "created_at": time.time(),
            "observation_date": training_date,
            "model_type": model_type,
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "independent_test_count": len(independent_test_samples),
            "historical_replay_count": len(historical_replay_samples),
            "historical_test_count": len(historical_test_samples),
            "future_holdout_count": len(future_holdout_samples),
            "total_samples": (
                len(train_samples)
                + len(val_samples)
                + len(independent_test_samples)
                + len(historical_replay_samples)
                + len(historical_test_samples)
                + len(future_holdout_samples)
            ),
            "identities": sorted(list(by_identity.keys()) + [s.person_id for s in historical_replay_samples]),
            "camera_ids": sorted({s.camera_id for s in (train_samples + independent_test_samples)}),
            "session_ids": sorted({s.session_id for s in (train_samples + independent_test_samples)}),
        }

        manifest_str = json.dumps(manifest_dict, sort_keys=True)
        manifest_sha256 = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            created_at=manifest_dict["created_at"],
            observation_date=training_date,
            model_type=model_type,
            total_samples=manifest_dict["total_samples"],
            train_count=manifest_dict["train_count"],
            val_count=manifest_dict["val_count"],
            independent_test_count=manifest_dict["independent_test_count"],
            historical_replay_count=manifest_dict["historical_replay_count"],
            historical_test_count=manifest_dict["historical_test_count"],
            future_holdout_count=manifest_dict["future_holdout_count"],
            identities=manifest_dict["identities"],
            camera_ids=manifest_dict["camera_ids"],
            session_ids=manifest_dict["session_ids"],
            manifest_sha256=manifest_sha256,
            metadata={"source_date": training_date, "modality": modality},
        )

        manifest_path = self.manifest_dir / f"{dataset_id}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        self._logger.info(
            f"[DATASET_BUILT] ID={dataset_id} date={training_date} type={model_type} "
            f"train={len(train_samples)} val={len(val_samples)} test={len(independent_test_samples)} "
            f"future_holdout={len(future_holdout_samples)} SHA256={manifest_sha256[:12]}..."
        )

        return (
            train_samples,
            val_samples,
            independent_test_samples,
            historical_replay_samples,
            historical_test_samples,
            future_holdout_samples,
            manifest,
        )

    def _verify_zero_leakage(
        self,
        train: list[DatasetSampleRecord],
        val: list[DatasetSampleRecord],
        test: list[DatasetSampleRecord],
        hist_replay: list[DatasetSampleRecord],
        hist_test: list[DatasetSampleRecord],
        future_holdout: list[DatasetSampleRecord],
    ) -> None:
        train_ids = {s.sample_id for s in train}
        val_ids = {s.sample_id for s in val}
        test_ids = {s.sample_id for s in test}
        future_ids = {s.sample_id for s in future_holdout}


        if train_ids.intersection(test_ids):
            raise ValueError(f"CRITICAL LEAKAGE: Training IDs found in independent test set: {train_ids.intersection(test_ids)}")
        if train_ids.intersection(val_ids):
            raise ValueError(f"CRITICAL LEAKAGE: Training IDs found in validation set: {train_ids.intersection(val_ids)}")
        if val_ids.intersection(test_ids):
            raise ValueError(f"CRITICAL LEAKAGE: Validation IDs found in independent test set: {val_ids.intersection(test_ids)}")
        if train_ids.intersection(future_ids):
            raise ValueError(f"CRITICAL LEAKAGE: Training IDs found in future holdout set: {train_ids.intersection(future_ids)}")


        train_sessions = {f"{s.person_id}_{s.session_id}_{s.track_id}" for s in train}
        test_sessions = {f"{s.person_id}_{s.session_id}_{s.track_id}" for s in test}
        overlap = train_sessions.intersection(test_sessions)
        if overlap:
            raise ValueError(f"CRITICAL TRACK LEAKAGE: Same track/session found across train and test: {overlap}")
