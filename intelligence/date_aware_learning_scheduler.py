import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from intelligence.operational_embedding_collector import (
    OperationalEmbeddingCollector,
)
from monitoring.logging_config import get_logger
from storage.embedding_database import EmbeddingDatabase


class LearningJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    PROMOTED = "PROMOTED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    SKIPPED_NO_NEW_DATA = "SKIPPED_NO_NEW_DATA"
    SKIPPED_BELOW_THRESHOLD = "SKIPPED_BELOW_THRESHOLD"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"
    ROLLED_BACK = "ROLLED_BACK"
    INTERRUPTED = "INTERRUPTED"


@dataclass
class LearningJobRecord:
    job_id: str
    training_date: str
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    status: LearningJobStatus = LearningJobStatus.PENDING
    candidate_version: str | None = None
    model_type: str = "dual_modal_fusion"
    new_embeddings_count: int = 0
    identities_count: int = 0
    identities: list[str] = field(default_factory=list)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str | None = None
    error_message: str | None = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningJobRecord":
        st_val = data.get("status", "PENDING")
        if isinstance(st_val, str):
            status = LearningJobStatus(st_val)
        else:
            status = LearningJobStatus.PENDING

        return cls(
            job_id=str(data["job_id"]),
            training_date=str(data["training_date"]),
            created_at=float(data.get("created_at", time.time())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            status=status,
            candidate_version=data.get("candidate_version"),
            model_type=str(data.get("model_type", "dual_modal_fusion")),
            new_embeddings_count=int(data.get("new_embeddings_count", 0)),
            identities_count=int(data.get("identities_count", 0)),
            identities=list(data.get("identities", [])),
            validation_metrics=dict(data.get("validation_metrics", {})),
            rejection_reason=data.get("rejection_reason"),
            error_message=data.get("error_message"),
            duration=float(data.get("duration", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


class DateAwareLearningScheduler:
    def __init__(
        self,
        jobs_file: str = "data/learning_jobs.json",
        collector: OperationalEmbeddingCollector | None = None,
        db: EmbeddingDatabase | None = None,
        min_training_embeddings: int = 4,
        min_identities: int = 2,
    ) -> None:
        self.jobs_file = Path(jobs_file)
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        self.collector = collector or OperationalEmbeddingCollector()
        self.db = db or EmbeddingDatabase()
        self.min_training_embeddings = max(1, int(min_training_embeddings))
        self.min_identities = max(1, int(min_identities))
        self._logger = get_logger("date_aware_scheduler")
        self._lock = threading.RLock()
        self._jobs_cache: dict[str, LearningJobRecord] | None = None
        self._last_mtime: float = 0.0


        self.recover_interrupted_jobs()

    def _load_jobs(self) -> dict[str, LearningJobRecord]:
        with self._lock:
            if not self.jobs_file.exists():
                self._jobs_cache = {}
                return {}
            try:
                mtime = self.jobs_file.stat().st_mtime
                if self._jobs_cache is not None and mtime == self._last_mtime:
                    return dict(self._jobs_cache)

                with open(self.jobs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                jobs = {}
                for k, v in data.get("jobs", {}).items():
                    jobs[k] = LearningJobRecord.from_dict(v)
                self._jobs_cache = jobs
                self._last_mtime = mtime
                return dict(jobs)
            except (OSError, json.JSONDecodeError, ValueError) as err:
                self._logger.warning(f"Failed to load learning jobs file: {err}")
                return {}

    def _save_jobs(self, jobs: dict[str, LearningJobRecord]) -> bool:
        with self._lock:
            tmp = self.jobs_file.with_suffix(".tmp")
            try:
                data = {"jobs": {k: v.to_dict() for k, v in jobs.items()}}
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                tmp.replace(self.jobs_file)
                self._jobs_cache = dict(jobs)
                self._last_mtime = self.jobs_file.stat().st_mtime
                return True
            except (OSError, ValueError) as err:
                self._logger.error(f"Failed to save learning jobs file: {err}")
                return False

    def list_jobs(self) -> list[LearningJobRecord]:
        jobs = self._load_jobs()
        return sorted(jobs.values(), key=lambda j: j.created_at)

    def get_job(self, job_id: str) -> LearningJobRecord | None:
        jobs = self._load_jobs()
        return jobs.get(job_id)

    def get_job_for_date_and_type(
        self, training_date: str, model_type: str
    ) -> LearningJobRecord | None:
        jobs = self._load_jobs()
        for j in reversed(list(jobs.values())):
            if (
                j.training_date == training_date
                and j.model_type == model_type
                and j.status
                not in (
                    LearningJobStatus.SKIPPED_NO_NEW_DATA,
                    LearningJobStatus.SKIPPED_BELOW_THRESHOLD,
                )
            ):
                return j
        return None

    def get_processed_dates(self, model_type: str | None = None) -> set[str]:
        jobs = self._load_jobs()
        processed = set()
        for j in jobs.values():
            if model_type and j.model_type != model_type:
                continue
            if j.status in (
                LearningJobStatus.PROMOTED,
                LearningJobStatus.VALIDATING,
                LearningJobStatus.RUNNING,
                LearningJobStatus.PENDING,
                LearningJobStatus.REJECTED,
            ):
                processed.add(j.training_date)
        return processed

    def recover_interrupted_jobs(self) -> list[LearningJobRecord]:
        with self._lock:
            jobs = self._load_jobs()
            recovered = []
            modified = False

            for j in jobs.values():
                if j.status in (LearningJobStatus.RUNNING, LearningJobStatus.VALIDATING):
                    self._logger.warning(
                        f"[CRASH_RECOVERY] Found interrupted job '{j.job_id}' for date '{j.training_date}' "
                        f"(status: {j.status.value}). Marking as INTERRUPTED."
                    )
                    j.status = LearningJobStatus.INTERRUPTED
                    j.error_message = "Job interrupted by system restart/shutdown"
                    j.completed_at = time.time()
                    recovered.append(j)
                    modified = True

            if modified:
                self._save_jobs(jobs)
            return recovered

    def scan_for_eligible_data(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            date_data: dict[str, dict[str, Any]] = {}


            eligible_obs = self.collector.get_training_eligible()
            for obs in eligible_obs:
                d = obs.observation_date or time.strftime("%Y-%m-%d", time.gmtime(obs.created_at))
                if d not in date_data:
                    date_data[d] = {
                        "observations": [],
                        "embeddings": [],
                        "identities": set(),
                    }
                date_data[d]["observations"].append(obs)
                ident = obs.verified_identity or obs.predicted_identity
                if ident and ident != "UNKNOWN":
                    date_data[d]["identities"].add(ident)


            for person in self.db.list_all_persons():
                if person.status != "ACTIVE":
                    continue
                for emb in person.gait_embeddings + person.appearance_embeddings:
                    if emb.status != "ACTIVE":
                        continue
                    d = emb.observation_date or time.strftime("%Y-%m-%d", time.gmtime(emb.created_at))
                    if d not in date_data:
                        date_data[d] = {
                            "observations": [],
                            "embeddings": [],
                            "identities": set(),
                        }
                    date_data[d]["embeddings"].append(emb)
                    date_data[d]["identities"].add(person.person_id)


            result = {}
            for d, val in sorted(date_data.items()):
                obs_list = val["observations"]
                emb_list = val["embeddings"]
                idents = sorted(val["identities"])
                tot = len(obs_list) + len(emb_list)
                if tot > 0:
                    result[d] = {
                        "observations": obs_list,
                        "embeddings": emb_list,
                        "identities": idents,
                        "total_count": tot,
                    }
            return result

    def get_unprocessed_dates(self, model_type: str | None = None) -> dict[str, dict[str, Any]]:
        all_dates = self.scan_for_eligible_data()
        processed = self.get_processed_dates(model_type=model_type)
        unprocessed = {d: info for d, info in all_dates.items() if d not in processed}
        return unprocessed

    def create_learning_job(
        self,
        training_date: str,
        model_type: str = "dual_modal_fusion",
        force: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> LearningJobRecord | None:
        with self._lock:
            jobs = self._load_jobs()


            if not force:
                for j in jobs.values():
                    if (
                        j.training_date == training_date
                        and j.model_type == model_type
                        and j.status
                        in (
                            LearningJobStatus.PENDING,
                            LearningJobStatus.RUNNING,
                            LearningJobStatus.VALIDATING,
                            LearningJobStatus.PROMOTED,
                        )
                    ):
                        self._logger.info(
                            f"[IDEMPOTENT_SKIP] Active or completed {model_type} job '{j.job_id}' "
                            f"already exists for date '{training_date}' (status: {j.status.value})"
                        )
                        return j


            all_dates = self.scan_for_eligible_data()
            date_info = all_dates.get(training_date)
            if not date_info and not force:
                self._logger.warning(f"No eligible data found for date '{training_date}'. Job creation skipped.")
                return None

            count = date_info["total_count"] if date_info else 0
            identities = date_info["identities"] if date_info else []


            if not force and (count < self.min_training_embeddings or len(identities) < self.min_identities):
                skip_id = f"CL-SKIP-{training_date.replace('-', '')}-{model_type[:4]}-{uuid.uuid4().hex[:4]}"
                skip_job = LearningJobRecord(
                    job_id=skip_id,
                    training_date=training_date,
                    status=LearningJobStatus.SKIPPED_BELOW_THRESHOLD,
                    model_type=model_type,
                    new_embeddings_count=count,
                    identities_count=len(identities),
                    identities=identities,
                    rejection_reason=(
                        f"Below threshold: {count}/{self.min_training_embeddings} embeddings, "
                        f"{len(identities)}/{self.min_identities} identities."
                    ),
                    completed_at=time.time(),
                    metadata=metadata or {},
                )
                jobs[skip_id] = skip_job
                self._save_jobs(jobs)
                self._logger.info(
                    f"[JOB_SKIPPED] Date '{training_date}' below minimum threshold "
                    f"({count}/{self.min_training_embeddings} embs, {len(identities)}/{self.min_identities} IDs). "
                    f"No training triggered."
                )
                return skip_job


            type_tag = model_type[:4].upper()
            job_id = f"CL-{training_date.replace('-', '')}-{type_tag}-{uuid.uuid4().hex[:6]}"
            job = LearningJobRecord(
                job_id=job_id,
                training_date=training_date,
                created_at=time.time(),
                status=LearningJobStatus.PENDING,
                model_type=model_type,
                new_embeddings_count=count,
                identities_count=len(identities),
                identities=identities,
                metadata=metadata or {},
            )

            jobs[job_id] = job
            self._save_jobs(jobs)

            self._logger.info(
                f"[LEARNING_JOB_CREATED] date={training_date} type={model_type} job={job_id} "
                f"new_embeddings={count} identities={len(identities)} status=PENDING"
            )
            return job

    def check_and_schedule_new_dates(
        self,
        model_type: str = "dual_modal_fusion",
        model_types: list[str] | None = None,
    ) -> list[LearningJobRecord]:
        with self._lock:
            types_to_schedule = model_types or [model_type]
            scheduled_jobs = []

            for mt in types_to_schedule:
                unprocessed = self.get_unprocessed_dates(model_type=mt)
                if not unprocessed:
                    self._logger.debug(
                        f"[DATE_SCAN] No new observation dates with eligible data found for {mt}. "
                        f"Zero jobs scheduled."
                    )
                    continue

                for date_str in sorted(unprocessed.keys()):
                    job = self.create_learning_job(
                        training_date=date_str,
                        model_type=mt,
                        force=False,
                    )
                    if job and job.status == LearningJobStatus.PENDING:
                        scheduled_jobs.append(job)

            return scheduled_jobs

    def update_job(self, job: LearningJobRecord) -> bool:
        with self._lock:
            jobs = self._load_jobs()
            jobs[job.job_id] = job
            return self._save_jobs(jobs)

