import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from monitoring.logging_config import get_logger


class ReferenceJobStatus(str, Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    VALIDATING = "VALIDATING"
    READY_TO_COMMIT = "READY_TO_COMMIT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobProgress:
    stage: str = "QUEUED"
    total_frames: int = 0
    frames_processed: int = 0
    fps: float = 0.0
    tracks_detected: int = 0
    selected_track_id: int | None = None
    valid_silhouettes: int = 0
    valid_sequences: int = 0
    embeddings_generated: int = 0
    embeddings_deduplicated: int = 0
    embeddings_committed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceJobRecord:
    job_id: str
    person_id: str
    video_path: str
    status: ReferenceJobStatus
    case_id: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    progress: JobProgress = field(default_factory=JobProgress)
    result: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    diagnostic_code: str | None = None
    owner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "person_id": self.person_id,
            "case_id": self.case_id,
            "video_path": self.video_path,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress.to_dict(),
            "result": self.result,
            "error_message": self.error_message,
            "diagnostic_code": self.diagnostic_code,
            "owner": self.owner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceJobRecord":
        prog_data = data.get("progress", {})
        progress = JobProgress(
            stage=prog_data.get("stage", "QUEUED"),
            total_frames=prog_data.get("total_frames", 0),
            frames_processed=prog_data.get("frames_processed", 0),
            fps=prog_data.get("fps", 0.0),
            tracks_detected=prog_data.get("tracks_detected", 0),
            selected_track_id=prog_data.get("selected_track_id"),
            valid_silhouettes=prog_data.get("valid_silhouettes", 0),
            valid_sequences=prog_data.get("valid_sequences", 0),
            embeddings_generated=prog_data.get("embeddings_generated", 0),
            embeddings_deduplicated=prog_data.get("embeddings_deduplicated", 0),
            embeddings_committed=prog_data.get("embeddings_committed", 0),
        )

        status_str = data.get("status", ReferenceJobStatus.QUEUED.value)
        try:
            status = ReferenceJobStatus(status_str)
        except ValueError:
            status = ReferenceJobStatus.FAILED

        return cls(
            job_id=str(data["job_id"]),
            person_id=str(data["person_id"]),
            case_id=str(data.get("case_id", "")),
            video_path=str(data.get("video_path", "")),
            status=status,
            created_at=float(data.get("created_at", time.time())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            progress=progress,
            result=dict(data.get("result", {})),
            error_message=data.get("error_message"),
            diagnostic_code=data.get("diagnostic_code"),
            owner=str(data.get("owner", "")),
        )


class ReferenceJobManager:
    _instance: "ReferenceJobManager | None" = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        jobs_dir: str = "data/reference_jobs",
        max_workers: int = 2,
    ) -> None:
        self.logger = get_logger("reference_job_manager")
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, ReferenceJobRecord] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="RefJobWorker")
        self._load_persisted_jobs()

    @classmethod
    def get_instance(cls, jobs_dir: str = "data/reference_jobs") -> "ReferenceJobManager":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls(jobs_dir=jobs_dir)
        return cls._instance

    def _load_persisted_jobs(self) -> None:
        count = 0
        for f in self.jobs_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                record = ReferenceJobRecord.from_dict(data)
                # Interrupted jobs on restart: mark as RETRY_REQUIRED or FAILED
                if record.status in (ReferenceJobStatus.PROCESSING, ReferenceJobStatus.QUEUED):
                    record.status = ReferenceJobStatus.FAILED
                    record.error_message = "Job interrupted by process shutdown or restart."
                    record.diagnostic_code = "PROCESS_RESTARTED"
                    self._persist_job(record)
                self._jobs[record.job_id] = record
                count += 1
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as err:
                self.logger.warning(f"Could not load reference job from {f.name}: {err}")
        self.logger.info(f"Loaded {count} reference jobs from {self.jobs_dir}")

    def _persist_job(self, job: ReferenceJobRecord) -> None:
        target = self.jobs_dir / f"{job.job_id}.json"
        tmp_target = self.jobs_dir / f"{job.job_id}.tmp"
        try:
            with open(tmp_target, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(), f, indent=2)
            tmp_target.replace(target)
        except OSError as e:
            self.logger.error(f"Failed to persist job {job.job_id} to disk: {e}")

    def create_job(
        self,
        person_id: str,
        video_path: str,
        case_id: str = "",
        owner: str = "",
    ) -> ReferenceJobRecord:
        normalized_person_id = person_id.strip()
        timestamp = int(time.time())
        unique_suffix = uuid.uuid4().hex[:8]
        job_id = f"ref_{normalized_person_id}_{timestamp}_{unique_suffix}"

        record = ReferenceJobRecord(
            job_id=job_id,
            person_id=normalized_person_id,
            case_id=case_id or normalized_person_id,
            video_path=video_path,
            status=ReferenceJobStatus.QUEUED,
            created_at=time.time(),
            owner=owner,
        )

        with self._lock:
            self._jobs[job_id] = record
            self._persist_job(record)

        self.logger.info(f"Created reference video job '{job_id}' for person '{normalized_person_id}' (owner: '{owner or 'none'}')")
        return record

    def get_job(self, job_id: str) -> ReferenceJobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50, owner: str | None = None) -> list[ReferenceJobRecord]:
        with self._lock:
            if owner:
                owner_clean = owner.strip().lower()
                filtered = [
                    j for j in self._jobs.values()
                    if not j.owner or j.owner.strip().lower() == owner_clean
                ]
                all_jobs = sorted(filtered, key=lambda j: j.created_at, reverse=True)
            else:
                all_jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return all_jobs[:limit]

    def update_progress(
        self,
        job_id: str,
        stage: str | None = None,
        status: ReferenceJobStatus | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            if status is not None:
                job.status = status
                if status == ReferenceJobStatus.PROCESSING and job.started_at is None:
                    job.started_at = time.time()

            if stage is not None:
                job.progress.stage = stage

            for k, v in kwargs.items():
                if hasattr(job.progress, k):
                    setattr(job.progress, k, v)

            self._persist_job(job)

    def complete_job(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            job.status = ReferenceJobStatus.COMPLETED
            job.progress.stage = "COMPLETED"
            job.completed_at = time.time()
            job.result = result
            job.error_message = None
            job.diagnostic_code = None
            self._persist_job(job)

        self.logger.info(f"Completed reference video job '{job_id}' successfully.")

    def fail_job(
        self,
        job_id: str,
        error_message: str,
        diagnostic_code: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            job.status = ReferenceJobStatus.FAILED
            job.progress.stage = "FAILED"
            job.completed_at = time.time()
            job.error_message = error_message
            job.diagnostic_code = diagnostic_code or "PROCESSING_ERROR"
            self._persist_job(job)

        self.logger.warning(f"Failed reference video job '{job_id}': [{job.diagnostic_code}] {error_message}")

    def submit_task(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return self._executor.submit(fn, *args, **kwargs)
