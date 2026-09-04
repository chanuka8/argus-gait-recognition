import json
import os
import pickle
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from monitoring.logging_config import get_logger


def _safe_atomic_replace(tmp_path: Path, target_path: Path, max_retries: int = 6) -> None:
    """Safely replaces target_path with tmp_path, handling Windows file locking delays."""
    for attempt in range(max_retries):
        try:
            tmp_path.replace(target_path)
            return
        except OSError:
            if attempt == max_retries - 1:
                try:
                    shutil.copyfile(str(tmp_path), str(target_path))
                    tmp_path.unlink(missing_ok=True)
                    return
                except OSError:
                    pass
                raise
            time.sleep(0.04 * (attempt + 1))


class ReferenceJobStatus(str, Enum):
    INGESTED = "INGESTED"
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    VALIDATING = "VALIDATING"
    VALIDATING_VIDEO = "VALIDATING_VIDEO"
    TRACKING = "TRACKING"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    EMBEDDING_GENERATION = "EMBEDDING_GENERATION"
    MATCHING = "MATCHING"
    PERSISTING = "PERSISTING"
    QUALITY_VALIDATION = "QUALITY_VALIDATION"
    READY_TO_COMMIT = "READY_TO_COMMIT"
    COMMITTED = "COMMITTED"
    ACTIVE = "ACTIVE"
    INTERRUPTED = "INTERRUPTED"
    RESUMING = "RESUMING"
    COMPLETED = "COMPLETED"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    FAILED_PROCESSING = "FAILED_PROCESSING"
    FAILED_COMMIT = "FAILED_COMMIT"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


RECOVERABLE_STATUSES = {
    ReferenceJobStatus.QUEUED,
    ReferenceJobStatus.PROCESSING,
    ReferenceJobStatus.VALIDATING,
    ReferenceJobStatus.VALIDATING_VIDEO,
    ReferenceJobStatus.TRACKING,
    ReferenceJobStatus.FEATURE_EXTRACTION,
    ReferenceJobStatus.EMBEDDING_GENERATION,
    ReferenceJobStatus.MATCHING,
    ReferenceJobStatus.PERSISTING,
    ReferenceJobStatus.READY_TO_COMMIT,
    ReferenceJobStatus.INTERRUPTED,
    ReferenceJobStatus.RESUMING,
}


@dataclass
class JobProgress:
    stage: str = "QUEUED"
    total_frames: int = 0
    frames_processed: int = 0
    last_safe_frame: int = 0
    fps: float = 0.0
    tracks_detected: int = 0
    selected_track_id: int | None = None
    valid_silhouettes: int = 0
    valid_sequences: int = 0
    completed_sequences: list[int] = field(default_factory=list)
    embeddings_generated: int = 0
    embeddings_deduplicated: int = 0
    embeddings_committed: int = 0
    persisted_embedding_ids: list[str] = field(default_factory=list)
    percent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceJobRecord:
    job_id: str
    person_id: str
    video_path: str = ""
    media_path: str = ""
    media_type: str = "video"  # "video" | "image"
    status: ReferenceJobStatus = ReferenceJobStatus.QUEUED
    case_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_checkpoint_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    progress: JobProgress = field(default_factory=JobProgress)
    result: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    diagnostic_code: str | None = None
    owner: str = ""
    original_filename: str = ""
    recovery_count: int = 0
    resumed: bool = False
    checkpoint_path: str = ""

    def __post_init__(self) -> None:
        if not self.media_path and self.video_path:
            self.media_path = self.video_path
        elif not self.video_path and self.media_path:
            self.video_path = self.media_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "person_id": self.person_id,
            "case_id": self.case_id,
            "video_path": self.video_path,
            "media_path": self.media_path,
            "media_type": self.media_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_checkpoint_at": self.last_checkpoint_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress.to_dict(),
            "result": self.result,
            "error_message": self.error_message,
            "diagnostic_code": self.diagnostic_code,
            "owner": self.owner,
            "original_filename": self.original_filename,
            "recovery_count": self.recovery_count,
            "resumed": self.resumed,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceJobRecord":
        prog_data = data.get("progress", {})
        progress = JobProgress(
            stage=prog_data.get("stage", "QUEUED"),
            total_frames=prog_data.get("total_frames", 0),
            frames_processed=prog_data.get("frames_processed", 0),
            last_safe_frame=prog_data.get("last_safe_frame", 0),
            fps=prog_data.get("fps", 0.0),
            tracks_detected=prog_data.get("tracks_detected", 0),
            selected_track_id=prog_data.get("selected_track_id"),
            valid_silhouettes=prog_data.get("valid_silhouettes", 0),
            valid_sequences=prog_data.get("valid_sequences", 0),
            completed_sequences=list(prog_data.get("completed_sequences", [])),
            embeddings_generated=prog_data.get("embeddings_generated", 0),
            embeddings_deduplicated=prog_data.get("embeddings_deduplicated", 0),
            embeddings_committed=prog_data.get("embeddings_committed", 0),
            persisted_embedding_ids=list(prog_data.get("persisted_embedding_ids", [])),
            percent=prog_data.get("percent", 0),
        )

        status_str = data.get("status", ReferenceJobStatus.QUEUED.value)
        try:
            status = ReferenceJobStatus(status_str)
        except ValueError:
            status = ReferenceJobStatus.FAILED

        v_path = str(data.get("video_path", ""))
        m_path = str(data.get("media_path", v_path))

        c_at = float(data.get("created_at", time.time()))
        u_at = float(data.get("updated_at", c_at))
        l_chk = float(data.get("last_checkpoint_at", u_at))

        return cls(
            job_id=str(data["job_id"]),
            person_id=str(data["person_id"]),
            case_id=str(data.get("case_id", "")),
            video_path=v_path or m_path,
            media_path=m_path or v_path,
            media_type=str(data.get("media_type", "video")),
            status=status,
            created_at=c_at,
            updated_at=u_at,
            last_checkpoint_at=l_chk,
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            progress=progress,
            result=dict(data.get("result", {})),
            error_message=data.get("error_message"),
            diagnostic_code=data.get("diagnostic_code"),
            owner=str(data.get("owner", "")),
            original_filename=str(data.get("original_filename", "")),
            recovery_count=int(data.get("recovery_count", 0)),
            resumed=bool(data.get("resumed", False)),
            checkpoint_path=str(data.get("checkpoint_path", "")),
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
        self.checkpoints_dir = self.jobs_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        self._lock = threading.RLock()
        self._jobs: dict[str, ReferenceJobRecord] = {}
        self._claimed_jobs: set[str] = set()
        self._shutdown_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="RefJobWorker")
        self._load_persisted_jobs()

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()

    @classmethod
    def get_instance(cls, jobs_dir: str = "data/reference_jobs") -> "ReferenceJobManager":
        if cls._instance is None or getattr(cls._instance._executor, "_shutdown", False):
            with cls._singleton_lock:
                if cls._instance is None or getattr(cls._instance._executor, "_shutdown", False):
                    cls._instance = cls(jobs_dir=jobs_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful in test suites)."""
        with cls._singleton_lock:
            if cls._instance is not None:
                try:
                    cls._instance.shutdown(timeout=1.0)
                except Exception as exc:  # noqa: BLE001
                    cls._instance.logger.debug("Reset instance shutdown error: %s", exc)
                cls._instance = None

    def _load_persisted_jobs(self) -> None:
        count = 0
        for f in self.jobs_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                record = ReferenceJobRecord.from_dict(data)

                # Active or queued jobs during shutdown/restart are marked INTERRUPTED for recovery
                if record.status in RECOVERABLE_STATUSES and record.status != ReferenceJobStatus.INTERRUPTED:
                    record.status = ReferenceJobStatus.INTERRUPTED
                    record.error_message = "Job interrupted by process shutdown or restart."
                    record.diagnostic_code = "PROCESS_INTERRUPTED"
                    self._persist_job(record)

                self._jobs[record.job_id] = record
                count += 1
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as err:
                self.logger.warning(f"Could not load reference job from {f.name}: {err}")
        self.logger.info(f"Loaded {count} reference jobs from {self.jobs_dir}")

    def _persist_job(self, job: ReferenceJobRecord) -> None:
        """Atomically persists the job record to disk using write-to-temp + fsync + replace."""
        job.updated_at = time.time()
        job.last_checkpoint_at = time.time()
        target = self.jobs_dir / f"{job.job_id}.json"
        tmp_target = self.jobs_dir / f"{job.job_id}.tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        try:
            with open(tmp_target, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            _safe_atomic_replace(tmp_target, target)
        except OSError as e:
            self.logger.error(f"Failed to persist job {job.job_id} to disk: {e}")
            if tmp_target.exists():
                try:
                    tmp_target.unlink(missing_ok=True)
                except OSError:
                    pass

    def save_checkpoint_data(self, job_id: str, data: dict[str, Any]) -> Path:
        """Persists companion intermediate pipeline data (e.g. tracks, crops, silhouettes)."""
        target = self.checkpoints_dir / f"{job_id}_checkpoint.pkl"
        tmp_target = self.checkpoints_dir / f"{job_id}_checkpoint.tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        try:
            with open(tmp_target, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            _safe_atomic_replace(tmp_target, target)
        finally:
            if tmp_target.exists():
                try:
                    tmp_target.unlink(missing_ok=True)
                except OSError:
                    pass

        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.checkpoint_path = str(target)
                self._persist_job(job)
        return target

    def load_checkpoint_data(self, job_id: str) -> dict[str, Any] | None:
        """Loads companion intermediate pipeline data if present and valid."""
        target = self.checkpoints_dir / f"{job_id}_checkpoint.pkl"
        if not target.exists() or target.stat().st_size == 0:
            return None
        try:
            with open(target, "rb") as f:
                return pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as e:
            self.logger.warning(f"Could not load checkpoint data for {job_id}: {e}")
            return None

    def delete_checkpoint_data(self, job_id: str) -> None:
        """Cleans up checkpoint data upon successful job completion."""
        target = self.checkpoints_dir / f"{job_id}_checkpoint.pkl"
        try:
            if target.exists():
                target.unlink(missing_ok=True)
        except OSError as e:
            self.logger.debug(f"Could not remove checkpoint data for {job_id}: {e}")

    def claim_job_for_recovery(self, job_id: str) -> bool:
        """Thread-safe claim mechanism preventing duplicate resumption scans or workers."""
        with self._lock:
            if job_id in self._claimed_jobs:
                return False
            self._claimed_jobs.add(job_id)
            return True

    def release_job_claim(self, job_id: str) -> None:
        """Releases the recovery claim when a job completes or fails."""
        with self._lock:
            self._claimed_jobs.discard(job_id)

    def is_job_claimed(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._claimed_jobs

    def create_job(
        self,
        person_id: str,
        video_path: str = "",
        case_id: str = "",
        owner: str = "",
        media_type: str = "video",
        media_path: str = "",
        original_filename: str = "",
    ) -> ReferenceJobRecord:
        normalized_person_id = person_id.strip()
        timestamp = int(time.time())
        unique_suffix = uuid.uuid4().hex[:8]
        prefix = "ref_vid" if media_type == "video" else "ref_img"
        job_id = f"{prefix}_{normalized_person_id}_{timestamp}_{unique_suffix}"

        v_path = video_path or media_path
        m_path = media_path or video_path

        record = ReferenceJobRecord(
            job_id=job_id,
            person_id=normalized_person_id,
            case_id=case_id or normalized_person_id,
            video_path=v_path,
            media_path=m_path,
            media_type=media_type,
            status=ReferenceJobStatus.QUEUED,
            created_at=time.time(),
            updated_at=time.time(),
            last_checkpoint_at=time.time(),
            owner=owner,
            original_filename=original_filename or Path(v_path).name,
        )

        with self._lock:
            self._jobs[job_id] = record
            self._persist_job(record)

        self.logger.info(
            f"Created reference {media_type} job '{job_id}' for person '{normalized_person_id}' (owner: '{owner or 'none'}')"
        )
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
                if status in (ReferenceJobStatus.PROCESSING, ReferenceJobStatus.RESUMING) and job.started_at is None:
                    job.started_at = time.time()

            if stage is not None:
                job.progress.stage = stage

            for k, v in kwargs.items():
                if hasattr(job.progress, k):
                    setattr(job.progress, k, v)
                elif hasattr(job, k):
                    setattr(job, k, v)

            self._persist_job(job)

    def checkpoint_job(
        self,
        job_id: str,
        stage: str,
        last_safe_frame: int,
        frames_processed: int,
        total_frames: int | None = None,
        status: ReferenceJobStatus | None = None,
        checkpoint_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Durable safe checkpoint writer. Guarantees checkpoint is persisted atomically with fsync."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            job.progress.stage = stage
            job.progress.last_safe_frame = last_safe_frame
            job.progress.frames_processed = frames_processed
            if total_frames is not None:
                job.progress.total_frames = total_frames
            if status is not None:
                job.status = status

            for k, v in kwargs.items():
                if hasattr(job.progress, k):
                    setattr(job.progress, k, v)
                elif hasattr(job, k):
                    setattr(job, k, v)

            if checkpoint_data:
                self.save_checkpoint_data(job_id, checkpoint_data)

            self._persist_job(job)

    def complete_job(
        self,
        job_id: str,
        result: dict[str, Any],
        status: ReferenceJobStatus = ReferenceJobStatus.COMPLETED,
        terminal_status: ReferenceJobStatus | None = None,
    ) -> None:
        effective_status = terminal_status if terminal_status is not None else status
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return

            job.status = effective_status
            job.progress.stage = "COMPLETED"
            job.progress.percent = 100
            if job.progress.total_frames > 0:
                job.progress.frames_processed = job.progress.total_frames
                job.progress.last_safe_frame = job.progress.total_frames
            job.completed_at = time.time()
            job.result = result
            job.error_message = None
            job.diagnostic_code = None
            self._persist_job(job)
            self.delete_checkpoint_data(job_id)
            self.release_job_claim(job_id)

        self.logger.info(f"Completed reference job '{job_id}' successfully with status '{job.status.value}'.")

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
            self.release_job_claim(job_id)

        self.logger.warning(f"Failed reference video job '{job_id}': [{job.diagnostic_code}] {error_message}")

    def submit_task(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self._executor is None or getattr(self._executor, "_shutdown", False):
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="RefJobWorker")
                self._shutdown_event.clear()
            return self._executor.submit(fn, *args, **kwargs)

    def recover_unfinished_jobs(
        self,
        processor: Any = None,
        gait_service_ref: Any = None,
    ) -> list[ReferenceJobRecord]:
        """Scans persisted jobs on startup, validates checkpoints and video existence,

        claims each unfinished job atomically, and automatically resumes processing.
        """
        recovered_jobs: list[ReferenceJobRecord] = []

        with self._lock:
            candidate_jobs = [
                j for j in self._jobs.values()
                if j.status in RECOVERABLE_STATUSES and j.status != ReferenceJobStatus.COMPLETED
            ]

        for job in candidate_jobs:
            if not self.claim_job_for_recovery(job.job_id):
                self.logger.info(f"[RECOVERY] Skipping already-claimed job: {job.job_id}")
                continue

            # Consistency check
            if not job.job_id or not job.person_id:
                self.fail_job(job.job_id, "Corrupted job record metadata", diagnostic_code="INVALID_CHECKPOINT")
                continue

            # Source media check
            v_path = Path(job.video_path or job.media_path)
            if job.media_type == "video" and (not v_path.exists() or v_path.stat().st_size == 0):
                self.fail_job(
                    job.job_id,
                    f"Source video file not found on disk during recovery: {v_path}",
                    diagnostic_code="VIDEO_NOT_FOUND",
                )
                continue

            # Update job state for resume
            with self._lock:
                job.status = ReferenceJobStatus.RESUMING
                job.resumed = True
                job.recovery_count += 1
                self._persist_job(job)

            self.logger.info(f"[RECOVERY] Found unfinished reference job: {job.job_id}")
            self.logger.info(f"[RECOVERY] Stage: {job.progress.stage}")
            self.logger.info(
                f"[RECOVERY] Last safe frame: {job.progress.last_safe_frame}/{job.progress.total_frames}"
            )
            self.logger.info(f"[RECOVERY] Completed sequences: {len(job.progress.completed_sequences)}")
            self.logger.info("[RECOVERY] Resuming job")

            recovered_jobs.append(job)

            if processor is not None:
                def _run_resume(j: ReferenceJobRecord = job) -> None:
                    try:
                        if j.media_type == "video":
                            res = processor.process_reference_video(
                                person_id=j.person_id,
                                video_path=j.video_path,
                                job_id=j.job_id,
                                case_id=j.case_id,
                                gait_service_ref=gait_service_ref,
                            )
                        else:
                            res = processor.process_reference_photos(
                                person_id=j.person_id,
                                photo_paths=[j.media_path],
                                job_id=j.job_id,
                                case_id=j.case_id,
                                gait_service_ref=gait_service_ref,
                            )
                        if res.get("success"):
                            self.logger.info(f"[RECOVERY] Job completed successfully: {j.job_id}")
                    except Exception as exc:  # noqa: BLE001
                        self.logger.error(f"[RECOVERY] Error during resumed job execution {j.job_id}: {exc}")
                        self.fail_job(j.job_id, f"Resumed processing failed: {exc}", diagnostic_code="RESUME_ERROR")
                    finally:
                        self.release_job_claim(j.job_id)

                self.submit_task(_run_resume)

        return recovered_jobs

    def shutdown(self, timeout: float = 5.0) -> None:
        """Bounded graceful shutdown: signals in-flight jobs, persists INTERRUPTED state,

        and shuts down worker pool cleanly.
        """
        self.logger.info("Initiating ReferenceJobManager graceful shutdown...")
        self._shutdown_event.set()

        # Mark in-flight jobs as INTERRUPTED so they are safely resumed on next startup
        with self._lock:
            for job in self._jobs.values():
                if job.status in (
                    ReferenceJobStatus.PROCESSING,
                    ReferenceJobStatus.RESUMING,
                    ReferenceJobStatus.TRACKING,
                    ReferenceJobStatus.FEATURE_EXTRACTION,
                    ReferenceJobStatus.MATCHING,
                    ReferenceJobStatus.PERSISTING,
                    ReferenceJobStatus.VALIDATING,
                    ReferenceJobStatus.VALIDATING_VIDEO,
                    ReferenceJobStatus.QUEUED,
                ):
                    job.status = ReferenceJobStatus.INTERRUPTED
                    job.error_message = "Job interrupted by graceful application shutdown."
                    job.diagnostic_code = "GRACEFUL_SHUTDOWN"
                    self._persist_job(job)

        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            if ReferenceJobManager._instance is self:
                ReferenceJobManager._instance = None
        self.logger.info("ReferenceJobManager worker pool shutdown complete.")
