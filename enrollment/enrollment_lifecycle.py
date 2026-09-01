import shutil
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from monitoring.logging_config import get_logger
from pipeline.detection.person_detector import PersonDetector
from pipeline.silhouette.extractor import SilhouetteExtractor
from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.steps.reid_feature_extraction import ReIDFeatureExtractionStep
from storage.embedding_database import EmbeddingDatabase


class EnrollmentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    EMBEDDING_GENERATED = "EMBEDDING_GENERATED"
    PERSISTENCE_VERIFIED = "PERSISTENCE_VERIFIED"
    FIREBASE_PERSISTED = "FIREBASE_PERSISTED"
    FIREBASE_PERSISTENCE_PENDING = "FIREBASE_PERSISTENCE_PENDING"
    RAW_MEDIA_DELETED = "RAW_MEDIA_DELETED"
    EMBEDDING_ONLY = "EMBEDDING_ONLY"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    PERSISTENCE_FAILED = "PERSISTENCE_FAILED"
    CLEANUP_FAILED = "CLEANUP_FAILED"
    RETRY_REQUIRED = "RETRY_REQUIRED"


@dataclass
class EnrollmentJobResult:
    job_id: str
    person_id: str
    status: EnrollmentStatus
    case_id: str = ""
    gait_embeddings_count: int = 0
    appearance_embeddings_count: int = 0
    firebase_persisted: bool = False
    raw_files_processed: list[str] = field(default_factory=list)
    raw_files_deleted: list[str] = field(default_factory=list)
    raw_files_retained: list[str] = field(default_factory=list)
    error_message: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class EnrollmentLifecycleManager:
    def __init__(
        self,
        db: EmbeddingDatabase | None = None,
        gait_extractor: FeatureExtractionStep | None = None,
        appearance_extractor: ReIDFeatureExtractionStep | None = None,
        detector: PersonDetector | None = None,
        firebase_store: Any | None = None,
        model_version: str = "v1.0.0",
    ) -> None:
        self._logger = get_logger("enrollment_lifecycle")
        self.db = db or EmbeddingDatabase()
        self.firebase_store = firebase_store or getattr(self.db, "firebase_store", None)
        self.gait_extractor = gait_extractor or FeatureExtractionStep()
        self.appearance_extractor = appearance_extractor or ReIDFeatureExtractionStep()
        self.detector = detector or PersonDetector()
        self.silhouette_extractor = SilhouetteExtractor(target_size=(64, 128))
        self.model_version = model_version

    def enroll_from_media(
        self,
        person_id: str,
        video_paths: list[Path | str] | None = None,
        photo_paths: list[Path | str] | None = None,
        gei_paths: list[Path | str] | None = None,
        session_id: str = "",
        case_id: str = "",
        auto_delete_raw: bool = True,
    ) -> EnrollmentJobResult:
        job_id = f"job_enroll_{person_id}_{int(time.time())}"
        v_paths = [Path(p) for p in (video_paths or []) if Path(p).exists()]
        p_paths = [Path(p) for p in (photo_paths or []) if Path(p).exists()]
        g_paths = [Path(p) for p in (gei_paths or []) if Path(p).exists()]

        all_input_paths = v_paths + p_paths + g_paths
        input_file_names = [str(p) for p in all_input_paths]

        result = EnrollmentJobResult(
            job_id=job_id,
            person_id=person_id,
            case_id=case_id,
            status=EnrollmentStatus.PROCESSING,
            raw_files_processed=input_file_names,
        )

        self._logger.info(
            f"[{job_id}] Starting enrollment lifecycle for '{person_id}' "
            f"({len(v_paths)} videos, {len(p_paths)} photos, {len(g_paths)} GEIs)"
        )

        gait_embeddings: list[np.ndarray] = []
        appearance_embeddings: list[np.ndarray] = []




        try:

            for g_path in g_paths:
                emb = self.gait_extractor.extract(g_path)
                if emb is not None and len(emb) == 256 and np.isfinite(emb).all():
                    gait_embeddings.append(emb)


            for p_path in p_paths:
                img = cv2.imread(str(p_path))
                crop = img
                if img is not None and img.size > 0 and self.detector is not None:
                    try:
                        dets = self.detector.detect(img)
                        if dets:
                            d = max(
                                dets,
                                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
                            )
                            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                            h, w = img.shape[:2]
                            crop = img[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
                    except (RuntimeError, ValueError, TypeError, cv2.error, OSError):
                        crop = img

                target_input = crop if crop is not None and getattr(crop, "size", 0) > 0 else p_path
                app_emb = self.appearance_extractor.extract(target_input)
                if app_emb is not None and len(app_emb) == 512 and np.isfinite(app_emb).all():
                    appearance_embeddings.append(app_emb)

            if not gait_embeddings and not appearance_embeddings:
                raise ValueError("No valid gait (256D) or appearance (512D) embeddings could be generated")

            result.status = EnrollmentStatus.EMBEDDING_GENERATED
            result.gait_embeddings_count = len(gait_embeddings)
            result.appearance_embeddings_count = len(appearance_embeddings)

        except (RuntimeError, ValueError, TypeError, OSError) as proc_err:
            result.status = EnrollmentStatus.PROCESSING_FAILED
            result.error_message = f"Embedding generation failed: {proc_err}"
            result.raw_files_retained = input_file_names
            result.completed_at = time.time()
            self._logger.error(
                f"[{job_id}] Processing failed for '{person_id}': {proc_err}. "
                f"SAFETY INVARIANT: Retained {len(input_file_names)} raw files for recovery."
            )
            return result




        try:
            persist_res = self.db.add_embeddings(
                person_id=person_id,
                gait_embeddings=gait_embeddings if gait_embeddings else None,
                appearance_embeddings=appearance_embeddings if appearance_embeddings else None,
                model_version=self.model_version,
                source_session_id=session_id or job_id,
            )

            if not persist_res.get("persistence_verified", False):
                raise RuntimeError("Database reported unverified persistence state")


            fb_results = persist_res.get("firebase_results", [])
            fb_all_verified = False
            if fb_results:
                fb_all_verified = all(r.get("success", False) for r in fb_results)
                result.firebase_persisted = fb_all_verified
                result.status = (
                    EnrollmentStatus.FIREBASE_PERSISTED
                    if fb_all_verified
                    else EnrollmentStatus.FIREBASE_PERSISTENCE_PENDING
                )
            else:
                result.status = EnrollmentStatus.PERSISTENCE_VERIFIED

            self._logger.info(
                f"[{job_id}] Local persistence verified for '{person_id}'. "
                f"Firebase status: {'PERSISTED' if fb_all_verified else 'LOCAL_ONLY/PENDING'}."
            )

        except (RuntimeError, ValueError, TypeError, OSError) as persist_err:
            result.status = EnrollmentStatus.PERSISTENCE_FAILED
            result.error_message = f"Persistence verification failed: {persist_err}"
            result.raw_files_retained = input_file_names
            result.completed_at = time.time()
            self._logger.error(
                f"[{job_id}] Persistence failed for '{person_id}': {persist_err}. "
                f"SAFETY INVARIANT: Retained {len(input_file_names)} raw files."
            )
            return result




        if auto_delete_raw:
            deleted_files = []
            retained_files = []
            cleanup_errors = []


            for f_path in all_input_paths:
                success, err = self.safe_delete_raw_file(f_path)
                if success:
                    deleted_files.append(str(f_path))
                else:
                    retained_files.append(str(f_path))
                    cleanup_errors.append(err)


            if case_id and self.firebase_store is not None:
                try:
                    self.firebase_store.delete_temporary_media(case_id)
                except Exception as fb_clean_err:  # noqa: BLE001
                    self._logger.warning(f"[{job_id}] Firebase Storage cleanup notice for {case_id}: {fb_clean_err}")

            result.raw_files_deleted = deleted_files
            result.raw_files_retained = retained_files

            if cleanup_errors:
                result.status = EnrollmentStatus.CLEANUP_FAILED
                result.error_message = (
                    f"Persistence succeeded, but cleanup encountered errors: {'; '.join(cleanup_errors)}"
                )
            else:
                result.status = EnrollmentStatus.EMBEDDING_ONLY
                self._logger.info(
                    f"[{job_id}] Successfully cleaned up {len(deleted_files)} raw media files. "
                    f"Subject '{person_id}' is now in EMBEDDING_ONLY state."
                )
        else:
            result.status = EnrollmentStatus.EMBEDDING_ONLY
            result.raw_files_retained = input_file_names

        result.completed_at = time.time()
        return result

    @staticmethod
    def safe_delete_raw_file(file_path: Path | str) -> tuple[bool, str | None]:
        p = Path(file_path)
        try:
            if p.exists():
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            return True, None
        except (OSError, PermissionError) as err:
            return False, f"Failed to delete {p.name}: {err}"
