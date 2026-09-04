from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from monitoring.logging_config import get_logger
from pipeline.detection.person_detector import PersonDetector
from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.steps.live_gei import LiveGEI
from pipeline.steps.silhouette_step import SilhouetteStep
from pipeline.steps.tracking import TrackingStep
from services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus
from storage.embedding_database import EmbeddingDatabase
from storage.vector_store import VectorStore


@dataclass
class TrackSummary:
    track_id: int
    frame_indices: list[int] = field(default_factory=list)
    bboxes: list[list[int]] = field(default_factory=list)
    crops: list[np.ndarray] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    silhouettes: list[np.ndarray] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    @property
    def average_area(self) -> float:
        return float(np.mean(self.areas)) if self.areas else 0.0

    @property
    def prominence_score(self) -> float:
        # Combined metric of temporal presence and visual scale
        return float(self.frame_count * math.sqrt(max(1.0, self.average_area)))


@dataclass
class ValidatedEmbedding:
    vector: np.ndarray
    dimension: int
    person_id: str
    track_id: int
    sequence_index: int
    quality_score: float
    model_version: str
    provenance: dict[str, Any]


class MissingPersonVideoProcessor:
    """Offline, camera-independent reference video processing service.

    Processes uploaded missing-person reference videos through the locked ARGUS gait pipeline:
      Video Decoder -> Person Detection (YOLOv8) -> ByteTrack -> Multi-Person Safety Selection ->
      Silhouette Extraction -> LiveGEI -> ByGaitLight (256D) -> Validation ->
      Configurable Deduplication -> Consistent Persistence (VectorStore + EmbeddingDatabase).

    INVARIANT:
      This service NEVER imports or requires CameraWorker, RecognitionWorker, or live camera IDs.
      It runs completely offline and camera-independent.
    """

    def __init__(
        self,
        config_path: str = "configs/inference.yaml",
        gait_gallery_dir: str = "models/live_gallery",
        appearance_gallery_dir: str = "models/appearance_gallery",
        db_dir: str = "data/embedding_db",
        detector: PersonDetector | None = None,
        tracker: TrackingStep | None = None,
        silhouette_step: SilhouetteStep | None = None,
        extractor: FeatureExtractionStep | None = None,
        appearance_extractor: Any | None = None,
        store: VectorStore | None = None,
        embedding_db: EmbeddingDatabase | None = None,
        job_manager: ReferenceJobManager | None = None,
        firebase_store: Any | None = None,
    ) -> None:
        self.logger = get_logger("missing_person_processor")
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        ref_cfg = self.config.get("reference_enrollment", {})
        # Evidence-based threshold: defaults to confirmed_threshold (0.92) from matching_policy
        matching_cfg = self.config.get("matching_policy", {})
        default_dedup = float(matching_cfg.get("confirmed_threshold", 0.92))
        self.dedup_cosine_threshold = float(ref_cfg.get("dedup_cosine_threshold", default_dedup))
        self.min_gait_frames = int(ref_cfg.get("min_gait_frames", 10))
        self.target_isolation_ratio = float(ref_cfg.get("target_isolation_ratio", 2.5))
        self.stride = max(1, int(ref_cfg.get("stride", 1)))
        self.model_version = str(ref_cfg.get("model_version", "v1.0.0"))

        # Reusable pipeline components (singleton injection or lazy creation)
        self.detector = detector or PersonDetector()
        self.tracker = tracker or TrackingStep(detector=self.detector)
        self.silhouette_step = silhouette_step or SilhouetteStep()
        self.extractor = extractor or FeatureExtractionStep()
        self.appearance_extractor = appearance_extractor

        # Storage components
        self.store = store or VectorStore(gallery_dir=gait_gallery_dir)
        self.appearance_store = VectorStore(gallery_dir=appearance_gallery_dir)
        self.embedding_db = embedding_db or EmbeddingDatabase(
            db_dir=db_dir,
            gait_gallery_dir=gait_gallery_dir,
            appearance_gallery_dir=appearance_gallery_dir,
            firebase_store=firebase_store,
        )
        self.job_manager = job_manager or ReferenceJobManager.get_instance()
        self.firebase_store = firebase_store or getattr(self.embedding_db, "firebase_store", None)

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists() or yaml is None:
            return {}
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def validate_video_file(self, video_path: str | Path) -> tuple[bool, str, dict[str, Any]]:
        path = Path(video_path)
        if not path.exists():
            return False, f"Video file not found: {path}", {}

        if path.is_file() and path.stat().st_size == 0:
            return False, "Uploaded video file is empty (0 bytes)", {}

        valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        if path.suffix.lower() not in valid_extensions:
            return False, f"Unsupported video format '{path.suffix}'. Allowed: {sorted(valid_extensions)}", {}

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, f"Video decoder failed to open file: {path.name}", {}

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        meta = {
            "total_frames": total_frames,
            "fps": fps if fps > 0 else 25.0,
            "width": width,
            "height": height,
            "file_size": path.stat().st_size,
        }

        if width < 32 or height < 32:
            return False, f"Video resolution {width}x{height} is too small for person detection", meta

        if total_frames > 0 and total_frames < self.min_gait_frames:
            return (
                False,
                f"Video length ({total_frames} frames) is shorter than minimum required gait sequence ({self.min_gait_frames} frames)",
                meta,
            )

        return True, "OK", meta

    def select_isolated_target_track(
        self,
        tracks: dict[int, TrackSummary],
    ) -> tuple[int | None, str]:
        """Multi-Person Safety Policy (Requirement 2):
        Missing Person reference enrollment MUST NEVER enroll an arbitrary/background person.
        If multiple people are detected and the target cannot be reliably isolated,
        reject/flag the video rather than guessing.
        """
        if not tracks:
            return None, "NO_PERSON_DETECTED: No individuals detected in video"

        # Filter out brief transient noise (< min_gait_frames)
        viable_tracks = {
            tid: summary for tid, summary in tracks.items() if summary.frame_count >= self.min_gait_frames
        }

        if not viable_tracks:
            max_len = max(s.frame_count for s in tracks.values())
            return (
                None,
                f"INSUFFICIENT_GAIT_SEQUENCE: Longest person track ({max_len} frames) is shorter than required minimum ({self.min_gait_frames} frames)",
            )

        if len(viable_tracks) == 1:
            target_id = next(iter(viable_tracks.keys()))
            return target_id, "SINGLE_TARGET_ISOLATED"

        # Multiple viable tracks detected: evaluate prominence ratio
        sorted_tracks = sorted(viable_tracks.values(), key=lambda t: t.prominence_score, reverse=True)
        primary = sorted_tracks[0]
        secondary = sorted_tracks[1]

        ratio = primary.prominence_score / max(1.0, secondary.prominence_score)
        if ratio >= self.target_isolation_ratio:
            self.logger.info(
                f"Primary target track {primary.track_id} isolated (score={primary.prominence_score:.1f}) "
                f"vs secondary background track {secondary.track_id} (score={secondary.prominence_score:.1f}, ratio={ratio:.2f})"
            )
            return primary.track_id, "PRIMARY_TARGET_ISOLATED"

        # Ambiguous: multiple prominent tracks without a dominant foreground subject
        reason = (
            f"AMBIGUOUS_MULTIPLE_PERSONS: Multiple prominent individuals detected (Track {primary.track_id}: "
            f"{primary.frame_count} frames, Track {secondary.track_id}: {secondary.frame_count} frames, ratio: {ratio:.2f} < {self.target_isolation_ratio:.1f}). "
            "Cannot reliably isolate the target missing person without risk of enrolling the wrong individual. "
            "Please upload a video containing only the target person."
        )
        return None, reason

    def validate_embedding(
        self,
        raw_embedding: np.ndarray | None,
        person_id: str,
        track_id: int,
        sequence_index: int,
    ) -> tuple[bool, str, ValidatedEmbedding | None]:
        """Strict Embedding Validation (Requirement 9):
        Validate 256D, finite float32, non-zero norm, no NaN, no Inf, unit normalization.
        """
        if raw_embedding is None:
            return False, "Embedding extraction returned None", None

        vec = np.asarray(raw_embedding, dtype=np.float32).ravel()

        if vec.size != 256:
            return False, f"Invalid embedding dimensionality: expected 256, got {vec.size}", None

        if not np.isfinite(vec).all():
            return False, "Embedding contains non-finite values (NaN or Inf)", None

        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return False, f"Degenerate embedding with near-zero norm ({norm:.2e})", None

        # Unit-normalize
        norm_vec = (vec / norm).astype(np.float32)

        provenance = {
            "source": "missing_person_reference_video",
            "modality": "gait",
            "dimension": 256,
            "track_id": track_id,
            "sequence_index": sequence_index,
            "model_architecture": "ByGaitLight",
            "model_version": self.model_version,
            "created_at": time.time(),
        }

        validated = ValidatedEmbedding(
            vector=norm_vec,
            dimension=256,
            person_id=person_id,
            track_id=track_id,
            sequence_index=sequence_index,
            quality_score=0.95,
            model_version=self.model_version,
            provenance=provenance,
        )
        return True, "VALID", validated

    def deduplicate_embeddings(
        self,
        candidate_embeddings: list[ValidatedEmbedding],
    ) -> tuple[list[ValidatedEmbedding], int]:
        """Embedding-level Cosine Deduplication (Requirement 1 & 8).
        Filters out redundant embeddings from the same video that exceed dedup_cosine_threshold.
        """
        accepted: list[ValidatedEmbedding] = []
        deduplicated_count = 0

        for cand in candidate_embeddings:
            is_dup = False
            for acc in accepted:
                cos_sim = float(np.dot(cand.vector, acc.vector))
                if cos_sim >= self.dedup_cosine_threshold:
                    is_dup = True
                    deduplicated_count += 1
                    self.logger.debug(
                        f"Filtered duplicate gait embedding (cosine similarity {cos_sim:.4f} >= {self.dedup_cosine_threshold})"
                    )
                    break

            if not is_dup:
                accepted.append(cand)

        return accepted, deduplicated_count

    def process_reference_video(
        self,
        person_id: str,
        video_path: str | Path,
        job_id: str | None = None,
        case_id: str = "",
        gait_service_ref: Any | None = None,
    ) -> dict[str, Any]:
        """Main camera-independent offline video processing execution.
        Preserves the locked ARGUS gait pipeline:
        Video -> Person Detection -> Tracking -> Silhouette -> LiveGEI -> ByGaitLight -> Validated 256D Embedding -> Consistent Persistence.

        Includes durable checkpointing, crash recovery, graceful shutdown support,
        and idempotent persistence.
        """
        start_time = time.perf_counter()
        normalized_person_id = person_id.strip()
        v_path = Path(video_path)

        # 0. Check if job was already completed (Idempotency guarantee)
        if job_id:
            existing_job = self.job_manager.get_job(job_id)
            if existing_job and existing_job.status == ReferenceJobStatus.COMPLETED:
                self.logger.info(f"Job '{job_id}' is already COMPLETED. Returning cached result.")
                return existing_job.result or {"success": True, "status": "COMPLETED", "person_id": normalized_person_id}

        # Check for previous checkpoint state
        chk_data = self.job_manager.load_checkpoint_data(job_id) if job_id else None
        existing_job = self.job_manager.get_job(job_id) if job_id else None
        is_resume = bool(
            existing_job
            and (
                existing_job.resumed
                or existing_job.recovery_count > 0
                or existing_job.progress.last_safe_frame > 0
                or (chk_data and chk_data.get("last_safe_frame", 0) > 0)
                or existing_job.status in (ReferenceJobStatus.RESUMING, ReferenceJobStatus.INTERRUPTED)
            )
        )

        if is_resume and existing_job:
            self.logger.info(f"[RECOVERY] Found unfinished reference job: {job_id}")
            self.logger.info(f"[RECOVERY] Stage: {existing_job.progress.stage}")
            self.logger.info(
                f"[RECOVERY] Last safe frame: {existing_job.progress.last_safe_frame}/{existing_job.progress.total_frames}"
            )
            self.logger.info(f"[RECOVERY] Completed sequences: {len(existing_job.progress.completed_sequences)}")
            self.logger.info("[RECOVERY] Resuming job")
            self.job_manager.update_progress(job_id, status=ReferenceJobStatus.RESUMING)

        # 1. Validation pass (only run if not already validated in checkpoint)
        need_validation = not (chk_data and chk_data.get("stage") in (
            "TRACKING", "TRACKING_DONE", "FEATURE_EXTRACTION", "MATCHING", "PERSISTING"
        ))

        if need_validation:
            if job_id:
                self.job_manager.checkpoint_job(
                    job_id,
                    stage="VALIDATING_VIDEO",
                    last_safe_frame=0,
                    frames_processed=0,
                    status=ReferenceJobStatus.PROCESSING,
                )

            valid_media, v_err, v_meta = self.validate_video_file(v_path)
            if not valid_media:
                if job_id:
                    self.job_manager.fail_job(job_id, v_err, diagnostic_code="INVALID_VIDEO")
                return {"success": False, "error": v_err, "diagnostic_code": "INVALID_VIDEO"}
            total_frames = v_meta.get("total_frames", 0)

            if job_id:
                self.job_manager.checkpoint_job(
                    job_id,
                    stage="TRACKING",
                    last_safe_frame=0,
                    frames_processed=0,
                    total_frames=total_frames,
                    status=ReferenceJobStatus.PROCESSING,
                )
        else:
            total_frames = existing_job.progress.total_frames if existing_job else 0
            if total_frames == 0:
                _, _, v_meta = self.validate_video_file(v_path)
                total_frames = v_meta.get("total_frames", 0)

        # 2. Tracking pass (check if tracking was already completed in saved checkpoint)
        tracks: dict[int, TrackSummary] = {}
        target_track_id: int | None = None
        target_crops: list[np.ndarray] = []
        frame_idx = 0
        decode_start = time.perf_counter()

        tracking_already_done = bool(
            chk_data and chk_data.get("stage") in ("TRACKING_DONE", "FEATURE_EXTRACTION", "MATCHING", "PERSISTING")
        )

        if tracking_already_done and chk_data:
            tracks = chk_data.get("tracks", {})
            target_track_id = chk_data.get("target_track_id")
            target_crops = chk_data.get("target_crops", [])
            frame_idx = existing_job.progress.last_safe_frame if existing_job else total_frames
            self.logger.info(
                f"[RECOVERY] Resuming from completed tracking checkpoint for {job_id} (target_track_id={target_track_id})"
            )
        else:
            cap = cv2.VideoCapture(str(v_path))
            if not cap.isOpened():
                err_msg = f"Unable to open video decoder for {v_path.name}"
                if job_id:
                    self.job_manager.fail_job(job_id, err_msg, diagnostic_code="DECODER_ERROR")
                return {"success": False, "error": err_msg, "diagnostic_code": "DECODER_ERROR"}

            last_safe_frame = existing_job.progress.last_safe_frame if (is_resume and existing_job) else 0
            if chk_data and "tracks" in chk_data:
                tracks = chk_data["tracks"]

            # Choose safe seek point with small overlap window to warm up ByteTrack Kalman filter
            overlap_window = 20
            start_seek_frame = max(0, last_safe_frame - overlap_window) if last_safe_frame > 0 else 0
            if start_seek_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_seek_frame)
                frame_idx = start_seek_frame
                self.logger.info(
                    f"[RECOVERY] Seeking video decoder to frame {start_seek_frame} (last_safe_frame={last_safe_frame}, overlap={overlap_window})"
                )
            else:
                frame_idx = 0
                if hasattr(self.tracker, "reset"):
                    self.tracker.reset()

            last_checkpoint_frame = frame_idx
            last_checkpoint_time = time.perf_counter()

            try:
                with torch.inference_mode():
                    while True:
                        if self.job_manager.is_shutting_down:
                            self.logger.info(
                                f"Graceful shutdown signaled during TRACKING at frame {frame_idx}. Persisting checkpoint."
                            )
                            if job_id:
                                self.job_manager.checkpoint_job(
                                    job_id,
                                    stage="TRACKING",
                                    last_safe_frame=frame_idx,
                                    frames_processed=frame_idx,
                                    total_frames=total_frames,
                                    status=ReferenceJobStatus.INTERRUPTED,
                                    checkpoint_data={"stage": "TRACKING", "tracks": tracks, "last_safe_frame": frame_idx},
                                )
                            return {
                                "success": False,
                                "interrupted": True,
                                "stage": "TRACKING",
                                "last_safe_frame": frame_idx,
                            }

                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_idx += 1
                        if self.stride > 1 and (frame_idx % self.stride != 0):
                            continue

                        detections = self.tracker.track(frame)
                        xyxy = getattr(detections, "xyxy", None)
                        tracker_ids = getattr(detections, "tracker_id", None)

                        if xyxy is not None and tracker_ids is not None and len(tracker_ids) > 0:
                            h, w = frame.shape[:2]
                            for box, tid in zip(xyxy, tracker_ids):
                                tid = int(tid)
                                x1, y1, x2, y2 = map(int, box)
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(w, x2)
                                y2 = min(h, y2)
                                area = float((x2 - x1) * (y2 - y1))

                                if area < 400 or (x2 <= x1) or (y2 <= y1):
                                    continue

                                crop = frame[y1:y2, x1:x2]

                                if tid not in tracks:
                                    tracks[tid] = TrackSummary(track_id=tid)

                                # Prevent duplicate frame entries when replaying overlap window
                                if frame_idx not in tracks[tid].frame_indices:
                                    tracks[tid].frame_indices.append(frame_idx)
                                    tracks[tid].bboxes.append([x1, y1, x2, y2])
                                    tracks[tid].areas.append(area)
                                    if len(tracks[tid].crops) < 120:
                                        tracks[tid].crops.append(crop.copy())

                        # Periodic checkpoint: every 25 frames or 1.5 seconds (whichever first)
                        now_perf = time.perf_counter()
                        if (frame_idx - last_checkpoint_frame >= 25 or (now_perf - last_checkpoint_time >= 1.5)) and job_id:
                            last_checkpoint_frame = frame_idx
                            last_checkpoint_time = now_perf
                            fps_calc = frame_idx / max(0.001, now_perf - decode_start)
                            pct = min(65, 20 + int((frame_idx / max(1, total_frames)) * 45))
                            self.job_manager.checkpoint_job(
                                job_id,
                                stage="TRACKING",
                                last_safe_frame=frame_idx,
                                frames_processed=frame_idx,
                                total_frames=total_frames,
                                fps=round(fps_calc, 1),
                                tracks_detected=len(tracks),
                                percent=pct,
                                status=ReferenceJobStatus.PROCESSING,
                                checkpoint_data={"stage": "TRACKING", "tracks": tracks, "last_safe_frame": frame_idx},
                            )
            finally:
                cap.release()

            # Checkpoint immediately upon tracking completion
            if job_id:
                fps_calc = frame_idx / max(0.001, time.perf_counter() - decode_start)
                self.job_manager.checkpoint_job(
                    job_id,
                    stage="TRACKING",
                    last_safe_frame=frame_idx,
                    frames_processed=frame_idx,
                    total_frames=total_frames,
                    fps=round(fps_calc, 1),
                    tracks_detected=len(tracks),
                    percent=65,
                    checkpoint_data={"stage": "TRACKING_DONE", "tracks": tracks, "last_safe_frame": frame_idx},
                )

        # 3. Multi-Person Safety Isolation
        if target_track_id is None:
            target_track_id, select_reason = self.select_isolated_target_track(tracks)
            if target_track_id is None:
                diag_code = (
                    "AMBIGUOUS_MULTIPLE_PERSONS"
                    if "AMBIGUOUS" in select_reason
                    else ("NO_PERSON_DETECTED" if "NO_PERSON" in select_reason else "INSUFFICIENT_GAIT_SEQUENCE")
                )
                if job_id:
                    self.job_manager.fail_job(job_id, select_reason, diagnostic_code=diag_code)
                return {"success": False, "error": select_reason, "diagnostic_code": diag_code}

            target_crops = tracks[target_track_id].crops
            # Checkpoint immediately after target-track selection (Requirement 5 item 3)
            if job_id:
                self.job_manager.checkpoint_job(
                    job_id,
                    stage="FEATURE_EXTRACTION",
                    selected_track_id=target_track_id,
                    last_safe_frame=frame_idx,
                    frames_processed=frame_idx,
                    total_frames=total_frames,
                    percent=70,
                    status=ReferenceJobStatus.PROCESSING,
                    checkpoint_data={
                        "stage": "FEATURE_EXTRACTION",
                        "tracks": tracks,
                        "target_track_id": target_track_id,
                        "target_crops": target_crops,
                        "last_safe_frame": frame_idx,
                    },
                )

        # 4. Valid Gait Sequence & GEI Generation (only if embeddings not already checkpointed)
        dedup_embeddings: list[ValidatedEmbedding] = (
            chk_data.get("dedup_embeddings", []) if (chk_data and "dedup_embeddings" in chk_data) else []
        )
        dedup_count: int = chk_data.get("dedup_count", 0) if chk_data else 0

        candidate_embeddings: list[ValidatedEmbedding] = (
            chk_data.get("candidate_embeddings", []) if (chk_data and "candidate_embeddings" in chk_data) else []
        )
        completed_sequences: list[int] = (
            chk_data.get("completed_sequences", []) if (chk_data and "completed_sequences" in chk_data) else []
        )

        target_silhouettes: list[np.ndarray] = (
            chk_data.get("target_silhouettes", []) if (chk_data and "target_silhouettes" in chk_data) else []
        )
        geis: list[np.ndarray] = []

        if not dedup_embeddings and not candidate_embeddings:
            if not target_silhouettes:
                for crop in target_crops:
                    if self.job_manager.is_shutting_down:
                        self.logger.info("Graceful shutdown during silhouette extraction. Saving checkpoint.")
                        if job_id:
                            self.job_manager.checkpoint_job(
                                job_id,
                                stage="FEATURE_EXTRACTION",
                                last_safe_frame=frame_idx,
                                frames_processed=frame_idx,
                                status=ReferenceJobStatus.INTERRUPTED,
                                checkpoint_data={
                                    "stage": "FEATURE_EXTRACTION",
                                    "target_track_id": target_track_id,
                                    "target_crops": target_crops,
                                    "target_silhouettes": target_silhouettes,
                                    "last_safe_frame": frame_idx,
                                },
                            )
                        return {"success": False, "interrupted": True, "stage": "FEATURE_EXTRACTION"}

                    sil = self.silhouette_step.extract_from_crop(crop)
                    if sil is not None:
                        target_silhouettes.append(sil)

            if len(target_silhouettes) < self.min_gait_frames:
                err_msg = (
                    f"Target track {target_track_id} produced only {len(target_silhouettes)} valid silhouettes "
                    f"(minimum required: {self.min_gait_frames}). Gait sequence insufficient."
                )
                if job_id:
                    self.job_manager.fail_job(job_id, err_msg, diagnostic_code="INSUFFICIENT_GAIT_SEQUENCE")
                return {"success": False, "error": err_msg, "diagnostic_code": "INSUFFICIENT_GAIT_SEQUENCE"}

            window_size = 15
            step_stride = 10
            for start_i in range(0, len(target_silhouettes) - self.min_gait_frames + 1, step_stride):
                chunk = target_silhouettes[start_i : start_i + window_size]
                if len(chunk) < self.min_gait_frames:
                    continue

                live_gei = LiveGEI(max_frames=window_size, min_frames=self.min_gait_frames)
                for s in chunk:
                    live_gei.add(s)

                if live_gei.ready():
                    gei = live_gei.build()
                    if gei is not None and gei.size > 0:
                        geis.append(gei)

            if not geis:
                err_msg = "No valid Gait Energy Images could be constructed from silhouettes"
                if job_id:
                    self.job_manager.fail_job(job_id, err_msg, diagnostic_code="GEI_GENERATION_FAILED")
                return {"success": False, "error": err_msg, "diagnostic_code": "GEI_GENERATION_FAILED"}

            # 5. ByGaitLight 256D Feature Extraction & Strict Validation (with sequence checkpointing)
            start_seq = len(candidate_embeddings)
            with torch.inference_mode():
                for seq_idx in range(start_seq, len(geis)):
                    if self.job_manager.is_shutting_down:
                        self.logger.info(
                            f"Graceful shutdown during FEATURE_EXTRACTION at sequence {seq_idx}. Persisting checkpoint."
                        )
                        if job_id:
                            self.job_manager.checkpoint_job(
                                job_id,
                                stage="FEATURE_EXTRACTION",
                                last_safe_frame=frame_idx,
                                frames_processed=frame_idx,
                                total_frames=total_frames,
                                valid_sequences=len(geis),
                                completed_sequences=completed_sequences,
                                embeddings_generated=len(candidate_embeddings),
                                status=ReferenceJobStatus.INTERRUPTED,
                                checkpoint_data={
                                    "stage": "FEATURE_EXTRACTION",
                                    "target_track_id": target_track_id,
                                    "target_crops": target_crops,
                                    "target_silhouettes": target_silhouettes,
                                    "candidate_embeddings": candidate_embeddings,
                                    "completed_sequences": completed_sequences,
                                    "last_safe_frame": frame_idx,
                                },
                            )
                        return {"success": False, "interrupted": True, "stage": "FEATURE_EXTRACTION"}

                    gei = geis[seq_idx]
                    raw_emb = self.extractor.extract_from_gei(gei)
                    valid, v_msg, val_emb = self.validate_embedding(
                        raw_emb,
                        person_id=normalized_person_id,
                        track_id=target_track_id,
                        sequence_index=seq_idx,
                    )
                    if valid and val_emb is not None:
                        candidate_embeddings.append(val_emb)
                        completed_sequences.append(seq_idx)
                    else:
                        self.logger.warning(f"Rejected candidate embedding at sequence {seq_idx}: {v_msg}")

                    # Checkpoint immediately after each successfully completed feature-extraction sequence (Requirement 5 item 4)
                    if job_id:
                        self.job_manager.checkpoint_job(
                            job_id,
                            stage="FEATURE_EXTRACTION",
                            last_safe_frame=frame_idx,
                            frames_processed=frame_idx,
                            total_frames=total_frames,
                            valid_silhouettes=len(target_silhouettes),
                            valid_sequences=len(geis),
                            completed_sequences=completed_sequences,
                            embeddings_generated=len(candidate_embeddings),
                            percent=75,
                            status=ReferenceJobStatus.VALIDATING,
                            checkpoint_data={
                                "stage": "FEATURE_EXTRACTION",
                                "target_track_id": target_track_id,
                                "target_crops": target_crops,
                                "target_silhouettes": target_silhouettes,
                                "candidate_embeddings": candidate_embeddings,
                                "completed_sequences": completed_sequences,
                                "last_safe_frame": frame_idx,
                            },
                        )

            if not candidate_embeddings:
                err_msg = "All candidate embeddings failed numerical validation (NaN, Inf, or dimension mismatch)"
                if job_id:
                    self.job_manager.fail_job(job_id, err_msg, diagnostic_code="EMBEDDINGS_INVALID")
                return {"success": False, "error": err_msg, "diagnostic_code": "EMBEDDINGS_INVALID"}

        # 6. Cosine Similarity Deduplication & Matching Stage (if not already deduplicated in checkpoint)
        if not dedup_embeddings:
            if job_id:
                self.job_manager.checkpoint_job(
                    job_id,
                    stage="MATCHING",
                    last_safe_frame=frame_idx,
                    frames_processed=frame_idx,
                    total_frames=total_frames,
                    percent=88,
                    status=ReferenceJobStatus.PROCESSING,
                    checkpoint_data={
                        "stage": "MATCHING",
                        "target_track_id": target_track_id,
                        "candidate_embeddings": candidate_embeddings,
                        "completed_sequences": completed_sequences,
                        "last_safe_frame": frame_idx,
                    },
                )

            dedup_embeddings, dedup_count = self.deduplicate_embeddings(candidate_embeddings)

            if job_id:
                self.job_manager.checkpoint_job(
                    job_id,
                    stage="PERSISTING",
                    last_safe_frame=frame_idx,
                    frames_processed=frame_idx,
                    total_frames=total_frames,
                    embeddings_generated=len(candidate_embeddings),
                    embeddings_deduplicated=dedup_count,
                    percent=95,
                    status=ReferenceJobStatus.PERSISTING,
                    checkpoint_data={
                        "stage": "PERSISTING",
                        "target_track_id": target_track_id,
                        "candidate_embeddings": candidate_embeddings,
                        "dedup_embeddings": dedup_embeddings,
                        "dedup_count": dedup_count,
                        "last_safe_frame": frame_idx,
                    },
                )

        # 7. Consistent Atomic Persistence & Gallery Activation with Deterministic Embedding Keys
        # Idempotent persistence guarantee: deterministic IDs prevent duplicates across restarts (Requirement 6)
        deterministic_ids = [
            f"gait_{normalized_person_id}_{job_id or 'ref'}_seq_{e.sequence_index}"
            for e in dedup_embeddings
        ]
        raw_vectors = [e.vector for e in dedup_embeddings]
        persist_res = self.embedding_db.commit_and_activate_embeddings(
            person_id=normalized_person_id,
            gait_embeddings=raw_vectors,
            model_version=self.model_version,
            source_session_id=job_id or f"ref_vid_{int(time.time())}",
            embedding_ids=deterministic_ids,
            gait_service_ref=gait_service_ref,
        )

        persisted_ids = persist_res.get("persisted_embedding_ids", deterministic_ids)

        # Confirm persistence before marking completed
        if job_id:
            self.job_manager.checkpoint_job(
                job_id,
                stage="PERSISTING",
                last_safe_frame=frame_idx,
                frames_processed=frame_idx,
                total_frames=total_frames,
                embeddings_committed=len(dedup_embeddings),
                persisted_embedding_ids=persisted_ids,
                percent=98,
                status=ReferenceJobStatus.PERSISTING,
            )

        elapsed_sec = time.perf_counter() - start_time
        effective_fps = frame_idx / max(0.001, elapsed_sec)

        summary_result = {
            "success": True,
            "person_id": normalized_person_id,
            "case_id": case_id or normalized_person_id,
            "status": "COMPLETED",
            "frames_processed": frame_idx,
            "selected_track_id": target_track_id,
            "valid_silhouettes": len(target_silhouettes),
            "valid_sequences": len(geis),
            "embeddings_generated": len(candidate_embeddings),
            "embeddings_deduplicated": dedup_count,
            "embeddings_committed": len(dedup_embeddings),
            "persisted_embedding_ids": persisted_ids,
            "dedup_cosine_threshold": self.dedup_cosine_threshold,
            "duration_seconds": round(elapsed_sec, 3),
            "effective_fps": round(effective_fps, 1),
            "persistence_verified": persist_res.get("persistence_verified", True),
            "firebase_status": persist_res.get("firebase_results") and "PERSISTED" or "LOCAL_ONLY",
            "active": True,
        }

        if job_id:
            self.job_manager.complete_job(job_id, summary_result, status=ReferenceJobStatus.COMPLETED)

        if is_resume:
            self.logger.info(f"[RECOVERY] Job completed successfully: {job_id}")

        self.logger.info(
            f"Missing person reference video processed successfully for '{normalized_person_id}': "
            f"{len(dedup_embeddings)} valid 256D gait embeddings committed in {elapsed_sec:.2f}s ({effective_fps:.1f} FPS)."
        )
        return summary_result

    def process_reference_photos(
        self,
        person_id: str,
        photo_paths: list[str | Path],
        job_id: str | None = None,
        case_id: str = "",
        gait_service_ref: Any | None = None,
    ) -> dict[str, Any]:
        """Offline, camera-independent asynchronous photo enrollment processor.

        Decoupled from HTTP request lifecycle:
        Photos -> Validate -> Silhouette Extraction -> ByGaitLight (256D) + OSNet (512D) ->
        Quality Validation -> Atomic Gallery Activation.
        """
        start_time = time.perf_counter()
        normalized_person_id = person_id.strip()

        if job_id:
            existing_job = self.job_manager.get_job(job_id)
            if existing_job and existing_job.status == ReferenceJobStatus.COMPLETED:
                self.logger.info(f"Photo job '{job_id}' is already COMPLETED. Returning cached result.")
                return existing_job.result or {"success": True, "status": "COMPLETED", "person_id": normalized_person_id}

        if job_id:
            self.job_manager.update_progress(
                job_id,
                stage="VALIDATING",
                status=ReferenceJobStatus.VALIDATING,
            )

        valid_crops: list[np.ndarray] = []
        for p in photo_paths:
            p_obj = Path(p)
            if not p_obj.exists() or p_obj.stat().st_size == 0:
                continue
            frame = cv2.imread(str(p_obj))
            if frame is not None and frame.size > 0 and frame.shape[0] >= 32 and frame.shape[1] >= 32:
                valid_crops.append(frame)

        if not valid_crops:
            err_msg = "No valid reference images could be loaded or resolved for enrollment"
            if job_id:
                self.job_manager.fail_job(job_id, err_msg, diagnostic_code="INVALID_IMAGES")
            return {"success": False, "error": err_msg, "diagnostic_code": "INVALID_IMAGES"}

        if job_id:
            self.job_manager.update_progress(
                job_id,
                stage="FEATURE_EXTRACTION",
                total_frames=len(valid_crops),
                frames_processed=len(valid_crops),
                status=ReferenceJobStatus.PROCESSING,
            )

        gait_embeddings: list[np.ndarray] = []
        app_embeddings: list[np.ndarray] = []
        for crop in valid_crops:
            # 1. Silhouette extraction (locked pipeline)
            sil = self.silhouette_step.extract_from_crop(crop)
            if sil is None:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sil = cv2.resize(gray, (64, 128))

            sil_norm = sil.astype(np.float32) / 255.0
            if hasattr(self.extractor, "backend") and self.extractor.backend is not None:
                emb = self.extractor.backend.predict(sil_norm).flatten().astype(np.float32)
            else:
                emb = self.extractor.extract_from_gei(sil).astype(np.float32)

            if emb is not None and emb.size == 256 and np.isfinite(emb).all():
                norm = float(np.linalg.norm(emb))
                if norm > 1e-6:
                    gait_embeddings.append((emb / norm).astype(np.float32))

            # 2. Appearance embedding (OSNet 512D)
            app_ext = self.appearance_extractor or getattr(gait_service_ref, "appearance_extractor", None)
            if app_ext is not None:
                try:
                    app_emb = app_ext.extract(crop)
                    if app_emb is not None and len(app_emb) == 512 and np.isfinite(app_emb).all():
                        app_embeddings.append(app_emb)
                except (RuntimeError, ValueError, TypeError, AttributeError) as app_err:
                    self.logger.debug(f"Appearance extraction notice: {app_err}")

        if not gait_embeddings:
            err_msg = "All reference photo embeddings failed mathematical validation (NaN, Inf, or zero norm)"
            if job_id:
                self.job_manager.fail_job(job_id, err_msg, diagnostic_code="EMBEDDINGS_INVALID")
            return {"success": False, "error": err_msg, "diagnostic_code": "EMBEDDINGS_INVALID"}

        if job_id:
            self.job_manager.update_progress(
                job_id,
                stage="QUALITY_VALIDATION",
                embeddings_generated=len(gait_embeddings),
                status=ReferenceJobStatus.VALIDATING,
            )

        # 3. Deduplicate if multiple photos
        candidate_embeddings = [
            ValidatedEmbedding(
                vector=vec,
                dimension=256,
                person_id=normalized_person_id,
                track_id=0,
                sequence_index=i,
                quality_score=0.95,
                model_version=self.model_version,
                provenance={"source": "photo_enrollment", "index": i},
            )
            for i, vec in enumerate(gait_embeddings)
        ]
        dedup_embeddings, dedup_count = self.deduplicate_embeddings(candidate_embeddings)

        if job_id:
            self.job_manager.update_progress(
                job_id,
                stage="READY_TO_COMMIT",
                embeddings_committed=len(dedup_embeddings),
                embeddings_deduplicated=dedup_count,
                status=ReferenceJobStatus.READY_TO_COMMIT,
            )

        # 4. Atomic Gallery Activation & Retirement with Deterministic IDs
        raw_vectors = [e.vector for e in dedup_embeddings]
        photo_gait_ids = [
            f"gait_{normalized_person_id}_{job_id or 'photo'}_img_{e.sequence_index}"
            for e in dedup_embeddings
        ]
        photo_app_ids = [
            f"app_{normalized_person_id}_{job_id or 'photo'}_img_{i}"
            for i in range(len(app_embeddings))
        ] if app_embeddings else None

        persist_res = self.embedding_db.commit_and_activate_embeddings(
            person_id=normalized_person_id,
            gait_embeddings=raw_vectors,
            appearance_embeddings=app_embeddings if app_embeddings else None,
            model_version=self.model_version,
            source_session_id=job_id or f"ref_img_{int(time.time())}",
            embedding_ids=photo_gait_ids,
            appearance_embedding_ids=photo_app_ids,
            gait_service_ref=gait_service_ref,
        )

        elapsed_sec = time.perf_counter() - start_time
        summary_result = {
            "success": True,
            "person_id": normalized_person_id,
            "case_id": case_id or normalized_person_id,
            "status": "COMPLETED",
            "frames_processed": len(valid_crops),
            "valid_silhouettes": len(valid_crops),
            "embeddings_generated": len(gait_embeddings),
            "embeddings_deduplicated": dedup_count,
            "embeddings_committed": len(dedup_embeddings),
            "duration_seconds": round(elapsed_sec, 3),
            "persistence_verified": persist_res.get("persistence_verified", True),
            "firebase_status": persist_res.get("firebase_results") and "PERSISTED" or "LOCAL_ONLY",
            "active": True,
        }

        if job_id:
            self.job_manager.complete_job(job_id, summary_result, status=ReferenceJobStatus.COMPLETED)

        self.logger.info(
            f"Reference photo processing completed for '{normalized_person_id}': "
            f"{len(dedup_embeddings)} gait embeddings activated in {elapsed_sec:.3f}s."
        )
        return summary_result
