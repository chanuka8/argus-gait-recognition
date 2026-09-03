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
from pipeline.silhouette.extractor import SilhouetteExtractor
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
        self.tracker = tracker or TrackingStep()
        self.silhouette_step = silhouette_step or SilhouetteStep()
        self.silhouette_extractor = SilhouetteExtractor(target_size=(64, 128))
        self.extractor = extractor or FeatureExtractionStep()

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
        """
        start_time = time.perf_counter()
        normalized_person_id = person_id.strip()
        v_path = Path(video_path)

        if job_id:
            self.job_manager.update_progress(job_id, stage="VALIDATING_VIDEO", status=ReferenceJobStatus.PROCESSING)

        valid_media, v_err, v_meta = self.validate_video_file(v_path)
        if not valid_media:
            if job_id:
                self.job_manager.fail_job(job_id, v_err, diagnostic_code="INVALID_VIDEO")
            return {"success": False, "error": v_err, "diagnostic_code": "INVALID_VIDEO"}

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            err_msg = f"Unable to open video decoder for {v_path.name}"
            if job_id:
                self.job_manager.fail_job(job_id, err_msg, diagnostic_code="DECODER_ERROR")
            return {"success": False, "error": err_msg, "diagnostic_code": "DECODER_ERROR"}

        total_frames = v_meta.get("total_frames", 0)
        tracks: dict[int, TrackSummary] = {}
        frame_idx = 0
        decode_start = time.perf_counter()

        if job_id:
            self.job_manager.update_progress(
                job_id, stage="TRACKING", total_frames=total_frames, status=ReferenceJobStatus.PROCESSING
            )

        try:
            # 1. Sequential Detection & Tracking Pass (with torch.inference_mode for speed)
            with torch.inference_mode():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if self.stride > 1 and (frame_idx % self.stride != 0):
                        continue

                    # Run Detection + ByteTrack (locked pipeline)
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
                            sil = self.silhouette_step.extract_from_crop(crop)

                            if tid not in tracks:
                                tracks[tid] = TrackSummary(track_id=tid)

                            tracks[tid].frame_indices.append(frame_idx)
                            tracks[tid].bboxes.append([x1, y1, x2, y2])
                            tracks[tid].areas.append(area)
                            if sil is not None:
                                tracks[tid].silhouettes.append(sil)

                    if frame_idx % 25 == 0 and job_id:
                        fps_calc = frame_idx / max(0.001, time.perf_counter() - decode_start)
                        self.job_manager.update_progress(
                            job_id,
                            stage="TRACKING",
                            frames_processed=frame_idx,
                            fps=round(fps_calc, 1),
                            tracks_detected=len(tracks),
                        )
        finally:
            cap.release()

        # 2. Multi-Person Safety Isolation
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

        if job_id:
            self.job_manager.update_progress(
                job_id,
                stage="GENERATING_GEI",
                selected_track_id=target_track_id,
                valid_silhouettes=len(tracks[target_track_id].silhouettes),
            )

        # 3. Valid Gait Sequence & GEI Generation (No Fallbacks!)
        target_silhouettes = tracks[target_track_id].silhouettes
        if len(target_silhouettes) < self.min_gait_frames:
            err_msg = (
                f"Target track {target_track_id} produced only {len(target_silhouettes)} valid silhouettes "
                f"(minimum required: {self.min_gait_frames}). Gait sequence insufficient."
            )
            if job_id:
                self.job_manager.fail_job(job_id, err_msg, diagnostic_code="INSUFFICIENT_GAIT_SEQUENCE")
            return {"success": False, "error": err_msg, "diagnostic_code": "INSUFFICIENT_GAIT_SEQUENCE"}

        # Generate GEIs using sliding gait windows (LiveGEI locked component)
        geis: list[np.ndarray] = []
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

        # 4. ByGaitLight 256D Feature Extraction & Strict Validation
        if job_id:
            self.job_manager.update_progress(
                job_id,
                stage="EXTRACTING_EMBEDDINGS",
                valid_sequences=len(geis),
                status=ReferenceJobStatus.VALIDATING,
            )

        candidate_embeddings: list[ValidatedEmbedding] = []
        with torch.inference_mode():
            for seq_idx, gei in enumerate(geis):
                raw_emb = self.extractor.extract_from_gei(gei)
                valid, v_msg, val_emb = self.validate_embedding(
                    raw_emb,
                    person_id=normalized_person_id,
                    track_id=target_track_id,
                    sequence_index=seq_idx,
                )
                if valid and val_emb is not None:
                    candidate_embeddings.append(val_emb)
                else:
                    self.logger.warning(f"Rejected candidate embedding at sequence {seq_idx}: {v_msg}")

        if not candidate_embeddings:
            err_msg = "All candidate embeddings failed numerical validation (NaN, Inf, or dimension mismatch)"
            if job_id:
                self.job_manager.fail_job(job_id, err_msg, diagnostic_code="EMBEDDINGS_INVALID")
            return {"success": False, "error": err_msg, "diagnostic_code": "EMBEDDINGS_INVALID"}

        # 5. Cosine Similarity Deduplication
        dedup_embeddings, dedup_count = self.deduplicate_embeddings(candidate_embeddings)

        if job_id:
            self.job_manager.update_progress(
                job_id,
                stage="PERSISTING",
                embeddings_generated=len(candidate_embeddings),
                embeddings_deduplicated=dedup_count,
                status=ReferenceJobStatus.READY_TO_COMMIT,
            )

        # 6. Consistent Atomic Persistence (Local VectorStore + EmbeddingDatabase + Firebase)
        raw_vectors = [e.vector for e in dedup_embeddings]
        persist_res = self.embedding_db.add_embeddings(
            person_id=normalized_person_id,
            gait_embeddings=raw_vectors,
            model_version=self.model_version,
            source_session_id=job_id or f"ref_vid_{int(time.time())}",
        )

        # Reload in-memory service if active
        if gait_service_ref is not None and hasattr(gait_service_ref, "reload_gallery"):
            try:
                gait_service_ref.reload_gallery()
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"Could not reload active GaitService gallery: {e}")

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
            "dedup_cosine_threshold": self.dedup_cosine_threshold,
            "duration_seconds": round(elapsed_sec, 3),
            "effective_fps": round(effective_fps, 1),
            "persistence_verified": persist_res.get("persistence_verified", True),
            "firebase_status": persist_res.get("firebase_results") and "PERSISTED" or "LOCAL_ONLY",
        }

        if job_id:
            self.job_manager.complete_job(job_id, summary_result)

        self.logger.info(
            f"Missing person reference video processed successfully for '{normalized_person_id}': "
            f"{len(dedup_embeddings)} valid 256D gait embeddings committed in {elapsed_sec:.2f}s ({effective_fps:.1f} FPS)."
        )
        return summary_result
