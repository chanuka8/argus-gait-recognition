import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from typing import Any

import cv2
import numpy as np

from intelligence.open_set_recognizer import OpenSetRecognizer
from monitoring.logging_config import get_logger
from pipeline.detection.person_detector import PersonDetector
from pipeline.gei.stream_gei_builder import StreamGEIBuilder
from pipeline.silhouette.extractor import SilhouetteExtractor
from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.steps.matching_step import MatchingStep
from pipeline.tracking.tracker import PersonTracker
from security_layer.credentials import sanitize_rtsp_url


@dataclass
class RecognitionResult:
    """Structured recognition result for a single tracked subject."""

    camera_id: str
    track_id: int
    identity: str
    similarity: float
    confidence: float
    decision: str
    status: str
    bbox: list[int]
    timestamp: float
    iso_timestamp: str
    gei_frames: int = 0
    appearance_identity: str = "UNKNOWN_PERSON"
    appearance_score: float = 0.0
    appearance_status: str = "UNKNOWN"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecognitionResultCache:
    """
    Thread-safe recognition result cache scoped by (camera_id, track_id).
    Enforces TTL expiration and track eviction to prevent stale overlays and memory growth.
    """

    def __init__(self, ttl_seconds: float = 2.5) -> None:
        self.ttl_seconds = max(0.1, float(ttl_seconds))
        self._cache: dict[tuple[str, int], RecognitionResult] = {}
        self._lock = threading.RLock()

    def get(self, camera_id: str, track_id: int) -> RecognitionResult | None:
        key = (camera_id, track_id)
        with self._lock:
            res = self._cache.get(key)
            if res is None:
                return None
            if time.monotonic() - res.timestamp > self.ttl_seconds:
                return None
            return res

    def put(self, result: RecognitionResult) -> None:
        key = (result.camera_id, result.track_id)
        with self._lock:
            self._cache[key] = result

    def get_active_tracks(self, camera_id: str) -> list[RecognitionResult]:
        now = time.monotonic()
        with self._lock:
            return [
                res
                for (cid, _), res in self._cache.items()
                if cid == camera_id and (now - res.timestamp <= self.ttl_seconds)
            ]

    def cleanup_inactive(self, camera_id: str, max_idle_seconds: float = 5.0) -> list[int]:
        now = time.monotonic()
        evicted = []
        with self._lock:
            for (cid, tid), res in list(self._cache.items()):
                if cid == camera_id and (now - res.timestamp > max_idle_seconds):
                    self._cache.pop((cid, tid), None)
                    evicted.append(tid)
        return evicted

    def clear_camera(self, camera_id: str) -> None:
        with self._lock:
            for cid, tid in list(self._cache.keys()):
                if cid == camera_id:
                    self._cache.pop((cid, tid), None)


class RecognitionWorker:
    """
    Decoupled Asynchronous Recognition Worker.
    Pulls latest frames from bounded queue, executes detection, tracking, GEI accumulation,
    ByGaitLight matching, and parallel Appearance embedding matching, and updates the shared recognition cache.
    """

    def __init__(
        self,
        camera_id: str,
        config: dict | None = None,
        cache: RecognitionResultCache | None = None,
        detector: PersonDetector | None = None,
        tracker: PersonTracker | None = None,
        silhouette_extractor: SilhouetteExtractor | None = None,
        gei_builder: StreamGEIBuilder | None = None,
        extractor: FeatureExtractionStep | None = None,
        matcher: MatchingStep | None = None,
        open_set_recognizer: OpenSetRecognizer | None = None,
        gallery_features: np.ndarray | None = None,
        gallery_labels: list | None = None,
        metadata: list | dict | None = None,
        appearance_extractor: Any | None = None,
        appearance_matcher: Any | None = None,
        appearance_gallery_features: np.ndarray | None = None,
        appearance_gallery_labels: list | None = None,
        appearance_metadata: Any | None = None,
        fusion_engine: Any | None = None,
        track_aggregator: Any | None = None,
        operational_collector: Any | None = None,
        event_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.config = config or {}
        self._logger = get_logger(f"recognition.{camera_id}")

        self.target_fps = max(1.0, float(self.config.get("target_fps", 8.0)))
        self._frame_interval = 1.0 / self.target_fps
        self.cooldown_seconds = max(0.1, float(self.config.get("cooldown_seconds", 1.5)))
        self.threshold = float(self.config.get("threshold", 0.85))

        self.cache = cache or RecognitionResultCache(ttl_seconds=float(self.config.get("result_ttl", 2.5)))

        self.detector = detector or PersonDetector()
        self.tracker = tracker or PersonTracker()
        self.silhouette_extractor = silhouette_extractor or SilhouetteExtractor(target_size=(64, 128))
        self.gei_builder = gei_builder or StreamGEIBuilder()
        self.extractor = extractor or FeatureExtractionStep()
        self.matcher = matcher or MatchingStep(threshold=self.threshold)
        self.open_set_recognizer = open_set_recognizer or OpenSetRecognizer(known_threshold=self.threshold)

        # Appearance Branch Components
        self.appearance_extractor = appearance_extractor
        if self.appearance_extractor is None:
            try:
                from intelligence.appearance_embedding import AppearanceEmbeddingExtractor

                self.appearance_extractor = AppearanceEmbeddingExtractor(
                    update_interval=int(self.config.get("appearance_update_interval", 8)),
                )
            except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
                self._logger.debug(f"Appearance extractor init skipped: {exc}")
                self.appearance_extractor = None

        self.appearance_matcher = appearance_matcher
        if self.appearance_matcher is None:
            try:
                from pipeline.steps.appearance_matching_step import AppearanceMatchingStep

                self.appearance_matcher = AppearanceMatchingStep(
                    threshold=float(self.config.get("appearance_threshold", 0.60)),
                )
            except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
                self._logger.debug(f"Appearance matcher init skipped: {exc}")
                self.appearance_matcher = None

        self.appearance_gallery_features = appearance_gallery_features
        self.appearance_gallery_labels = list(appearance_gallery_labels) if appearance_gallery_labels is not None else []
        if isinstance(appearance_metadata, dict):
            self.appearance_metadata = appearance_metadata
        elif isinstance(appearance_metadata, list) and len(appearance_metadata) > 0 and isinstance(appearance_metadata[0], dict):
            self.appearance_metadata = {str(m.get("person_id", m.get("label", i))): m for i, m in enumerate(appearance_metadata)}
        else:
            self.appearance_metadata = {str(lbl): {"status": "ACTIVE", "enabled": True} for lbl in self.appearance_gallery_labels}

        self.fusion_engine = fusion_engine
        if self.fusion_engine is None:
            try:
                from intelligence.dual_modal_fusion import DualModalFusion

                fusion_cfg = self.config.get("dual_modal_fusion", self.config.get("fusion", {}))
                self.fusion_engine = DualModalFusion.from_config(fusion_cfg)
            except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
                self._logger.debug(f"Dual-Modal Fusion init skipped: {exc}")
                self.fusion_engine = None

        # Temporal Track Aggregation Component
        self.track_aggregator = track_aggregator
        if self.track_aggregator is None:
            try:
                from intelligence.track_identity_aggregator import TrackIdentityAggregator

                temp_cfg = self.config.get("temporal_aggregation", self.config.get("temporal_verification", {}))
                if temp_cfg.get("enabled", True):
                    self.track_aggregator = TrackIdentityAggregator.from_config(self.config)
            except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
                self._logger.debug(f"TrackIdentityAggregator init skipped: {exc}")
                self.track_aggregator = None

        # Operational Observation Collector Component
        self.operational_collector = operational_collector
        if self.operational_collector is None:
            try:
                from intelligence.operational_embedding_collector import OperationalEmbeddingCollector

                self.operational_collector = OperationalEmbeddingCollector()
            except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
                self._logger.debug(f"OperationalEmbeddingCollector init skipped: {exc}")
                self.operational_collector = None

        self._input_queue: Queue[np.ndarray] = Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

        self._last_recognition_at: str | None = None
        self.gallery_features = gallery_features
        self.gallery_labels = list(gallery_labels) if gallery_labels is not None else []
        if isinstance(metadata, dict):
            self.metadata = metadata
        elif isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
            self.metadata = {str(m.get("person_id", m.get("label", i))): m for i, m in enumerate(metadata)}
        else:
            self.metadata = {str(lbl): {"status": "ACTIVE", "enabled": True} for lbl in self.gallery_labels}

        self.event_callback = event_callback

        self._last_recognition_times: dict[int, float] = {}
        self._frame_count = 0
        self._recognition_count = 0
        self._active_track_count = 0

    def update_appearance_gallery(
        self,
        gallery_features: np.ndarray | None,
        gallery_labels: list[str] | None,
        metadata: Any | None = None,
    ) -> None:
        """Update reference appearance gallery features, labels, and metadata at runtime."""
        with self._lock:
            self.appearance_gallery_features = gallery_features
            self.appearance_gallery_labels = list(gallery_labels) if gallery_labels is not None else []
            if isinstance(metadata, dict):
                self.appearance_metadata = metadata
            elif isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
                self.appearance_metadata = {str(m.get("person_id", m.get("label", i))): m for i, m in enumerate(metadata)}
            else:
                self.appearance_metadata = {str(lbl): {"status": "ACTIVE", "enabled": True} for lbl in self.appearance_gallery_labels}
            self._logger.info(
                f"Updated live appearance gallery for camera {self.camera_id}: {len(self.appearance_gallery_labels)} identities"
            )

    def update_gallery(
        self,
        gallery_features: np.ndarray | None,
        gallery_labels: list[str] | None,
        metadata: Any | None = None,
    ) -> None:
        """Update reference gallery features, labels, and metadata at runtime."""
        with self._lock:
            self.gallery_features = gallery_features
            self.gallery_labels = list(gallery_labels) if gallery_labels is not None else []
            if isinstance(metadata, dict):
                self.metadata = metadata
            elif isinstance(metadata, list) and len(metadata) > 0 and isinstance(metadata[0], dict):
                self.metadata = {str(m.get("person_id", m.get("label", i))): m for i, m in enumerate(metadata)}
            else:
                self.metadata = {str(lbl): {"status": "ACTIVE", "enabled": True} for lbl in self.gallery_labels}
            self._logger.info(
                f"Updated live recognition gallery for camera {self.camera_id}: {len(self.gallery_labels)} identities"
            )

    def put_frame(self, frame: np.ndarray) -> bool:
        """
        Put latest captured frame into recognition queue with non-blocking drop policy.
        Never blocks the camera capture thread.
        """
        if self._stop_event.is_set() or frame is None or frame.size == 0:
            return False

        try:
            self._input_queue.put_nowait(frame)
            return True
        except Full:
            try:
                self._input_queue.get_nowait()
            except Empty:
                pass
            try:
                self._input_queue.put_nowait(frame)
                return True
            except Full:
                return False

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._recognition_loop,
                name=f"ARGUS-Recognition-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()
            self._logger.info(f"Recognition worker started for camera {self.camera_id}")
            return True

    def stop(self, timeout: float = 3.0) -> bool:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except Empty:
                break

        self.cache.clear_camera(self.camera_id)
        self._logger.info(f"Recognition worker stopped for camera {self.camera_id}")
        return True

    def is_alive(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "recognition_active": self.is_alive(),
                "processed_recognition_frames": self._frame_count,
                "recognitions_performed": self._recognition_count,
                "active_tracks": self._active_track_count,
                "last_recognition_at": self._last_recognition_at,
                "queue_size": self._input_queue.qsize(),
            }

    def _recognition_loop(self) -> None:
        """Core decoupled detection, tracking, silhouette, GEI, and gait matching loop."""
        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            try:
                frame = self._input_queue.get(timeout=0.1)
            except Empty:
                continue

            self._frame_count += 1
            now = time.monotonic()
            iso_now = datetime.now(timezone.utc).isoformat()

            try:
                raw_detections = self.detector.detect(frame)

                tracked_objects = self.tracker.update(raw_detections, frame.shape)
                with self._lock:
                    self._active_track_count = len(tracked_objects)

                for obj in tracked_objects:
                    track_id = int(obj["track_id"])
                    bbox = [int(b) for b in obj["bbox"]]

                    # Safely crop person for Branch A (Appearance) and silhouette
                    h, w = frame.shape[:2]
                    x1 = max(0, min(w - 1, bbox[0]))
                    y1 = max(0, min(h - 1, bbox[1]))
                    x2 = max(0, min(w, bbox[2]))
                    y2 = max(0, min(h, bbox[3]))
                    crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

                    # Branch B (Gait): Silhouette & GEI accumulation
                    silhouette = self.silhouette_extractor.extract_from_frame(frame, bbox)
                    if silhouette is not None:
                        self.gei_builder.add_silhouette(track_id, silhouette)

                    # Branch A (Appearance): Gated 512D appearance embedding extraction & matching
                    app_identity = "UNKNOWN_PERSON"
                    app_score = 0.0
                    app_status = "UNKNOWN"

                    if self.appearance_extractor is not None and crop is not None and getattr(crop, "size", 0) > 0:
                        try:
                            app_emb = self.appearance_extractor.extract(
                                crop=crop,
                                track_id=track_id,
                                frame_index=self._frame_count,
                                track_reliable=True,
                                recognition_deferred=False,
                            )
                            if (
                                app_emb is not None
                                and self.appearance_matcher is not None
                                and self.appearance_gallery_features is not None
                                and len(self.appearance_gallery_features) > 0
                            ):
                                matched_app_id, matched_app_score = self.appearance_matcher.match(
                                    query_feature=app_emb,
                                    gallery_features=self.appearance_gallery_features,
                                    gallery_labels=self.appearance_gallery_labels,
                                    metadata=self.appearance_metadata,
                                    unknown_label="UNKNOWN_PERSON",
                                )
                                app_identity = str(matched_app_id)
                                app_score = float(matched_app_score)
                                if app_identity != "UNKNOWN_PERSON" and app_score >= self.appearance_matcher.threshold:
                                    app_status = "MATCH"
                                else:
                                    app_status = "UNKNOWN"

                            # P0 Target Continual-Learning Hook: Record valid appearance observation
                            if self.operational_collector is not None and app_emb is not None:
                                try:
                                    self.operational_collector.record_observation(
                                        camera_id=self.camera_id,
                                        track_id=track_id,
                                        vector=app_emb,
                                        predicted_identity=app_identity,
                                        confidence=float(app_score),
                                        modality="appearance",
                                        quality_score=float(app_score) if app_identity != "UNKNOWN_PERSON" else 0.85,
                                        model_name="OSNet-x0.25",
                                        model_version="v1.0.0",
                                        metadata={
                                            "bbox": bbox,
                                            "frame_count": self._frame_count,
                                            "app_status": app_status,
                                        },
                                    )
                                except (RuntimeError, ValueError, TypeError, OSError) as obs_err:
                                    self._logger.debug(f"Failed to record appearance observation: {obs_err}")

                        except (RuntimeError, ValueError, TypeError, OSError) as app_err:
                            self._logger.debug(f"Appearance matching error for track {track_id}: {app_err}")

                    cached = self.cache.get(self.camera_id, track_id)
                    last_rec_time = self._last_recognition_times.get(track_id, 0.0)
                    time_since_rec = now - last_rec_time

                    gei_ready = self.gei_builder.is_ready(track_id)
                    gei_count = self.gei_builder.get_frame_count(track_id)

                    if cached is not None and cached.status == "CONFIRMED" and time_since_rec < self.cooldown_seconds:
                        cached.bbox = bbox
                        cached.timestamp = now
                        cached.iso_timestamp = iso_now
                        cached.gei_frames = gei_count
                        cached.appearance_identity = app_identity
                        cached.appearance_score = round(app_score, 4)
                        cached.appearance_status = app_status
                        cached.details["appearance"] = {
                            "identity": app_identity,
                            "score": round(app_score, 4),
                            "status": app_status,
                        }
                        cached.details["gait"] = {
                            "identity": cached.identity,
                            "score": round(cached.similarity, 4),
                            "status": cached.status,
                        }
                        self.cache.put(cached)
                        continue

                    if gei_ready and time_since_rec >= self.cooldown_seconds:
                        gei = self.gei_builder.build_gei(track_id)
                        if gei is not None:
                            gait_identity, gait_similarity, gait_decision, gait_status, gait_embedding = self._recognize_gei(gei)

                            final_identity = gait_identity
                            final_similarity = gait_similarity
                            final_decision = gait_decision
                            final_status = gait_status
                            fusion_details = None

                            # P0 Target Continual-Learning Hook: Record valid gait observation
                            if self.operational_collector is not None and gait_embedding is not None:
                                try:
                                    self.operational_collector.record_observation(
                                        camera_id=self.camera_id,
                                        track_id=track_id,
                                        vector=gait_embedding,
                                        predicted_identity=gait_identity,
                                        confidence=float(gait_similarity),
                                        modality="gait",
                                        quality_score=float(gait_similarity) if gait_identity != "UNKNOWN" else 0.85,
                                        model_name="ByGaitLight",
                                        model_version="v1.0.0",
                                        metadata={
                                            "bbox": bbox,
                                            "gei_frames": gei_count,
                                            "decision": gait_decision,
                                            "status": gait_status,
                                            "frame_count": self._frame_count,
                                        },
                                    )
                                except (RuntimeError, ValueError, TypeError, OSError) as obs_err:
                                    self._logger.debug(f"Failed to record gait observation: {obs_err}")

                            if self.fusion_engine is not None and self.fusion_engine.is_enabled():
                                decision_res = self.fusion_engine.decide_identity(
                                    gait_identity=gait_identity,
                                    gait_score=gait_similarity,
                                    appearance_identity=app_identity,
                                    appearance_score=app_score,
                                    gait_threshold=self.threshold,
                                    appearance_threshold=self.appearance_matcher.threshold if self.appearance_matcher else 0.60,
                                    crop=crop,
                                    gei_frame_count=gei_count,
                                    gei=gei,
                                    track_reliability=1.0,
                                )
                                final_identity = decision_res["final_identity"]
                                final_similarity = decision_res["final_score"]
                                final_decision = decision_res["decision"]
                                final_status = decision_res["status"]
                                fusion_details = decision_res

                            temporal_details = None
                            if self.track_aggregator is not None:
                                agg_res = self.track_aggregator.update(
                                    track_id=track_id,
                                    identity=final_identity,
                                    score=final_similarity,
                                    modality_state=fusion_details.get("modality_state", final_status) if fusion_details else final_status,
                                    details=fusion_details,
                                )
                                if agg_res.get("is_aggregated", False):
                                    final_identity = agg_res["identity"]
                                    final_similarity = agg_res["confidence"]
                                    final_decision = agg_res["decision"]
                                    final_status = agg_res["status"]
                                    temporal_details = agg_res

                            res = RecognitionResult(
                                camera_id=self.camera_id,
                                track_id=track_id,
                                identity=final_identity,
                                similarity=final_similarity,
                                confidence=final_similarity,
                                decision=final_decision,
                                status=final_status,
                                bbox=bbox,
                                timestamp=now,
                                iso_timestamp=iso_now,
                                gei_frames=gei_count,
                                appearance_identity=app_identity,
                                appearance_score=round(app_score, 4),
                                appearance_status=app_status,
                                details={
                                    "appearance": {
                                        "identity": app_identity,
                                        "score": round(app_score, 4),
                                        "status": app_status,
                                    },
                                    "gait": {
                                        "identity": gait_identity,
                                        "score": round(gait_similarity, 4),
                                        "status": gait_status,
                                    },
                                    **({"dual_modal": fusion_details} if fusion_details is not None else {}),
                                    **({"temporal_aggregation": temporal_details} if temporal_details is not None else {}),
                                },
                            )
                            self.cache.put(res)
                            self._last_recognition_times[track_id] = now
                            self._last_recognition_at = iso_now
                            self._recognition_count += 1

                            if final_status == "CONFIRMED" and self.event_callback is not None:
                                try:
                                    self.event_callback(
                                        {
                                            "event_id": f"evt_{self.camera_id}_{track_id}_{int(now)}",
                                            "camera_id": self.camera_id,
                                            "track_id": track_id,
                                            "person_id": final_identity,
                                            "similarity": final_similarity,
                                            "timestamp": iso_now,
                                            "bbox": bbox,
                                            "appearance": {
                                                "identity": app_identity,
                                                "score": round(app_score, 4),
                                                "status": app_status,
                                            },
                                            "gait": {
                                                "identity": gait_identity,
                                                "score": round(gait_similarity, 4),
                                                "status": gait_status,
                                            },
                                            **({"dual_modal": fusion_details} if fusion_details is not None else {}),
                                            **({"temporal_aggregation": temporal_details} if temporal_details is not None else {}),
                                        }
                                    )
                                except (RuntimeError, ValueError, TypeError, OSError) as cb_err:
                                    self._logger.debug(f"Event callback error: {cb_err}")

                            continue

                    provisional_status = "TRACKING" if gei_count >= 3 else "DETECTION"
                    provisional_identity = "UNKNOWN" if cached is None else cached.identity
                    provisional_similarity = 0.0 if cached is None else cached.similarity

                    final_prov_identity = provisional_identity
                    final_prov_similarity = provisional_similarity
                    final_prov_status = provisional_status
                    final_prov_decision = provisional_status
                    fusion_prov_details = None

                    if (
                        self.fusion_engine is not None
                        and self.fusion_engine.is_enabled()
                        and app_status == "MATCH"
                    ):
                        decision_res = self.fusion_engine.decide_identity(
                            gait_identity=provisional_identity,
                            gait_score=provisional_similarity,
                            appearance_identity=app_identity,
                            appearance_score=app_score,
                            gait_threshold=self.threshold,
                            appearance_threshold=self.appearance_matcher.threshold if self.appearance_matcher else 0.60,
                            crop=crop,
                            gei_frame_count=gei_count,
                            track_reliability=1.0,
                        )
                        if decision_res["modality_state"] == "APPEARANCE_ONLY":
                            final_prov_identity = decision_res["final_identity"]
                            final_prov_similarity = decision_res["final_score"]
                            final_prov_status = decision_res["status"]
                            final_prov_decision = decision_res["decision"]
                            fusion_prov_details = decision_res

                    temporal_prov_details = None
                    if self.track_aggregator is not None and app_status == "MATCH":
                        agg_res = self.track_aggregator.update(
                            track_id=track_id,
                            identity=final_prov_identity,
                            score=final_prov_similarity,
                            modality_state=fusion_prov_details.get("modality_state", final_prov_status) if fusion_prov_details else final_prov_status,
                            details=fusion_prov_details,
                        )
                        if agg_res.get("is_aggregated", False) and agg_res["status"] in ("CONFIRMED", "REVIEW_REQUIRED"):
                            final_prov_identity = agg_res["identity"]
                            final_prov_similarity = agg_res["confidence"]
                            final_prov_decision = agg_res["decision"]
                            final_prov_status = agg_res["status"]
                            temporal_prov_details = agg_res

                    res = RecognitionResult(
                        camera_id=self.camera_id,
                        track_id=track_id,
                        identity=final_prov_identity,
                        similarity=final_prov_similarity,
                        confidence=final_prov_similarity,
                        decision=final_prov_decision,
                        status=final_prov_status,
                        bbox=bbox,
                        timestamp=now,
                        iso_timestamp=iso_now,
                        gei_frames=gei_count,
                        appearance_identity=app_identity,
                        appearance_score=round(app_score, 4),
                        appearance_status=app_status,
                        details={
                            "appearance": {
                                "identity": app_identity,
                                "score": round(app_score, 4),
                                "status": app_status,
                            },
                            "gait": {
                                "identity": provisional_identity,
                                "score": round(provisional_similarity, 4),
                                "status": provisional_status,
                            },
                            **({"dual_modal": fusion_prov_details} if fusion_prov_details is not None else {}),
                            **({"temporal_aggregation": temporal_prov_details} if temporal_prov_details is not None else {}),
                        },
                    )
                    self.cache.put(res)

                if self._frame_count % 150 == 0:
                    evicted = self.tracker.cleanup_inactive(max_idle_seconds=5.0)
                    if evicted:
                        if self.appearance_extractor is not None:
                            for tid in evicted:
                                try:
                                    self.appearance_extractor.clear_track(tid)
                                except (KeyError, ValueError, RuntimeError, TypeError, OSError) as exc:
                                    self._logger.debug(f"Error clearing appearance track {tid}: {exc}")
                        if self.track_aggregator is not None:
                            for tid in evicted:
                                try:
                                    self.track_aggregator.on_track_lost(tid)
                                except (KeyError, ValueError, RuntimeError, TypeError, OSError) as exc:
                                    self._logger.debug(f"Error notifying track aggregator on lost track {tid}: {exc}")
                    self.gei_builder.cleanup_inactive(max_idle_seconds=6.0)
                    self.cache.cleanup_inactive(self.camera_id, max_idle_seconds=5.0)

            except (RuntimeError, ValueError, TypeError, cv2.error, OSError) as err:
                self._logger.error(f"Recognition error in frame {self._frame_count}: {sanitize_rtsp_url(str(err))}")

            elapsed = time.monotonic() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _recognize_gei(self, gei: np.ndarray) -> tuple[str, float, str, str, np.ndarray | None]:
        """Run ByGaitLight CNN embedding extraction and gallery matching."""
        try:
            with self._lock:
                g_features = self.gallery_features
                g_labels = self.gallery_labels
                g_meta = self.metadata

            embedding = self.extractor.extract_from_gei(gei)
            if embedding is None or len(embedding) == 0:
                return "UNKNOWN", 0.0, "UNKNOWN_PERSON", "UNKNOWN", None

            if g_features is None or len(g_labels) == 0:
                return "UNKNOWN", 0.0, "UNKNOWN_PERSON", "UNKNOWN", embedding

            if not isinstance(g_meta, dict):
                g_meta = {str(lbl): {"status": "ACTIVE", "enabled": True} for lbl in g_labels}

            identity, score = self.matcher.match(
                query_feature=embedding,
                gallery_features=g_features,
                gallery_labels=g_labels,
                metadata=g_meta,
            )

            top_matches = self.matcher.top_k_matches(
                query_feature=embedding,
                gallery_features=g_features,
                gallery_labels=g_labels,
                metadata=g_meta,
                k=5,
            )
            decision_res = self.open_set_recognizer.evaluate_open_set_decision(top_matches)

            if decision_res.state.value == "KNOWN" and identity != "UNKNOWN" and score >= self.threshold:
                return identity, float(score), "CONFIRMED_MATCH", "CONFIRMED", embedding
            elif decision_res.state.value == "UNCERTAIN":
                return identity, float(score), "REVIEW_REQUIRED", "UNKNOWN", embedding
            else:
                return "UNKNOWN", float(score), "UNKNOWN_PERSON", "UNKNOWN", embedding

        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            self._logger.warning(f"Gait matching failed: {sanitize_rtsp_url(str(exc))}")
            return "UNKNOWN", 0.0, "UNKNOWN_PERSON", "UNKNOWN", None
