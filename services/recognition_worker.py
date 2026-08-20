"""
Real-Time Decoupled Recognition Worker and Cache for ARGUS AI.

Orchestrates person detection, tracking, silhouette extraction, GEI generation,
ByGaitLight CNN feature extraction, and VectorStore gallery matching on an
asynchronous thread decoupled from camera capture.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from monitoring.logging_config import get_logger
from pipeline.detection.person_detector import PersonDetector
from pipeline.gei.stream_gei_builder import StreamGEIBuilder
from pipeline.silhouette.extractor import SilhouetteExtractor
from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.steps.matching_step import MatchingStep
from pipeline.tracking.tracker import PersonTracker
from intelligence.open_set_recognizer import OpenSetRecognizer
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
    bbox: List[int]
    timestamp: float
    iso_timestamp: str
    gei_frames: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RecognitionResultCache:
    """
    Thread-safe recognition result cache scoped by (camera_id, track_id).
    Enforces TTL expiration and track eviction to prevent stale overlays and memory growth.
    """

    def __init__(self, ttl_seconds: float = 2.5) -> None:
        self.ttl_seconds = max(0.1, float(ttl_seconds))
        self._cache: Dict[Tuple[str, int], RecognitionResult] = {}
        self._lock = threading.RLock()

    def get(self, camera_id: str, track_id: int) -> Optional[RecognitionResult]:
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

    def get_active_tracks(self, camera_id: str) -> List[RecognitionResult]:
        now = time.monotonic()
        with self._lock:
            return [
                res
                for (cid, _), res in self._cache.items()
                if cid == camera_id and (now - res.timestamp <= self.ttl_seconds)
            ]

    def cleanup_inactive(self, camera_id: str, max_idle_seconds: float = 5.0) -> List[int]:
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
            for (cid, tid) in list(self._cache.keys()):
                if cid == camera_id:
                    self._cache.pop((cid, tid), None)


class RecognitionWorker:
    """
    Decoupled Asynchronous Recognition Worker.
    Pulls latest frames from bounded queue, executes detection, tracking, GEI accumulation,
    and ByGaitLight matching, and updates the shared recognition cache.
    """

    def __init__(
        self,
        camera_id: str,
        config: Optional[dict] = None,
        cache: Optional[RecognitionResultCache] = None,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[PersonTracker] = None,
        silhouette_extractor: Optional[SilhouetteExtractor] = None,
        gei_builder: Optional[StreamGEIBuilder] = None,
        extractor: Optional[FeatureExtractionStep] = None,
        matcher: Optional[MatchingStep] = None,
        open_set_recognizer: Optional[OpenSetRecognizer] = None,
        gallery_features: Optional[np.ndarray] = None,
        gallery_labels: Optional[list] = None,
        metadata: Optional[list] = None,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.camera_id = camera_id
        self.config = config or {}
        self._logger = get_logger(f"recognition.{camera_id}")

        self.target_fps = max(1.0, float(self.config.get("target_fps", 8.0)))
        self._frame_interval = 1.0 / self.target_fps
        self.cooldown_seconds = max(0.1, float(self.config.get("cooldown_seconds", 1.5)))
        self.threshold = float(self.config.get("threshold", 0.85))

        self.cache = cache or RecognitionResultCache(
            ttl_seconds=float(self.config.get("result_ttl", 2.5))
        )

        self.detector = detector or PersonDetector()
        self.tracker = tracker or PersonTracker()
        self.silhouette_extractor = silhouette_extractor or SilhouetteExtractor(target_size=(64, 128))
        self.gei_builder = gei_builder or StreamGEIBuilder()
        self.extractor = extractor or FeatureExtractionStep()
        self.matcher = matcher or MatchingStep(threshold=self.threshold)
        self.open_set_recognizer = open_set_recognizer or OpenSetRecognizer(known_threshold=self.threshold)

        self.gallery_features = gallery_features
        self.gallery_labels = gallery_labels or []
        self.metadata = metadata or []
        self.event_callback = event_callback

        self._input_queue: Queue[np.ndarray] = Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._last_recognition_at: Optional[str] = None
        self._last_recognition_times: Dict[int, float] = {}
        self._frame_count = 0
        self._recognition_count = 0
        self._active_track_count = 0

    def update_gallery(
        self,
        gallery_features: Optional[np.ndarray],
        gallery_labels: Optional[list],
        metadata: Optional[list] = None,
    ) -> None:
        """Update live gallery embeddings thread-safely."""
        with self._lock:
            self.gallery_features = gallery_features
            self.gallery_labels = gallery_labels or []
            self.metadata = metadata or []

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
                # Discard stale frame and insert newest
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

        # Clear queues and cache
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
                # 1. Person Detection
                raw_detections = self.detector.detect(frame)

                # 2. Multi-Object Tracking
                tracked_objects = self.tracker.update(raw_detections, frame.shape)
                with self._lock:
                    self._active_track_count = len(tracked_objects)

                for obj in tracked_objects:
                    track_id = int(obj["track_id"])
                    bbox = [int(b) for b in obj["bbox"]]

                    # 3. Silhouette Extraction
                    silhouette = self.silhouette_extractor.extract_from_frame(frame, bbox)
                    if silhouette is not None:
                        self.gei_builder.add_silhouette(track_id, silhouette)

                    # 4. Check Cooldown / Cache State
                    cached = self.cache.get(self.camera_id, track_id)
                    last_rec_time = self._last_recognition_times.get(track_id, 0.0)
                    time_since_rec = now - last_rec_time

                    gei_ready = self.gei_builder.is_ready(track_id)
                    gei_count = self.gei_builder.get_frame_count(track_id)

                    # If already confirmed and within cooldown, maintain identity with updated bbox
                    if cached is not None and cached.status == "CONFIRMED" and time_since_rec < self.cooldown_seconds:
                        cached.bbox = bbox
                        cached.timestamp = now
                        cached.iso_timestamp = iso_now
                        cached.gei_frames = gei_count
                        self.cache.put(cached)
                        continue

                    # 5. Execute Gait Recognition if GEI is ready
                    if gei_ready and time_since_rec >= self.cooldown_seconds:
                        gei = self.gei_builder.build_gei(track_id)
                        if gei is not None:
                            identity, similarity, decision, status = self._recognize_gei(gei)

                            res = RecognitionResult(
                                camera_id=self.camera_id,
                                track_id=track_id,
                                identity=identity,
                                similarity=similarity,
                                confidence=similarity,
                                decision=decision,
                                status=status,
                                bbox=bbox,
                                timestamp=now,
                                iso_timestamp=iso_now,
                                gei_frames=gei_count,
                            )
                            self.cache.put(res)
                            self._last_recognition_times[track_id] = now
                            self._last_recognition_at = iso_now
                            self._recognition_count += 1

                            if status == "CONFIRMED" and self.event_callback is not None:
                                try:
                                    self.event_callback({
                                        "event_id": f"evt_{self.camera_id}_{track_id}_{int(now)}",
                                        "camera_id": self.camera_id,
                                        "track_id": track_id,
                                        "person_id": identity,
                                        "similarity": similarity,
                                        "timestamp": iso_now,
                                        "bbox": bbox,
                                    })
                                except Exception as cb_err:
                                    self._logger.debug(f"Event callback error: {cb_err}")

                            continue

                    # 6. Provisional Tracking / Detection State if not yet recognized
                    provisional_status = "TRACKING" if gei_count >= 3 else "DETECTION"
                    provisional_identity = "UNKNOWN" if cached is None else cached.identity
                    provisional_similarity = 0.0 if cached is None else cached.similarity

                    res = RecognitionResult(
                        camera_id=self.camera_id,
                        track_id=track_id,
                        identity=provisional_identity,
                        similarity=provisional_similarity,
                        confidence=provisional_similarity,
                        decision=provisional_status,
                        status=provisional_status,
                        bbox=bbox,
                        timestamp=now,
                        iso_timestamp=iso_now,
                        gei_frames=gei_count,
                    )
                    self.cache.put(res)

                # Periodic cleanup of inactive tracks
                if self._frame_count % 150 == 0:
                    self.tracker.cleanup_inactive(max_idle_seconds=5.0)
                    self.gei_builder.cleanup_inactive(max_idle_seconds=6.0)
                    self.cache.cleanup_inactive(self.camera_id, max_idle_seconds=5.0)

            except Exception as err:
                self._logger.error(f"Recognition error in frame {self._frame_count}: {sanitize_rtsp_url(str(err))}")

            # Pacing to maintain target recognition FPS
            elapsed = time.monotonic() - loop_start
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _recognize_gei(self, gei: np.ndarray) -> Tuple[str, float, str, str]:
        """Run ByGaitLight CNN embedding extraction and gallery matching."""
        try:
            with self._lock:
                g_features = self.gallery_features
                g_labels = self.gallery_labels
                g_meta = self.metadata

            if g_features is None or len(g_labels) == 0:
                return "UNKNOWN", 0.0, "UNKNOWN_PERSON", "UNKNOWN"

            embedding = self.extractor.extract_from_gei(gei)
            if embedding is None or len(embedding) == 0:
                return "UNKNOWN", 0.0, "UNKNOWN_PERSON", "UNKNOWN"

            # Gallery matching
            identity, score = self.matcher.match(
                query_feature=embedding,
                gallery_features=g_features,
                gallery_labels=g_labels,
                metadata=g_meta,
            )

            # Open-set evaluation
            top_matches = self.matcher.top_k_matches(
                query_feature=embedding,
                gallery_features=g_features,
                gallery_labels=g_labels,
                metadata=g_meta,
                k=5,
            )
            decision_res = self.open_set_recognizer.evaluate_open_set_decision(top_matches)

            if decision_res.state.value == "KNOWN" and identity != "UNKNOWN" and score >= self.threshold:
                return identity, float(score), "CONFIRMED_MATCH", "CONFIRMED"
            elif decision_res.state.value == "UNCERTAIN":
                return identity, float(score), "REVIEW_REQUIRED", "UNKNOWN"
            else:
                return "UNKNOWN", float(score), "UNKNOWN_PERSON", "UNKNOWN"

        except Exception as exc:
            self._logger.warning(f"Gait matching failed: {sanitize_rtsp_url(str(exc))}")
            return "UNKNOWN", 0.0, "UNKNOWN_PERSON", "UNKNOWN"
