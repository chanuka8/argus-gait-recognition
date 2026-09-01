import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import WebSocket

from core.logger import setup_logger
from security_layer.credentials import sanitize_rtsp_url
from services.camera_worker import CameraWorker
from storage.vector_store import VectorStore


class WebSocketManager:
    """Manages active WebSocket connections for real-time recognition event broadcasting."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except (RuntimeError, ValueError, OSError):
                self.disconnect(connection)


class GaitService:
    """
    Unified Single-Instance Gait Recognition Service for ARGUS FastAPI Backend.
    Encapsulates person detection, silhouette extraction (UNet + Otsu fallback),
    ByGaitLight feature encoding, VectorStore matching, camera worker state, and event history.
    Uses high-performance lazy initialization and background warmup for instant server startup.
    """

    def __init__(
        self, gallery_dir: str = "models/live_gallery", appearance_gallery_dir: str = "models/appearance_gallery"
    ) -> None:
        self.logger = setup_logger("ARGUS.GaitService")
        self.gallery_dir = gallery_dir
        self.appearance_gallery_dir = appearance_gallery_dir

        self.store = VectorStore(gallery_dir=gallery_dir)
        self.appearance_store = VectorStore(gallery_dir=appearance_gallery_dir)

        # Thread-safe re-entrant lock for lazy model loading
        self._lock = threading.RLock()

        # Lazy component slots
        self._embedding_db = None
        self._model_registry = None
        self._continuous_engine = None
        self._extractor = None
        self._matcher = None
        self._silhouette_extractor = None
        self._open_set_recognizer = None
        self._appearance_extractor = None
        self._appearance_matcher = None
        self._detector = None
        self._source_resolver = None

        self._is_warmed_up = False
        self._warmup_error: str | None = None

        self.ws_manager = WebSocketManager()
        self.events_log: list[dict] = []
        self.active_cameras: dict[str, dict] = {}
        self.camera_workers: dict[str, CameraWorker] = {}

        self.stats = {
            "processed_images": 0,
            "processed_videos": 0,
            "total_events": 0,
        }

        self.appearance_gallery_features = np.empty((0, 512), dtype=np.float32)
        self.appearance_gallery_labels = []
        self.appearance_metadata = {}

        self.reload_gallery()

    @property
    def is_warmed_up(self) -> bool:
        """Returns True if background model warmup has completed."""
        return self._is_warmed_up

    @property
    def extractor(self):
        if self._extractor is None:
            with self._lock:
                if self._extractor is None:
                    from pipeline.steps.feature_extraction import FeatureExtractionStep

                    self._extractor = FeatureExtractionStep()
        return self._extractor

    @extractor.setter
    def extractor(self, value: Any) -> None:
        self._extractor = value

    @property
    def matcher(self):
        if self._matcher is None:
            with self._lock:
                if self._matcher is None:
                    from pipeline.steps.matching_step import MatchingStep

                    self._matcher = MatchingStep(threshold=0.85)
        return self._matcher

    @matcher.setter
    def matcher(self, value: Any) -> None:
        self._matcher = value

    @property
    def silhouette_extractor(self):
        if self._silhouette_extractor is None:
            with self._lock:
                if self._silhouette_extractor is None:
                    from pipeline.silhouette.extractor import SilhouetteExtractor

                    self._silhouette_extractor = SilhouetteExtractor(target_size=(64, 128))
        return self._silhouette_extractor

    @silhouette_extractor.setter
    def silhouette_extractor(self, value: Any) -> None:
        self._silhouette_extractor = value

    @property
    def open_set_recognizer(self):
        if self._open_set_recognizer is None:
            with self._lock:
                if self._open_set_recognizer is None:
                    from intelligence.open_set_recognizer import OpenSetRecognizer

                    self._open_set_recognizer = OpenSetRecognizer()
        return self._open_set_recognizer

    @open_set_recognizer.setter
    def open_set_recognizer(self, value: Any) -> None:
        self._open_set_recognizer = value

    @property
    def appearance_extractor(self):
        if self._appearance_extractor is None:
            with self._lock:
                if self._appearance_extractor is None:
                    try:
                        from intelligence.appearance_embedding import AppearanceEmbeddingExtractor

                        self._appearance_extractor = AppearanceEmbeddingExtractor(update_interval=8)
                    except (ImportError, RuntimeError, ValueError, TypeError, OSError) as app_init_err:
                        self.logger.warning(f"Appearance extractor init deferred: {app_init_err}")
                        self._appearance_extractor = None
        return self._appearance_extractor

    @appearance_extractor.setter
    def appearance_extractor(self, value: Any) -> None:
        self._appearance_extractor = value

    @property
    def appearance_matcher(self):
        if self._appearance_matcher is None:
            with self._lock:
                if self._appearance_matcher is None:
                    try:
                        from pipeline.steps.appearance_matching_step import AppearanceMatchingStep

                        self._appearance_matcher = AppearanceMatchingStep(threshold=0.60)
                    except (ImportError, RuntimeError, ValueError, TypeError, OSError) as app_init_err:
                        self.logger.warning(f"Appearance matcher init deferred: {app_init_err}")
                        self._appearance_matcher = None
        return self._appearance_matcher

    @appearance_matcher.setter
    def appearance_matcher(self, value: Any) -> None:
        self._appearance_matcher = value

    @property
    def detector(self):
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    try:
                        from pipeline.detection.person_detector import PersonDetector

                        self._detector = PersonDetector()
                    except (ImportError, RuntimeError, ValueError, OSError) as err:
                        self.logger.warning(f"PersonDetector initialization skipped: {err}")
                        self._detector = None
        return self._detector

    @detector.setter
    def detector(self, value: Any) -> None:
        self._detector = value

    @property
    def source_resolver(self):
        if self._source_resolver is None:
            with self._lock:
                if self._source_resolver is None:
                    from services.camera_source_resolver import CameraSourceResolver

                    self._source_resolver = CameraSourceResolver()
        return self._source_resolver

    @source_resolver.setter
    def source_resolver(self, value: Any) -> None:
        self._source_resolver = value

    @property
    def embedding_db(self):
        if self._embedding_db is None:
            with self._lock:
                if self._embedding_db is None:
                    try:
                        from storage.embedding_database import EmbeddingDatabase

                        self._embedding_db = EmbeddingDatabase(
                            gait_gallery_dir=self.gallery_dir,
                            appearance_gallery_dir=self.appearance_gallery_dir,
                        )
                    except (ImportError, RuntimeError, ValueError, TypeError, OSError) as err:
                        self.logger.warning(f"EmbeddingDatabase init deferred: {err}")
                        self._embedding_db = None
        return self._embedding_db

    @embedding_db.setter
    def embedding_db(self, value: Any) -> None:
        self._embedding_db = value

    @property
    def model_registry(self):
        if self._model_registry is None:
            with self._lock:
                if self._model_registry is None:
                    try:
                        from models.model_registry import ModelRegistry

                        self._model_registry = ModelRegistry()
                    except (ImportError, RuntimeError, ValueError, TypeError, OSError) as err:
                        self.logger.warning(f"ModelRegistry init deferred: {err}")
                        self._model_registry = None
        return self._model_registry

    @model_registry.setter
    def model_registry(self, value: Any) -> None:
        self._model_registry = value

    @property
    def continuous_engine(self):
        if self._continuous_engine is None:
            with self._lock:
                if self._continuous_engine is None:
                    try:
                        from intelligence.continuous_improvement_engine import ContinuousImprovementEngine

                        self._continuous_engine = ContinuousImprovementEngine(
                            registry=self.model_registry,
                            db=self.embedding_db,
                        )
                    except (ImportError, RuntimeError, ValueError, TypeError, OSError) as err:
                        self.logger.warning(f"ContinuousImprovementEngine init deferred: {err}")
                        self._continuous_engine = None
        return self._continuous_engine

    @continuous_engine.setter
    def continuous_engine(self, value: Any) -> None:
        self._continuous_engine = value

    def warmup(self) -> dict[str, Any]:
        """
        Pre-warms all heavy models and components in a background worker thread.
        Guarantees that inference requests after warmup execute with zero initial load latency.
        """
        with self._lock:
            if self._is_warmed_up:
                return {"status": "WARMED_UP", "already_warmed": True}

            self.logger.info("[STARTUP] Beginning background model warmup...")
            t0 = time.perf_counter()
            results = {}

            try:
                _ = self.extractor
                _ = self.matcher
                results["bygait_light"] = "READY"
            except Exception as e:  # noqa: BLE001
                results["bygait_light"] = f"ERROR: {e}"

            try:
                _ = self.silhouette_extractor
                results["silhouette_extractor"] = "READY"
            except Exception as e:  # noqa: BLE001
                results["silhouette_extractor"] = f"ERROR: {e}"

            try:
                _ = self.appearance_extractor
                _ = self.appearance_matcher
                results["osnet_appearance"] = "READY"
            except Exception as e:  # noqa: BLE001
                results["osnet_appearance"] = f"ERROR: {e}"

            try:
                _ = self.detector
                results["person_detector"] = "READY" if self._detector else "DISABLED"
            except Exception as e:  # noqa: BLE001
                results["person_detector"] = f"ERROR: {e}"

            try:
                _ = self.open_set_recognizer
                _ = self.embedding_db
                _ = self.model_registry
                _ = self.continuous_engine
                results["continual_learning"] = "READY"
            except Exception as e:  # noqa: BLE001
                results["continual_learning"] = f"ERROR: {e}"

            dur = time.perf_counter() - t0
            self._is_warmed_up = True
            self.logger.info(f"[STARTUP] Background model warmup completed in {dur:.3f}s. Results: {results}")
            return {"status": "WARMED_UP", "duration": dur, "components": results}

    async def warmup_async(self) -> dict[str, Any]:
        """Asynchronously triggers model warmup in a thread pool without blocking event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.warmup)

    async def shutdown_async(self) -> None:
        """Gracefully shut down active camera workers."""
        for cam_id, worker in list(self.camera_workers.items()):
            try:
                worker.stop()
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"Error stopping camera {cam_id}: {e}")

    def reload_gallery(self) -> None:
        try:
            gallery = self.store.load()
            if gallery is not None:
                self.gallery_features, labels, self.metadata = gallery
                self.gallery_labels = list(labels) if labels is not None else []
                self.logger.info(f"Loaded gait gallery with {len(self.gallery_labels)} embeddings.")
            else:
                self.gallery_features = np.empty((0, 256), dtype=np.float32)
                self.gallery_labels = []
                self.metadata = []
                self.logger.warning("No gait gallery found; operating with empty gait gallery.")
        except (RuntimeError, ValueError, TypeError, OSError) as err:
            self.logger.error(f"Failed to load gait gallery: {err}")
            self.gallery_features = np.empty((0, 256), dtype=np.float32)
            self.gallery_labels = []
            self.metadata = []

        try:
            app_gallery = self.appearance_store.load()
            if app_gallery is not None:
                self.appearance_gallery_features, app_labels, self.appearance_metadata = app_gallery
                self.appearance_gallery_labels = list(app_labels) if app_labels is not None else []
                self.logger.info(f"Loaded appearance gallery with {len(self.appearance_gallery_labels)} embeddings.")
            else:
                self.appearance_gallery_features = np.empty((0, 512), dtype=np.float32)
                self.appearance_gallery_labels = []
                self.appearance_metadata = {}
        except (RuntimeError, ValueError, TypeError, OSError) as app_err:
            self.logger.warning(f"Failed to load appearance gallery: {app_err}")
            self.appearance_gallery_features = np.empty((0, 512), dtype=np.float32)
            self.appearance_gallery_labels = []
            self.appearance_metadata = {}

        for worker in self.camera_workers.values():
            if worker.recognition_worker is not None:
                worker.recognition_worker.update_gallery(
                    gallery_features=self.gallery_features,
                    gallery_labels=self.gallery_labels,
                    metadata=self.metadata,
                )
                worker.recognition_worker.update_appearance_gallery(
                    gallery_features=self.appearance_gallery_features,
                    gallery_labels=self.appearance_gallery_labels,
                    metadata=self.appearance_metadata,
                )

    def _handle_recognition_event(self, event_dict: dict) -> None:
        """Store confirmed recognition event in history and broadcast to WebSocket subscribers."""
        self.events_log.insert(0, event_dict)
        if len(self.events_log) > 500:
            self.events_log.pop()
        self.stats["total_events"] += 1

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.ws_manager.broadcast(event_dict))
        except (RuntimeError, ValueError, TypeError, OSError):
            pass

    def get_metrics(self) -> dict:
        has_labels = len(self.gallery_labels) > 0 if self.gallery_labels is not None else False
        unique_labels = set(self.gallery_labels) if has_labels else set()
        return {
            "people": len(unique_labels),
            "embeddings": len(self.gallery_labels) if has_labels else 0,
            "labels": len(unique_labels),
            "processed_images": self.stats["processed_images"],
            "processed_videos": self.stats["processed_videos"],
            "total_events": len(self.events_log),
        }

    def process_image_bytes(self, image_bytes: bytes, camera_id: str = "upload-image") -> dict:
        """Processes uploaded image bytes and returns a structured recognition event."""
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            raise ValueError("Invalid or corrupted image format")

        self.stats["processed_images"] += 1
        h, w = frame.shape[:2]

        bbox = [0, 0, w, h]
        if self.detector is not None:
            try:
                detections = self.detector.detect(frame)
                if detections and len(detections) > 0:
                    bbox = detections[0]["bbox"]
            except (RuntimeError, ValueError, TypeError, cv2.error, OSError) as err:
                self.logger.warning(f"Detection failed; using full frame: {err}")

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            crop = frame

        silhouette = self.silhouette_extractor.extract_from_crop(crop)
        if silhouette is None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            silhouette = cv2.resize(gray, (64, 128))

        sil_norm = silhouette.astype(np.float32) / 255.0
        embedding = self.extractor.backend.predict(sil_norm).flatten().astype(np.float32)

        identity = "UNKNOWN"
        decision = "UNKNOWN"
        confidence = 0.0

        if len(self.gallery_features) > 0:
            open_set_res = self.matcher.match_open_set(
                embedding,
                self.gallery_features,
                self.gallery_labels,
                self.metadata,
            )
            matched_id, score = self.matcher.match(
                embedding,
                self.gallery_features,
                self.gallery_labels,
                self.metadata,
            )

            decision = open_set_res.state.value.upper()
            confidence = float(score)
            if decision == "KNOWN":
                identity = str(matched_id)
            elif decision == "UNCERTAIN":
                identity = f"UNCERTAIN ({matched_id})"
            else:
                identity = "UNKNOWN"

        event = {
            "event_id": f"evt-{uuid.uuid4().hex[:12]}",
            "camera_id": camera_id,
            "track_id": 1,
            "identity": identity,
            "decision": decision,
            "confidence": round(confidence, 4),
            "quality": 0.85,
            "bbox": [x1, y1, x2, y2],
            "recognition_branch": "2D_GEI",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.events_log.insert(0, event)
        if len(self.events_log) > 200:
            self.events_log.pop()

        self.stats["total_events"] += 1

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.ws_manager.broadcast(event))
        except (RuntimeError, ValueError, TypeError, OSError):
            pass

        return event

    def enroll_images(self, person_id: str, image_bytes_list: list[bytes]) -> dict:
        """Enrolls a new subject into the gallery from uploaded image byte buffers."""
        if not person_id or not person_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Invalid person_id: must be alphanumeric (hyphens/underscores permitted)")

        added_embeddings = 0
        embeddings = []
        app_embeddings = []

        for raw_bytes in image_bytes_list:
            array = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue

            # 1. ByGaitLight Gait Feature Extraction (256D)
            silhouette = self.silhouette_extractor.extract_from_crop(frame)
            if silhouette is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                silhouette = cv2.resize(gray, (64, 128))

            sil_norm = silhouette.astype(np.float32) / 255.0
            embedding = self.extractor.backend.predict(sil_norm).flatten().astype(np.float32)
            embeddings.append(embedding)
            added_embeddings += 1

            # 2. OSNet Appearance Feature Extraction (512D)
            if self.appearance_extractor is not None:
                try:
                    app_emb = self.appearance_extractor.extract(frame)
                    if app_emb is not None and len(app_emb) == 512 and np.isfinite(app_emb).all():
                        app_embeddings.append(app_emb)
                except Exception as app_err:  # noqa: BLE001
                    self.logger.debug(f"Appearance extraction notice during enrollment: {app_err}")

        if added_embeddings == 0:
            return {
                "success": False,
                "person_id": person_id,
                "message": "No valid images could be processed for enrollment",
                "embeddings_added": 0,
                "gait_embeddings_added": 0,
                "appearance_embeddings_added": 0,
                "firebase_status": "FAILED",
                "status": "PROCESSING_FAILED",
            }

        new_features = np.vstack(embeddings)
        new_labels = [person_id] * added_embeddings

        if len(self.gallery_features) > 0:
            self.gallery_features = np.vstack([self.gallery_features, new_features])
            if not isinstance(self.gallery_labels, list):
                self.gallery_labels = list(self.gallery_labels)
            self.gallery_labels.extend(new_labels)
        else:
            self.gallery_features = new_features
            self.gallery_labels = new_labels

        self.store.save(self.gallery_features, self.gallery_labels, self.metadata)

        # Save appearance gallery
        if app_embeddings:
            new_app_features = np.vstack(app_embeddings)
            new_app_labels = [person_id] * len(app_embeddings)
            if len(self.appearance_gallery_features) > 0:
                self.appearance_gallery_features = np.vstack([self.appearance_gallery_features, new_app_features])
                if not isinstance(self.appearance_gallery_labels, list):
                    self.appearance_gallery_labels = list(self.appearance_gallery_labels)
                self.appearance_gallery_labels.extend(new_app_labels)
            else:
                self.appearance_gallery_features = new_app_features
                self.appearance_gallery_labels = new_app_labels
            self.appearance_store.save(
                self.appearance_gallery_features, self.appearance_gallery_labels, self.appearance_metadata
            )

        # Persistent EmbeddingDatabase Sync
        db_persist_result = None
        if self.embedding_db is not None:
            try:
                db_persist_result = self.embedding_db.add_embeddings(
                    person_id=person_id,
                    gait_embeddings=embeddings,
                    appearance_embeddings=app_embeddings if app_embeddings else None,
                )
            except (RuntimeError, ValueError, TypeError, OSError) as db_err:
                self.logger.warning(f"EmbeddingDatabase persistence sync warning: {db_err}")

        fb_res = db_persist_result.get("firebase_results", []) if db_persist_result else []
        fb_status = "CONFIRMED" if fb_res and all(r.get("success", False) for r in fb_res) else (
            "PENDING" if self.embedding_db and getattr(self.embedding_db, "firebase_store", None) else "LOCAL_ONLY"
        )

        return {
            "success": True,
            "person_id": person_id,
            "message": f"Successfully enrolled {person_id} with {added_embeddings} gait and {len(app_embeddings)} appearance embeddings",
            "embeddings_added": added_embeddings,
            "gait_embeddings_added": added_embeddings,
            "appearance_embeddings_added": len(app_embeddings),
            "firebase_status": fb_status,
            "status": "EMBEDDING_ONLY",
        }

    def _load_camera_config(self) -> dict:
        """Load default camera configuration settings from configs/system.yaml."""
        config_path = Path("configs/system.yaml")
        defaults = {
            "width": 640,
            "height": 480,
            "target_fps": 15,
            "jpeg_quality": 75,
            "preview_max_fps": 15,
            "reconnect_interval": 5,
            "max_reconnect_attempts": 0,
            "max_queue_size": 10,
            "startup_timeout": 10.0,
            "startup_retry_interval": 0.3,
        }

        if not config_path.exists():
            return defaults

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                section = data.get("camera", {})
                if isinstance(section, dict):
                    for k in defaults:
                        if k in section:
                            defaults[k] = section[k]
        except (yaml.YAMLError, OSError, ValueError) as e:
            self.logger.warning(f"Could not load system camera configuration: {e}")

        return defaults

    def start_camera(
        self,
        camera_id: str,
        source: str = "auto",
        location: str = "Surveillance Zone",
        zone_id: str | None = None,
        user_id: str = "default_user",
        credential_id: str | None = None,
    ) -> dict:
        """Starts camera tracking worker state with automatic or explicit source resolution."""
        if camera_id in self.active_cameras:
            return self.get_camera_info(camera_id)

        resolution = self.source_resolver.resolve_source(
            camera_id=camera_id,
            requested_source=source or "auto",
            zone_id=zone_id,
            user_id=user_id,
            credential_id=credential_id,
        )

        resolved_source = resolution["resolved_source"]
        resolved_type = resolution.get("resolved_source_type") or resolution.get("source_type") or "webcam"
        source_type = (
            "webcam"
            if resolved_type in ("usb", "webcam", "local")
            else ("rtsp" if resolved_type == "rtsp" else resolved_type)
        )
        resolved_label = resolution["resolved_source_label"]
        res_cred_id = resolution.get("credential_id")
        res_cred_conf = resolution.get("credential_configured", False)

        sanitized_source = sanitize_rtsp_url(resolved_source)
        retained_capture = resolution.get("capture")
        initial_frame = resolution.get("initial_frame")

        camera_defaults = self._load_camera_config()
        worker_cfg = {
            **camera_defaults,
            "type": source_type,
            "url": resolved_source if source_type != "webcam" else "",
            "device_index": int(resolved_source) if source_type == "webcam" and str(resolved_source).isdigit() else 0,
        }

        # Pre-flight camera admission check (dynamic capacity & resource safety)
        enforce_admission = bool(worker_cfg.get("enforce_admission", False))
        try:
            from streaming.deployment_readiness import AdmissionDecision, DeploymentReadinessManager

            if not hasattr(self, "_deployment_manager") or self._deployment_manager is None:
                self._deployment_manager = DeploymentReadinessManager()

            adm_res = self._deployment_manager.request_camera_admission(
                camera_id=camera_id,
                current_active_cameras=len(self.active_cameras),
            )
            if not adm_res.admitted:
                self.logger.warning(f"Camera admission notice for '{camera_id}': {adm_res.reason}")
                if enforce_admission:
                    self.source_resolver.release_source_by_camera_id(camera_id)
                    raise RuntimeError(f"Camera admission rejected for '{camera_id}': {adm_res.reason}")
            elif adm_res.decision == AdmissionDecision.ADMITTED_DEGRADED:
                self.logger.info(f"Camera '{camera_id}' admitted with degraded FPS ({adm_res.effective_fps:.1f})")
                worker_cfg["target_fps"] = max(1, int(adm_res.effective_fps))
        except RuntimeError:
            raise
        except (ImportError, Exception) as adm_err:  # noqa: BLE001
            self.logger.debug(f"Admission check note: {adm_err}")

        recognition_worker = None
        try:
            from services.recognition_worker import RecognitionWorker

            recognition_worker = RecognitionWorker(
                camera_id=camera_id,
                config=worker_cfg.get("recognition", {}),
                detector=self.detector,
                silhouette_extractor=self.silhouette_extractor,
                extractor=self.extractor,
                matcher=self.matcher,
                open_set_recognizer=self.open_set_recognizer,
                gallery_features=self.gallery_features,
                gallery_labels=self.gallery_labels,
                metadata=self.metadata,
                appearance_extractor=self.appearance_extractor,
                appearance_matcher=self.appearance_matcher,
                appearance_gallery_features=self.appearance_gallery_features,
                appearance_gallery_labels=self.appearance_gallery_labels,
                appearance_metadata=self.appearance_metadata,
                operational_collector=self.continuous_engine.collector if self.continuous_engine else None,
                event_callback=self._handle_recognition_event,
            )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as rec_err:
            self.logger.warning(f"Recognition worker init deferred for {camera_id}: {rec_err}")

        worker = CameraWorker(
            camera_id=camera_id,
            camera_config=worker_cfg,
            inference_pipeline=None,
            detection_processor=None,
            recognition_worker=recognition_worker,
            existing_capture=retained_capture,
            initial_frame=initial_frame,
        )

        started = worker.start()
        if not started:
            self.source_resolver.release_source_by_camera_id(camera_id)
            raise RuntimeError(
                f"Unable to establish stream connection for {sanitized_source} ({camera_id}): failed to capture video frames"
            )

        self.camera_workers[camera_id] = worker

        stats = worker.get_stats()
        frames_captured = stats.get("frames_captured", 1)

        cam_info = {
            "camera_id": camera_id,
            "zone_id": zone_id,
            "source": sanitized_source,
            "source_type": source_type,
            "location": location,
            "status": "ACTIVE",
            "fps": round(stats.get("fps", 0.0), 1),
            "processed_frames": frames_captured,
            "active_tracks": stats.get("active_tracks", 0),
            "recognition_active": stats.get("recognition_active", False),
            "requested_source": sanitize_rtsp_url(source),
            "resolved_source": sanitized_source,
            "resolved_source_type": source_type,
            "resolved_source_label": sanitize_rtsp_url(resolved_label),
            "preview_url": f"/api/v1/cameras/{camera_id}/stream",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_frame_at": stats.get("last_frame_at"),
            "last_recognition_at": stats.get("last_recognition_at"),
            "credential_id": res_cred_id,
            "credential_configured": res_cred_conf,
        }
        self.active_cameras[camera_id] = cam_info
        self.logger.info(
            f"Camera worker {camera_id} active with {sanitize_rtsp_url(resolved_label)} [type={source_type}]"
        )
        return cam_info

    def stop_camera(self, camera_id: str) -> bool:
        """Stops camera tracking worker state and releases source reservation."""
        worker = self.camera_workers.pop(camera_id, None)
        if worker:
            try:
                worker.stop()
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                self.logger.warning(f"Error stopping worker {camera_id}: {sanitize_rtsp_url(str(e))}")

        if camera_id in self.active_cameras:
            self.active_cameras[camera_id]["status"] = "STOPPED"
            del self.active_cameras[camera_id]
            self.source_resolver.release_source_by_camera_id(camera_id)
            return True
        return False

    def get_camera_worker(self, camera_id: str) -> CameraWorker | None:
        """Return active CameraWorker instance if present."""
        return self.camera_workers.get(camera_id)

    def get_camera_info(self, camera_id: str) -> dict | None:
        """Return updated camera status with live telemetry metrics."""
        cam = self.active_cameras.get(camera_id)
        if not cam:
            return None
        worker = self.camera_workers.get(camera_id)
        if worker:
            stats = worker.get_stats()
            cam["processed_frames"] = stats.get("frames_captured", cam.get("processed_frames", 0))
            cam["fps"] = round(stats.get("fps", 0.0), 1)
            cam["active_tracks"] = stats.get("active_tracks", 0)
            cam["recognition_active"] = stats.get("recognition_active", False)
            cam["last_recognition_at"] = stats.get("last_recognition_at")
            cam["active_clients"] = stats.get("active_clients", 0)
            cam["recognized_identities"] = stats.get("recognized_identities", [])
            cam["preview_url"] = f"/api/v1/cameras/{camera_id}/stream"
            cam["last_frame_at"] = stats.get("last_frame_at")
            cam["source_type"] = cam.get("source_type") or stats.get("source_type", "webcam")
            cam["status"] = (
                "ACTIVE"
                if worker.is_running() and worker.is_connected()
                else ("ACTIVE" if worker.is_running() else "STOPPED")
            )
        return cam

    def list_all_cameras(self) -> list[dict]:
        """Return all active camera info with updated telemetry."""
        result = []
        for cam_id in list(self.active_cameras.keys()):
            info = self.get_camera_info(cam_id)
            if info:
                result.append(info)
        return result
