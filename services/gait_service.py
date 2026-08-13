import asyncio
from datetime import datetime, timezone
import uuid

import cv2
import numpy as np
from fastapi import WebSocket

from core.logger import setup_logger
from intelligence.open_set_recognizer import OpenSetRecognizer
from pipeline.detection.person_detector import PersonDetector
from pipeline.silhouette.extractor import SilhouetteExtractor
from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.steps.matching_step import MatchingStep
from services.camera_source import AutoSourceResolver, ZoneSourceBindingRegistry, resolve_source_type
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
            except Exception:
                self.disconnect(connection)


class GaitService:
    """
    Unified Single-Instance Gait Recognition Service for ARGUS FastAPI Backend.
    Encapsulates person detection, silhouette extraction (UNet + Otsu fallback),
    ByGaitLight feature encoding, VectorStore matching, thread-safe zone-to-source
    camera worker lifecycle, and event history.
    """

    def __init__(self, gallery_dir: str = "models/live_gallery") -> None:
        self.logger = setup_logger("ARGUS.GaitService")
        self.gallery_dir = gallery_dir

        self.store = VectorStore(gallery_dir=gallery_dir)
        self.extractor = FeatureExtractionStep()
        self.matcher = MatchingStep(threshold=0.85)
        self.silhouette_extractor = SilhouetteExtractor(target_size=(64, 128))
        self.open_set_recognizer = OpenSetRecognizer()

        # Detector fallback if YOLO weights exist
        self.detector = None
        try:
            self.detector = PersonDetector()
        except Exception as err:
            self.logger.warning(f"PersonDetector initialization skipped: {err}")

        self.ws_manager = WebSocketManager()
        self.events_log: list[dict] = []

        # Thread-safe auto-source resolver & zone binding registry
        self.auto_resolver = AutoSourceResolver()
        self.zone_registry = ZoneSourceBindingRegistry()

        self.stats = {
            "processed_images": 0,
            "processed_videos": 0,
            "total_events": 0,
        }

        self.reload_gallery()

    @property
    def active_cameras(self) -> dict[str, dict]:
        """Backward compatibility dictionary for active camera info."""
        workers = self.zone_registry.get_all_workers()
        return {cam_id: worker.to_dict() for cam_id, worker in workers.items()}

    def reload_gallery(self) -> None:
        try:
            gallery = self.store.load()
            if gallery is not None:
                self.gallery_features, labels, self.metadata = gallery
                self.gallery_labels = list(labels) if labels is not None else []
                self.logger.info(f"Loaded gallery with {len(self.gallery_labels)} embeddings.")
            else:
                self.gallery_features = np.empty((0, 256), dtype=np.float32)
                self.gallery_labels = []
                self.metadata = []
                self.logger.warning("No gallery found; operating with empty gallery.")
        except Exception as err:
            self.logger.error(f"Failed to load gallery: {err}")
            self.gallery_features = np.empty((0, 256), dtype=np.float32)
            self.gallery_labels = []
            self.metadata = []

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

        # 1. Detect person bounding box or default to full frame crop
        bbox = [0, 0, w, h]
        if self.detector is not None:
            try:
                detections = self.detector.detect(frame)
                if detections and len(detections) > 0:
                    bbox = detections[0]["bbox"]  # [x1, y1, x2, y2]
            except Exception as err:
                self.logger.warning(f"Detection failed; using full frame: {err}")

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
        if crop.size == 0:
            crop = frame

        # 2. Extract silhouette mask (UNet primary + Otsu fallback)
        silhouette = self.silhouette_extractor.extract_from_crop(crop)
        if silhouette is None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            silhouette = cv2.resize(gray, (64, 128))

        # 3. Extract gait embedding using ByGaitLight encoder
        sil_norm = silhouette.astype(np.float32) / 255.0
        embedding = self.extractor.backend.predict(sil_norm).flatten().astype(np.float32)

        # 4. Open-set recognition matching against gallery
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

        # Async broadcast event to WebSocket clients
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.ws_manager.broadcast(event))
        except Exception:
            pass

        return event

    def enroll_images(self, person_id: str, image_bytes_list: list[bytes]) -> dict:
        """Enrolls a new subject into the gallery from uploaded image byte buffers."""
        if not person_id or not person_id.isalnum():
            raise ValueError("Invalid person_id: must be alphanumeric")

        added_embeddings = 0
        embeddings = []

        for raw_bytes in image_bytes_list:
            array = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                continue

            silhouette = self.silhouette_extractor.extract_from_crop(frame)
            if silhouette is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                silhouette = cv2.resize(gray, (64, 128))

            sil_norm = silhouette.astype(np.float32) / 255.0
            embedding = self.extractor.backend.predict(sil_norm).flatten().astype(np.float32)
            embeddings.append(embedding)
            added_embeddings += 1

        if added_embeddings == 0:
            return {
                "success": False,
                "person_id": person_id,
                "message": "No valid images could be processed for enrollment",
                "embeddings_added": 0,
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

        return {
            "success": True,
            "person_id": person_id,
            "message": f"Successfully enrolled {person_id} with {added_embeddings} embeddings",
            "embeddings_added": added_embeddings,
        }

    def discover_sources(self, max_index: int = 5, force_refresh: bool = False, backend: str = "dshow") -> list[dict]:
        """Discovers available local USB webcam sources."""
        sources = self.auto_resolver.discover_usb_sources(max_index=max_index, force_refresh=force_refresh, backend_preference=backend)
        return [s.to_dict() for s in sources]

    def probe_camera(self, source: str, source_type: str = "auto", backend: str = "auto") -> dict:
        """Probes a selected camera source without starting a worker."""
        info = self.auto_resolver.probe_source(source_val=source, requested_type=source_type, backend=backend)
        return info.to_dict()

    def start_camera(
        self,
        camera_id: str,
        source: str,
        zone_id: str = "Z01",
        source_type: str = "auto",
        capture_backend: str = "auto",
        location: str = "Surveillance Zone",
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
    ) -> dict:
        """Starts a real single-owner camera worker in the specified zone."""
        # 1. Resolve source_type and typed value
        resolved_type, typed_val = resolve_source_type(source, source_type)
        device_index = typed_val if resolved_type == "usb" else None

        # 2. Check duplicate zone binding
        if self.zone_registry.is_zone_active(zone_id, exclude_camera_id=camera_id):
            raise ValueError(f"CONFLICT: Zone '{zone_id}' already has an active camera worker.")

        # 3. Check duplicate USB device index binding
        if resolved_type == "usb" and device_index is not None:
            if self.zone_registry.is_usb_active(device_index, exclude_camera_id=camera_id):
                raise ValueError(f"CONFLICT: USB webcam device index {device_index} is already active in another zone.")

        # If camera_id exists, stop old instance
        existing_worker = self.zone_registry.get_worker(camera_id)
        if existing_worker is not None:
            existing_worker.stop()
            self.zone_registry.unregister_binding(camera_id)

        # 4. Instantiate worker
        worker = CameraWorker(
            camera_id=camera_id,
            source=typed_val,
            zone_id=zone_id,
            source_type=resolved_type,
            capture_backend=capture_backend,
            location=location,
            target_width=width,
            target_height=height,
            target_fps=fps,
            gait_service=self,
        )

        # 5. Register binding
        self.zone_registry.register_binding(camera_id, zone_id, resolved_type, device_index, worker)

        # 6. Start worker and await first frame startup handshake
        success = worker.start(startup_timeout=5.0)
        if not success:
            self.zone_registry.unregister_binding(camera_id)
            err_msg = worker.error_message or f"Failed to start camera worker '{camera_id}' for source '{source}'."
            raise ValueError(err_msg)

        return worker.to_dict()

    def stop_camera(self, camera_id: str) -> bool:
        """Stops an active camera worker and releases zone and USB reservations."""
        worker = self.zone_registry.get_worker(camera_id)
        if worker is not None:
            worker.stop()
            self.zone_registry.unregister_binding(camera_id)
            return True
        return False

    def shutdown(self) -> None:
        """Shuts down all active camera workers during application teardown."""
        self.zone_registry.shutdown_all()
