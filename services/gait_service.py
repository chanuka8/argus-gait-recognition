from services.camera_worker import CameraWorker
from typing import Optional, List
from pathlib import Path
import yaml
from services.camera_source_resolver import CameraSourceResolver
from security_layer.credentials import sanitize_rtsp_url
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
    ByGaitLight feature encoding, VectorStore matching, camera worker state, and event history.
    """

    def __init__(self, gallery_dir: str = "models/live_gallery") -> None:
        self.logger = setup_logger("ARGUS.GaitService")
        self.gallery_dir = gallery_dir

        self.store = VectorStore(gallery_dir=gallery_dir)
        self.extractor = FeatureExtractionStep()
        self.matcher = MatchingStep(threshold=0.85)
        self.silhouette_extractor = SilhouetteExtractor(target_size=(64, 128))
        self.open_set_recognizer = OpenSetRecognizer()

        self.detector = None
        try:
            self.detector = PersonDetector()
        except Exception as err:
            self.logger.warning(f"PersonDetector initialization skipped: {err}")

        self.ws_manager = WebSocketManager()
        self.events_log: list[dict] = []
        self.active_cameras: dict[str, dict] = {}
        self.source_resolver = CameraSourceResolver()
        self.camera_workers: dict[str, CameraWorker] = {}

        self.stats = {
            "processed_images": 0,
            "processed_videos": 0,
            "total_events": 0,
        }

        self.reload_gallery()

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

        for worker in self.camera_workers.values():
            if worker.recognition_worker is not None:
                worker.recognition_worker.update_gallery(
                    gallery_features=self.gallery_features,
                    gallery_labels=self.gallery_labels,
                    metadata=self.metadata,
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
        except Exception:
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
            except Exception as err:
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
        except Exception as e:
            self.logger.warning(f"Could not load system camera configuration: {e}")

        return defaults

    def start_camera(
        self,
        camera_id: str,
        source: str = "auto",
        location: str = "Surveillance Zone",
        zone_id: Optional[str] = None,
        user_id: str = "default_user",
        credential_id: Optional[str] = None,
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
        source_type = "webcam" if resolved_type in ("usb", "webcam", "local") else ("rtsp" if resolved_type == "rtsp" else resolved_type)
        resolved_label = resolution["resolved_source_label"]
        res_cred_id = resolution.get("credential_id")
        res_cred_conf = resolution.get("credential_configured", False)

        sanitized_source = sanitize_rtsp_url(resolved_source)

        camera_defaults = self._load_camera_config()
        worker_cfg = {
            **camera_defaults,
            "type": source_type,
            "url": resolved_source if source_type != "webcam" else "",
            "device_index": int(resolved_source) if source_type == "webcam" and str(resolved_source).isdigit() else 0,
        }

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
                event_callback=self._handle_recognition_event,
            )
        except Exception as rec_err:
            self.logger.warning(f"Recognition worker init deferred for {camera_id}: {rec_err}")

        worker = CameraWorker(
            camera_id=camera_id,
            camera_config=worker_cfg,
            inference_pipeline=None,
            detection_processor=None,
            recognition_worker=recognition_worker,
        )

        started = worker.start()
        if not started:
            self.source_resolver.release_source_by_camera_id(camera_id)
            raise RuntimeError(f"Unable to establish stream connection for {sanitized_source} ({camera_id}): failed to capture video frames")

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
        self.logger.info(f"Camera worker {camera_id} active with {sanitize_rtsp_url(resolved_label)} [type={source_type}]")
        return cam_info

    def stop_camera(self, camera_id: str) -> bool:
        """Stops camera tracking worker state and releases source reservation."""
        worker = self.camera_workers.pop(camera_id, None)
        if worker:
            try:
                worker.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping worker {camera_id}: {sanitize_rtsp_url(str(e))}")

        if camera_id in self.active_cameras:
            self.active_cameras[camera_id]["status"] = "STOPPED"
            del self.active_cameras[camera_id]
            self.source_resolver.release_source_by_camera_id(camera_id)
            return True
        return False

    def get_camera_worker(self, camera_id: str) -> Optional[CameraWorker]:
        """Return active CameraWorker instance if present."""
        return self.camera_workers.get(camera_id)

    def get_camera_info(self, camera_id: str) -> Optional[dict]:
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
            cam["status"] = "ACTIVE" if worker.is_running() and worker.is_connected() else ("ACTIVE" if worker.is_running() else "STOPPED")
        return cam

    def list_all_cameras(self) -> List[dict]:
        """Return all active camera info with updated telemetry."""
        result = []
        for cam_id in list(self.active_cameras.keys()):
            info = self.get_camera_info(cam_id)
            if info:
                result.append(info)
        return result
