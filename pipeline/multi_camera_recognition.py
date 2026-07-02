from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch
import yaml

from models.architectures.bygait_light import ByGaitLight
from pipeline.steps.centroid_matching_step import CentroidMatchingStep
from pipeline.steps.live_gei import LiveGEI
from pipeline.steps.matching_step import MatchingStep
from pipeline.steps.silhouette_step import SilhouetteStep
from pipeline.steps.tracking import TrackingStep
from security_layer.security_engine import SecurityEngine
from storage.vector_store import VectorStore
from streaming.multi_stream_engine import MultiStreamEngine
from utils.alert_manager import AlertManager
from utils.event_logger import EventLogger
from utils.prediction_smoother import PredictionSmoother
from utils.box_stabilizer import BoxStabilizer
from utils.display_renderer import DetectionDisplayRenderer, load_display_config
from utils.detection_reporter import DetectionReporter, load_reporting_config


def _load_matching_policy() -> dict:
    """Load matching policy from inference config with safe defaults."""
    config_path = Path("configs/inference.yaml")

    defaults = {
        "confirmed_threshold": 0.92,
        "verify_low": 0.85,
        "verify_high": 0.92,
        "low_confidence_low": 0.70,
        "low_confidence_high": 0.85,
        "unknown_ceiling": 0.70,
        "centroid_threshold": 0.85,
        "margin": 0.05,
        "top_k": 5,
        "min_stable_votes": 3,
        "history_size": 10,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    policy = data.get("matching_policy", {})

    if not isinstance(policy, dict):
        return defaults

    merged = {}
    for key, default_value in defaults.items():
        merged[key] = policy.get(key, default_value)

    return merged


def _load_crowd_control_config() -> dict:
    config_path = Path("configs/inference.yaml")

    defaults = {
        "enabled": True,
        "max_tracked_people_per_camera": 50,
        "max_recognitions_per_frame": 3,
        "recognition_queue_size": 100,
        "track_timeout_frames": 90,
        "min_box_height": 60,
        "min_box_area_ratio": 0.003,
        "priority_update_interval": 10,
        "alert_cooldown_seconds": 5,
        "disable_debug_windows": True,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    cc = data.get("crowd_control", {})

    if not isinstance(cc, dict):
        return defaults

    merged = {}

    for key, default_value in defaults.items():
        merged[key] = cc.get(key, default_value)

    return merged


def _load_box_stability_config() -> dict:
    config_path = Path("configs/inference.yaml")

    defaults = {
        "enabled": True,
        "ema_alpha": 0.35,
        "min_detection_confidence": 0.35,
        "min_iou_keep": 0.25,
        "max_missed_frames": 8,
        "max_jump_ratio": 0.35,
        "use_stable_box_for_display": True,
        "use_stable_box_for_silhouette": True,
        "prune_after_frames": 30,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    bs = data.get("box_stability", {})

    if not isinstance(bs, dict):
        return defaults

    merged = {}

    for key, default_value in defaults.items():
        merged[key] = bs.get(key, default_value)

    return merged


class CameraWorkerState:

    """
    Isolated per-camera mutable state.

    Each camera gets its own tracker (YOLO + ByteTrack),
    silhouette processor, GEI buffers, frame counters,
    recognition results, and prediction smoother.
    This prevents cross-camera track ID collisions and
    state corruption.
    """

    def __init__(
        self,
        camera_id: str,
        policy: dict,
        gei_frames: int = 15,
        recognition_interval: int = 10,
    ) -> None:
        self.camera_id = camera_id

        # Per-camera stateful components
        self.tracker = TrackingStep()
        self.silhouette_step = SilhouetteStep()

        # Per-camera mutable dictionaries
        self.buffers: dict[int, LiveGEI] = {}
        self.last_results: dict[int, dict] = {}
        self.frame_counters: dict[int, int] = {}

        self.gei_frames = gei_frames
        self.recognition_interval = recognition_interval

        self.smoother = PredictionSmoother(
            history_size=policy["history_size"],
            min_stable_votes=policy["min_stable_votes"],
        )

        self.box_stability_config = _load_box_stability_config()
        self.box_stabilizer = BoxStabilizer(self.box_stability_config)

        self.queue = []
        self.track_frames = {}
        self.last_recognition_frame = {}
        self.last_seen_frame = {}
        self.current_frame_index = 0
        self.interval_processed = 0
        self.interval_skipped = 0



class MultiCameraRecognitionPipeline:
    """
    Multi-camera orchestrator using Option B architecture.

    Manages per-camera worker threads with isolated state
    and shared read-only model/gallery resources.
    """

    def __init__(
        self,
        cameras_config_path: str = "configs/cameras.yaml",
        model_path: str = "runs/exp_001/best_model.pth",
        threshold: float = 0.85,
        alert_threshold: float = 0.90,
        security_threshold: float = 0.90,
        gei_frames: int = 15,
        recognition_interval: int = 10,
        gallery_dir: str = "models/live_gallery",
    ) -> None:
        # Load camera configuration
        self.cameras_config = self._load_cameras_config(
            cameras_config_path,
        )

        self.camera_list = self.cameras_config.get(
            "cameras", [],
        )

        self.orchestrator_config = self.cameras_config.get(
            "orchestrator", {},
        )

        # Filter to enabled cameras and apply max limit
        max_cameras = self.orchestrator_config.get(
            "max_cameras", 2,
        )

        self.camera_list = [
            cam for cam in self.camera_list
            if cam.get("enabled", True)
        ][:max_cameras]

        if not self.camera_list:
            raise RuntimeError(
                "No enabled cameras found in cameras.yaml"
            )

        # Load matching policy
        self.policy = _load_matching_policy()
        self.cc_config = _load_crowd_control_config()
        self.threshold = threshold
        self.gei_frames = gei_frames
        self.recognition_interval = recognition_interval

        # Shared resources (read-only, thread-safe)

        # Shared: neural network model (eval mode)
        self.model = self._load_model(model_path)

        # Shared: gallery embeddings
        gait_gallery = VectorStore(
            gallery_dir=gallery_dir,
        ).load()

        if gait_gallery is None:
            self.gallery_features = None
            self.gallery_labels = None
            self.metadata = {}
        else:
            (
                self.gallery_features,
                self.gallery_labels,
                self.metadata,
            ) = gait_gallery

        # Shared: stateless matchers
        self.matcher = MatchingStep(
            threshold=threshold,
        )

        self.centroid_matcher = CentroidMatchingStep(
            threshold=self.policy["centroid_threshold"],
            margin=self.policy["margin"],
            top_k=self.policy["top_k"],
        )

        # Shared: thread-safe loggers
        self.event_logger = EventLogger()

        self.alert_manager = AlertManager(
            confidence_threshold=alert_threshold,
        )

        self.security_engine = SecurityEngine(
            confidence_threshold=security_threshold,
        )

        # Per-camera isolated state

        self.workers: dict[str, CameraWorkerState] = {}

        for cam_cfg in self.camera_list:
            camera_id = str(cam_cfg["id"])

            self.workers[camera_id] = CameraWorkerState(
                camera_id=camera_id,
                policy=self.policy,
                gei_frames=gei_frames,
                recognition_interval=recognition_interval,
            )

        # Multi-stream engine

        queue_max_size = self.orchestrator_config.get(
            "queue_max_size", 20,
        )

        self.stream_engine = MultiStreamEngine(
            camera_configs=self.camera_list,
            queue_max_size=queue_max_size,
        )

        # GUI configuration
        self.show_gui = self.orchestrator_config.get(
            "show_gui", True,
        )

        # CCTV display renderer (shared, stateless) and detection reporter (shared, thread-safe)
        self.renderer = DetectionDisplayRenderer(load_display_config())
        self.reporter = DetectionReporter(
            config=load_reporting_config(),
            source_mode="multi-camera",
        )

        # Per-camera location lookup (fallback: "Unknown Location")
        self.camera_locations: dict[str, str] = {}
        for cam_cfg in self.camera_list:
            cid = str(cam_cfg["id"])
            self.camera_locations[cid] = cam_cfg.get("location", "Unknown Location")

        # Frame storage for main-thread rendering
        self.latest_frames: dict[str, np.ndarray] = {}
        self.frame_lock = threading.Lock()

        # Control flag
        self.running = False

    # Config loading

    @staticmethod
    def _load_cameras_config(
        config_path: str,
    ) -> dict:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Cameras config not found: {config_path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_model(
        self,
        model_path: str,
    ) -> ByGaitLight:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        model = ByGaitLight()

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

        filtered = {}
        for key, value in checkpoint.items():
            if key.startswith("backbone."):
                filtered[key.replace("backbone.", "")] = value

        model.load_state_dict(
            filtered,
            strict=True,
        )

        model.eval()

        return model

    # Processing helpers

    @staticmethod
    def _crop_person(
        frame: np.ndarray,
        box,
    ) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _gei_to_embedding(
        self,
        gei: np.ndarray,
    ) -> np.ndarray:
        gei_float = gei.astype(np.float32) / 255.0

        tensor = torch.from_numpy(
            gei_float,
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(
                tensor,
            ).cpu().numpy().flatten()

        return embedding

    def _adaptive_decision(
        self,
        embedding: np.ndarray,
        flat_identity: str,
        flat_score: float,
    ) -> tuple[str, float, str]:
        """
        Adaptive hybrid matching decision policy.
        Identical to single-camera logic.
        """
        confirmed_threshold = self.policy["confirmed_threshold"]
        verify_low = self.policy["verify_low"]
        verify_high = self.policy["verify_high"]
        low_confidence_low = self.policy["low_confidence_low"]
        low_confidence_high = self.policy["low_confidence_high"]
        unknown_ceiling = self.policy["unknown_ceiling"]

        if flat_identity == "UNKNOWN":
            return "UNKNOWN", flat_score, "UNKNOWN_PERSON"

        if flat_score >= confirmed_threshold:
            return flat_identity, flat_score, "CONFIRMED_MATCH"

        if verify_low <= flat_score < verify_high:
            centroid_identity, _ = self.centroid_matcher.match(
                embedding,
                self.gallery_features,
                self.gallery_labels,
                self.metadata,
                mode="centroid_margin_topk",
            )

            if centroid_identity == flat_identity:
                return flat_identity, flat_score, "VERIFIED_MATCH"
            else:
                return flat_identity, flat_score, "REVIEW_REQUIRED"

        if low_confidence_low <= flat_score < low_confidence_high:
            return flat_identity, flat_score, "LOW_CONFIDENCE"

        if flat_score < unknown_ceiling:
            return "UNKNOWN", flat_score, "UNKNOWN_PERSON"

        return flat_identity, flat_score, "LOW_CONFIDENCE"

    @staticmethod
    def _should_recognize(
        worker: CameraWorkerState,
        track_id: int,
    ) -> bool:
        if track_id not in worker.frame_counters:
            worker.frame_counters[track_id] = 0

        worker.frame_counters[track_id] += 1

        return (
            worker.frame_counters[track_id]
            % worker.recognition_interval
            == 0
        )

    # Recognition

    def _recognize_track(
        self,
        worker: CameraWorkerState,
        track_id: int,
        is_predicted: bool = False,
    ) -> None:
        gei = worker.buffers[track_id].build()

        if gei is None:
            return

        embedding = self._gei_to_embedding(gei)

        # Flat matching
        matches = self.matcher.top_k_matches(
            embedding,
            self.gallery_features,
            self.gallery_labels,
            self.metadata,
            k=1,
        )

        if not matches:
            flat_identity = "UNKNOWN"
            flat_score = 0.0
        else:
            flat_identity = matches[0][0]
            flat_score = matches[0][1]

        # Adaptive hybrid decision
        raw_identity, score, decision = self._adaptive_decision(
            embedding=embedding,
            flat_identity=flat_identity,
            flat_score=flat_score,
        )

        # Prediction smoothing
        stable_identity = worker.smoother.update(
            track_id,
            raw_identity,
            score=score,
            threshold=self.threshold,
        )

        # Severity classification
        if stable_identity == "UNKNOWN":
            score = 0.0
            decision = "UNKNOWN_PERSON"
            severity = "HIGH"
        elif decision == "CONFIRMED_MATCH":
            severity = "INFO"
        elif decision == "VERIFIED_MATCH":
            severity = "INFO"
        elif decision == "REVIEW_REQUIRED":
            severity = "MEDIUM"
        elif decision == "LOW_CONFIDENCE":
            severity = "MEDIUM"
        else:
            severity = "INFO"

        # Thread-safe logging with camera_id
        self.event_logger.log(
            track_id=track_id,
            identity=stable_identity,
            score=score,
            camera_id=worker.camera_id,
        )

        if not is_predicted:
            self.alert_manager.evaluate(
                track_id=track_id,
                identity=stable_identity,
                score=score,
                decision=decision,
                camera_id=worker.camera_id,
            )

        self.security_engine.evaluate(
            track_id=track_id,
            identity=stable_identity,
            score=score,
            camera_id=worker.camera_id,
        )

        # Store result in per-camera state
        worker.last_results[track_id] = {
            "identity": stable_identity,
            "raw_identity": raw_identity,
            "score": score,
            "gait_score": float(score),
            "severity": severity,
            "decision": decision,
        }

    # Drawing

    def _draw_track(
        self,
        frame: np.ndarray,
        box,
        track_id: int,
        worker: CameraWorkerState,
        is_predicted: bool = False,
    ) -> None:
        result = worker.last_results.get(
            track_id,
            {
                "identity": "TRACKING" if is_predicted else "COLLECTING",
                "raw_identity": "TRACKING" if is_predicted else "COLLECTING",
                "score": 0.0,
                "severity": "INFO",
                "decision": "TRACKING" if is_predicted else "COLLECTING",
            },
        )

        identity = str(result["identity"])
        score = float(result["score"])
        decision = str(result.get("decision", "COLLECTING"))

        if is_predicted and decision in ("COLLECTING", "TRACKING"):
            identity = "TRACKING"
            decision = "TRACKING"

        self.renderer.draw(
            frame=frame,
            box=box,
            track_id=track_id,
            identity=identity,
            score=score,
            decision=decision,
            camera_id=worker.camera_id,
        )

        # Auto-report (cooldown-gated, status-filtered)
        status = self.renderer.get_status(decision)
        bbox = list(map(int, box))
        location = self.camera_locations.get(worker.camera_id, "Unknown Location")
        self.reporter.report(
            camera_id=worker.camera_id,
            location=location,
            track_id=track_id,
            identity=identity,
            status=status,
            score=score,
            bbox=bbox,
            frame=frame,
        )

    # Frame processing

    def _process_camera_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
    ) -> None:
        """
        Process a single frame for a specific camera.

        Runs detection, tracking, silhouette extraction,
        GEI accumulation, and recognition using isolated
        per-camera state.
        """
        worker = self.workers[camera_id]

        try:
            detections = worker.tracker.track(frame)

            xyxy = detections.xyxy
            tracker_ids = detections.tracker_id

            # Build list of raw detections for stabilizer
            raw_detections = []
            if tracker_ids is not None:
                confidences = getattr(detections, "confidence", None)
                for i in range(len(tracker_ids)):
                    raw_detections.append((
                        int(tracker_ids[i]),
                        xyxy[i],
                        float(confidences[i]) if confidences is not None else 1.0
                    ))

            # Update stabilizer
            stable_results = worker.box_stabilizer.update(raw_detections, frame.shape)

            if not self.cc_config.get("enabled", True):
                for track_id, (stable_box, is_valid, is_predicted) in stable_results.items():
                    if not is_valid:
                        continue

                    raw_detected = (tracker_ids is not None) and (track_id in tracker_ids)
                    raw_box = None
                    if raw_detected:
                        raw_box = xyxy[list(tracker_ids).index(track_id)]

                    silhouette = None
                    if raw_detected:
                        box_to_crop = stable_box if worker.box_stability_config.get("use_stable_box_for_silhouette", True) else raw_box
                        crop = self._crop_person(frame, box_to_crop)
                        if crop is not None:
                            silhouette = worker.silhouette_step.extract_from_crop(crop)
                            if silhouette is not None:
                                if track_id not in worker.buffers:
                                    worker.buffers[track_id] = LiveGEI(max_frames=worker.gei_frames)
                                worker.buffers[track_id].add(silhouette)

                    if track_id in worker.buffers and worker.buffers[track_id].ready() and self._should_recognize(worker, track_id):
                        if not (is_predicted and not worker.buffers[track_id].ready()):
                            self._recognize_track(worker, track_id, is_predicted)

                    if self.show_gui:
                        box_to_draw = stable_box if worker.box_stability_config.get("use_stable_box_for_display", True) else (raw_box if raw_detected else stable_box)
                        self._draw_track(frame, box_to_draw, track_id, worker, is_predicted)
            else:
                worker.current_frame_index += 1
                h, w = frame.shape[:2]
                frame_area = h * w

                valid_detections = []
                for track_id, (stable_box, is_valid, is_predicted) in stable_results.items():
                    if not is_valid:
                        continue

                    # Bounding box coordinates for display
                    box_to_use = stable_box if worker.box_stability_config.get("use_stable_box_for_display", True) else stable_box
                    x1, y1, x2, y2 = box_to_use
                    box_h = y2 - y1
                    box_w = x2 - x1
                    box_area = box_w * box_h
                    area_ratio = box_area / frame_area if frame_area > 0 else 0

                    if box_h >= self.cc_config["min_box_height"] and area_ratio >= self.cc_config["min_box_area_ratio"]:
                        raw_detected = (tracker_ids is not None) and (track_id in tracker_ids)
                        raw_box = None
                        if raw_detected:
                            raw_box = xyxy[list(tracker_ids).index(track_id)]
                        valid_detections.append((track_id, stable_box, raw_box, box_area, is_predicted, raw_detected))

                max_track = self.cc_config["max_tracked_people_per_camera"]
                valid_detections = valid_detections[:max_track]

                active_track_ids = set()

                for box_item in valid_detections:
                    track_id, stable_box, raw_box, box_area, is_predicted, raw_detected = box_item
                    active_track_ids.add(track_id)
                    worker.last_seen_frame[track_id] = worker.current_frame_index

                    if raw_detected:
                        worker.track_frames[track_id] = worker.track_frames.get(track_id, 0) + 1
                        box_to_crop = stable_box if worker.box_stability_config.get("use_stable_box_for_silhouette", True) else raw_box
                        crop = self._crop_person(frame, box_to_crop)
                        if crop is not None:
                            silhouette = worker.silhouette_step.extract_from_crop(crop)
                            if silhouette is not None:
                                if track_id not in worker.buffers:
                                    worker.buffers[track_id] = LiveGEI(max_frames=worker.gei_frames)
                                worker.buffers[track_id].add(silhouette)

                    if track_id in worker.buffers and worker.buffers[track_id].ready() and self._should_recognize(worker, track_id):
                        if not (is_predicted and not worker.buffers[track_id].ready()):
                            not_rec_rec = worker.current_frame_index - worker.last_recognition_frame.get(track_id, 0)
                            priority = (
                                not_rec_rec,
                                1 if worker.buffers[track_id].ready() else 0,
                                worker.track_frames.get(track_id, 0),
                                box_area
                            )
                            queued_item = next((item for item in worker.queue if item["track_id"] == track_id), None)
                            if queued_item is not None:
                                queued_item["priority"] = priority
                                queued_item["box"] = stable_box
                            else:
                                worker.queue.append({
                                    "track_id": track_id,
                                    "priority": priority,
                                    "box": stable_box
                                })

                    if self.show_gui:
                        box_to_draw = stable_box if worker.box_stability_config.get("use_stable_box_for_display", True) else (raw_box if raw_detected else stable_box)
                        self._draw_track(frame, box_to_draw, track_id, worker, is_predicted)

                # Periodic update: sort queue, drop excess, remove inactive, log stats
                update_interval = self.cc_config["priority_update_interval"]
                if worker.current_frame_index % update_interval == 0:
                    worker.queue = [item for item in worker.queue if item["track_id"] in active_track_ids]
                    worker.queue.sort(key=lambda x: x["priority"], reverse=True)

                    q_max = self.cc_config["recognition_queue_size"]
                    if len(worker.queue) > q_max:
                        dropped = len(worker.queue) - q_max
                        worker.interval_skipped += dropped
                        worker.queue = worker.queue[:q_max]

                    timeout = self.cc_config["track_timeout_frames"]
                    inactive_tracks = []
                    for tid in list(worker.buffers.keys()):
                        if worker.current_frame_index - worker.last_seen_frame.get(tid, 0) > timeout:
                            inactive_tracks.append(tid)

                    for tid in inactive_tracks:
                        worker.buffers.pop(tid, None)
                        worker.frame_counters.pop(tid, None)
                        worker.last_results.pop(tid, None)
                        worker.track_frames.pop(tid, None)
                        worker.last_recognition_frame.pop(tid, None)
                        worker.last_seen_frame.pop(tid, None)
                        worker.smoother.history.pop(tid, None)
                        worker.smoother.confirmed_identities.pop(tid, None)

                    print(
                        f"[CROWD_CONTROL] Camera={camera_id} | "
                        f"Tracked={len(active_track_ids)} | "
                        f"Queued={len(worker.queue)} | "
                        f"Processed={worker.interval_processed} | "
                        f"Skipped={worker.interval_skipped}"
                    )
                    worker.interval_processed = 0
                    worker.interval_skipped = 0

                max_rec = self.cc_config["max_recognitions_per_frame"]
                processed_this_frame = 0
                while worker.queue and processed_this_frame < max_rec:
                    item = worker.queue.pop(0)
                    tid = item["track_id"]
                    if tid in worker.buffers:
                        track_pred = False
                        if tid in stable_results:
                            track_pred = stable_results[tid][2]
                        self._recognize_track(worker, tid, track_pred)
                        worker.last_recognition_frame[tid] = worker.current_frame_index
                        worker.interval_processed += 1
                        processed_this_frame += 1


        except Exception as e:
            print(
                f"[ERROR] Camera {camera_id} "
                f"processing error: {e}"
            )

        # Push annotated frame for main-thread rendering
        with self.frame_lock:
            self.latest_frames[camera_id] = frame

    # Worker thread

    def _camera_loop(
        self,
        camera_id: str,
    ) -> None:
        """
        Worker thread loop for a single camera.

        Reads frames from the multi-stream engine and
        processes them using isolated per-camera state.
        If the stream dies, the worker exits gracefully.
        """
        print(
            f"[MULTI-CAM] Camera {camera_id} "
            f"worker started"
        )

        while self.running:
            ret, frame = self.stream_engine.read(camera_id)

            if not ret:
                stream = self.stream_engine.streams.get(
                    camera_id,
                )

                if stream and not stream.is_opened():
                    print(
                        f"[ERROR] Camera {camera_id} "
                        f"stream died: {stream.error}"
                    )
                    break

                # No frame available yet, brief sleep
                time.sleep(0.01)
                continue

            self._process_camera_frame(camera_id, frame)

        print(
            f"[MULTI-CAM] Camera {camera_id} "
            f"worker stopped"
        )

    # Main entry point

    def run(self) -> None:
        """
        Start the multi-camera recognition pipeline.

        Camera worker threads process frames independently.
        GUI rendering happens in the main thread only.
        Press Q to quit.
        """
        print("\n" + "=" * 60)
        print("ARGUS MULTI-CAMERA RECOGNITION")
        print("=" * 60)
        print(
            f"Cameras configured: "
            f"{len(self.camera_list)}"
        )

        for cam in self.camera_list:
            print(
                f"  - {cam['id']}: "
                f"source={cam.get('source', 0)}, "
                f"fps={cam.get('target_fps', 5)}"
            )

        print(f"Threshold: {self.threshold:.2f}")
        print(
            f"Confirmed threshold: "
            f"{self.policy['confirmed_threshold']:.2f}"
        )
        print(
            f"Verify range: "
            f"[{self.policy['verify_low']:.2f}, "
            f"{self.policy['verify_high']:.2f})"
        )
        print(f"GUI: {self.show_gui}")
        print("Adaptive hybrid matching enabled")
        print("Per-camera state isolation enabled")
        print("Thread-safe logging enabled")
        print("Press Q to quit")
        print("=" * 60)

        # Start all camera streams
        start_results = self.stream_engine.start_all()

        active = [
            cid for cid, ok in start_results.items()
            if ok
        ]

        if not active:
            print(
                "[ERROR] No cameras could be started. "
                "Check your cameras.yaml configuration."
            )
            return

        print(f"\n[MULTI-CAM] Active cameras: {active}")

        self.running = True

        # Start per-camera worker threads
        threads = []

        for camera_id in active:
            t = threading.Thread(
                target=self._camera_loop,
                args=(camera_id,),
                daemon=True,
                name=f"worker-{camera_id}",
            )
            t.start()
            threads.append(t)

        # Main thread: GUI rendering loop
        try:
            while self.running:
                if self.show_gui:
                    with self.frame_lock:
                        frames_to_show = dict(
                            self.latest_frames,
                        )

                    for camera_id, frame in (
                        frames_to_show.items()
                    ):
                        window_name = (
                            f"ARGUS - {camera_id}"
                        )
                        cv2.imshow(window_name, frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print(
                            "\n[MULTI-CAM] "
                            "Quit signal received"
                        )
                        self.running = False
                        break
                else:
                    time.sleep(0.1)

                # Check if all cameras are dead
                any_alive = any(
                    self.stream_engine.streams[cid].is_opened()
                    for cid in active
                    if cid in self.stream_engine.streams
                )

                if not any_alive:
                    print(
                        "[MULTI-CAM] "
                        "All cameras stopped"
                    )
                    self.running = False
                    break

        except KeyboardInterrupt:
            print(
                "\n[MULTI-CAM] Interrupted by user"
            )
            self.running = False

        # Cleanup

        self.stream_engine.stop_all()

        for t in threads:
            t.join(timeout=5)

        if self.show_gui:
            cv2.destroyAllWindows()

        # Print statistics
        print("\n=== MULTI-CAM STATS ===")

        for camera_id, stats in (
            self.stream_engine.stats().items()
        ):
            print(
                f"  {camera_id}: "
                f"read={stats['frames_read']}, "
                f"dropped={stats['frames_dropped']}, "
                f"error={stats['error']}"
            )

        print("Multi-camera recognition stopped.")
