from pathlib import Path

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
from streaming.stream_engine import StreamEngine
from utils.alert_manager import AlertManager
from utils.event_logger import EventLogger
from utils.prediction_smoother import PredictionSmoother
from utils.box_stabilizer import BoxStabilizer
from utils.display_renderer import DetectionDisplayRenderer, load_display_config
from utils.detection_reporter import DetectionReporter, load_reporting_config



def _load_matching_policy() -> dict:
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


class LiveRecognitionPipeline:

    def __init__(
        self,
        model_path: str = "runs/exp_001/best_model.pth",
        threshold: float = 0.85,
        alert_threshold: float = 0.90,
        security_threshold: float = 0.90,
        gei_frames: int = 15,
        recognition_interval: int = 10,
        history_size: int = 10,
        gallery_dir: str = "models/live_gallery",
    ) -> None:
        self.stream = StreamEngine()
        self.tracker = TrackingStep()
        self.silhouette_step = SilhouetteStep()

        self.policy = _load_matching_policy()
        self.cc_config = _load_crowd_control_config()
        self.box_stability_config = _load_box_stability_config()
        self.box_stabilizer = BoxStabilizer(self.box_stability_config)

        # CCTV display renderer and detection reporter
        self.renderer = DetectionDisplayRenderer(load_display_config())
        self.reporter = DetectionReporter(
            config=load_reporting_config(),
            source_mode="live",
        )
        self.camera_id = "cam_00"
        self.camera_location = "Unknown Location"


        self.matcher = MatchingStep(
            threshold=threshold,
        )

        self.centroid_matcher = CentroidMatchingStep(
            threshold=self.policy["centroid_threshold"],
            margin=self.policy["margin"],
            top_k=self.policy["top_k"],
        )

        self.threshold = threshold

        self.model = self._load_model(
            model_path,
        )

        gait_gallery = VectorStore(
            gallery_dir=gallery_dir,
        ).load()

        if gait_gallery is None:
            self.gallery_features = None
            self.gallery_labels = None
            self.metadata = {}
        else:
            self.gallery_features, self.gallery_labels, self.metadata = gait_gallery

        self.gei_frames = gei_frames
        self.recognition_interval = recognition_interval

        self.buffers: dict[int, LiveGEI] = {}
        self.last_results: dict[int, dict] = {}
        self.frame_counters: dict[int, int] = {}

        self.event_logger = EventLogger()
        self.smoother = PredictionSmoother(
            history_size=self.policy["history_size"],
            min_stable_votes=self.policy["min_stable_votes"],
        )

        self.alert_manager = AlertManager(
            confidence_threshold=alert_threshold,
        )

        self.security_engine = SecurityEngine(
            confidence_threshold=security_threshold,
        )

        self.queue = []
        self.track_frames = {}
        self.last_recognition_frame = {}
        self.last_seen_frame = {}
        self.current_frame_index = 0
        self.interval_processed = 0
        self.interval_skipped = 0


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

    def _crop_person(
        self,
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
        gei = gei.astype(np.float32) / 255.0

        tensor = torch.from_numpy(
            gei,
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(
                tensor,
            ).cpu().numpy().flatten()

        return embedding

    def _should_recognize(
        self,
        track_id: int,
    ) -> bool:
        if track_id not in self.frame_counters:
            self.frame_counters[track_id] = 0

        self.frame_counters[track_id] += 1

        return (
            self.frame_counters[track_id]
            % self.recognition_interval
            == 0
        )

    def _adaptive_decision(
        self,
        embedding: np.ndarray,
        flat_identity: str,
        flat_score: float,
    ) -> tuple[str, float, str]:
        """
        Adaptive hybrid matching decision policy.

        Returns:
            (identity, score, decision)

        Decision levels:
            CONFIRMED_MATCH     -> flat_score >= confirmed_threshold
            VERIFIED_MATCH      -> verify_low <= flat_score < verify_high,
                                   centroid verification agrees
            REVIEW_REQUIRED     -> verification disagrees
            LOW_CONFIDENCE      -> low_confidence_low <= flat_score < low_confidence_high
            UNKNOWN_PERSON      -> flat_score < unknown_ceiling or flat is UNKNOWN
        """
        confirmed_threshold = self.policy["confirmed_threshold"]
        verify_low = self.policy["verify_low"]
        verify_high = self.policy["verify_high"]
        low_confidence_low = self.policy["low_confidence_low"]
        low_confidence_high = self.policy["low_confidence_high"]
        unknown_ceiling = self.policy["unknown_ceiling"]

        # If flat matcher returned UNKNOWN
        if flat_identity == "UNKNOWN":
            return "UNKNOWN", flat_score, "UNKNOWN_PERSON"

        # High confidence confirmed match
        if flat_score >= confirmed_threshold:
            return flat_identity, flat_score, "CONFIRMED_MATCH"

        # Mid-range: run centroid, margin, and top-k verification
        if verify_low <= flat_score < verify_high:
            centroid_identity, centroid_score = self.centroid_matcher.match(
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

        # Low confidence zone
        if low_confidence_low <= flat_score < low_confidence_high:
            return flat_identity, flat_score, "LOW_CONFIDENCE"

        # Below floor: unknown
        if flat_score < unknown_ceiling:
            return "UNKNOWN", flat_score, "UNKNOWN_PERSON"

        # Fallback: treat as low confidence
        return flat_identity, flat_score, "LOW_CONFIDENCE"

    def _match_gait(
        self,
        gei: np.ndarray,
    ) -> tuple[str, float, str]:
        """
        Run flat matching first, then apply adaptive decision policy.

        Returns:
            (identity, score, decision)
        """
        embedding = self._gei_to_embedding(
            gei,
        )

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

        identity, score, decision = self._adaptive_decision(
            embedding=embedding,
            flat_identity=flat_identity,
            flat_score=flat_score,
        )

        return identity, score, decision

    def _final_identity(
        self,
        track_id: int,
        raw_identity: str,
        score: float,
    ) -> str:
        return self.smoother.update(
            track_id,
            raw_identity,
            score=score,
            threshold=self.threshold,
        )

    def _recognize_track(
        self,
        track_id: int,
        is_predicted: bool = False,
    ) -> None:
        gei = self.buffers[track_id].build()

        if gei is None:
            return

        raw_gait_identity, gait_score, decision = self._match_gait(
            gei,
        )

        stable_identity = self._final_identity(
            track_id=track_id,
            raw_identity=raw_gait_identity,
            score=gait_score,
        )

        if stable_identity == "UNKNOWN":
            score = 0.0
            severity = "HIGH"
            decision = "UNKNOWN_PERSON"
        else:
            score = gait_score

            if decision == "CONFIRMED_MATCH":
                severity = "INFO"
            elif decision == "VERIFIED_MATCH":
                severity = "INFO"
            elif decision == "REVIEW_REQUIRED":
                severity = "MEDIUM"
            elif decision == "LOW_CONFIDENCE":
                severity = "MEDIUM"
            else:
                severity = "INFO"

        self.event_logger.log(
            track_id=track_id,
            identity=stable_identity,
            score=score,
        )

        if not is_predicted:
            self.alert_manager.evaluate(
                track_id=track_id,
                identity=stable_identity,
                score=score,
                decision=decision,
            )

        self.last_results[track_id] = {
            "identity": stable_identity,
            "raw_identity": raw_gait_identity,
            "score": score,
            "gait_score": float(gait_score),
            "severity": severity,
            "decision": decision,
        }


        if not self.cc_config.get("disable_debug_windows", True):
            cv2.imshow(
                f"GEI Track {track_id}",
                gei,
            )


    def _draw_track(
        self,
        frame: np.ndarray,
        box,
        track_id: int,
        is_predicted: bool = False,
    ) -> None:
        result = self.last_results.get(
            track_id,
            {
                "identity": "TRACKING" if is_predicted else "COLLECTING",
                "raw_identity": "TRACKING" if is_predicted else "COLLECTING",
                "score": 0.0,
                "gait_score": 0.0,
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
            camera_id=self.camera_id,
        )

        # Auto-report (cooldown-gated, status-filtered)
        status = self.renderer.get_status(decision)
        bbox = list(map(int, box))
        self.reporter.report(
            camera_id=self.camera_id,
            location=self.camera_location,
            track_id=track_id,
            identity=identity,
            status=status,
            score=score,
            bbox=bbox,
            frame=frame,
        )

    def run(
        self,
    ) -> None:
        if not self.stream.is_opened():
            print("Camera not found.")
            return

        print("ARGUS adaptive hybrid live recognition started")
        print("Live gallery enabled")
        print(f"Gait threshold: {self.threshold:.2f}")
        print(f"Confirmed threshold: {self.policy['confirmed_threshold']:.2f}")
        print(f"Verify range: [{self.policy['verify_low']:.2f}, {self.policy['verify_high']:.2f})")
        print(f"Low confidence range: [{self.policy['low_confidence_low']:.2f}, {self.policy['low_confidence_high']:.2f})")
        print(f"Unknown ceiling: {self.policy['unknown_ceiling']:.2f}")
        print("Adaptive hybrid matching enabled")
        print("Prediction smoothing enabled")
        print("Unknown-safe smoothing enabled")
        print("Event logging enabled")
        print("Alert system enabled")
        print("Security layer enabled")
        print("Press Q to quit")

        while True:
            ret, frame = self.stream.read()

            if not ret:
                print("Failed to read frame.")
                break

            detections = self.tracker.track(
                frame,
            )

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
            stable_results = self.box_stabilizer.update(raw_detections, frame.shape)

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
                        box_to_crop = stable_box if self.box_stability_config.get("use_stable_box_for_silhouette", True) else raw_box
                        crop = self._crop_person(frame, box_to_crop)
                        if crop is not None:
                            silhouette = self.silhouette_step.extract_from_crop(crop)
                            if silhouette is not None:
                                if track_id not in self.buffers:
                                    self.buffers[track_id] = LiveGEI(max_frames=self.gei_frames)
                                self.buffers[track_id].add(silhouette)

                    if track_id in self.buffers and self.buffers[track_id].ready() and self._should_recognize(track_id):
                        if not (is_predicted and not self.buffers[track_id].ready()):
                            self._recognize_track(track_id, is_predicted)

                    box_to_draw = stable_box if self.box_stability_config.get("use_stable_box_for_display", True) else (raw_box if raw_detected else stable_box)
                    self._draw_track(frame, box_to_draw, track_id, is_predicted)

                    if not self.cc_config.get("disable_debug_windows", True) and raw_detected and silhouette is not None:
                        cv2.imshow(
                            f"Silhouette Track {track_id}",
                            silhouette,
                        )
            else:
                self.current_frame_index += 1
                h, w = frame.shape[:2]
                frame_area = h * w

                valid_detections = []
                for track_id, (stable_box, is_valid, is_predicted) in stable_results.items():
                    if not is_valid:
                        continue

                    # Bounding box coordinates for display
                    box_to_use = stable_box if self.box_stability_config.get("use_stable_box_for_display", True) else stable_box
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
                    self.last_seen_frame[track_id] = self.current_frame_index

                    silhouette = None
                    if raw_detected:
                        self.track_frames[track_id] = self.track_frames.get(track_id, 0) + 1
                        box_to_crop = stable_box if self.box_stability_config.get("use_stable_box_for_silhouette", True) else raw_box
                        crop = self._crop_person(frame, box_to_crop)
                        if crop is not None:
                            silhouette = self.silhouette_step.extract_from_crop(crop)
                            if silhouette is not None:
                                if track_id not in self.buffers:
                                    self.buffers[track_id] = LiveGEI(max_frames=self.gei_frames)
                                self.buffers[track_id].add(silhouette)

                    if track_id in self.buffers and self.buffers[track_id].ready() and self._should_recognize(track_id):
                        if not (is_predicted and not self.buffers[track_id].ready()):
                            not_rec_rec = self.current_frame_index - self.last_recognition_frame.get(track_id, 0)
                            priority = (
                                not_rec_rec,
                                1 if self.buffers[track_id].ready() else 0,
                                self.track_frames.get(track_id, 0),
                                box_area
                            )
                            queued_item = next((item for item in self.queue if item["track_id"] == track_id), None)
                            if queued_item is not None:
                                queued_item["priority"] = priority
                                queued_item["box"] = stable_box
                            else:
                                self.queue.append({
                                    "track_id": track_id,
                                    "priority": priority,
                                    "box": stable_box
                                })

                    box_to_draw = stable_box if self.box_stability_config.get("use_stable_box_for_display", True) else (raw_box if raw_detected else stable_box)
                    self._draw_track(frame, box_to_draw, track_id, is_predicted)

                    if not self.cc_config.get("disable_debug_windows", True) and raw_detected and silhouette is not None:
                        cv2.imshow(
                            f"Silhouette Track {track_id}",
                            silhouette,
                        )

                # Periodic update: sort queue, drop excess, remove inactive, log stats
                update_interval = self.cc_config["priority_update_interval"]
                if self.current_frame_index % update_interval == 0:
                    self.queue = [item for item in self.queue if item["track_id"] in active_track_ids]
                    self.queue.sort(key=lambda x: x["priority"], reverse=True)

                    q_max = self.cc_config["recognition_queue_size"]
                    if len(self.queue) > q_max:
                        dropped = len(self.queue) - q_max
                        self.interval_skipped += dropped
                        self.queue = self.queue[:q_max]

                    timeout = self.cc_config["track_timeout_frames"]
                    inactive_tracks = []
                    for tid in list(self.buffers.keys()):
                        if self.current_frame_index - self.last_seen_frame.get(tid, 0) > timeout:
                            inactive_tracks.append(tid)

                    for tid in inactive_tracks:
                        self.buffers.pop(tid, None)
                        self.frame_counters.pop(tid, None)
                        self.last_results.pop(tid, None)
                        self.track_frames.pop(tid, None)
                        self.last_recognition_frame.pop(tid, None)
                        self.last_seen_frame.pop(tid, None)
                        self.smoother.history.pop(tid, None)
                        self.smoother.confirmed_identities.pop(tid, None)

                    print(
                        f"[CROWD_CONTROL] Camera=default | "
                        f"Tracked={len(active_track_ids)} | "
                        f"Queued={len(self.queue)} | "
                        f"Processed={self.interval_processed} | "
                        f"Skipped={self.interval_skipped}"
                    )
                    self.interval_processed = 0
                    self.interval_skipped = 0

                max_rec = self.cc_config["max_recognitions_per_frame"]
                processed_this_frame = 0
                while self.queue and processed_this_frame < max_rec:
                    item = self.queue.pop(0)
                    tid = item["track_id"]
                    if tid in self.buffers:
                        track_pred = False
                        if tid in stable_results:
                            track_pred = stable_results[tid][2]
                        self._recognize_track(tid, track_pred)
                        self.last_recognition_frame[tid] = self.current_frame_index
                        self.interval_processed += 1
                        processed_this_frame += 1


            cv2.imshow(
                "ARGUS Multi-Person Live Recognition",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stream.release()
        cv2.destroyAllWindows()