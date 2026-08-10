from pathlib import Path

import supervision as sv
from ultralytics import YOLO
import yaml


class TrackingStep:
    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        config_path: str = "configs/detection.yaml",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        resolved_model_path = model_path if model_path is not None else self.config.get("model_path", "models/weights/yolov8n.pt")
        self.model_path = Path(resolved_model_path)

        raw_conf = confidence if confidence is not None else self.config.get("confidence", 0.4)
        self.confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) and 0.0 <= raw_conf <= 1.0 else 0.4

        raw_iou = self.config.get("iou_threshold", 0.45)
        self.iou_threshold = float(raw_iou) if isinstance(raw_iou, (int, float)) and 0.0 <= raw_iou <= 1.0 else 0.45

        raw_classes = self.config.get("classes", [0])
        self.classes = list(raw_classes) if isinstance(raw_classes, list) else [0]

        self.device = str(self.config.get("device", "cpu")).lower()
        if self.device not in {"cpu", "cuda", "0", "1", "auto"}:
            self.device = "cpu"

        raw_imgsz = self.config.get("img_size", 640)
        self.img_size = int(raw_imgsz) if isinstance(raw_imgsz, int) and raw_imgsz > 0 else 640

        if self.model_path.exists():
            self.detector = YOLO(str(self.model_path))
        else:
            self.detector = YOLO("yolov8n.pt")

        self.tracker = sv.ByteTrack()

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        if not config_path.exists():
            return {}
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def track(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return sv.Detections.empty()

        kwargs = {
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "classes": self.classes,
            "verbose": False,
            "imgsz": self.img_size,
        }
        if self.device and self.device != "auto":
            kwargs["device"] = self.device

        result = self.detector(
            frame,
            **kwargs,
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        detections = self.tracker.update_with_detections(detections)

        return detections

