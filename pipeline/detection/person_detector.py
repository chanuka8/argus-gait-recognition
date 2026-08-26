import threading
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from automation.device_manager import DeviceManager
from monitoring.logging_config import get_logger


class PersonDetector:
    def __init__(self, config_path: str = "configs/detection.yaml") -> None:
        self.logger = get_logger("detection")
        self.config = self._load_config(config_path)
        self.lock = threading.Lock()

        model_path = Path(self.config.get("model_path", "models/weights/yolov8n.pt"))

        raw_conf = self.config.get("confidence", 0.4)
        self.confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) and 0.0 <= raw_conf <= 1.0 else 0.4

        raw_iou = self.config.get("iou_threshold", 0.45)
        self.iou_threshold = float(raw_iou) if isinstance(raw_iou, (int, float)) and 0.0 <= raw_iou <= 1.0 else 0.45

        raw_classes = self.config.get("classes", [0])
        self.classes = list(raw_classes) if isinstance(raw_classes, list) else [0]

        raw_device = str(self.config.get("device", "auto")).lower()
        if raw_device not in {"cpu", "cuda", "cuda:0", "0", "1", "auto"}:
            raw_device = "auto"
        self.device = raw_device
        self.runtime_device = DeviceManager.get_instance().resolve_component_device(self.device)

        raw_imgsz = self.config.get("img_size", 640)
        self.img_size = int(raw_imgsz) if isinstance(raw_imgsz, int) and raw_imgsz > 0 else 640

        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            self.model = YOLO("yolov8n.pt")

        if self.runtime_device:
            try:
                self.model.to(self.runtime_device)
            except Exception:
                pass

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        defaults = {
            "model_path": "models/weights/yolov8n.pt",
            "confidence": 0.4,
            "iou_threshold": 0.45,
            "classes": [0],
            "device": "cpu",
            "img_size": 640,
        }

        if not path.exists():
            return defaults

        try:
            with open(path, encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
                if not isinstance(data, dict):
                    return defaults
                for key, val in defaults.items():
                    data.setdefault(key, val)
                return data
        except Exception:
            return defaults

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []

        kwargs = {
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "classes": self.classes,
            "verbose": False,
            "imgsz": self.img_size,
        }
        if self.runtime_device:
            kwargs["device"] = self.runtime_device

        with self.lock:
            results = self.model(
                frame,
                **kwargs,
            )

        detections = []

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            box = xyxy[i].tolist()
            score = float(confidences[i])

            detections.append({
                "track_input": frame,
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "confidence": score,
            })

        return detections
