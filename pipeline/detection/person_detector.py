import threading
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from monitoring.logging_config import get_logger


class PersonDetector:
    def __init__(self, config_path: str = "configs/detection.yaml") -> None:
        self.logger = get_logger("detection")
        self.config = self._load_config(config_path)
        self.lock = threading.Lock()

        model_path = Path(self.config.get("model_path", "models/weights/yolov8n.pt"))
        self.confidence = float(self.config.get("confidence", 0.4))
        self.classes = self.config.get("classes", [0])

        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            self.model = YOLO("yolov8n.pt")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        defaults = {
            "model_path": "models/weights/yolov8n.pt",
            "confidence": 0.4,
            "classes": [0],
        }

        if not path.exists():
            return defaults

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
                for key, val in defaults.items():
                    data.setdefault(key, val)
                return data
        except Exception:
            return defaults

    def detect(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []

        with self.lock:
            results = self.model(
                frame,
                conf=self.confidence,
                classes=self.classes,
                verbose=False,
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
