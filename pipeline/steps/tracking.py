from pathlib import Path

import supervision as sv
from ultralytics import YOLO


class TrackingStep:
    def __init__(
        self,
        model_path: str = "models/weights/yolov8n.pt",
        confidence: float = 0.4,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence

        if self.model_path.exists():
            self.detector = YOLO(str(self.model_path))
        else:
            self.detector = YOLO("yolov8n.pt")

        self.tracker = sv.ByteTrack()

    def track(self, frame):
        result = self.detector(
            frame,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        detections = self.tracker.update_with_detections(detections)

        return detections