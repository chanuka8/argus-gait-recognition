from pathlib import Path

from ultralytics import YOLO


class DetectionStep:
    def __init__(
        self,
        model_path: str = "models/weights/yolov8n.pt",
        confidence: float = 0.4,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence

        if not self.model_path.exists():
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(str(self.model_path))

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )

        return results