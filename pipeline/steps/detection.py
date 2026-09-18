from pathlib import Path

from ultralytics import YOLO

from core.paths import resolve_app_path


class DetectionStep:
    def __init__(
        self,
        model_path: str | Path = "models/weights/yolov8n.pt",
        confidence: float = 0.4,
    ) -> None:
        self.model_path = resolve_app_path(model_path)
        self.confidence = confidence


        from security_layer.model_integrity import (
            ROLE_PERSON_DETECTOR,
            get_model_verifier,
            verify_model,
        )

        verifier = get_model_verifier()
        if verifier.is_strict_mode():
            verified_path = verify_model(self.model_path, expected_role=ROLE_PERSON_DETECTOR)
            self.model = YOLO(str(verified_path))
        else:
            if not self.model_path.exists():
                self.model = YOLO("yolov8n.pt")
            else:
                verified_path = verify_model(self.model_path, expected_role=ROLE_PERSON_DETECTOR)
                self.model = YOLO(str(verified_path))

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[0],
            verbose=False,
        )

        return results
