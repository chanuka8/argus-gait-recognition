import numpy as np

from pipeline.steps.silhouette_step import SilhouetteStep


class SilhouetteExtractor:
    def __init__(self, target_size: tuple[int, int] = (64, 128)) -> None:
        self.target_size = target_size
        self.step = SilhouetteStep(target_size=target_size)

    def extract_from_crop(self, crop: np.ndarray) -> np.ndarray | None:
        return self.step.extract_from_crop(crop)

    def extract_from_frame(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        if frame is None or frame.size == 0 or len(bbox) < 4:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        return self.extract_from_crop(crop)
