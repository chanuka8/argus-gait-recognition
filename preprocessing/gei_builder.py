import cv2
import numpy as np


class GEIBuilder:
    def __init__(self, size=(64, 128)):
        self.frames = []
        self.size = size
        self.max_frames = 30

    def add_frame(self, silhouette: np.ndarray) -> None:
        if silhouette is None:
            return

        frame = cv2.resize(silhouette, self.size)
        frame = (frame > 0).astype(np.float32)

        self.frames.append(frame)

        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    def is_ready(self, min_frames: int = 15) -> bool:
        return len(self.frames) >= min_frames

    def build(self) -> np.ndarray | None:
        if not self.is_ready():
            return None

        gei = np.mean(self.frames, axis=0)

        gei = (gei * 255).astype(np.uint8)

        return gei

    def reset(self) -> None:
        self.frames.clear()