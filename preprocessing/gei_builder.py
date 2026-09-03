import cv2
import numpy as np


class GEIBuilder:
    def __init__(self, size: tuple[int, int] = (64, 128)) -> None:
        self.frames: list[np.ndarray] = []
        self.size = size  # (width, height)
        self.max_frames = 30
        self.running_sum: np.ndarray | None = None

    def add_frame(self, silhouette: np.ndarray | None) -> None:
        if silhouette is None or silhouette.size == 0:
            return

        expected_shape = (self.size[1], self.size[0])
        if silhouette.shape[:2] == expected_shape:
            frame = silhouette
        else:
            frame = cv2.resize(silhouette, self.size, interpolation=cv2.INTER_NEAREST)

        norm = (frame > 0).astype(np.float32, copy=False)
        self.frames.append(norm)

        if self.running_sum is None:
            self.running_sum = norm.copy()
        else:
            self.running_sum += norm

        if len(self.frames) > self.max_frames:
            popped = self.frames.pop(0)
            self.running_sum -= popped

    def is_ready(self, min_frames: int = 15) -> bool:
        return len(self.frames) >= min_frames

    def build(self) -> np.ndarray | None:
        if not self.is_ready() or self.running_sum is None:
            return None

        count = len(self.frames)
        if count == 0:
            return None

        # O(1) vectorized mean and uint8 conversion
        gei = np.clip((self.running_sum / count) * 255.0, 0, 255).astype(np.uint8)
        return gei

    def reset(self) -> None:
        self.frames.clear()
        self.running_sum = None
