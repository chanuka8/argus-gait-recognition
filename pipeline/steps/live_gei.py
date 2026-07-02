import cv2
import numpy as np


class LiveGEI:

    def __init__(
        self,
        max_frames: int = 15,
        min_frames: int | None = None,
        size=(64, 128),
    ):
        self.max_frames = max_frames
        self.min_frames = min_frames if min_frames is not None else max(1, min(10, max_frames))
        self.size = size

        self.frames = []
        self.valid_frames = 0
        self.rejected_frames = 0

    def add(self, silhouette):

        if silhouette is None:
            self.rejected_frames += 1
            return

        frame = cv2.resize(
            silhouette,
            self.size,
        )

        frame = (frame > 0).astype(
            np.float32
        )

        self.frames.append(frame)
        self.valid_frames += 1

        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    def ready(self):

        return len(self.frames) >= self.min_frames

    def build(self):

        if not self.ready():
            return None

        gei = np.mean(
            self.frames,
            axis=0,
        )

        gei = (
            gei * 255
        ).astype(np.uint8)

        return gei

    def count(self):

        return len(self.frames)

    def clear(self):

        self.frames.clear()
        self.valid_frames = 0
        self.rejected_frames = 0