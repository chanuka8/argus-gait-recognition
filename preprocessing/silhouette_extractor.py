import cv2
import numpy as np


class SilhouetteExtractor:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False,
        )

    def extract(self, frame: np.ndarray) -> np.ndarray:
        mask = self.background_subtractor.apply(frame)

        _, binary = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY,
        )

        return binary