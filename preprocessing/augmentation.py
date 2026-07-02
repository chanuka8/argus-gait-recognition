import cv2
import numpy as np


class Augmentation:
    @staticmethod
    def horizontal_flip(image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    @staticmethod
    def gaussian_noise(
        image: np.ndarray,
        sigma: float = 10.0,
    ) -> np.ndarray:

        noise = np.random.normal(
            0,
            sigma,
            image.shape,
        )

        noisy = image.astype(np.float32) + noise

        noisy = np.clip(noisy, 0, 255)

        return noisy.astype(np.uint8)