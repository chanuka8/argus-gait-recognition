from pathlib import Path

import cv2
import numpy as np


class AppearanceFeatureExtractionStep:
    def __init__(
        self,
        size: tuple[int, int] = (128, 256),
        hist_bins: int = 32,
    ) -> None:
        self.size = size
        self.hist_bins = hist_bins

    def _read_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        image = cv2.imread(
            str(image_path),
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image

    def _resize(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        return cv2.resize(
            image,
            self.size,
        )

    def _color_histogram(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [
                self.hist_bins,
                self.hist_bins,
                8,
            ],
            [
                0,
                180,
                0,
                256,
                0,
                256,
            ],
        )

        hist = cv2.normalize(
            hist,
            hist,
        ).flatten()

        return hist.astype(
            np.float32,
        )

    def _shape_feature(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        projection_x = np.mean(
            edges,
            axis=0,
        )

        projection_y = np.mean(
            edges,
            axis=1,
        )

        feature = np.concatenate(
            [
                projection_x,
                projection_y,
            ]
        )

        return feature.astype(
            np.float32,
        )

    def extract_from_array(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        image = self._resize(
            image,
        )

        color = self._color_histogram(
            image,
        )

        shape = self._shape_feature(
            image,
        )

        feature = np.concatenate(
            [
                color,
                shape,
            ]
        )

        norm = np.linalg.norm(
            feature,
        )

        feature = feature / (
            norm + 1e-8
        )

        return feature.astype(
            np.float32,
        )

    def extract(
        self,
        image_path,
    ) -> np.ndarray:
        path = Path(
            image_path,
        )

        image = self._read_image(
            path,
        )

        return self.extract_from_array(
            image,
        )