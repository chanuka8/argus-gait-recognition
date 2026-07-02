from pathlib import Path

import cv2
import numpy as np
import torch

from models.architectures.bygait_light import ByGaitLight


class FeatureExtractionStep:
    def __init__(
        self,
        model_path: str = "runs/exp_001/best_model.pth",
        image_size: tuple[int, int] = (64, 128),
        binary_threshold: int = 20,
    ) -> None:
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.binary_threshold = binary_threshold
        self.model = self._load_model()

    def _load_model(
        self,
    ) -> ByGaitLight:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        model = ByGaitLight()

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
        )

        filtered = {}

        for key, value in checkpoint.items():
            if key.startswith("backbone."):
                filtered[key.replace("backbone.", "")] = value

        model.load_state_dict(
            filtered,
            strict=True,
        )

        model.eval()

        return model

    def _read_grayscale(
        self,
        image_path: Path,
    ) -> np.ndarray:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image

    def _normalize_to_silhouette(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        image = cv2.resize(
            image,
            self.image_size,
        )

        return image.astype(np.float32) / 255.0

    def _load_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        image = self._read_grayscale(
            image_path,
        )

        return self._normalize_to_silhouette(
            image,
        )

    def extract(
        self,
        image_path,
    ) -> np.ndarray:
        path = Path(
            image_path,
        )

        image = self._load_image(
            path,
        )

        tensor = torch.from_numpy(
            image,
        ).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(
                tensor,
            ).cpu().numpy().flatten()

        norm = np.linalg.norm(
            embedding,
        )

        embedding = embedding / (
            norm + 1e-8
        )

        return embedding.astype(
            np.float32,
        )