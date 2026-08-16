from pathlib import Path

import cv2
import numpy as np
import torch

from models.architectures.bygait_light import ByGaitLight


from models.inference.backend import get_inference_backend, load_inference_backend_config


class FeatureExtractionStep:
    def __init__(
        self,
        model_path: str = None,
        image_size: tuple[int, int] = (64, 128),
        binary_threshold: int = 20,
        backend_config: dict = None,
    ) -> None:
        self.backend_config = backend_config or load_inference_backend_config()
        resolved_path = model_path or self.backend_config.get("model_path") or "runs/exp_001/best_model.pth"
        self.model_path = Path(resolved_path)
        self.image_size = image_size
        self.binary_threshold = binary_threshold
        self.backend = get_inference_backend(
            config=self.backend_config,
            model_path=str(self.model_path) if self.model_path.exists() else None,
        )
        self.model = getattr(self.backend, "model", None)

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

        embedding = self.backend.predict(image).flatten()
        return embedding.astype(
            np.float32,
        )
