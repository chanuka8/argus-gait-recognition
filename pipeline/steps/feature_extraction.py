from pathlib import Path

import cv2
import numpy as np
import torch

from models.architectures.bygait_light import ByGaitLight
from models.inference.backend import get_inference_backend, load_inference_backend_config


class FeatureExtractionStep:
    def __init__(
        self,
        model_path: str | None = None,
        image_size: tuple[int, int] = (64, 128),
        binary_threshold: int = 20,
        backend_config: dict | None = None,
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
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

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
            raise ValueError(f"Unable to read image: {image_path}")

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

    def extract_from_gei(
        self,
        gei: np.ndarray,
    ) -> np.ndarray:
        """Extract a 256-d L2-normalized embedding directly from an in-memory GEI array."""
        if gei is None or gei.size == 0:
            return np.empty((0, 256), dtype=np.float32)

        if gei.ndim == 3 and gei.shape[2] == 3:
            gei = cv2.cvtColor(gei, cv2.COLOR_BGR2GRAY)

        h, w = gei.shape[:2]
        if (w, h) != self.image_size:
            gei = cv2.resize(gei, self.image_size)

        if gei.dtype == np.uint8 or gei.max() > 1.0:
            norm_gei = gei.astype(np.float32) / 255.0
        else:
            norm_gei = gei.astype(np.float32)

        embedding = self.backend.predict(norm_gei).flatten()
        return embedding.astype(np.float32)
