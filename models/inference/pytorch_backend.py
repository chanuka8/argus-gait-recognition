"""
PyTorch Reference Inference Backend for ARGUS AI.

Wraps ByGaitLight CNN model for reference CPU/GPU execution.
Guarantees L2 normalization semantics and input tensor shape handling.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from models.architectures.bygait_light import ByGaitLight
from models.inference.backend import BaseInferenceBackend


class PyTorchBackend(BaseInferenceBackend):
    """PyTorch reference backend implementation."""

    def __init__(
        self,
        config: Optional[dict] = None,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(config=config)
        self.model_path = Path(model_path or "runs/exp_001/best_model.pth")
        self.device = self._resolve_device(self.device_str)
        self.model = self._load_model()
        self.warmup()

    def _resolve_device(self, device_str: str) -> torch.device:
        """Resolve target PyTorch execution device."""
        if device_str in ("cuda", "gpu") and torch.cuda.is_available():
            return torch.device("cuda")
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def _load_model(self) -> ByGaitLight:
        """Instantiate ByGaitLight model and load weights if checkpoint exists."""
        model = ByGaitLight()
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location="cpu")
                filtered = {}
                for key, value in checkpoint.items():
                    if key.startswith("backbone."):
                        filtered[key.replace("backbone.", "")] = value
                    elif key in model.state_dict():
                        filtered[key] = value

                model.load_state_dict(filtered, strict=False)
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint from {self.model_path}: {e}")

        model.to(self.device)
        model.eval()

        if self.precision == "fp16" and self.device.type == "cuda":
            model.half()

        return model

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Execute PyTorch forward inference and return L2-normalized numpy embedding array.

        Args:
            x: Input array or tensor of shape (B, 1, 64, 128) or (64, 128) or (1, 64, 128).

        Returns:
            L2-normalized float32 numpy array of shape (B, 256).
        """
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).float()
        else:
            tensor = x.float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            output = self.model(tensor)
            if self.precision == "fp16":
                output = output.float()

            embeddings = output.cpu().numpy()

        return self.normalize_embedding(embeddings)
