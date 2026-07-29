"""
ONNX Runtime Inference Backend for ARGUS AI.

Executes ONNX models using onnxruntime with lazy optional imports
and transparent fallback to PyTorch backend.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from models.inference.backend import BaseInferenceBackend


class ONNXBackend(BaseInferenceBackend):
    """ONNX Runtime execution engine with lazy import and safe fallback."""

    def __init__(
        self,
        config: Optional[dict] = None,
        model_path: Optional[str] = None,
    ) -> None:
        super().__init__(config=config)
        self.onnx_path = Path(self.config.get("onnx_path", "models/engines/bygait_light.onnx"))
        self.session = None
        self.input_name = None
        self._fallback_backend = None
        self._initialized = False
        self._init_session(model_path=model_path)

    def _init_session(self, model_path: Optional[str] = None) -> None:
        """Initialize ONNX Runtime inference session lazily."""
        try:
            import onnxruntime as ort

            if not self.onnx_path.exists():
                raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device_str == "cpu":
                providers = ["CPUExecutionProvider"]

            self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._initialized = True
            self.warmup()
        except Exception as e:
            self._initialized = False
            if self.allow_fallback:
                self.logger.warning(f"ONNX initialization failed ({e}). Falling back to PyTorch.")
                from models.inference.pytorch_backend import PyTorchBackend

                self._fallback_backend = PyTorchBackend(config=self.config, model_path=model_path)

    def is_available(self) -> bool:
        """Check if ONNX session initialized successfully."""
        return self._initialized and self.session is not None

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Execute ONNX inference and return L2-normalized numpy embedding array.

        Args:
            x: Input array or tensor of shape (B, 1, 64, 128) or (64, 128) or (1, 64, 128).

        Returns:
            L2-normalized float32 numpy array of shape (B, 256).
        """
        if not self.is_available():
            if self._fallback_backend is not None:
                return self._fallback_backend.predict(x)
            raise RuntimeError("ONNX backend is not available and fallback is disabled.")

        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy().astype(np.float32)
        else:
            arr = x.astype(np.float32)

        if arr.ndim == 2:
            arr = np.expand_dims(np.expand_dims(arr, 0), 0)
        elif arr.ndim == 3:
            arr = np.expand_dims(arr, 0)

        outputs = self.session.run(None, {self.input_name: arr})
        embeddings = outputs[0]
        return self.normalize_embedding(embeddings)
