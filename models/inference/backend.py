"""
Inference Backend Base Interface and Factory for ARGUS AI.

Defines the abstract BaseInferenceBackend interface and factory get_inference_backend()
to instantiate execution engines (pytorch, onnxruntime, tensorrt, auto) with safe fallback logic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml

from monitoring.logging_config import get_logger


def load_inference_backend_config() -> dict:
    """Load inference_backend configuration section from configs/inference.yaml."""
    config_path = Path("configs/inference.yaml")
    defaults = {
        "backend": "pytorch",
        "device": "auto",
        "precision": "fp32",
        "engine_path": "models/engines/bygait_light_fp16.engine",
        "onnx_path": "models/engines/bygait_light.onnx",
        "allow_fallback": True,
        "warmup_iterations": 3,
        "dynamic_batch": False,
        "max_batch_size": 1,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return defaults

    section = data.get("inference_backend", {})
    if not isinstance(section, dict):
        return defaults

    merged = dict(defaults)
    for key, val in defaults.items():
        if key in section:
            merged[key] = section[key]

    return merged


class BaseInferenceBackend(ABC):
    """Abstract Base Class for ARGUS model inference execution engines."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_inference_backend_config()
        self.logger = get_logger("detection")
        self.backend_name = str(self.config.get("backend", "pytorch")).lower()
        self._requested_backend = self.backend_name
        self.device_str = str(self.config.get("device", "auto")).lower()
        self.precision = str(self.config.get("precision", "fp32")).lower()
        self.allow_fallback = bool(self.config.get("allow_fallback", True))
        self.warmup_iterations = int(self.config.get("warmup_iterations", 3))
        self._fallback_used = False
        self._selection_fallback_used = False
        self._attempted_backends = [self.backend_name]
        self._fallback_reason: Optional[str] = None
        self._fallback_backend: Optional["BaseInferenceBackend"] = None
        self._execution_provider = "PyTorch-CPU"

    @property
    def requested_backend(self) -> str:
        """The backend requested by caller or configuration."""
        return self._requested_backend

    @requested_backend.setter
    def requested_backend(self, value: str) -> None:
        self._requested_backend = str(value).lower()

    @property
    def active_backend(self) -> str:
        """The actual execution engine currently active for inference."""
        if self.fallback_used and self._fallback_backend is not None:
            return self._fallback_backend.active_backend
        return self.backend_name

    @property
    def fallback_used(self) -> bool:
        """Boolean flag indicating if active backend is a PyTorch fallback."""
        return self._fallback_used

    @fallback_used.setter
    def fallback_used(self, value: bool) -> None:
        self._fallback_used = bool(value)

    @property
    def selection_fallback_used(self) -> bool:
        """Boolean flag indicating if backend selection chain involved a fallback."""
        return self._selection_fallback_used

    @selection_fallback_used.setter
    def selection_fallback_used(self, value: bool) -> None:
        self._selection_fallback_used = bool(value)

    @property
    def attempted_backends(self) -> list:
        """List of backends attempted during selection in order."""
        return self._attempted_backends

    @attempted_backends.setter
    def attempted_backends(self, value: list) -> None:
        self._attempted_backends = list(value)

    @property
    def fallback_reason(self) -> Optional[str]:
        """Concise sanitized reason explaining why fallback occurred, or None."""
        if self._fallback_used or self._selection_fallback_used:
            return self._fallback_reason
        return None

    @fallback_reason.setter
    def fallback_reason(self, value: Optional[str]) -> None:
        self._fallback_reason = value

    @property
    def execution_provider(self) -> str:
        """The active low-level execution provider (e.g. CPUExecutionProvider, PyTorch-CPU)."""
        if self.fallback_used and self._fallback_backend is not None:
            return self._fallback_backend.execution_provider
        return self._execution_provider

    @execution_provider.setter
    def execution_provider(self, value: str) -> None:
        self._execution_provider = str(value)

    @abstractmethod
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Execute forward inference on GEI input tensor/array and return L2-normalized embedding.

        Args:
            x: Input array or tensor of shape (B, 1, 64, 128) or (64, 128) or (1, 64, 128).

        Returns:
            L2-normalized float32 numpy array of shape (B, 256).
        """
        pass

    def warmup(self, sample_input: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """Run warmup iterations on dummy or provided input tensor."""
        if self.warmup_iterations <= 0:
            return

        if sample_input is None:
            sample_input = np.zeros((1, 1, 64, 128), dtype=np.float32)

        for _ in range(self.warmup_iterations):
            try:
                self.predict(sample_input)
            except Exception:
                break

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """Apply L2 normalization on output embedding array."""
        if embedding.ndim == 1:
            norm = np.linalg.norm(embedding)
            return (embedding / (norm + 1e-8)).astype(np.float32)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return (embedding / (norm + 1e-8)).astype(np.float32)


def get_inference_backend(
    config: Optional[dict] = None,
    model_path: Optional[str] = None,
) -> BaseInferenceBackend:
    """
    Factory function to instantiate configured inference engine backend with safe fallback.

    Args:
        config: Dictionary containing inference_backend settings.
        model_path: Optional path to ByGaitLight PyTorch model checkpoint.

    Returns:
        Initialized BaseInferenceBackend instance.
    """
    cfg = config or load_inference_backend_config()
    requested_backend = str(cfg.get("backend", "pytorch")).lower()
    allow_fallback = bool(cfg.get("allow_fallback", True))
    logger = get_logger("detection")

    from models.inference.pytorch_backend import PyTorchBackend

    if requested_backend == "pytorch":
        backend = PyTorchBackend(config=cfg, model_path=model_path)
        backend.requested_backend = "pytorch"
        backend.attempted_backends = ["pytorch"]
        backend.selection_fallback_used = False
        return backend

    if requested_backend == "onnxruntime":
        reason = None
        try:
            from models.inference.onnx_backend import ONNXBackend

            backend = ONNXBackend(config=cfg, model_path=model_path)
            backend.requested_backend = "onnxruntime"
            backend.attempted_backends = ["onnxruntime"]
            if backend.is_available():
                backend.selection_fallback_used = False
                return backend
            reason = backend.fallback_reason or f"ONNX model file not found at {backend.onnx_path}"
        except Exception as e:
            reason = str(e)
            logger.warning(f"ONNX backend unavailable ({reason}). Falling back to PyTorch.")

        if not allow_fallback:
            raise RuntimeError(f"ONNX backend unavailable and fallback disabled: {reason}")

        fb = PyTorchBackend(config=cfg, model_path=model_path)
        fb.requested_backend = "onnxruntime"
        fb.attempted_backends = ["onnxruntime", "pytorch"]
        fb.fallback_used = True
        fb.selection_fallback_used = True
        fb.fallback_reason = reason
        return fb

    if requested_backend == "tensorrt":
        reason = None
        try:
            from models.inference.tensorrt_backend import TensorRTBackend

            backend = TensorRTBackend(config=cfg, model_path=model_path)
            backend.requested_backend = "tensorrt"
            backend.attempted_backends = ["tensorrt"]
            if backend.is_available():
                backend.selection_fallback_used = False
                return backend
            reason = backend.fallback_reason or f"TensorRT engine not available at {backend.engine_path}"
        except Exception as e:
            reason = str(e)

        if not allow_fallback:
            raise RuntimeError(f"TensorRT backend unavailable and fallback disabled: {reason}")

        logger.warning(f"TensorRT backend unavailable ({reason}). Falling back to PyTorch.")
        fb = PyTorchBackend(config=cfg, model_path=model_path)
        fb.requested_backend = "tensorrt"
        fb.attempted_backends = ["tensorrt", "pytorch"]
        fb.fallback_used = True
        fb.selection_fallback_used = True
        fb.fallback_reason = reason
        return fb

    if requested_backend == "auto":
        attempted = []
        reasons = []

        # 1. Try TensorRT
        attempted.append("tensorrt")
        try:
            from models.inference.tensorrt_backend import TensorRTBackend

            backend = TensorRTBackend(config=cfg, model_path=model_path)
            backend.requested_backend = "auto"
            backend.attempted_backends = list(attempted)
            if backend.is_available():
                backend.selection_fallback_used = False
                return backend
            if backend.fallback_reason:
                reasons.append(f"TensorRT unavailable ({backend.fallback_reason})")
        except Exception as e:
            reasons.append(f"TensorRT unavailable ({e})")

        # 2. Try ONNX Runtime
        attempted.append("onnxruntime")
        try:
            from models.inference.onnx_backend import ONNXBackend

            backend = ONNXBackend(config=cfg, model_path=model_path)
            backend.requested_backend = "auto"
            backend.attempted_backends = list(attempted)
            if backend.is_available():
                backend.selection_fallback_used = True
                backend.fallback_reason = "; ".join(reasons) + "; ONNX Runtime selected"
                return backend
            if backend.fallback_reason:
                reasons.append(f"ONNX unavailable ({backend.fallback_reason})")
        except Exception as e:
            reasons.append(f"ONNX unavailable ({e})")

        # 3. Fallback to PyTorch
        attempted.append("pytorch")
        if not allow_fallback:
            raise RuntimeError(f"Auto backend failed to find available accelerated engine: {'; '.join(reasons)}")

        combined_reason = "; ".join(reasons) if reasons else "Accelerated backends unavailable"
        fb = PyTorchBackend(config=cfg, model_path=model_path)
        fb.requested_backend = "auto"
        fb.attempted_backends = list(attempted)
        fb.fallback_used = True
        fb.selection_fallback_used = True
        fb.fallback_reason = combined_reason
        return fb

    if allow_fallback:
        logger.warning(f"Invalid backend '{requested_backend}'. Falling back to PyTorch.")
        fb = PyTorchBackend(config=cfg, model_path=model_path)
        fb.requested_backend = requested_backend
        fb.attempted_backends = [requested_backend, "pytorch"]
        fb.fallback_used = True
        fb.selection_fallback_used = True
        fb.fallback_reason = f"Unsupported backend '{requested_backend}'"
        return fb

    raise ValueError(f"Unsupported inference backend: {requested_backend}")
