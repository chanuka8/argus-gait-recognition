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
        self.device_str = str(self.config.get("device", "auto")).lower()
        self.precision = str(self.config.get("precision", "fp32")).lower()
        self.allow_fallback = bool(self.config.get("allow_fallback", True))
        self.warmup_iterations = int(self.config.get("warmup_iterations", 3))

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

    # PyTorch default reference backend
    from models.inference.pytorch_backend import PyTorchBackend

    if requested_backend == "pytorch":
        return PyTorchBackend(config=cfg, model_path=model_path)

    if requested_backend == "onnxruntime":
        try:
            from models.inference.onnx_backend import ONNXBackend
            backend = ONNXBackend(config=cfg, model_path=model_path)
            if backend.is_available():
                return backend
        except Exception as e:
            if not allow_fallback:
                raise e
            logger.warning(f"ONNX backend unavailable ({e}). Falling back to PyTorch.")

        return PyTorchBackend(config=cfg, model_path=model_path)

    if requested_backend == "tensorrt":
        try:
            from models.inference.tensorrt_backend import TensorRTBackend
            backend = TensorRTBackend(config=cfg, model_path=model_path)
            if backend.is_available():
                return backend
        except Exception as e:
            if not allow_fallback:
                raise e
            logger.warning(f"TensorRT backend unavailable ({e}). Falling back to PyTorch.")

        return PyTorchBackend(config=cfg, model_path=model_path)

    if requested_backend == "auto":
        # Try TensorRT, then ONNX, then PyTorch
        try:
            from models.inference.tensorrt_backend import TensorRTBackend
            backend = TensorRTBackend(config=cfg, model_path=model_path)
            if backend.is_available():
                return backend
        except Exception:
            pass

        try:
            from models.inference.onnx_backend import ONNXBackend
            backend = ONNXBackend(config=cfg, model_path=model_path)
            if backend.is_available():
                return backend
        except Exception:
            pass

        return PyTorchBackend(config=cfg, model_path=model_path)

    if allow_fallback:
        logger.warning(f"Invalid backend '{requested_backend}'. Falling back to PyTorch.")
        return PyTorchBackend(config=cfg, model_path=model_path)

    raise ValueError(f"Unsupported inference backend: {requested_backend}")
