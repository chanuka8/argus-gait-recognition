"""
TensorRT Inference Backend for ARGUS AI.

Executes TensorRT engines using lazy optional imports (tensorrt)
and transparent fallback to PyTorch backend.
"""

from pathlib import Path

import numpy as np
import torch

from models.inference.backend import BaseInferenceBackend


class TensorRTBackend(BaseInferenceBackend):
    """TensorRT execution engine with lazy import and safe fallback."""

    def __init__(
        self,
        config: dict | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self.backend_name = "tensorrt"
        self.engine_path = Path(self.config.get("engine_path", "models/engines/bygait_light_fp16.engine"))
        self.engine = None
        self.context = None
        self._fallback_backend = None
        self._initialized = False
        self._init_engine(model_path=model_path)

    def _init_engine(self, model_path: str | None = None) -> None:
        """Initialize TensorRT engine lazily."""
        try:
            import tensorrt as trt

            if not self.engine_path.exists():
                raise FileNotFoundError(f"TensorRT engine file not found: {self.engine_path}")

            logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                if self.engine is not None:
                    self.context = self.engine.create_execution_context()
                    self.execution_provider = "TensorRT-CUDA"
                    self._initialized = True
                    self._fallback_used = False
                    self._fallback_reason = None
                    self.warmup()
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            self._initialized = False
            reason = str(e)
            self.fallback_reason = reason
            if self.allow_fallback:
                self.fallback_used = True
                from models.inference.pytorch_backend import PyTorchBackend

                self._fallback_backend = PyTorchBackend(config=self.config, model_path=model_path)
                self._fallback_backend.requested_backend = self.requested_backend
                self._fallback_backend.fallback_used = True
                self._fallback_backend.fallback_reason = reason

    def is_available(self) -> bool:
        """Check if TensorRT engine initialized successfully."""
        return self._initialized and self.context is not None

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Execute TensorRT inference and return L2-normalized numpy embedding array.

        Args:
            x: Input array or tensor of shape (B, 1, 128, 64) or (128, 64) or (1, 128, 64).

        Returns:
            L2-normalized float32 numpy array of shape (B, 256).
        """
        if not self.is_available():
            if self._fallback_backend is not None:
                return self._fallback_backend.predict(x)
            raise RuntimeError("TensorRT backend is not available and fallback is disabled.")

        # Fallback to backend execution if context execution fails at runtime
        try:
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy().astype(np.float32)
            else:
                arr = x.astype(np.float32)

            if arr.ndim == 2:
                arr = np.expand_dims(np.expand_dims(arr, 0), 0)
            elif arr.ndim == 3:
                arr = np.expand_dims(arr, 0)

            # TensorRT binding execution logic using PyTorch CUDA tensors if available
            gpu_input = torch.from_numpy(arr).cuda()
            gpu_output = torch.empty((arr.shape[0], 256), dtype=torch.float32, device="cuda")

            bindings = [int(gpu_input.data_ptr()), int(gpu_output.data_ptr())]
            self.context.execute_v2(bindings)

            embeddings = gpu_output.cpu().numpy()
            return self.normalize_embedding(embeddings)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            if self.allow_fallback:
                self.logger.warning(f"TensorRT execution failed ({e}). Falling back to PyTorch.")
                if self._fallback_backend is None:
                    from models.inference.pytorch_backend import PyTorchBackend

                    self._fallback_backend = PyTorchBackend(config=self.config)
                return self._fallback_backend.predict(x)
            raise
