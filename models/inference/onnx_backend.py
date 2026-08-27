"""
ONNX Runtime Inference Backend for ARGUS AI.

Executes ONNX models using onnxruntime with lazy optional imports
and transparent fallback to PyTorch backend.
"""

from pathlib import Path

import numpy as np
import torch

from models.inference.backend import BaseInferenceBackend


class ONNXBackend(BaseInferenceBackend):
    """ONNX Runtime execution engine with lazy import and safe fallback."""

    def __init__(
        self,
        config: dict | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self.backend_name = "onnxruntime"
        self.onnx_path = Path(self.config.get("onnx_path", "models/engines/bygait_light.onnx"))
        self.session = None
        self.input_name = None
        self._fallback_backend = None
        self._initialized = False
        self._init_session(model_path=model_path)

    def _init_session(self, model_path: str | None = None) -> None:
        """Initialize ONNX Runtime inference session lazily."""
        try:
            from automation.device_manager import DeviceManager
            from automation.dll_manager import setup_cuda_dll_paths

            setup_cuda_dll_paths()

            import onnxruntime as ort

            if not self.onnx_path.exists():
                raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

            dm = DeviceManager.get_instance()
            target_device = dm.resolve_component_device(self.device_str)

            available_providers = ort.get_available_providers()
            providers = []
            if "cuda" in target_device and "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            ort_state = getattr(ort.capi, "onnxruntime_pybind11_state", None)
            ort_exceptions = (
                (
                    RuntimeError,
                    OSError,
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    ort_state.Fail,
                    ort_state.InvalidProtobuf,
                    ort_state.InvalidGraph,
                    ort_state.EPFail,
                    ort_state.EngineError,
                    ort_state.InvalidArgument,
                    ort_state.NoSuchFile,
                    ort_state.NoModel,
                    ort_state.RuntimeException,
                )
                if ort_state is not None
                else (RuntimeError, OSError, ValueError, TypeError, KeyError, AttributeError)
            )

            try:
                self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
            except ort_exceptions as sess_err:
                raise RuntimeError(f"Failed to initialize ONNX session: {sess_err}") from sess_err
            self.input_name = self.session.get_inputs()[0].name
            active_providers = self.session.get_providers() if hasattr(self.session, "get_providers") else providers
            self.execution_provider = active_providers[0] if active_providers else "CPUExecutionProvider"
            self._initialized = True
            self._fallback_used = False
            self._fallback_reason = None
            self.warmup()
        except (ImportError, RuntimeError, OSError, ValueError, KeyError) as e:
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
        """Check if ONNX session initialized successfully."""
        return self._initialized and self.session is not None

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Execute ONNX inference and return L2-normalized numpy embedding array.

        Args:
            x: Input array or tensor of shape (B, 1, 128, 64) or (128, 64) or (1, 128, 64).

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
