import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import yaml

from monitoring.logging_config import get_logger


class BackendStatus(str, Enum):
    READY = "READY"
    AVAILABLE = "AVAILABLE"
    NOT_INSTALLED = "NOT_INSTALLED"
    MODEL_MISSING = "MODEL_MISSING"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    UNSUPPORTED = "UNSUPPORTED"
    FALLBACK_ACTIVE = "FALLBACK_ACTIVE"


@dataclass
class BackendCapability:
    backend_name: str
    supported_devices: list = field(default_factory=list)
    supported_precisions: list = field(default_factory=list)
    dynamic_batching: bool = False
    max_batch_size: int = 1


@dataclass
class BackendHealth:
    backend: str
    status: BackendStatus
    is_available: bool
    execution_provider: str
    error_message: str | None = None


def load_inference_backend_config() -> dict:
    config_path = Path("configs/inference.yaml")
    defaults = {
        "backend": "pytorch",
        "device": "auto",
        "precision": "fp32",
        "model_path": "runs/exp_001/best_model.pth",
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
    except (yaml.YAMLError, OSError, ValueError, KeyError):
        return defaults

    section = data.get("inference_backend", {})
    if not isinstance(section, dict):
        return defaults

    merged = dict(defaults)
    for key in defaults:
        if key in section:
            merged[key] = section[key]

    return merged


class BaseInferenceBackend(ABC):
    def __init__(self, config: dict | None = None) -> None:
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
        self._fallback_reason: str | None = None
        self._fallback_backend: BaseInferenceBackend | None = None
        self._execution_provider = "PyTorch-CPU"

    @property
    def requested_backend(self) -> str:
        return self._requested_backend

    @requested_backend.setter
    def requested_backend(self, value: str) -> None:
        self._requested_backend = str(value).lower()

    @property
    def active_backend(self) -> str:
        if self.fallback_used and self._fallback_backend is not None:
            return self._fallback_backend.active_backend
        return self.backend_name

    @property
    def fallback_used(self) -> bool:
        return self._fallback_used

    @fallback_used.setter
    def fallback_used(self, value: bool) -> None:
        self._fallback_used = bool(value)

    @property
    def selection_fallback_used(self) -> bool:
        return self._selection_fallback_used

    @selection_fallback_used.setter
    def selection_fallback_used(self, value: bool) -> None:
        self._selection_fallback_used = bool(value)

    @property
    def attempted_backends(self) -> list:
        return self._attempted_backends

    @attempted_backends.setter
    def attempted_backends(self, value: list) -> None:
        self._attempted_backends = list(value)

    @property
    def fallback_reason(self) -> str | None:
        if self._fallback_used or self._selection_fallback_used:
            return self._fallback_reason
        return None

    @fallback_reason.setter
    def fallback_reason(self, value: str | None) -> None:
        self._fallback_reason = value

    @property
    def execution_provider(self) -> str:
        if self.fallback_used and self._fallback_backend is not None:
            return self._fallback_backend.execution_provider
        return self._execution_provider

    @execution_provider.setter
    def execution_provider(self, value: str) -> None:
        self._execution_provider = str(value)

    @property
    def metadata(self) -> dict:
        return {
            "requested_backend": self.requested_backend,
            "active_backend": self.active_backend,
            "execution_provider": self.execution_provider,
            "allow_fallback": self.allow_fallback,
            "fallback_used": bool(self.fallback_used or self.selection_fallback_used),
            "fallback_reason": self.fallback_reason,
            "attempted_backends": self.attempted_backends,
        }

    @abstractmethod
    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        pass

    def warmup(self, sample_input: np.ndarray | torch.Tensor | None = None) -> None:
        if self.warmup_iterations <= 0:
            return

        if sample_input is None:
            sample_input = np.zeros((1, 1, 128, 64), dtype=np.float32)

        for _ in range(self.warmup_iterations):
            try:
                self.predict(sample_input)
            except (RuntimeError, ValueError, TypeError, OSError):
                break

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        if embedding.ndim == 1:
            norm = np.linalg.norm(embedding)
            return (embedding / (norm + 1e-8)).astype(np.float32)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return (embedding / (norm + 1e-8)).astype(np.float32)


class BackendValidator:
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_inference_backend_config()

    def check_pytorch(self) -> BackendHealth:
        try:
            import torch

            model_path = Path(self.config.get("model_path") or "runs/exp_001/best_model.pth")
            status = BackendStatus.READY if model_path.exists() else BackendStatus.AVAILABLE
            return BackendHealth(
                backend="pytorch",
                status=status,
                is_available=True,
                execution_provider="PyTorch-CPU" if not torch.cuda.is_available() else "PyTorch-CUDA",
            )
        except ImportError as e:
            return BackendHealth(
                backend="pytorch",
                status=BackendStatus.NOT_INSTALLED,
                is_available=False,
                execution_provider="None",
                error_message=str(e),
            )

    def check_onnxruntime(self) -> BackendHealth:
        try:
            import onnxruntime as ort

            onnx_path = Path(self.config.get("onnx_path", "models/engines/bygait_light.onnx"))
            if not onnx_path.exists():
                return BackendHealth(
                    backend="onnxruntime",
                    status=BackendStatus.MODEL_MISSING,
                    is_available=False,
                    execution_provider="None",
                    error_message=f"ONNX model file not found at {onnx_path}",
                )
            providers = ort.get_available_providers()
            provider = "CPUExecutionProvider"
            if "CUDAExecutionProvider" in providers:
                provider = "CUDAExecutionProvider"
            return BackendHealth(
                backend="onnxruntime",
                status=BackendStatus.READY,
                is_available=True,
                execution_provider=provider,
            )
        except ImportError as e:
            return BackendHealth(
                backend="onnxruntime",
                status=BackendStatus.NOT_INSTALLED,
                is_available=False,
                execution_provider="None",
                error_message=str(e),
            )

    def run_smoke_test(self, backend: BaseInferenceBackend) -> bool:
        dummy = np.zeros((1, 1, 128, 64), dtype=np.float32)
        try:
            out = backend.predict(dummy)
            return bool(out is not None and getattr(out, "shape", None) == (1, 256))
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
            return False


class BackendReport:
    def __init__(
        self,
        backend: BaseInferenceBackend,
        output_path: str = "outputs/reports/backend_report.json",
    ) -> None:
        self.backend = backend
        self.output_path = Path(output_path)

    def generate(self) -> dict:
        validator = BackendValidator(self.backend.config)
        smoke_test = validator.run_smoke_test(self.backend)

        if self.backend.active_backend == "onnxruntime":
            m_path = str(self.backend.config.get("onnx_path", "models/engines/bygait_light.onnx"))
        else:
            m_path = str(self.backend.config.get("model_path", "runs/exp_001/best_model.pth"))

        report_data = {
            "requested_backend": self.backend.requested_backend,
            "active_backend": self.backend.active_backend,
            "execution_provider": self.backend.execution_provider,
            "fallback_used": bool(self.backend.fallback_used or self.backend.selection_fallback_used),
            "fallback_reason": self.backend.fallback_reason,
            "model_path": m_path,
            "initialization_result": "SUCCESS" if self.backend is not None else "FAILED",
            "inference_smoke_test_result": "PASSED" if smoke_test else "FAILED",
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)

        return report_data


def generate_backend_report(
    backend: BaseInferenceBackend,
    output_path: str = "outputs/reports/backend_report.json",
) -> dict:
    reporter = BackendReport(backend=backend, output_path=output_path)
    return reporter.generate()


def get_inference_backend(
    config: dict | None = None,
    model_path: str | None = None,
) -> BaseInferenceBackend:
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
        except (ImportError, RuntimeError, OSError, ValueError) as e:
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
        except (ImportError, RuntimeError, OSError, ValueError) as e:
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
        attempted = ["onnxruntime"]
        reasons = []

        try:
            from models.inference.onnx_backend import ONNXBackend

            backend = ONNXBackend(config=cfg, model_path=model_path)
            backend.requested_backend = "auto"
            backend.attempted_backends = list(attempted)
            if backend.is_available():
                backend.selection_fallback_used = False
                return backend
            if backend.fallback_reason:
                reasons.append(f"ONNX unavailable ({backend.fallback_reason})")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            reasons.append(f"ONNX unavailable ({e})")


        attempted.append("pytorch")
        if not allow_fallback:
            raise RuntimeError(f"Auto backend failed to find available accelerated engine: {'; '.join(reasons)}")

        combined_reason = "; ".join(reasons) if reasons else "ONNX Runtime backend unavailable"
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
