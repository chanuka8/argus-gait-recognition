"""
Centralized Authoritative Runtime Device Manager for ARGUS AI.

Acts as the single source of truth across the entire system for compute device
and backend decisions. Eliminates ad-hoc 'cuda' vs 'cpu' branching throughout the
codebase.

Architecture:
    HardwareDetector → EnvironmentValidator → DeviceManager → [YOLO / ONNX / ByGaitLight / PyTorch]
"""

import threading
from dataclasses import dataclass
from typing import Any, Optional

from automation.environment_validator import (
    EnvironmentState,
    EnvironmentValidationReport,
    EnvironmentValidator,
)


@dataclass
class DeviceInfo:
    backend: str
    device: str
    cuda_available: bool
    gpu_name: str | None
    vram_mb: float
    pytorch_version: str | None
    cuda_version: str | None
    onnx_provider: str
    status: str


class DeviceManager:
    """
    Thread-safe Singleton Device Manager for ARGUS AI runtime inference.
    """

    _instance: Optional["DeviceManager"] = None
    _lock = threading.RLock()

    def __init__(self, force_refresh: bool = False, force_cpu: bool = False) -> None:
        DeviceManager._instance = self
        self._validator = EnvironmentValidator()
        self._report: EnvironmentValidationReport | None = None
        self._force_cpu = force_cpu
        self._refresh(force_cpu=force_cpu)

    def _refresh(self, force_cpu: bool = False) -> None:
        """Run authoritative environment evaluation and bind device state."""
        self._force_cpu = force_cpu
        self._report = self._validator.validate(force_cpu=force_cpu)

    @classmethod
    def get_instance(cls, force_refresh: bool = False, force_cpu: bool | None = None) -> "DeviceManager":
        """Retrieve or create the singleton DeviceManager instance."""
        with cls._lock:
            target_force_cpu = (
                force_cpu
                if force_cpu is not None
                else (cls._instance._force_cpu if cls._instance is not None else False)
            )
            if (
                cls._instance is None
                or force_refresh
                or (force_cpu is not None and force_cpu != cls._instance._force_cpu)
            ):
                cls._instance = cls(force_refresh=force_refresh, force_cpu=target_force_cpu)
            return cls._instance

    @property
    def validation_report(self) -> EnvironmentValidationReport:
        if self._report is None:
            self._refresh(force_cpu=self._force_cpu)
        assert self._report is not None
        return self._report

    @property
    def backend(self) -> str:
        """Authoritative backend: 'cuda' | 'cpu'."""
        if self.validation_report.state == EnvironmentState.CUDA_READY:
            return "cuda"
        return "cpu"

    @property
    def device(self) -> str:
        """Authoritative runtime device string: 'cuda:0' | 'cpu'."""
        if self.validation_report.state == EnvironmentState.CUDA_READY:
            return "cuda:0"
        return "cpu"

    @property
    def torch_device(self) -> Any:
        """Authoritative PyTorch device object."""
        try:
            import torch

            return torch.device(self.device)
        except (ImportError, RuntimeError, ValueError):
            return None

    @property
    def is_cuda(self) -> bool:
        """Boolean indicating whether CUDA is active and fully verified."""
        return self.backend == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Boolean indicating whether CPU fallback is active."""
        return self.backend == "cpu"

    @property
    def cuda_available(self) -> bool:
        """Boolean indicating whether CUDA hardware and runtime are available."""
        return self.is_cuda

    @property
    def gpu_name(self) -> str | None:
        """Name of the active GPU or None in CPU mode."""
        if self.validation_report.hardware.gpu.present:
            return self.validation_report.hardware.gpu.gpu_name
        return None

    @property
    def vram_mb(self) -> float:
        """GPU VRAM total in MB or 0.0 in CPU mode."""
        if self.validation_report.hardware.gpu.present:
            return self.validation_report.hardware.gpu.vram_mb
        return 0.0

    @property
    def pytorch_version(self) -> str | None:
        """Installed PyTorch version string."""
        try:
            import torch

            return getattr(torch, "__version__", None)
        except (ImportError, AttributeError):
            return None

    @property
    def cuda_version(self) -> str | None:
        """PyTorch CUDA build version or None."""
        try:
            import torch

            return getattr(torch.version, "cuda", None)
        except (ImportError, AttributeError):
            return None

    @property
    def onnx_provider(self) -> str:
        """Active ONNX Runtime execution provider."""
        if self.is_cuda and self.validation_report.cuda_report:
            return self.validation_report.cuda_report.onnx_selected_provider
        return "CPUExecutionProvider"

    @property
    def status(self) -> EnvironmentState:
        """Current EnvironmentState enum value."""
        return self.validation_report.state

    def resolve_component_device(self, requested: str | None = None) -> str:
        """
        Arbitrate requested device strings ('auto', 'cuda', 'gpu', 'cpu') against authoritative state.
        Never yields 'cuda' if the system is not CUDA_READY.
        """
        if not requested or requested.lower() in ("auto", "default"):
            return self.device

        req_clean = requested.lower().strip()
        if req_clean in ("cuda", "cuda:0", "gpu"):
            if self.is_cuda:
                return "cuda:0"
            return "cpu"

        return "cpu"

    def summary(self) -> dict[str, Any]:
        """Structured telemetry dictionary for logs, API responses, and manifest generation."""
        return {
            "backend": self.backend,
            "device": self.device,
            "gpu": self.gpu_name,
            "gpu_name": self.gpu_name,
            "vram_mb": self.vram_mb,
            "cuda_available": self.cuda_available,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "onnx_provider": self.onnx_provider,
            "status": self.status.value,
        }
