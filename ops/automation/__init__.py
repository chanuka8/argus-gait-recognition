from ops.automation.cuda_detector import CudaDetector
from ops.automation.device_manager import DeviceManager
from ops.automation.dll_manager import setup_cuda_dll_paths
from ops.automation.environment_validator import (
    ComputeBackend,
    EnvironmentState,
    EnvironmentValidator,
)
from ops.automation.hardware_detector import HardwareDetector
from ops.automation.onnx_manager import OnnxManager
from ops.automation.pytorch_manager import PyTorchManager

__all__ = [
    "ComputeBackend",
    "CudaDetector",
    "DeviceManager",
    "EnvironmentBootstrap",
    "EnvironmentState",
    "EnvironmentValidator",
    "HardwareDetector",
    "OnnxManager",
    "PyTorchManager",
    "setup_cuda_dll_paths",
]


def __getattr__(name: str):
    if name == "EnvironmentBootstrap":
        from ops.automation.bootstrap import EnvironmentBootstrap

        return EnvironmentBootstrap
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
