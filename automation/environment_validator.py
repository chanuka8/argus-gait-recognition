from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from automation.cuda_detector import CudaDetectionReport, CudaDetector
from automation.hardware_detector import HardwareDetector, HardwareProfile


class EnvironmentState(str, Enum):
    DETECTING = "DETECTING"
    INSTALLING = "INSTALLING"
    VALIDATING = "VALIDATING"
    CUDA_READY = "CUDA_READY"
    CPU_READY = "CPU_READY"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
    FAILED = "FAILED"


class ComputeBackend(str, Enum):
    CUDA = "CUDA"
    CPU = "CPU"


@dataclass
class EnvironmentValidationReport:
    state: EnvironmentState
    target_compute: ComputeBackend
    active_compute: ComputeBackend
    active_device: str
    is_healthy: bool
    requires_repair: bool
    repair_action: str
    hardware: HardwareProfile
    cuda_report: CudaDetectionReport | None = None
    cpu_validation_passed: bool = False
    details: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "target_compute": self.target_compute.value,
            "active_compute": self.active_compute.value,
            "active_device": self.active_device,
            "is_healthy": self.is_healthy,
            "requires_repair": self.requires_repair,
            "repair_action": self.repair_action,
            "cpu_validation_passed": self.cpu_validation_passed,
            "hardware": self.hardware.to_dict(),
            "cuda_report": self.cuda_report.to_dict() if self.cuda_report else None,
            "details": self.details,
            "errors": self.errors,
        }


class EnvironmentValidator:
    def __init__(self, weights_dir: str = "models/weights") -> None:
        self.weights_dir = Path(weights_dir)
        self.cuda_detector = CudaDetector(weights_dir=str(self.weights_dir))

    @staticmethod
    def validate_cpu_pipeline() -> tuple[bool, list[str], list[str]]:
        details: list[str] = []
        errors: list[str] = []
        cpu_ok = True


        try:
            import torch

            a = torch.zeros((256, 256), device="cpu")
            b = torch.ones((256, 256), device="cpu")
            c = a @ b
            if c.shape == (256, 256):
                details.append("PyTorch CPU tensor matmul passed")
            else:
                cpu_ok = False
                errors.append(f"PyTorch CPU matmul shape mismatch: {c.shape}")
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError, ImportError) as e:
            cpu_ok = False
            errors.append(f"PyTorch CPU tensor failed: {e}")


        try:
            import torch

            from models.architectures.bygait_light import ByGaitLight

            model = ByGaitLight().to("cpu")
            model.eval()
            dummy_gei = torch.randn(1, 1, 128, 64, device="cpu")
            with torch.no_grad():
                emb = model(dummy_gei)
            if emb.shape == (1, 256):
                norm = torch.norm(emb, p=2, dim=-1).item()
                if abs(norm - 1.0) <= 1e-3:
                    details.append(f"ByGaitLight CPU forward pass passed (norm: {norm:.4f})")
                else:
                    cpu_ok = False
                    errors.append(f"ByGaitLight CPU invalid norm: {norm}")
            else:
                cpu_ok = False
                errors.append(f"ByGaitLight CPU invalid shape: {emb.shape}")
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError, ImportError) as e:
            cpu_ok = False
            errors.append(f"ByGaitLight CPU forward pass failed: {e}")


        try:
            import numpy as np
            import onnxruntime as ort

            model_candidates = [
                Path("models/weights/silhouette_segmenter.onnx"),
                Path("models/engines/silhouette_segmenter.onnx"),
            ]
            model_path = next((p for p in model_candidates if p.exists()), None)
            if model_path:
                sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                in_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name
                dummy = np.zeros((1, 3, 256, 256), dtype=np.float32)
                out = sess.run([out_name], {in_name: dummy})
                if out is not None and len(out) > 0:
                    details.append("ONNX CPUExecutionProvider silhouette inference passed")
                else:
                    cpu_ok = False
                    errors.append("ONNX CPU inference returned empty output")
            else:
                details.append("Silhouette ONNX model not found, Otsu fallback available")
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError, ImportError) as e:
            cpu_ok = False
            errors.append(f"ONNX CPU inference failed: {e}")

        return cpu_ok, details, errors

    def validate(self, force_cpu: bool = False) -> EnvironmentValidationReport:
        hw = HardwareDetector.detect()
        details: list[str] = []
        errors: list[str] = []

        if force_cpu or not hw.gpu.present:
            cpu_ok, cpu_details, cpu_errors = self.validate_cpu_pipeline()
            details.extend(cpu_details)
            errors.extend(cpu_errors)

            state = EnvironmentState.CPU_READY if cpu_ok else EnvironmentState.FAILED
            repair_action = "NONE" if cpu_ok else "INSTALL_CPU_DEPENDENCIES"

            return EnvironmentValidationReport(
                state=state,
                target_compute=ComputeBackend.CPU,
                active_compute=ComputeBackend.CPU,
                active_device="cpu",
                is_healthy=cpu_ok,
                requires_repair=not cpu_ok,
                repair_action=repair_action,
                hardware=hw,
                cuda_report=None,
                cpu_validation_passed=cpu_ok,
                details=details,
                errors=errors,
            )


        cuda_report = self.cuda_detector.run_full_detection(gpu_info=hw.gpu)

        if cuda_report.all_cuda_stages_passed:
            details.append(f"All 6 CUDA pipeline stages verified on {hw.gpu.gpu_name}")
            return EnvironmentValidationReport(
                state=EnvironmentState.CUDA_READY,
                target_compute=ComputeBackend.CUDA,
                active_compute=ComputeBackend.CUDA,
                active_device="cuda:0",
                is_healthy=True,
                requires_repair=False,
                repair_action="NONE",
                hardware=hw,
                cuda_report=cuda_report,
                cpu_validation_passed=True,
                details=details,
                errors=[],
            )


        errors.extend(cuda_report.failure_reasons)


        needs_pytorch_repair = not (
            cuda_report.pytorch_installed and cuda_report.pytorch_cuda_build and cuda_report.cuda_is_available
        )
        needs_onnx_repair = not cuda_report.onnx_cuda_passed

        if needs_pytorch_repair or needs_onnx_repair:
            repair_action = "INSTALL_CUDA_PYTORCH" if needs_pytorch_repair else "INSTALL_ONNX_GPU"
            return EnvironmentValidationReport(
                state=EnvironmentState.REPAIR_REQUIRED,
                target_compute=ComputeBackend.CUDA,
                active_compute=ComputeBackend.CPU,
                active_device="cpu",
                is_healthy=False,
                requires_repair=True,
                repair_action=repair_action,
                hardware=hw,
                cuda_report=cuda_report,
                cpu_validation_passed=False,
                details=details,
                errors=errors,
            )


        cpu_ok, cpu_details, cpu_errors = self.validate_cpu_pipeline()
        details.extend(cpu_details)
        errors.extend(cpu_errors)

        state = EnvironmentState.CPU_READY if cpu_ok else EnvironmentState.FAILED
        return EnvironmentValidationReport(
            state=state,
            target_compute=ComputeBackend.CUDA,
            active_compute=ComputeBackend.CPU,
            active_device="cpu",
            is_healthy=cpu_ok,
            requires_repair=False,
            repair_action="CPU_FALLBACK_ACTIVE",
            hardware=hw,
            cuda_report=cuda_report,
            cpu_validation_passed=cpu_ok,
            details=details,
            errors=errors,
        )
