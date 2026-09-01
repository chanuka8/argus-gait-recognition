"""
Multi-Stage CUDA Hardware & Compute Pipeline Detector for ARGUS AI.

Evaluates CUDA readiness across the entire pipeline hierarchy:
1. NVIDIA hardware presence and driver readiness.
2. PyTorch CUDA build capability and runtime device availability.
3. Actual CUDA tensor allocation and synchronized matrix multiplication.
4. ByGaitLight CNN execution on CUDA ([1, 256] embedding, unit L2 norm).
5. Ultralytics YOLOv8 runtime execution device verification.
6. ONNX Runtime CUDAExecutionProvider initialization and silhouette inference.

CUDA is NEVER reported as READY based on torch.cuda.is_available() alone.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automation.dll_manager import setup_cuda_dll_paths
from automation.hardware_detector import HardwareDetector, NvidiaGpuInfo


@dataclass
class CudaStageResult:
    stage_name: str
    passed: bool
    details: str
    error: str | None = None


@dataclass
class CudaDetectionReport:
    hardware_gpu_detected: bool
    gpu_name: str | None
    driver_version: str | None
    vram_mb: float
    cuda_driver_version: str | None
    pytorch_installed: bool
    pytorch_version: str | None
    pytorch_cuda_build: str | None
    cuda_is_available: bool
    device_count: int
    tensor_probe_passed: bool
    bygait_probe_passed: bool
    yolo_cuda_passed: bool
    yolo_runtime_device: str
    onnx_cuda_passed: bool
    onnx_runtime_version: str | None
    onnx_selected_provider: str
    onnx_providers_available: list[str] = field(default_factory=list)
    all_cuda_stages_passed: bool = False
    stages: list[CudaStageResult] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "driver_version": self.driver_version,
            "vram_mb": self.vram_mb,
            "cuda_driver_version": self.cuda_driver_version,
            "pytorch_version": self.pytorch_version,
            "pytorch_cuda_build": self.pytorch_cuda_build,
            "cuda_is_available": self.cuda_is_available,
            "tensor_probe": self.tensor_probe_passed,
            "bygait_probe": self.bygait_probe_passed,
            "yolo_cuda": self.yolo_cuda_passed,
            "yolo_runtime_device": self.yolo_runtime_device,
            "onnx_cuda": self.onnx_cuda_passed,
            "onnx_provider": self.onnx_selected_provider,
            "all_cuda_stages_passed": self.all_cuda_stages_passed,
            "failure_reasons": self.failure_reasons,
        }


class CudaDetector:
    """Rigorous multi-stage CUDA pipeline validator."""

    def __init__(self, weights_dir: str = "models/weights") -> None:
        self.weights_dir = Path(weights_dir)
        setup_cuda_dll_paths()

    def probe_pytorch_cuda_build(self) -> tuple[bool, str | None, str | None, bool, int, str | None]:
        """Inspect PyTorch build, CUDA capability, and device enumeration."""
        try:
            import torch

            version = getattr(torch, "__version__", None)
            cuda_build = getattr(torch.version, "cuda", None)
            is_avail = torch.cuda.is_available()
            count = torch.cuda.device_count() if is_avail else 0
            return True, version, cuda_build, is_avail, count, None
        except ImportError as imp_err:
            return False, None, None, False, 0, f"PyTorch import failed: {imp_err}"
        except (RuntimeError, ValueError, AttributeError, OSError) as err:
            return False, None, None, False, 0, f"PyTorch probe error: {err}"

    def probe_cuda_tensor_execution(self) -> tuple[bool, str, str | None]:
        """Execute real CUDA memory allocation and synchronized 1024x1024 MatMul."""
        try:
            import torch

            if not torch.cuda.is_available():
                return False, "CUDA not available", "torch.cuda.is_available() is False"

            a = torch.zeros((1024, 1024), device="cuda")
            b = torch.ones((1024, 1024), device="cuda")
            c = a @ b
            torch.cuda.synchronize()

            if c.shape == (1024, 1024):
                return True, "1024x1024 Tensor MatMul synchronized successfully on CUDA", None
            return False, "Shape mismatch in tensor matmul", f"Expected (1024, 1024), got {c.shape}"
        except (RuntimeError, ValueError, TypeError, AttributeError) as err:
            return False, "CUDA tensor execution failed", str(err)

    def probe_bygait_cuda_execution(self) -> tuple[bool, str, str | None]:
        """Verify ByGaitLight forward pass on CUDA (output shape [1, 256], unit L2 norm)."""
        try:
            import torch

            from models.architectures.bygait_light import ByGaitLight

            if not torch.cuda.is_available():
                return False, "CUDA not available for ByGaitLight", "torch.cuda.is_available() is False"

            model = ByGaitLight().to("cuda")
            model.eval()

            dummy_gei = torch.randn(1, 1, 128, 64, device="cuda")
            with torch.no_grad():
                emb = model(dummy_gei)

            if emb.shape != (1, 256):
                return False, "Invalid embedding shape", f"Expected (1, 256), got {list(emb.shape)}"

            norm = torch.norm(emb, p=2, dim=-1).item()
            if abs(norm - 1.0) > 1e-4:
                return False, "Invalid L2 normalization", f"Expected norm ~1.0, got {norm:.6f}"

            return True, f"ByGaitLight forward pass verified on CUDA: shape {list(emb.shape)}, norm {norm:.4f}", None
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as err:
            return False, "ByGaitLight CUDA verification failed", str(err)

    def probe_yolo_cuda_execution(self) -> tuple[bool, str, str, str | None]:
        """Verify YOLOv8 runtime execution on CUDA directly."""
        try:
            import numpy as np
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                return False, "cpu", "CUDA is not available for YOLO", None

            model_path = self.weights_dir / "yolov8n.pt"
            if not model_path.exists():
                model_path = Path("models/weights/yolov8n.pt")

            model = YOLO(str(model_path) if model_path.exists() else "yolov8n.pt")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = model(dummy_frame, device="cuda:0", verbose=False)

            runtime_dev = str(next(model.model.parameters()).device)
            is_cuda = "cuda" in runtime_dev

            if is_cuda:
                return True, runtime_dev, f"YOLO runtime confirmed on {runtime_dev}", None
            return False, runtime_dev, f"YOLO running on {runtime_dev} (expected cuda:0)", None
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError, ImportError) as err:
            return False, "cpu", "YOLO runtime probe error", str(err)

    def probe_onnx_cuda_execution(self) -> tuple[bool, str | None, str, list[str], str | None]:
        """Verify ONNX Runtime CUDA provider initialization and real silhouette inference."""
        setup_cuda_dll_paths()
        try:
            import numpy as np
            import onnxruntime as ort

            version = getattr(ort, "__version__", None)
            providers = ort.get_available_providers()
            cuda_avail = "CUDAExecutionProvider" in providers

            if not cuda_avail:
                return (
                    False,
                    version,
                    "CPUExecutionProvider",
                    providers,
                    "CUDAExecutionProvider not in available ONNX providers",
                )

            model_candidates = [
                self.weights_dir / "silhouette_segmenter.onnx",
                Path("models/weights/silhouette_segmenter.onnx"),
                Path("models/engines/silhouette_segmenter.onnx"),
            ]
            model_path = next((p for p in model_candidates if p.exists()), None)

            if not model_path:
                return False, version, "CPUExecutionProvider", providers, "Silhouette ONNX model file not found"

            sess = ort.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active_providers = sess.get_providers()
            selected = active_providers[0] if active_providers else "CPUExecutionProvider"

            if selected != "CUDAExecutionProvider":
                return (
                    False,
                    version,
                    selected,
                    providers,
                    f"ONNX initialized with {selected} instead of CUDAExecutionProvider",
                )

            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            dummy_in = np.random.randn(1, 3, 256, 256).astype(np.float32)
            out = sess.run([out_name], {in_name: dummy_in})

            if out is None or len(out) == 0:
                return False, version, selected, providers, "ONNX silhouette inference returned empty output"

            return True, version, selected, providers, None
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError, ImportError) as err:
            return False, None, "CPUExecutionProvider", [], f"ONNX CUDA probe failed: {err}"

    def run_full_detection(self, gpu_info: NvidiaGpuInfo | None = None) -> CudaDetectionReport:
        """
        Execute comprehensive 6-phase CUDA readiness audit.
        """
        if gpu_info is None:
            gpu_info = HardwareDetector.detect_nvidia_gpu()

        stages: list[CudaStageResult] = []
        failures: list[str] = []


        hw_passed = gpu_info.present and bool(gpu_info.driver_version)
        stages.append(
            CudaStageResult(
                stage_name="Hardware GPU Detection",
                passed=hw_passed,
                details=f"GPU: {gpu_info.gpu_name}, Driver: {gpu_info.driver_version}, VRAM: {gpu_info.vram_mb:.0f} MB",
                error=gpu_info.error if not hw_passed else None,
            )
        )
        if not hw_passed:
            failures.append(f"Hardware GPU not usable: {gpu_info.error or 'No NVIDIA GPU'}")


        pt_inst, pt_ver, pt_cuda, pt_avail, dev_count, pt_err = self.probe_pytorch_cuda_build()
        pt_passed = pt_inst and bool(pt_cuda) and pt_avail and (dev_count > 0)
        stages.append(
            CudaStageResult(
                stage_name="PyTorch CUDA Build",
                passed=pt_passed,
                details=f"PyTorch: {pt_ver}, CUDA Build: {pt_cuda}, Available: {pt_avail}, Devices: {dev_count}",
                error=pt_err if not pt_passed else None,
            )
        )
        if not pt_passed:
            failures.append(f"PyTorch CUDA build not ready: {pt_err or 'No CUDA in PyTorch build'}")


        tensor_passed, tensor_details, tensor_err = self.probe_cuda_tensor_execution()
        stages.append(
            CudaStageResult(
                stage_name="CUDA Tensor MatMul Execution",
                passed=tensor_passed,
                details=tensor_details,
                error=tensor_err,
            )
        )
        if not tensor_passed:
            failures.append(f"CUDA tensor execution failed: {tensor_err}")


        bygait_passed, bygait_details, bygait_err = self.probe_bygait_cuda_execution()
        stages.append(
            CudaStageResult(
                stage_name="ByGaitLight CNN CUDA Execution",
                passed=bygait_passed,
                details=bygait_details,
                error=bygait_err,
            )
        )
        if not bygait_passed:
            failures.append(f"ByGaitLight CUDA inference failed: {bygait_err}")


        yolo_passed, yolo_dev, yolo_details, yolo_err = self.probe_yolo_cuda_execution()
        stages.append(
            CudaStageResult(
                stage_name="YOLO PersonDetector Runtime Device",
                passed=yolo_passed,
                details=f"Runtime Device: {yolo_dev} ({yolo_details})",
                error=yolo_err,
            )
        )
        if not yolo_passed:
            failures.append(f"YOLO CUDA execution not active: {yolo_details}")


        onnx_passed, onnx_ver, onnx_sel, onnx_provs, onnx_err = self.probe_onnx_cuda_execution()
        stages.append(
            CudaStageResult(
                stage_name="ONNX Runtime Silhouette CUDA Inference",
                passed=onnx_passed,
                details=f"Provider: {onnx_sel}, Version: {onnx_ver}",
                error=onnx_err,
            )
        )
        if not onnx_passed:
            failures.append(f"ONNX CUDA execution failed: {onnx_err}")

        all_passed = hw_passed and pt_passed and tensor_passed and bygait_passed and yolo_passed and onnx_passed

        return CudaDetectionReport(
            hardware_gpu_detected=gpu_info.present,
            gpu_name=gpu_info.gpu_name,
            driver_version=gpu_info.driver_version,
            vram_mb=gpu_info.vram_mb,
            cuda_driver_version=gpu_info.cuda_driver_version,
            pytorch_installed=pt_inst,
            pytorch_version=pt_ver,
            pytorch_cuda_build=pt_cuda,
            cuda_is_available=pt_avail,
            device_count=dev_count,
            tensor_probe_passed=tensor_passed,
            bygait_probe_passed=bygait_passed,
            yolo_cuda_passed=yolo_passed,
            yolo_runtime_device=yolo_dev,
            onnx_cuda_passed=onnx_passed,
            onnx_runtime_version=onnx_ver,
            onnx_selected_provider=onnx_sel,
            onnx_providers_available=onnx_provs,
            all_cuda_stages_passed=all_passed,
            stages=stages,
            failure_reasons=failures,
        )
