"""Cheap, torch-free hardware/RAM profile computed once at process start.

Reuses the existing SystemProfile/RuntimeParameters model in
app.streaming.deployment_readiness (rather than inventing a parallel one),
but builds the HardwareCapabilityReport from CPU/RAM (psutil) and GPU
presence (nvidia-smi subprocess, via ops.automation.hardware_detector) only
- never from HardwareCapabilityDetector._detect_gpu()/_detect_cuda(), which
import torch and would reintroduce the eager-torch-import cost this module
exists to avoid. Safe to call from the startup critical path.
"""

from app.streaming.deployment_readiness import (
    CUDAInfo,
    GPUInfo,
    HardwareCapabilityDetector,
    HardwareCapabilityReport,
    RuntimeParameters,
    SystemProfileEngine,
)

_params: RuntimeParameters | None = None


def _compute() -> RuntimeParameters:
    detector = HardwareCapabilityDetector()
    cpu = detector._detect_cpu()
    ram = detector._detect_ram()

    gpu = GPUInfo()
    cuda = CUDAInfo()
    try:
        from ops.automation.hardware_detector import HardwareDetector

        nv = HardwareDetector.detect_nvidia_gpu()
        if nv.present:
            gpu = GPUInfo(
                available=True,
                vendor="NVIDIA",
                model=nv.gpu_name or "Unknown",
                vram_total_mb=nv.vram_mb,
                vram_free_mb=nv.vram_mb,
            )
            cuda = CUDAInfo(available=True, cuda_version=nv.cuda_driver_version or "N/A")
    except (ImportError, OSError):
        pass

    report = HardwareCapabilityReport(cpu=cpu, ram=ram, gpu=gpu, cuda=cuda)
    params = SystemProfileEngine.select_profile(report)

    # SystemProfileEngine's tiers are keyed almost entirely on GPU/VRAM and
    # total RAM; a machine with a capable GPU but no free RAM headroom (this
    # dev machine: 6GB GPU, 8.3GB total RAM, ~90%+ used) still lands in a
    # high-concurrency tier. Available RAM is what actually determines
    # whether the OS starts paging, so it overrides the tier here.
    if ram.available_mb < 1536.0 and params.profile_name != "LOW_RESOURCE":
        params = RuntimeParameters(
            profile_name="LOW_RESOURCE",
            worker_count=1,
            detector_batch_size=1,
            osnet_batch_size=2,
            queue_depth=2,
            max_processing_fps=10.0,
            frame_dropping_policy="aggressive_stale",
            memory_safety_threshold_pct=75.0,
            vram_safety_threshold_mb=200.0,
            concurrent_inference_limit=1,
            enable_gpu=params.enable_gpu,
            device_name=params.device_name,
        )

    return params


def get_runtime_parameters() -> RuntimeParameters:
    global _params
    if _params is None:
        _params = _compute()
    return _params


def is_low_resource() -> bool:
    return get_runtime_parameters().profile_name == "LOW_RESOURCE"
