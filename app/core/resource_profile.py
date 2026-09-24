"""Cheap, torch-free hardware/RAM profile computed once at process start.

Reuses the existing SystemProfile/RuntimeParameters model in
app.streaming.deployment_readiness (rather than inventing a parallel one),
but builds the HardwareCapabilityReport from CPU/RAM (psutil) and GPU
presence (nvidia-smi subprocess, via ops.automation.hardware_detector) only
- never from HardwareCapabilityDetector._detect_gpu()/_detect_cuda(), which
import torch and would reintroduce the eager-torch-import cost this module
exists to avoid. Safe to call from the startup critical path.
"""

import os

from app.streaming.deployment_readiness import (
    CUDAInfo,
    GPUInfo,
    HardwareCapabilityDetector,
    HardwareCapabilityReport,
    RuntimeParameters,
    SystemProfileEngine,
)

_params: RuntimeParameters | None = None

# Conservative default, derived from measured evidence on an 8GB-class dev
# machine: normal ML startup takes 6-18s with adequate headroom, but took
# over 2 hours when available RAM dropped to roughly 270-470MB (severe
# Windows page-fault thrashing, not a slow-but-working startup). This sits
# comfortably above that observed danger zone with real margin, while still
# being reachable on an 8GB machine under ordinary desktop load. This is
# deliberately a *different, lower-level* gate than the existing 1536MB
# LOW_RESOURCE tier threshold below - LOW_RESOURCE only tunes batch sizes
# for an already-safe-to-start warmup, it was never a safety gate on
# whether to start warmup at all.
_DEFAULT_MIN_ML_STARTUP_HEADROOM_MB = 1024.0


def min_ml_startup_headroom_mb() -> float:
    """ARGUS_MIN_ML_STARTUP_HEADROOM_MB, validated, with a safe default."""
    raw = os.environ.get("ARGUS_MIN_ML_STARTUP_HEADROOM_MB")
    if raw is None:
        return _DEFAULT_MIN_ML_STARTUP_HEADROOM_MB
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_MIN_ML_STARTUP_HEADROOM_MB
    if value <= 0:
        return _DEFAULT_MIN_ML_STARTUP_HEADROOM_MB
    return value


def current_available_ram_mb() -> float:
    """Fresh (uncached) available-RAM reading.

    Unlike get_runtime_parameters(), which computes its profile once per
    process and caches it forever, this re-measures on every call - it's
    meant to be checked immediately before any heavy ML construction, not
    just once at process start, since available RAM on a shared desktop
    machine can swing by gigabytes over a session.
    """
    return HardwareCapabilityDetector()._detect_ram().available_mb


def has_sufficient_ml_startup_headroom() -> tuple[bool, float, float]:
    """Returns (sufficient, available_mb, threshold_mb), all freshly measured."""
    available = current_available_ram_mb()
    threshold = min_ml_startup_headroom_mb()
    return available >= threshold, available, threshold


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
