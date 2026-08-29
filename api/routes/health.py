"""
Implements:
- /health/live    — Liveness probe (process alive)
- /health/ready   — Readiness probe (system ready to process)
- /health/degraded — Degradation check
- /health/cameras — Per-camera health
- /health/workers — Inference worker health
- /health/system  — Full resource health + capacity estimation

One disconnected camera does NOT make the entire system unhealthy.
Readiness represents system capability, not perfection of every configured camera.
"""

import os
import time

from fastapi import APIRouter

health_router = APIRouter(prefix="/health", tags=["health"])

# Runtime reference — set by server.py at startup
_runtime = None
_start_time = time.monotonic()


def set_runtime(runtime) -> None:
    """Set the production runtime reference for health checks."""
    global _runtime
    _runtime = runtime


def _get_resource_snapshot() -> dict:
    """Collect current system resource metrics."""
    metrics = {
        "cpu_percent": 0.0,
        "ram_percent": 0.0,
        "ram_used_mb": 0.0,
        "gpu_name": "N/A",
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
        "vram_percent": 0.0,
    }
    try:
        import psutil
        metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        metrics["ram_percent"] = mem.percent
        metrics["ram_used_mb"] = round(mem.used / (1024 * 1024), 1)
    except (ImportError, OSError):
        pass

    try:
        import torch
        if torch.cuda.is_available():
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
            metrics["vram_used_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
            metrics["vram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
            )
            if metrics["vram_total_mb"] > 0:
                metrics["vram_percent"] = round(
                    metrics["vram_used_mb"] / metrics["vram_total_mb"] * 100.0, 1
                )
    except (ImportError, RuntimeError):
        pass

    return metrics


@health_router.get("/live")
def health_live():
    """Liveness probe — process is alive and responding."""
    return {
        "status": "alive",
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "pid": os.getpid(),
    }


@health_router.get("/ready")
def health_ready():
    """Readiness probe — system is ready to accept and process camera streams.

    One disconnected camera does NOT make the system 'not ready'.
    Readiness requires: process alive + at least one worker available.
    """
    ready = True
    reasons = []

    if _runtime is not None:
        workers = _runtime.worker_pool.get_worker_info()
        if not workers:
            ready = False
            reasons.append("no_workers_available")

        if _runtime.shutdown_manager.is_shutting_down:
            ready = False
            reasons.append("shutting_down")
    else:
        # No runtime configured = standalone API mode, still ready
        pass

    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "reasons": reasons,
    }


@health_router.get("/degraded")
def health_degraded():
    """Check if system is operational but degraded."""
    degraded = False
    degradation_reasons = []

    if _runtime is not None:
        pressure = _runtime.resource_manager.pressure.value
        if pressure in ("ELEVATED", "SATURATED", "CRITICAL"):
            degraded = True
            degradation_reasons.append(f"resource_pressure_{pressure.lower()}")

        cameras = _runtime.camera_state_machine.get_all_cameras()
        from streaming.production_runtime import CameraState
        failed_count = sum(
            1 for c in cameras.values()
            if c.connection_state == CameraState.FAILED
        )
        reconnecting_count = sum(
            1 for c in cameras.values()
            if c.connection_state == CameraState.RECONNECTING
        )
        if failed_count > 0:
            degraded = True
            degradation_reasons.append(f"{failed_count}_cameras_failed")
        if reconnecting_count > 0:
            degraded = True
            degradation_reasons.append(f"{reconnecting_count}_cameras_reconnecting")

    return {
        "status": "degraded" if degraded else "healthy",
        "degraded": degraded,
        "reasons": degradation_reasons,
    }


@health_router.get("/cameras")
def health_cameras():
    """Per-camera health report."""
    if _runtime is None:
        return {"cameras": {}, "total": 0}

    camera_health = _runtime.get_camera_health()
    return {
        "cameras": camera_health,
        "total": len(camera_health),
    }


@health_router.get("/workers")
def health_workers():
    """Inference worker health report."""
    if _runtime is None:
        return {"workers": {}, "total": 0}

    worker_info = _runtime.worker_pool.get_worker_info()
    return {
        "workers": worker_info,
        "total": len(worker_info),
    }


@health_router.get("/system")
def health_system():
    """Full system resource health snapshot with capacity estimation."""
    resources = _get_resource_snapshot()

    result = {
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "pid": os.getpid(),
        "resources": resources,
    }

    if _runtime is not None:
        system_health = _runtime.get_system_health()
        result["cameras"] = system_health.get("cameras", {})
        result["workers"] = system_health.get("workers", {})
        result["resource_pressure"] = system_health.get("resource_pressure", "UNKNOWN")
        result["processing_rate_factor"] = system_health.get("processing_rate_factor", 1.0)
        result["frame_quality"] = system_health.get("frame_quality", {})
        result["poisoning_guard"] = system_health.get("poisoning_guard", {})
        result["model_swapper"] = system_health.get("model_swapper", {})
        result["events_emitted"] = system_health.get("events_emitted", 0)

    return result


@health_router.get("/hardware")
def health_hardware():
    """Hardware capability discovery report without hardcoded environment assumptions."""
    from streaming.deployment_readiness import HardwareCapabilityDetector

    detector = HardwareCapabilityDetector()
    return detector.discover().to_dict()


@health_router.get("/models")
def health_models():
    """Resource and throughput profiles for pipeline neural network models."""
    from streaming.deployment_readiness import ModelProfileRegistry

    registry = ModelProfileRegistry()
    return registry.get_all_profiles()


@health_router.get("/capacity")
def health_capacity():
    """Comprehensive multi-factorial camera capacity estimation."""
    from streaming.deployment_readiness import DeploymentReadinessManager

    mgr = DeploymentReadinessManager()
    summary = mgr.get_deployment_summary()
    return summary.get("capacity", {})


@health_router.get("/admission")
def health_admission():
    """Current camera admission status and headroom."""
    from streaming.deployment_readiness import DeploymentReadinessManager

    mgr = DeploymentReadinessManager()
    summary = mgr.get_deployment_summary()
    return {
        "status": "OPERATIONAL",
        "system_profile": summary.get("system_profile", "AUTO"),
        "capacity": summary.get("capacity", {}),
        "runtime_parameters": summary.get("runtime_parameters", {}),
    }


@health_router.get("/deployment")
def health_deployment():
    """Complete deployment readiness diagnostic report."""
    from streaming.deployment_readiness import DeploymentReadinessManager

    mgr = DeploymentReadinessManager()
    return mgr.get_deployment_summary()
