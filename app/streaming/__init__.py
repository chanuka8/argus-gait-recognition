"""Lazy package exports.

Importing `app.streaming` (or any single submodule, e.g.
`app.streaming.deployment_readiness`) used to eagerly import every sibling
module below, several of which (`production_multicamera_engine`,
`production_runtime`) import torch at module scope - turning a request for
the torch-free `deployment_readiness` helpers into a ~70s+ cold torch
import. Names in `__all__` are still importable via
`from app.streaming import X`; the owning submodule now loads lazily on
first access (PEP 562) instead of at package-import time.
"""

import importlib
from typing import Any

__all__ = [
    "AdaptiveInferencePolicy",
    "AdaptiveResourceManager",
    "AdmissionDecision",
    "AdmissionResult",
    "BufferQueue",
    "CameraAdmissionController",
    "CameraLoadBalancer",
    "CameraScheduler",
    "CameraState",
    "CameraStateMachine",
    "CameraWorkerPool",
    "CapacityEstimator",
    "CentralStreamScheduler",
    "DataPoisoningGuard",
    "DeploymentMode",
    "DeploymentReadinessManager",
    "FPSGovernor",
    "FPSPolicy",
    "FrameDropper",
    "FramePacket",
    "FrameQualityGate",
    "GPUMemoryGuard",
    "GracefulShutdownManager",
    "HardwareCapabilityDetector",
    "HardwareCapabilityReport",
    "HardwareProfile",
    "InferenceQualityMode",
    "ModelProfileRegistry",
    "ModelResourceProfile",
    "MultiStreamEngine",
    "NetworkBandwidthEstimator",
    "PerformanceOptimizer",
    "ProductionCapacityEstimator",
    "ProductionMultiCameraEngine",
    "ProductionSurveillanceRuntime",
    "ReconnectEngine",
    "ResilientWorkerPool",
    "RuntimeParameters",
    "SafeModelSwapper",
    "SecurityAuditor",
    "StorageSafetyAuditor",
    "StreamIngestionQueue",
    "StructuredEventLogger",
    "SystemProfile",
    "SystemProfileEngine",
    "detect_hardware_profile",
]

_EXPORTS: dict[str, str] = {
    "BufferQueue": "app.streaming.buffer_queue",
    "CameraScheduler": "app.streaming.camera_scheduler",
    "AdaptiveInferencePolicy": "app.streaming.deployment_readiness",
    "AdmissionDecision": "app.streaming.deployment_readiness",
    "AdmissionResult": "app.streaming.deployment_readiness",
    "CameraAdmissionController": "app.streaming.deployment_readiness",
    "DeploymentMode": "app.streaming.deployment_readiness",
    "DeploymentReadinessManager": "app.streaming.deployment_readiness",
    "GPUMemoryGuard": "app.streaming.deployment_readiness",
    "HardwareCapabilityDetector": "app.streaming.deployment_readiness",
    "HardwareCapabilityReport": "app.streaming.deployment_readiness",
    "InferenceQualityMode": "app.streaming.deployment_readiness",
    "ModelProfileRegistry": "app.streaming.deployment_readiness",
    "ModelResourceProfile": "app.streaming.deployment_readiness",
    "NetworkBandwidthEstimator": "app.streaming.deployment_readiness",
    "ProductionCapacityEstimator": "app.streaming.deployment_readiness",
    "RuntimeParameters": "app.streaming.deployment_readiness",
    "SecurityAuditor": "app.streaming.deployment_readiness",
    "StorageSafetyAuditor": "app.streaming.deployment_readiness",
    "SystemProfile": "app.streaming.deployment_readiness",
    "SystemProfileEngine": "app.streaming.deployment_readiness",
    "FrameDropper": "app.streaming.frame_dropper",
    "CameraLoadBalancer": "app.streaming.load_balancer",
    "MultiStreamEngine": "app.streaming.multi_stream_engine",
    "PerformanceOptimizer": "app.streaming.performance_optimizer",
    "CentralStreamScheduler": "app.streaming.production_multicamera_engine",
    "FramePacket": "app.streaming.production_multicamera_engine",
    "HardwareProfile": "app.streaming.production_multicamera_engine",
    "ProductionMultiCameraEngine": "app.streaming.production_multicamera_engine",
    "StreamIngestionQueue": "app.streaming.production_multicamera_engine",
    "detect_hardware_profile": "app.streaming.production_multicamera_engine",
    "AdaptiveResourceManager": "app.streaming.production_runtime",
    "CameraState": "app.streaming.production_runtime",
    "CameraStateMachine": "app.streaming.production_runtime",
    "CapacityEstimator": "app.streaming.production_runtime",
    "DataPoisoningGuard": "app.streaming.production_runtime",
    "FPSGovernor": "app.streaming.production_runtime",
    "FPSPolicy": "app.streaming.production_runtime",
    "FrameQualityGate": "app.streaming.production_runtime",
    "GracefulShutdownManager": "app.streaming.production_runtime",
    "ProductionSurveillanceRuntime": "app.streaming.production_runtime",
    "ReconnectEngine": "app.streaming.production_runtime",
    "ResilientWorkerPool": "app.streaming.production_runtime",
    "SafeModelSwapper": "app.streaming.production_runtime",
    "StructuredEventLogger": "app.streaming.production_runtime",
    "CameraWorkerPool": "app.streaming.worker_pool",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
