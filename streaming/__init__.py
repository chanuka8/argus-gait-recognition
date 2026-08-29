"""
Streaming module for ARGUS AI multi-stream and Phase 4 optimization.
"""

from streaming.buffer_queue import BufferQueue
from streaming.camera_scheduler import CameraScheduler
from streaming.frame_dropper import FrameDropper
from streaming.load_balancer import CameraLoadBalancer
from streaming.multi_stream_engine import MultiStreamEngine
from streaming.performance_optimizer import PerformanceOptimizer
from streaming.production_runtime import (
    AdaptiveResourceManager,
    CameraState,
    CameraStateMachine,
    CapacityEstimator,
    DataPoisoningGuard,
    FPSGovernor,
    FPSPolicy,
    FrameQualityGate,
    GracefulShutdownManager,
    ProductionSurveillanceRuntime,
    ReconnectEngine,
    ResilientWorkerPool,
    SafeModelSwapper,
    StructuredEventLogger,
)
from streaming.worker_pool import CameraWorkerPool

__all__ = [
    "AdaptiveResourceManager",
    "BufferQueue",
    "CameraLoadBalancer",
    "CameraScheduler",
    "CameraState",
    "CameraStateMachine",
    "CameraWorkerPool",
    "CapacityEstimator",
    "DataPoisoningGuard",
    "FPSGovernor",
    "FPSPolicy",
    "FrameDropper",
    "FrameQualityGate",
    "GracefulShutdownManager",
    "MultiStreamEngine",
    "PerformanceOptimizer",
    "ProductionSurveillanceRuntime",
    "ReconnectEngine",
    "ResilientWorkerPool",
    "SafeModelSwapper",
    "StructuredEventLogger",
]
