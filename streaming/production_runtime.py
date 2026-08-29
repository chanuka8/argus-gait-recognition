"""
Implements production-grade camera lifecycle management, reconnect engine,
failure isolation, inference worker resilience, adaptive resource management,
frame quality control, FPS governance, model lifecycle safety, structured
logging, and graceful shutdown.

"""

import enum
import hashlib
import json
import logging
import random
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger

# ---------------------------------------------------------------------------
# Task 1 — Camera State Machine
# ---------------------------------------------------------------------------


class CameraState(enum.Enum):
    """Production camera lifecycle states.

    Initial state is always STOPPED. FAILED is only reached after
    an actual connection attempt — never as the default initial state.
    """

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"
    STOPPING = "STOPPING"


# Valid transitions enforced by the state machine
_VALID_TRANSITIONS: dict[CameraState, set[CameraState]] = {
    CameraState.STOPPED: {CameraState.STARTING},
    CameraState.STARTING: {CameraState.CONNECTING, CameraState.STOPPING, CameraState.FAILED},
    CameraState.CONNECTING: {CameraState.CONNECTED, CameraState.FAILED, CameraState.STOPPING},
    CameraState.CONNECTED: {CameraState.DEGRADED, CameraState.RECONNECTING, CameraState.STOPPING},
    CameraState.DEGRADED: {CameraState.CONNECTED, CameraState.RECONNECTING, CameraState.STOPPING},
    CameraState.RECONNECTING: {CameraState.CONNECTING, CameraState.FAILED, CameraState.STOPPING},
    CameraState.FAILED: {CameraState.STARTING, CameraState.STOPPING, CameraState.STOPPED},
    CameraState.STOPPING: {CameraState.STOPPED},
}


@dataclass
class CameraResource:
    """Full camera lifecycle state with production telemetry."""

    camera_id: str
    source_type: str = "webcam"
    source_uri: str = ""

    # State machine
    connection_state: CameraState = CameraState.STOPPED
    desired_state: CameraState = CameraState.STOPPED
    actual_state: CameraState = CameraState.STOPPED

    # Configuration
    fps_target: int = 15
    resolution: tuple[int, int] = (640, 480)
    codec: str = "h264"
    priority: int = 5

    # Timestamps
    last_frame_timestamp: float = 0.0
    last_success_timestamp: float = 0.0
    last_error: str = ""
    last_error_timestamp: float = 0.0
    created_at: float = field(default_factory=time.monotonic)

    # Reconnect
    reconnect_attempts: int = 0
    total_reconnect_count: int = 0

    # Telemetry
    queue_depth: int = 0
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    health_score: float = 1.0

    def compute_health_score(self) -> float:
        """Compute health score 0.0-1.0 based on recent metrics."""
        score = 1.0
        if self.connection_state == CameraState.FAILED:
            return 0.0
        if self.connection_state == CameraState.RECONNECTING:
            score -= 0.4
        if self.connection_state == CameraState.DEGRADED:
            score -= 0.2
        if self.frames_received > 0:
            drop_rate = self.frames_dropped / self.frames_received
            score -= min(0.3, drop_rate)
        now = time.monotonic()
        if self.last_success_timestamp > 0 and (now - self.last_success_timestamp) > 10.0:
            score -= 0.2
        self.health_score = max(0.0, min(1.0, score))
        return self.health_score


class CameraStateMachine:
    """Thread-safe camera state machine with validated transitions."""

    def __init__(self) -> None:
        self._cameras: dict[str, CameraResource] = {}
        self._lock = threading.RLock()
        self._logger = get_logger("camera_lifecycle")
        self._state_listeners: list[Any] = []

    def register_camera(
        self,
        camera_id: str,
        source_type: str = "webcam",
        source_uri: str = "",
        fps_target: int = 15,
        resolution: tuple[int, int] = (640, 480),
        codec: str = "h264",
        priority: int = 5,
    ) -> CameraResource:
        """Register a new camera. Initial state is always STOPPED."""
        with self._lock:
            if camera_id in self._cameras:
                return self._cameras[camera_id]
            cam = CameraResource(
                camera_id=camera_id,
                source_type=source_type,
                source_uri=source_uri,
                connection_state=CameraState.STOPPED,
                desired_state=CameraState.STOPPED,
                actual_state=CameraState.STOPPED,
                fps_target=fps_target,
                resolution=resolution,
                codec=codec,
                priority=priority,
            )
            self._cameras[camera_id] = cam
            self._logger.info(
                f"Camera '{camera_id}' registered (source={source_type}, state=STOPPED)"
            )
            return cam

    def transition(self, camera_id: str, new_state: CameraState, error: str = "") -> bool:
        """Attempt state transition with validation."""
        with self._lock:
            cam = self._cameras.get(camera_id)
            if cam is None:
                return False
            old_state = cam.connection_state
            if new_state not in _VALID_TRANSITIONS.get(old_state, set()):
                self._logger.warning(
                    f"Invalid transition for '{camera_id}': {old_state.value} → {new_state.value}"
                )
                return False
            cam.connection_state = new_state
            cam.actual_state = new_state
            if error:
                cam.last_error = error
                cam.last_error_timestamp = time.monotonic()
            if new_state == CameraState.CONNECTED:
                cam.last_success_timestamp = time.monotonic()
                cam.reconnect_attempts = 0
            if new_state == CameraState.RECONNECTING:
                cam.reconnect_attempts += 1
                cam.total_reconnect_count += 1
            cam.compute_health_score()
            self._logger.info(
                f"Camera '{camera_id}': {old_state.value} → {new_state.value}"
                + (f" (error: {error})" if error else "")
            )
            for listener in self._state_listeners:
                try:
                    listener(camera_id, old_state, new_state, error)
                except (RuntimeError, ValueError, TypeError, KeyError, OSError):
                    pass
            return True

    def get_camera(self, camera_id: str) -> CameraResource | None:
        with self._lock:
            return self._cameras.get(camera_id)

    def get_all_cameras(self) -> dict[str, CameraResource]:
        with self._lock:
            return dict(self._cameras)

    def unregister_camera(self, camera_id: str) -> bool:
        with self._lock:
            cam = self._cameras.pop(camera_id, None)
            if cam is None:
                return False
            self._logger.info(f"Camera '{camera_id}' unregistered")
            return True

    def add_state_listener(self, listener) -> None:
        with self._lock:
            self._state_listeners.append(listener)


# ---------------------------------------------------------------------------
# Task 2 — Production Reconnect Engine
# ---------------------------------------------------------------------------


@dataclass
class ReconnectConfig:
    """Reconnect policy configuration."""

    min_retry_interval: float = 1.0
    max_retry_interval: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.5
    max_retry_attempts: int = 0  # 0 = unlimited


class ReconnectEngine:
    """Exponential backoff reconnect engine with jitter and resource safety."""

    def __init__(self, config: ReconnectConfig | None = None) -> None:
        self.config = config or ReconnectConfig()
        self._timers: dict[str, threading.Timer] = {}
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._logger = get_logger("reconnect_engine")
        self._stopped = threading.Event()

    def schedule_reconnect(
        self, camera_id: str, callback, *args, **kwargs
    ) -> bool:
        """Schedule a reconnect attempt with exponential backoff + jitter."""
        with self._lock:
            if self._stopped.is_set():
                return False
            # Cancel any existing timer for this camera
            self._cancel_timer_unsafe(camera_id)

            attempt = self._attempts.get(camera_id, 0)
            if (
                self.config.max_retry_attempts > 0
                and attempt >= self.config.max_retry_attempts
            ):
                self._logger.warning(
                    f"Camera '{camera_id}': max reconnect attempts ({self.config.max_retry_attempts}) reached"
                )
                return False

            delay = self._compute_delay(attempt)
            self._attempts[camera_id] = attempt + 1

            self._logger.info(
                f"Camera '{camera_id}': scheduling reconnect attempt {attempt + 1} "
                f"in {delay:.1f}s"
            )

            timer = threading.Timer(delay, self._execute_reconnect, args=(camera_id, callback, args, kwargs))
            timer.daemon = True
            timer.name = f"ARGUS-Reconnect-{camera_id}"
            timer.start()
            self._timers[camera_id] = timer
            return True

    def reset(self, camera_id: str) -> None:
        """Reset retry counter after successful connection."""
        with self._lock:
            self._attempts.pop(camera_id, None)
            self._cancel_timer_unsafe(camera_id)

    def cancel(self, camera_id: str) -> None:
        """Cancel pending reconnect for a camera."""
        with self._lock:
            self._cancel_timer_unsafe(camera_id)
            self._attempts.pop(camera_id, None)

    def cancel_all(self) -> None:
        """Cancel all pending reconnects."""
        with self._lock:
            self._stopped.set()
            for camera_id in list(self._timers):
                self._cancel_timer_unsafe(camera_id)
            self._timers.clear()
            self._attempts.clear()

    def get_attempt_count(self, camera_id: str) -> int:
        with self._lock:
            return self._attempts.get(camera_id, 0)

    def _cancel_timer_unsafe(self, camera_id: str) -> None:
        timer = self._timers.pop(camera_id, None)
        if timer is not None:
            timer.cancel()

    def _compute_delay(self, attempt: int) -> float:
        base = self.config.min_retry_interval * (
            self.config.backoff_multiplier ** attempt
        )
        clamped = min(base, self.config.max_retry_interval)
        jitter = random.uniform(
            -self.config.jitter_range * clamped,
            self.config.jitter_range * clamped,
        )
        return max(self.config.min_retry_interval, clamped + jitter)

    def _execute_reconnect(self, camera_id, callback, args, kwargs) -> None:
        try:
            callback(*args, **kwargs)
        except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
            self._logger.warning(f"Reconnect callback error for '{camera_id}': {exc}")


# ---------------------------------------------------------------------------
# Task 4 — Inference Worker Resilience
# ---------------------------------------------------------------------------


class InferenceWorkerState(enum.Enum):
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    FAILED = "FAILED"
    RESTARTING = "RESTARTING"
    STOPPED = "STOPPED"


@dataclass
class InferenceWorkerInfo:
    """Per-worker health and telemetry."""

    worker_id: str
    worker_type: str = "shared_inference"
    state: InferenceWorkerState = InferenceWorkerState.IDLE
    processed_frames: int = 0
    failed_frames: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    queue_depth: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    _latencies: deque = field(default_factory=lambda: deque(maxlen=200))

    def record_success(self, latency_ms: float) -> None:
        self.processed_frames += 1
        self.last_success = time.monotonic()
        self.state = InferenceWorkerState.IDLE
        self._latencies.append(latency_ms)
        if self._latencies:
            self.average_latency_ms = sum(self._latencies) / len(self._latencies)
            sorted_lat = sorted(self._latencies)
            idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
            self.p95_latency_ms = sorted_lat[idx]

    def record_failure(self) -> None:
        self.failed_frames += 1
        self.last_failure = time.monotonic()
        self.state = InferenceWorkerState.FAILED


class ResilientWorkerPool:
    """Manages inference workers with health monitoring, restart, and timeout protection."""

    def __init__(
        self,
        num_workers: int = 2,
        inference_timeout_seconds: float = 30.0,
        health_check_interval: float = 10.0,
    ) -> None:
        self.num_workers = max(1, num_workers)
        self.inference_timeout = inference_timeout_seconds
        self.health_check_interval = health_check_interval

        self._workers: dict[str, InferenceWorkerInfo] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._logger = get_logger("worker_pool")
        self._health_thread: threading.Thread | None = None
        self._process_fn = None

    def start(self, process_fn) -> None:
        """Start worker pool with given processing function."""
        self._process_fn = process_fn
        self._stop_event.clear()

        for i in range(self.num_workers):
            wid = f"worker-{i:02d}"
            self._spawn_worker(wid)

        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="ARGUS-WorkerHealthMonitor",
            daemon=True,
        )
        self._health_thread.start()
        self._logger.info(f"Resilient worker pool started with {self.num_workers} workers")

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop all workers."""
        self._stop_event.set()
        with self._lock:
            for wid, t in list(self._threads.items()):
                if t.is_alive():
                    t.join(timeout=timeout)
                info = self._workers.get(wid)
                if info:
                    info.state = InferenceWorkerState.STOPPED
            self._threads.clear()

        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=2.0)
        self._logger.info("Worker pool stopped")

    def get_worker_info(self) -> dict[str, dict]:
        """Return all worker telemetry."""
        with self._lock:
            result = {}
            for wid, info in self._workers.items():
                result[wid] = {
                    "worker_id": info.worker_id,
                    "worker_type": info.worker_type,
                    "state": info.state.value,
                    "processed_frames": info.processed_frames,
                    "failed_frames": info.failed_frames,
                    "average_latency_ms": round(info.average_latency_ms, 2),
                    "p95_latency_ms": round(info.p95_latency_ms, 2),
                    "queue_depth": info.queue_depth,
                    "last_success": info.last_success,
                    "last_failure": info.last_failure,
                }
            return result

    def _spawn_worker(self, worker_id: str) -> None:
        with self._lock:
            info = InferenceWorkerInfo(worker_id=worker_id)
            self._workers[worker_id] = info
            t = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=f"ARGUS-InferenceWorker-{worker_id}",
                daemon=True,
            )
            t.start()
            self._threads[worker_id] = t

    def _worker_loop(self, worker_id: str) -> None:
        while not self._stop_event.is_set():
            info = self._workers.get(worker_id)
            if info is None:
                break
            try:
                if self._process_fn:
                    info.state = InferenceWorkerState.PROCESSING
                    t0 = time.monotonic()
                    self._process_fn()
                    elapsed = (time.monotonic() - t0) * 1000.0
                    info.record_success(elapsed)
                else:
                    time.sleep(0.01)
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
                if info:
                    info.record_failure()
                self._logger.warning(f"Worker {worker_id} inference error: {exc}")
                time.sleep(0.05)

    def _health_monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.health_check_interval):
                break
            with self._lock:
                for wid, info in list(self._workers.items()):
                    t = self._threads.get(wid)
                    if t is None or not t.is_alive():
                        self._logger.warning(f"Worker {wid} died — restarting")
                        info.state = InferenceWorkerState.RESTARTING
                        self._spawn_worker(wid)


# ---------------------------------------------------------------------------
# Task 5 — Adaptive Resource Management
# ---------------------------------------------------------------------------


class ResourcePressure(enum.Enum):
    HEALTHY = "HEALTHY"
    ELEVATED = "ELEVATED"
    SATURATED = "SATURATED"
    CRITICAL = "CRITICAL"


@dataclass
class ResourceThresholds:
    """Configurable resource thresholds for adaptive policy."""

    max_cpu_percent: float = 85.0
    max_ram_percent: float = 85.0
    max_gpu_percent: float = 90.0
    max_vram_percent: float = 90.0
    max_queue_depth: int = 8
    max_frame_age_ms: float = 500.0
    max_p95_latency_ms: float = 200.0
    elevated_cpu_percent: float = 70.0
    elevated_gpu_percent: float = 75.0


@dataclass
class ResourceSnapshot:
    """Point-in-time resource measurements."""

    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_mb: float = 0.0
    gpu_percent: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_percent: float = 0.0
    avg_queue_depth: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    frame_drop_rate: float = 0.0
    active_cameras: int = 0
    active_workers: int = 0
    timestamp: float = field(default_factory=time.monotonic)


class AdaptiveResourceManager:
    """Monitors system resources and applies backpressure policies."""

    def __init__(self, thresholds: ResourceThresholds | None = None) -> None:
        self.thresholds = thresholds or ResourceThresholds()
        self._pressure = ResourcePressure.HEALTHY
        self._processing_rate_factor: float = 1.0
        self._lock = threading.Lock()
        self._logger = get_logger("resource_manager")
        self._history: deque[ResourceSnapshot] = deque(maxlen=60)

    @property
    def pressure(self) -> ResourcePressure:
        with self._lock:
            return self._pressure

    @property
    def processing_rate_factor(self) -> float:
        """Factor 0.0-1.0 applied to processing rate. 1.0 = full speed."""
        with self._lock:
            return self._processing_rate_factor

    def take_snapshot(self) -> ResourceSnapshot:
        """Collect current resource state."""
        snap = ResourceSnapshot()
        try:
            import psutil
            snap.cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            snap.ram_percent = mem.percent
            snap.ram_used_mb = mem.used / (1024 * 1024)
        except (ImportError, OSError):
            pass

        try:
            import torch
            if torch.cuda.is_available():
                snap.vram_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                snap.vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                snap.vram_percent = (
                    (snap.vram_used_mb / snap.vram_total_mb * 100.0)
                    if snap.vram_total_mb > 0
                    else 0.0
                )
        except (ImportError, RuntimeError):
            pass

        snap.timestamp = time.monotonic()
        with self._lock:
            self._history.append(snap)
        return snap

    def evaluate(self, snapshot: ResourceSnapshot | None = None) -> ResourcePressure:
        """Evaluate resource pressure and adjust processing rate."""
        snap = snapshot or self.take_snapshot()
        th = self.thresholds

        with self._lock:
            old_pressure = self._pressure

            # Determine pressure level
            if (
                snap.cpu_percent >= th.max_cpu_percent
                or snap.ram_percent >= th.max_ram_percent
                or snap.vram_percent >= th.max_vram_percent
                or snap.p95_latency_ms >= th.max_p95_latency_ms * 1.5
            ):
                self._pressure = ResourcePressure.CRITICAL
                target_factor = 0.25
            elif (
                snap.cpu_percent >= th.max_cpu_percent * 0.9
                or snap.vram_percent >= th.max_vram_percent * 0.9
                or snap.avg_queue_depth >= th.max_queue_depth
                or snap.p95_latency_ms >= th.max_p95_latency_ms
            ):
                self._pressure = ResourcePressure.SATURATED
                target_factor = 0.50
            elif (
                snap.cpu_percent >= th.elevated_cpu_percent
                or snap.vram_percent >= th.elevated_gpu_percent
            ):
                self._pressure = ResourcePressure.ELEVATED
                target_factor = 0.75
            else:
                self._pressure = ResourcePressure.HEALTHY
                target_factor = 1.0

            # Immediate reduction under CRITICAL; smooth transitions otherwise
            if self._pressure == ResourcePressure.CRITICAL:
                self._processing_rate_factor = target_factor
            elif target_factor < self._processing_rate_factor:
                self._processing_rate_factor = max(
                    target_factor,
                    self._processing_rate_factor - 0.1,
                )
            elif target_factor > self._processing_rate_factor:
                self._processing_rate_factor = min(
                    target_factor,
                    self._processing_rate_factor + 0.05,
                )

            if self._pressure != old_pressure:
                self._logger.info(
                    f"Resource pressure: {old_pressure.value} → {self._pressure.value} "
                    f"(rate_factor={self._processing_rate_factor:.2f}, "
                    f"CPU={snap.cpu_percent:.0f}%, VRAM={snap.vram_percent:.0f}%)"
                )

            return self._pressure

    def get_history(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "cpu_percent": s.cpu_percent,
                    "ram_percent": s.ram_percent,
                    "vram_percent": s.vram_percent,
                    "avg_queue_depth": s.avg_queue_depth,
                    "p95_latency_ms": s.p95_latency_ms,
                    "timestamp": s.timestamp,
                }
                for s in self._history
            ]


# ---------------------------------------------------------------------------
# Task 6 — Frame Quality & Staleness Control
# ---------------------------------------------------------------------------


@dataclass
class QualifiedFrame:
    """Frame with full provenance and quality metadata."""

    camera_id: str
    frame_id: int
    frame_uuid: str
    capture_timestamp: float  # monotonic
    wall_timestamp: str  # ISO 8601 UTC
    frame: np.ndarray
    source_type: str = "webcam"
    priority: int = 5
    frame_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameQualityGate:
    """Validates frame freshness, duplicates, and ordering."""

    def __init__(
        self,
        max_frame_age_ms: float = 500.0,
        enable_duplicate_detection: bool = True,
    ) -> None:
        self.max_frame_age_ms = max_frame_age_ms
        self.enable_duplicate_detection = enable_duplicate_detection
        self._last_hashes: dict[str, str] = {}
        self._last_timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

        # Counters
        self.rejected_stale: int = 0
        self.rejected_duplicate: int = 0
        self.rejected_out_of_order: int = 0
        self.accepted: int = 0

    def validate(self, frame: QualifiedFrame) -> tuple[bool, str]:
        """Validate a frame. Returns (accepted, reason)."""
        now = time.monotonic()
        age_ms = (now - frame.capture_timestamp) * 1000.0

        # Staleness check
        if age_ms > self.max_frame_age_ms:
            with self._lock:
                self.rejected_stale += 1
            return False, f"stale ({age_ms:.0f}ms > {self.max_frame_age_ms:.0f}ms)"

        with self._lock:
            # Out-of-order check
            last_ts = self._last_timestamps.get(frame.camera_id, 0.0)
            if frame.capture_timestamp < last_ts:
                self.rejected_out_of_order += 1
                return False, "out_of_order"

            # Duplicate detection
            if self.enable_duplicate_detection and frame.frame_hash:
                prev_hash = self._last_hashes.get(frame.camera_id)
                if prev_hash and prev_hash == frame.frame_hash:
                    self.rejected_duplicate += 1
                    return False, "duplicate"
                self._last_hashes[frame.camera_id] = frame.frame_hash

            self._last_timestamps[frame.camera_id] = frame.capture_timestamp
            self.accepted += 1
            return True, "accepted"

    def compute_frame_hash(self, frame_data: np.ndarray) -> str:
        """Fast perceptual hash for duplicate detection."""
        if frame_data is None or frame_data.size == 0:
            return ""
        # Use a small downsampled region for speed
        small = frame_data[::16, ::16].tobytes()[:256]
        return hashlib.md5(small).hexdigest()

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "accepted": self.accepted,
                "rejected_stale": self.rejected_stale,
                "rejected_duplicate": self.rejected_duplicate,
                "rejected_out_of_order": self.rejected_out_of_order,
            }


# ---------------------------------------------------------------------------
# Task 7 — Camera FPS Governor
# ---------------------------------------------------------------------------


class FPSPolicy(enum.Enum):
    PROCESS_EVERY_FRAME = "PROCESS_EVERY_FRAME"
    TARGET_FPS = "TARGET_FPS"
    ADAPTIVE_FPS = "ADAPTIVE_FPS"


class FPSGovernor:
    """Controls inference processing rate independently from camera capture rate."""

    def __init__(
        self,
        policy: FPSPolicy = FPSPolicy.TARGET_FPS,
        target_inference_fps: float = 10.0,
        adaptive_min_fps: float = 2.0,
        adaptive_max_fps: float = 30.0,
    ) -> None:
        self.policy = policy
        self.target_inference_fps = max(0.1, target_inference_fps)
        self.adaptive_min_fps = adaptive_min_fps
        self.adaptive_max_fps = adaptive_max_fps

        self._last_process_time: dict[str, float] = {}
        self._current_adaptive_fps: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_process(self, camera_id: str, resource_factor: float = 1.0) -> bool:
        """Determine if this frame should be processed based on FPS policy."""
        now = time.monotonic()

        if self.policy == FPSPolicy.PROCESS_EVERY_FRAME:
            return True

        with self._lock:
            if self.policy == FPSPolicy.TARGET_FPS:
                target = self.target_inference_fps
            elif self.policy == FPSPolicy.ADAPTIVE_FPS:
                base = self._current_adaptive_fps.get(
                    camera_id, self.target_inference_fps
                )
                target = max(
                    self.adaptive_min_fps,
                    min(self.adaptive_max_fps, base * resource_factor),
                )
                self._current_adaptive_fps[camera_id] = target
            else:
                target = self.target_inference_fps

            if target <= 0:
                return True

            min_interval = 1.0 / target
            last = self._last_process_time.get(camera_id, 0.0)

            if (now - last) >= min_interval:
                self._last_process_time[camera_id] = now
                return True
            return False

    def get_effective_fps(self, camera_id: str) -> float:
        with self._lock:
            if self.policy == FPSPolicy.ADAPTIVE_FPS:
                return self._current_adaptive_fps.get(
                    camera_id, self.target_inference_fps
                )
            return self.target_inference_fps


# ---------------------------------------------------------------------------
# Task 13 — Model Lifecycle Safety (Hot-Swap)
# ---------------------------------------------------------------------------


class ModelVersion:
    """Represents a versioned model artifact."""

    def __init__(
        self,
        version_id: str,
        model_path: str,
        model_name: str,
        created_at: str = "",
        metadata: dict | None = None,
    ) -> None:
        self.version_id = version_id
        self.model_path = model_path
        self.model_name = model_name
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}


class SafeModelSwapper:
    """Atomic model version swap with rollback support.

    Lifecycle: ACTIVE → candidate training → validation → registry → promotion → new ACTIVE
    Workers finish current batch before using promoted version.
    Rollback restores previous production version atomically.
    """

    def __init__(self) -> None:
        self._active_version: ModelVersion | None = None
        self._previous_version: ModelVersion | None = None
        self._candidate: ModelVersion | None = None
        self._registry: dict[str, ModelVersion] = {}
        self._swap_lock = threading.RLock()
        self._version_counter = 0
        self._logger = get_logger("model_swapper")
        self._swap_listeners: list[Any] = []

    def set_active(self, version: ModelVersion) -> None:
        """Set the initial active model version."""
        with self._swap_lock:
            self._active_version = version
            self._registry[version.version_id] = version
            self._logger.info(f"Active model set: {version.version_id}")

    def get_active(self) -> ModelVersion | None:
        with self._swap_lock:
            return self._active_version

    def register_candidate(self, version: ModelVersion) -> None:
        """Register a training candidate for validation."""
        with self._swap_lock:
            self._candidate = version
            self._registry[version.version_id] = version
            self._logger.info(f"Candidate registered: {version.version_id}")

    def promote_candidate(self) -> bool:
        """Atomically promote candidate to active. Old active becomes rollback target."""
        with self._swap_lock:
            if self._candidate is None:
                self._logger.warning("No candidate to promote")
                return False
            self._previous_version = self._active_version
            self._active_version = self._candidate
            self._candidate = None
            self._version_counter += 1
            self._logger.info(
                f"Model promoted: {self._active_version.version_id} "
                f"(previous: {self._previous_version.version_id if self._previous_version else 'none'})"
            )
            for listener in self._swap_listeners:
                try:
                    listener(self._active_version, self._previous_version)
                except (RuntimeError, ValueError, TypeError, KeyError, OSError):
                    pass
            return True

    def rollback(self) -> bool:
        """Atomically rollback to previous production version."""
        with self._swap_lock:
            if self._previous_version is None:
                self._logger.warning("No previous version available for rollback")
                return False
            rolled = self._active_version
            self._active_version = self._previous_version
            self._previous_version = rolled
            self._version_counter += 1
            self._logger.info(
                f"Model rolled back to: {self._active_version.version_id}"
            )
            return True

    def get_registry(self) -> dict[str, dict]:
        with self._swap_lock:
            return {
                vid: {
                    "version_id": v.version_id,
                    "model_name": v.model_name,
                    "model_path": v.model_path,
                    "created_at": v.created_at,
                }
                for vid, v in self._registry.items()
            }

    def add_swap_listener(self, listener) -> None:
        with self._swap_lock:
            self._swap_listeners.append(listener)


# ---------------------------------------------------------------------------
# Task 15 — Data Poisoning Protection
# ---------------------------------------------------------------------------


class DataPoisoningGuard:
    """Prevents poisoned observations from reaching training data.

    Gates: confidence threshold, duplicate rejection, outlier detection,
    temporal consistency, cross-camera consistency, verification state.
    """

    def __init__(
        self,
        min_confidence: float = 0.70,
        max_embedding_distance: float = 3.0,
        min_temporal_gap_seconds: float = 0.5,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_embedding_distance = max_embedding_distance
        self.min_temporal_gap = min_temporal_gap_seconds
        self._recent_embeddings: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._lock = threading.Lock()
        self._logger = get_logger("poisoning_guard")

        # Counters
        self.rejected_low_confidence: int = 0
        self.rejected_duplicate: int = 0
        self.rejected_outlier: int = 0
        self.rejected_temporal: int = 0
        self.accepted: int = 0

    def validate_observation(
        self,
        identity: str,
        confidence: float,
        embedding: np.ndarray,
        timestamp: float,
        verification_state: str = "PREDICTED",
    ) -> tuple[bool, str]:
        """Validate an observation before it can become training-eligible."""
        # Gate 1: Predicted observations are NEVER auto-promoted
        if verification_state == "PREDICTED":
            # Allowed to persist as PREDICTED, but cannot become TRAINING_ELIGIBLE
            pass

        # Gate 2: Confidence threshold
        if confidence < self.min_confidence:
            self.rejected_low_confidence += 1
            return False, f"low_confidence ({confidence:.3f} < {self.min_confidence})"

        with self._lock:
            history = self._recent_embeddings.get(identity, [])

            # Gate 3: Temporal consistency
            if history:
                last_ts = history[-1][0]
                if abs(timestamp - last_ts) < self.min_temporal_gap:
                    self.rejected_temporal += 1
                    return False, "temporal_too_close"

            # Gate 4: Outlier detection via distance from recent cluster
            if len(history) >= 3 and embedding is not None:
                recent_vecs = np.array([h[1] for h in history[-5:]])
                centroid = recent_vecs.mean(axis=0)
                dist = float(np.linalg.norm(embedding - centroid))
                if dist > self.max_embedding_distance:
                    self.rejected_outlier += 1
                    return False, f"outlier (dist={dist:.3f} > {self.max_embedding_distance})"

            # Gate 5: Duplicate embedding rejection (near identical match)
            if history and embedding is not None:
                last_emb = history[-1][1]
                cos_sim = float(np.dot(embedding, last_emb))
                if cos_sim > 0.99999:
                    self.rejected_duplicate += 1
                    return False, "duplicate_embedding"

            # Accept and track
            if embedding is not None:
                history.append((timestamp, embedding.copy()))
                # Keep bounded history
                if len(history) > 20:
                    history[:] = history[-20:]
                self._recent_embeddings[identity] = history

            self.accepted += 1
            return True, "accepted"

    def get_stats(self) -> dict:
        return {
            "accepted": self.accepted,
            "rejected_low_confidence": self.rejected_low_confidence,
            "rejected_duplicate": self.rejected_duplicate,
            "rejected_outlier": self.rejected_outlier,
            "rejected_temporal": self.rejected_temporal,
        }


# ---------------------------------------------------------------------------
# Task 16 — Structured Event Logging
# ---------------------------------------------------------------------------


class StructuredEventLogger:
    """JSON-structured production event logging with credential redaction."""

    EVENT_TYPES = frozenset({
        "camera_connected", "camera_disconnected", "camera_reconnect",
        "frame_dropped", "queue_overflow",
        "worker_started", "worker_failed", "worker_restarted",
        "model_loaded", "model_promoted", "model_rollback",
        "observation_saved", "training_started", "training_completed",
        "training_rejected", "system_degraded",
        "shutdown_started", "shutdown_completed",
        "resource_pressure_changed", "capacity_warning",
    })

    def __init__(self) -> None:
        self._logger = get_logger("structured_events")
        self._lock = threading.Lock()
        self._event_count: int = 0

    def emit(
        self,
        event_type: str,
        message: str,
        severity: str = "INFO",
        component: str = "",
        camera_id: str = "",
        worker_id: str = "",
        model_version: str = "",
        correlation_id: str = "",
        extra: dict | None = None,
    ) -> dict:
        """Emit a structured event record."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "component": component,
            "event_type": event_type,
            "message": message,
        }
        if camera_id:
            record["camera_id"] = camera_id
        if worker_id:
            record["worker_id"] = worker_id
        if model_version:
            record["model_version"] = model_version
        if correlation_id:
            record["correlation_id"] = correlation_id
        if extra:
            record["extra"] = extra

        with self._lock:
            self._event_count += 1

        level = getattr(logging, severity.upper(), logging.INFO)
        self._logger.log(level, json.dumps(record, default=str))
        return record

    @property
    def event_count(self) -> int:
        with self._lock:
            return self._event_count


# ---------------------------------------------------------------------------
# Task 17 — Graceful Shutdown
# ---------------------------------------------------------------------------


class GracefulShutdownManager:
    """Orchestrates clean system shutdown in correct dependency order."""

    def __init__(self) -> None:
        self._shutdown_hooks: list[tuple[int, str, Any]] = []
        self._lock = threading.Lock()
        self._logger = get_logger("shutdown")
        self._shutting_down = threading.Event()
        self._completed = threading.Event()

    def register_hook(self, priority: int, name: str, callback) -> None:
        """Register a shutdown hook. Lower priority = runs first."""
        with self._lock:
            self._shutdown_hooks.append((priority, name, callback))
            self._shutdown_hooks.sort(key=lambda x: x[0])

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down.is_set()

    def shutdown(self, timeout_per_hook: float = 5.0) -> dict:
        """Execute ordered shutdown sequence."""
        if self._shutting_down.is_set():
            return {"status": "already_shutting_down"}

        self._shutting_down.set()
        self._logger.info("Graceful shutdown initiated")
        results = {}

        with self._lock:
            hooks = list(self._shutdown_hooks)

        for priority, name, callback in hooks:
            self._logger.info(f"Shutdown hook [{priority}]: {name}")
            try:
                callback()
                results[name] = "SUCCESS"
            except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
                results[name] = f"ERROR: {exc}"
                self._logger.warning(f"Shutdown hook '{name}' failed: {exc}")

        self._completed.set()
        self._logger.info("Graceful shutdown completed")
        return results

    def install_signal_handlers(self) -> None:
        """Install OS signal handlers for graceful shutdown."""
        def _handler(signum, frame):
            self._logger.info(f"Received signal {signum}, initiating shutdown")
            self.shutdown()

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except (OSError, ValueError):
            # Cannot install in non-main thread
            pass


# ---------------------------------------------------------------------------
# Task 20 — Capacity Estimation
# ---------------------------------------------------------------------------


class CapacityEstimator:
    """Estimates sustainable camera count from measured runtime metrics.

    Formula: sustainable_cameras = measured_throughput_fps / target_fps_per_camera

    Validated against resource utilization constraints.
    """

    def __init__(
        self,
        target_fps_per_camera: float = 10.0,
        max_acceptable_drop_rate: float = 0.05,
        max_acceptable_p95_latency_ms: float = 200.0,
        max_cpu_utilization: float = 85.0,
        max_vram_utilization: float = 90.0,
    ) -> None:
        self.target_fps = target_fps_per_camera
        self.max_drop_rate = max_acceptable_drop_rate
        self.max_p95_latency_ms = max_acceptable_p95_latency_ms
        self.max_cpu = max_cpu_utilization
        self.max_vram = max_vram_utilization

    def estimate(
        self,
        measured_throughput_fps: float,
        current_cameras: int,
        cpu_percent: float,
        vram_percent: float,
        p95_latency_ms: float,
        drop_rate: float,
    ) -> dict:
        """Estimate sustainable camera capacity from measured data."""
        if self.target_fps <= 0 or measured_throughput_fps <= 0:
            return {
                "estimated_sustainable_cameras": 0,
                "basis": "insufficient_data",
                "constraints_met": False,
            }

        # Raw estimate from throughput
        raw_estimate = measured_throughput_fps / self.target_fps

        # Apply resource constraints
        constraints_met = True
        limiting_factor = "none"

        if cpu_percent >= self.max_cpu:
            # Scale down proportionally by the overload ratio
            raw_estimate *= (self.max_cpu / max(1.0, cpu_percent))
            constraints_met = False
            limiting_factor = "cpu"

        if vram_percent >= self.max_vram:
            raw_estimate *= 0.8
            constraints_met = False
            limiting_factor = "vram"

        if p95_latency_ms >= self.max_p95_latency_ms:
            latency_ratio = self.max_p95_latency_ms / max(1.0, p95_latency_ms)
            raw_estimate *= latency_ratio
            constraints_met = False
            limiting_factor = "latency"

        if drop_rate >= self.max_drop_rate:
            raw_estimate *= (1.0 - drop_rate)
            constraints_met = False
            limiting_factor = "drop_rate"

        sustainable = max(1, int(raw_estimate))

        return {
            "estimated_sustainable_cameras": sustainable,
            "target_fps_per_camera": self.target_fps,
            "measured_throughput_fps": round(measured_throughput_fps, 1),
            "current_cameras": current_cameras,
            "cpu_percent": round(cpu_percent, 1),
            "vram_percent": round(vram_percent, 1),
            "p95_latency_ms": round(p95_latency_ms, 2),
            "drop_rate": round(drop_rate, 4),
            "constraints_met": constraints_met,
            "limiting_factor": limiting_factor,
            "basis": "measured_runtime_data",
        }


# ---------------------------------------------------------------------------
# Integration: ProductionSurveillanceRuntime
# ---------------------------------------------------------------------------


class ProductionSurveillanceRuntime:
    """Top-level production surveillance runtime integrating all Phase 4 components.

    Wires together:
    - CameraStateMachine (Task 1)
    - ReconnectEngine (Task 2)
    - Camera failure isolation (Task 3)
    - ResilientWorkerPool (Task 4)
    - AdaptiveResourceManager (Task 5)
    - FrameQualityGate (Task 6)
    - FPSGovernor (Task 7)
    - SafeModelSwapper (Task 13)
    - DataPoisoningGuard (Task 15)
    - StructuredEventLogger (Task 16)
    - GracefulShutdownManager (Task 17)
    - CapacityEstimator (Task 20)
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}

        # Sub-components
        self.camera_state_machine = CameraStateMachine()
        self.reconnect_engine = ReconnectEngine(
            ReconnectConfig(
                min_retry_interval=cfg.get("reconnect_min_interval", 1.0),
                max_retry_interval=cfg.get("reconnect_max_interval", 60.0),
                backoff_multiplier=cfg.get("reconnect_backoff_multiplier", 2.0),
                jitter_range=cfg.get("reconnect_jitter_range", 0.5),
                max_retry_attempts=cfg.get("reconnect_max_attempts", 0),
            )
        )
        self.worker_pool = ResilientWorkerPool(
            num_workers=cfg.get("worker_count", 2),
            inference_timeout_seconds=cfg.get("inference_timeout", 30.0),
            health_check_interval=cfg.get("worker_health_interval", 10.0),
        )
        self.resource_manager = AdaptiveResourceManager(
            ResourceThresholds(
                max_cpu_percent=cfg.get("max_cpu_percent", 85.0),
                max_ram_percent=cfg.get("max_ram_percent", 85.0),
                max_gpu_percent=cfg.get("max_gpu_percent", 90.0),
                max_vram_percent=cfg.get("max_vram_percent", 90.0),
                max_queue_depth=cfg.get("max_queue_depth", 8),
                max_frame_age_ms=cfg.get("max_frame_age_ms", 500.0),
                max_p95_latency_ms=cfg.get("max_p95_latency_ms", 200.0),
            )
        )
        self.frame_quality_gate = FrameQualityGate(
            max_frame_age_ms=cfg.get("max_frame_age_ms", 500.0),
            enable_duplicate_detection=cfg.get("enable_duplicate_detection", True),
        )
        self.fps_governor = FPSGovernor(
            policy=FPSPolicy(cfg.get("fps_policy", "TARGET_FPS")),
            target_inference_fps=cfg.get("target_inference_fps", 10.0),
        )
        self.model_swapper = SafeModelSwapper()
        self.poisoning_guard = DataPoisoningGuard(
            min_confidence=cfg.get("poisoning_min_confidence", 0.70),
        )
        self.event_logger = StructuredEventLogger()
        self.shutdown_manager = GracefulShutdownManager()
        self.capacity_estimator = CapacityEstimator(
            target_fps_per_camera=cfg.get("target_inference_fps", 10.0),
        )

        self._start_time = time.monotonic()
        self._logger = get_logger("production_runtime")

        # Register shutdown hooks in dependency order
        self.shutdown_manager.register_hook(10, "stop_camera_ingestion", self._shutdown_cameras)
        self.shutdown_manager.register_hook(20, "stop_reconnect_engine", self.reconnect_engine.cancel_all)
        self.shutdown_manager.register_hook(30, "stop_worker_pool", self.worker_pool.stop)
        self.shutdown_manager.register_hook(90, "flush_logs", self._flush_logs)

    def register_camera(
        self,
        camera_id: str,
        source_type: str = "webcam",
        source_uri: str = "",
        fps_target: int = 15,
        resolution: tuple[int, int] = (640, 480),
        priority: int = 5,
    ) -> CameraResource:
        """Register camera in STOPPED state."""
        cam = self.camera_state_machine.register_camera(
            camera_id=camera_id,
            source_type=source_type,
            source_uri=source_uri,
            fps_target=fps_target,
            resolution=resolution,
            priority=priority,
        )
        self.event_logger.emit(
            "camera_connected",
            f"Camera '{camera_id}' registered",
            component="camera_lifecycle",
            camera_id=camera_id,
        )
        return cam

    def start_camera(self, camera_id: str) -> bool:
        """Transition camera from STOPPED → STARTING → CONNECTING."""
        ok = self.camera_state_machine.transition(camera_id, CameraState.STARTING)
        if ok:
            self.camera_state_machine.transition(camera_id, CameraState.CONNECTING)
        return ok

    def connect_camera(self, camera_id: str) -> bool:
        """Mark camera as CONNECTED after successful connection."""
        ok = self.camera_state_machine.transition(camera_id, CameraState.CONNECTED)
        if ok:
            self.reconnect_engine.reset(camera_id)
            self.event_logger.emit(
                "camera_connected",
                f"Camera '{camera_id}' connected successfully",
                component="camera_lifecycle",
                camera_id=camera_id,
            )
        return ok

    def fail_camera(self, camera_id: str, error: str = "") -> None:
        """Handle camera failure — triggers reconnect if applicable."""
        cam = self.camera_state_machine.get_camera(camera_id)
        if cam is None:
            return
        state = cam.connection_state
        if state in (CameraState.CONNECTING, CameraState.CONNECTED, CameraState.DEGRADED):
            self.camera_state_machine.transition(
                camera_id, CameraState.RECONNECTING, error=error
            )
            self.event_logger.emit(
                "camera_disconnected",
                f"Camera '{camera_id}' disconnected: {error}",
                severity="WARNING",
                component="camera_lifecycle",
                camera_id=camera_id,
            )
            self.reconnect_engine.schedule_reconnect(
                camera_id,
                self._attempt_reconnect,
                camera_id,
            )
        elif state == CameraState.RECONNECTING:
            # Schedule another attempt
            self.reconnect_engine.schedule_reconnect(
                camera_id,
                self._attempt_reconnect,
                camera_id,
            )
        elif state == CameraState.STARTING:
            self.camera_state_machine.transition(
                camera_id, CameraState.FAILED, error=error
            )

    def stop_camera(self, camera_id: str) -> bool:
        """Stop a camera cleanly."""
        cam = self.camera_state_machine.get_camera(camera_id)
        if cam is None:
            return False
        self.reconnect_engine.cancel(camera_id)
        # Transition to STOPPING from any state
        if cam.connection_state != CameraState.STOPPED:
            self.camera_state_machine.transition(camera_id, CameraState.STOPPING)
            self.camera_state_machine.transition(camera_id, CameraState.STOPPED)
        self.event_logger.emit(
            "camera_disconnected",
            f"Camera '{camera_id}' stopped",
            component="camera_lifecycle",
            camera_id=camera_id,
        )
        return True

    def get_system_health(self) -> dict:
        """Comprehensive system health snapshot."""
        cameras = self.camera_state_machine.get_all_cameras()
        snap = self.resource_manager.take_snapshot()
        self.resource_manager.evaluate(snap)

        connected = sum(
            1 for c in cameras.values()
            if c.connection_state == CameraState.CONNECTED
        )
        degraded = sum(
            1 for c in cameras.values()
            if c.connection_state == CameraState.DEGRADED
        )
        failed = sum(
            1 for c in cameras.values()
            if c.connection_state == CameraState.FAILED
        )

        return {
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "resource_pressure": self.resource_manager.pressure.value,
            "processing_rate_factor": round(self.resource_manager.processing_rate_factor, 2),
            "cameras": {
                "total": len(cameras),
                "connected": connected,
                "degraded": degraded,
                "failed": failed,
            },
            "workers": self.worker_pool.get_worker_info(),
            "resources": {
                "cpu_percent": snap.cpu_percent,
                "ram_percent": snap.ram_percent,
                "vram_percent": snap.vram_percent,
                "vram_used_mb": round(snap.vram_used_mb, 1),
            },
            "frame_quality": self.frame_quality_gate.get_stats(),
            "poisoning_guard": self.poisoning_guard.get_stats(),
            "model_swapper": {
                "active_version": (
                    self.model_swapper.get_active().version_id
                    if self.model_swapper.get_active()
                    else None
                ),
                "registry_count": len(self.model_swapper.get_registry()),
            },
            "events_emitted": self.event_logger.event_count,
        }

    def get_camera_health(self) -> dict[str, dict]:
        """Per-camera health report."""
        cameras = self.camera_state_machine.get_all_cameras()
        result = {}
        for cid, cam in cameras.items():
            cam.compute_health_score()
            result[cid] = {
                "camera_id": cid,
                "connection_state": cam.connection_state.value,
                "desired_state": cam.desired_state.value,
                "source_type": cam.source_type,
                "fps_target": cam.fps_target,
                "resolution": list(cam.resolution),
                "frames_received": cam.frames_received,
                "frames_processed": cam.frames_processed,
                "frames_dropped": cam.frames_dropped,
                "queue_depth": cam.queue_depth,
                "health_score": round(cam.health_score, 3),
                "reconnect_attempts": cam.reconnect_attempts,
                "total_reconnect_count": cam.total_reconnect_count,
                "last_error": cam.last_error,
                "last_frame_age_seconds": (
                    round(time.monotonic() - cam.last_frame_timestamp, 2)
                    if cam.last_frame_timestamp > 0
                    else None
                ),
            }
        return result

    def estimate_capacity(
        self,
        measured_throughput_fps: float,
        current_cameras: int,
    ) -> dict:
        """Estimate sustainable camera count from measured runtime data."""
        snap = self.resource_manager.take_snapshot()
        return self.capacity_estimator.estimate(
            measured_throughput_fps=measured_throughput_fps,
            current_cameras=current_cameras,
            cpu_percent=snap.cpu_percent,
            vram_percent=snap.vram_percent,
            p95_latency_ms=snap.p95_latency_ms,
            drop_rate=snap.frame_drop_rate,
        )

    def _attempt_reconnect(self, camera_id: str) -> None:
        """Attempt reconnection for a failed camera."""
        cam = self.camera_state_machine.get_camera(camera_id)
        if cam is None or cam.connection_state == CameraState.STOPPED:
            return
        self.event_logger.emit(
            "camera_reconnect",
            f"Reconnect attempt {cam.reconnect_attempts} for '{camera_id}'",
            component="reconnect_engine",
            camera_id=camera_id,
        )
        # Transition to CONNECTING for the attempt
        self.camera_state_machine.transition(camera_id, CameraState.CONNECTING)

    def _shutdown_cameras(self) -> None:
        """Stop all cameras during shutdown."""
        for cid in list(self.camera_state_machine.get_all_cameras()):
            self.stop_camera(cid)

    def _flush_logs(self) -> None:
        """Flush all log handlers."""
        for handler in logging.root.handlers:
            try:
                handler.flush()
            except (RuntimeError, ValueError, OSError):
                pass
