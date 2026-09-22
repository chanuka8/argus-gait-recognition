import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Self

import numpy as np


@dataclass
class ProfilingSummary:
    enabled: bool
    window_seconds: float
    fps: float
    metrics: dict[str, dict[str, float]]


class PerformanceProfiler:
    """Lightweight, thread-safe production performance profiler.

    Near-zero overhead when disabled. Collects aggregated statistics in RAM when enabled.
    Zero per-frame disk logging.
    """

    _instance: "PerformanceProfiler | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, window_size: int = 100) -> None:
        self.enabled = os.environ.get("ARGUS_PROFILING_ENABLED", "0").lower() in ("1", "true", "yes")
        self.window_size = max(10, int(window_size))
        self._lock = threading.Lock()
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._last_report_time = time.monotonic()

    @classmethod
    def get_instance(cls) -> "PerformanceProfiler":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def record_metric(self, name: str, duration_ms: float) -> None:
        """Record timing sample. If disabled, exits immediately with minimal overhead."""
        if not self.enabled:
            return
        with self._lock:
            self._samples[name].append(duration_ms)

    def record_frame(self) -> None:
        """Increment frame counter."""
        if not self.enabled:
            return
        with self._lock:
            self._frame_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Produce statistical summary of recorded metrics."""
        if not self.enabled:
            return {"enabled": False, "status": "PROFILING_DISABLED"}

        now = time.monotonic()
        with self._lock:
            elapsed = max(0.001, now - self._start_time)
            fps = round(self._frame_count / elapsed, 2)
            summary: dict[str, Any] = {
                "enabled": True,
                "uptime_sec": round(elapsed, 2),
                "fps": fps,
                "metrics": {},
            }
            for name, samples in self._samples.items():
                if not samples:
                    continue
                arr = np.array(samples, dtype=np.float64)
                summary["metrics"][name] = {
                    "count": len(samples),
                    "mean_ms": round(float(np.mean(arr)), 3),
                    "p50_ms": round(float(np.median(arr)), 3),
                    "p95_ms": round(float(np.percentile(arr, 95)), 3),
                    "min_ms": round(float(np.min(arr)), 3),
                    "max_ms": round(float(np.max(arr)), 3),
                }
            return summary


class TimedSection:
    """Zero-overhead context manager when profiler is disabled."""

    __slots__ = ("_name", "_profiler", "_t0")

    def __init__(self, name: str) -> None:
        self._name = name
        self._profiler = PerformanceProfiler.get_instance()
        self._t0 = 0.0

    def __enter__(self) -> Self:
        if self._profiler.enabled:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._profiler.enabled and self._t0 > 0.0:
            duration_ms = (time.perf_counter() - self._t0) * 1000.0
            self._profiler.record_metric(self._name, duration_ms)
