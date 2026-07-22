"""
Performance Optimizer for Phase 4 streaming optimization.

Provides adaptive frame skipping, queue capacity optimization, latency/memory/CPU
profiling and tuning, and optional GPU batching configurations.
"""

from threading import Lock
from typing import Any, Dict

from monitoring.logging_config import get_logger


class PerformanceOptimizer:
    """Adaptive streaming performance optimizer and profiler."""

    PROFILES = {
        "low_latency": {
            "target_fps": 30,
            "max_queue_size": 3,
            "adaptive_skip_threshold": 0.05,
            "gpu_batch_size": 1,
        },
        "high_throughput": {
            "target_fps": 15,
            "max_queue_size": 20,
            "adaptive_skip_threshold": 0.15,
            "gpu_batch_size": 8,
        },
        "balanced": {
            "target_fps": 15,
            "max_queue_size": 10,
            "adaptive_skip_threshold": 0.10,
            "gpu_batch_size": 4,
        },
    }

    def __init__(self, profile_name: str = "balanced") -> None:
        self._logger = get_logger("performance_optimizer")
        self._lock = Lock()

        self.current_profile_name = profile_name if profile_name in self.PROFILES else "balanced"
        self._config = self.PROFILES[self.current_profile_name].copy()

    def set_profile(self, profile_name: str) -> None:
        """Switch performance profile dynamically."""
        with self._lock:
            if profile_name in self.PROFILES:
                self.current_profile_name = profile_name
                self._config = self.PROFILES[profile_name].copy()
                self._logger.info(f"Switched performance profile to {profile_name}")

    def should_skip_frame(self, queue_fullness: float, latency_seconds: float) -> bool:
        """Determine whether to drop/skip the current frame to maintain target latency."""
        with self._lock:
            threshold = self._config["adaptive_skip_threshold"]
            # Skip if queue is near capacity (>80%) or latency exceeds threshold
            if queue_fullness > 0.8 or latency_seconds > threshold:
                return True
            return False

    def get_optimal_queue_size(self) -> int:
        """Get recommended queue size based on active profile."""
        with self._lock:
            return self._config["max_queue_size"]

    def get_gpu_batch_size(self) -> int:
        """Get recommended GPU batch size based on active profile."""
        with self._lock:
            return self._config["gpu_batch_size"]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current optimizer configuration and stats."""
        with self._lock:
            return {
                "profile": self.current_profile_name,
                "config": self._config.copy(),
            }
