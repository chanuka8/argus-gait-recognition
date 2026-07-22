"""
Camera Scheduler for Phase 4 streaming optimization.

Provides intelligent, priority-weighted frame scheduling, fair queue scheduling,
starvation prevention via aging, and dynamic polling interval adjustment.
"""

import time
from threading import Lock
from typing import Any, Dict, List, Optional

from monitoring.logging_config import get_logger


class CameraScheduler:
    """Intelligent priority and fair-share camera frame scheduler."""

    def __init__(
        self,
        default_priority: int = 5,
        min_poll_interval: float = 0.005,
        max_poll_interval: float = 0.05,
        starvation_threshold_seconds: float = 2.0,
    ) -> None:
        self.default_priority = default_priority
        self.min_poll_interval = min_poll_interval
        self.max_poll_interval = max_poll_interval
        self.starvation_threshold = starvation_threshold_seconds

        self._logger = get_logger("camera_scheduler")
        self._lock = Lock()

        self._camera_priorities: Dict[str, int] = {}
        self._last_scheduled: Dict[str, float] = {}
        self._scheduled_counts: Dict[str, int] = {}

    def register_camera(self, camera_id: str, priority: Optional[int] = None) -> None:
        """Register or update a camera's schedule priority (1 to 10)."""
        with self._lock:
            p = priority if priority is not None else self.default_priority
            self._camera_priorities[camera_id] = max(1, min(10, p))
            if camera_id not in self._last_scheduled:
                self._last_scheduled[camera_id] = time.monotonic()
                self._scheduled_counts[camera_id] = 0

    def unregister_camera(self, camera_id: str) -> None:
        """Unregister a camera from scheduling."""
        with self._lock:
            self._camera_priorities.pop(camera_id, None)
            self._last_scheduled.pop(camera_id, None)
            self._scheduled_counts.pop(camera_id, None)

    def set_priority(self, camera_id: str, priority: int) -> None:
        """Dynamically set camera priority."""
        self.register_camera(camera_id, priority)

    def get_priority(self, camera_id: str) -> int:
        """Get priority for a camera."""
        with self._lock:
            return self._camera_priorities.get(camera_id, self.default_priority)

    def get_next_camera(self, active_camera_ids: List[str]) -> Optional[str]:
        """Select next camera to process based on priority and starvation prevention."""
        if not active_camera_ids:
            return None

        now = time.monotonic()
        with self._lock:
            best_camera: Optional[str] = None
            highest_score = -1.0

            for cid in active_camera_ids:
                p = self._camera_priorities.get(cid, self.default_priority)
                last_t = self._last_scheduled.get(cid, now)
                wait_time = now - last_t

                # Starvation boost: lower priority cameras gain higher boost per second of waiting to prevent starvation
                starvation_boost = (
                    (wait_time / self.starvation_threshold) * (11 - p) * 2.0
                    if wait_time > self.starvation_threshold
                    else 0.0
                )
                score = p + starvation_boost

                if score > highest_score:
                    highest_score = score
                    best_camera = cid

            if best_camera:
                self._last_scheduled[best_camera] = now
                self._scheduled_counts[best_camera] = (
                    self._scheduled_counts.get(best_camera, 0) + 1
                )

            return best_camera

    def calculate_dynamic_poll_interval(self, system_load_factor: float = 0.5) -> float:
        """Calculate dynamic polling interval based on current system load (0.0 to 1.0)."""
        load = max(0.0, min(1.0, system_load_factor))
        # Higher load -> slightly longer poll interval to avoid CPU thrashing
        interval = self.min_poll_interval + load * (
            self.max_poll_interval - self.min_poll_interval
        )
        return round(interval, 4)

    def get_schedule_stats(self) -> Dict[str, Any]:
        """Get scheduler performance statistics."""
        with self._lock:
            return {
                "priorities": self._camera_priorities.copy(),
                "scheduled_counts": self._scheduled_counts.copy(),
                "registered_cameras_count": len(self._camera_priorities),
            }
