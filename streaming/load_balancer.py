"""
Camera Load Balancer for Phase 4 multi-camera workload distribution.

Distributes cameras across worker nodes/threads, performs automatic rebalancing
when overload is detected, and supports dynamic worker migration.
"""

from threading import Lock
from typing import Any, Dict, List, Optional

from monitoring.logging_config import get_logger


class CameraLoadBalancer:
    """Dynamic camera load balancer and migration engine."""

    def __init__(
        self,
        imbalance_threshold: float = 0.3,
        max_cameras_per_worker: int = 10,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.max_cameras_per_worker = max_cameras_per_worker

        self._logger = get_logger("load_balancer")
        self._lock = Lock()

        # worker_id -> list of camera_ids
        self._worker_assignments: Dict[str, List[str]] = {}
        # camera_id -> estimated weight / load
        self._camera_weights: Dict[str, float] = {}

    def register_worker(self, worker_id: str) -> None:
        """Register a worker to handle camera streams."""
        with self._lock:
            if worker_id not in self._worker_assignments:
                self._worker_assignments[worker_id] = []

    def unregister_worker(self, worker_id: str) -> List[str]:
        """Unregister a worker and return orphaned cameras for re-assignment."""
        with self._lock:
            orphaned = self._worker_assignments.pop(worker_id, [])
            return orphaned

    def set_camera_weight(self, camera_id: str, fps: float = 15.0, width: int = 640, height: int = 480) -> None:
        """Set estimated workload weight for a camera based on resolution and FPS."""
        with self._lock:
            # Normalized pixel-rate weight
            weight = (width * height * fps) / (640 * 480 * 15)
            self._camera_weights[camera_id] = max(0.1, weight)

    def assign_camera(self, camera_id: str, fps: float = 15.0, width: int = 640, height: int = 480) -> Optional[str]:
        """Assign a new camera to the least loaded worker."""
        self.set_camera_weight(camera_id, fps, width, height)

        with self._lock:
            if not self._worker_assignments:
                return None

            best_worker = min(
                self._worker_assignments.keys(),
                key=lambda w: (
                    len(self._worker_assignments[w]) >= self.max_cameras_per_worker,
                    self._get_worker_load_unlocked(w),
                ),
            )

            if len(self._worker_assignments[best_worker]) >= self.max_cameras_per_worker:
                self._logger.warning("All workers at max camera capacity!")

            self._worker_assignments[best_worker].append(camera_id)
            self._logger.info(f"Assigned camera {camera_id} to worker {best_worker}")
            return best_worker

    def unassign_camera(self, camera_id: str) -> Optional[str]:
        """Remove a camera assignment."""
        with self._lock:
            self._camera_weights.pop(camera_id, None)
            for worker_id, cameras in self._worker_assignments.items():
                if camera_id in cameras:
                    cameras.remove(camera_id)
                    return worker_id
            return None

    def migrate_camera(self, camera_id: str, target_worker_id: str) -> bool:
        """Migrate a camera explicitly to a target worker."""
        with self._lock:
            if target_worker_id not in self._worker_assignments:
                return False

            old_worker = None
            for w_id, cameras in self._worker_assignments.items():
                if camera_id in cameras:
                    old_worker = w_id
                    cameras.remove(camera_id)
                    break

            self._worker_assignments[target_worker_id].append(camera_id)
            self._logger.info(f"Migrated camera {camera_id} from {old_worker} to {target_worker_id}")
            return True

    def check_and_rebalance(self) -> Dict[str, str]:
        """Check for workload imbalance and generate camera migrations {camera_id: new_worker_id}."""
        with self._lock:
            if len(self._worker_assignments) < 2:
                return {}

            loads = {w: self._get_worker_load_unlocked(w) for w in self._worker_assignments}
            max_w = max(loads, key=loads.get)
            min_w = min(loads, key=loads.get)

            if loads[max_w] - loads[min_w] < self.imbalance_threshold:
                return {}

            # Migrate one camera from max_w to min_w if possible
            max_cams = self._worker_assignments[max_w]
            if len(max_cams) <= 1:
                return {}

            cam_to_move = max_cams[-1]
            max_cams.remove(cam_to_move)
            self._worker_assignments[min_w].append(cam_to_move)

            self._logger.info(f"Auto-rebalanced: moved camera {cam_to_move} from {max_w} to {min_w}")
            return {cam_to_move: min_w}

    def _get_worker_load_unlocked(self, worker_id: str) -> float:
        cameras = self._worker_assignments.get(worker_id, [])
        return sum(self._camera_weights.get(cid, 1.0) for cid in cameras)

    def get_assignment_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            return {
                "workers_count": len(self._worker_assignments),
                "assignments": {w: cams.copy() for w, cams in self._worker_assignments.items()},
                "worker_loads": {w: self._get_worker_load_unlocked(w) for w in self._worker_assignments},
            }
