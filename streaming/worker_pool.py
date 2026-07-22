"""
Camera Worker Pool for Phase 4 multi-camera processing.

Manages a pool of camera worker instances with dynamic scaling, thread safety,
automatic recovery on failure, graceful shutdown, and health reporting.
"""

from threading import Lock
from typing import Any, Dict, List, Optional

from monitoring.logging_config import get_logger
from services.camera_worker import CameraWorker


class CameraWorkerPool:
    """Thread-safe worker pool for scaling camera ingestion & processing."""

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 16,
        inference_pipeline: Optional[Any] = None,
        detection_processor: Optional[Any] = None,
    ) -> None:
        self.min_workers = max(1, min_workers)
        self.max_workers = max(self.min_workers, max_workers)
        self._inference_pipeline = inference_pipeline
        self._detection_processor = detection_processor

        self._logger = get_logger("worker_pool")
        self._lock = Lock()

        self._workers: Dict[str, CameraWorker] = {}
        self._worker_health: Dict[str, bool] = {}

    def add_worker_for_camera(self, camera_id: str, camera_config: dict) -> Optional[CameraWorker]:
        """Create and register a new camera worker in the pool."""
        with self._lock:
            if len(self._workers) >= self.max_workers:
                self._logger.warning(f"Worker pool max limit ({self.max_workers}) reached!")
                return None

            if camera_id in self._workers:
                return self._workers[camera_id]

            config = camera_config.copy()
            config["id"] = camera_id

            worker = CameraWorker(
                camera_id=camera_id,
                camera_config=config,
                inference_pipeline=self._inference_pipeline,
                detection_processor=self._detection_processor,
            )

            self._workers[camera_id] = worker
            self._worker_health[camera_id] = True
            self._logger.info(f"Worker added to pool for camera {camera_id}")
            return worker

    def remove_worker(self, camera_id: str, timeout: float = 5.0) -> bool:
        """Remove and gracefully stop a worker."""
        with self._lock:
            if camera_id not in self._workers:
                return False

            worker = self._workers.pop(camera_id)
            self._worker_health.pop(camera_id, None)

        stopped = worker.stop(timeout=timeout)
        self._logger.info(f"Worker removed for camera {camera_id}")
        return stopped

    def start_worker(self, camera_id: str) -> bool:
        """Start a specific camera worker in the pool."""
        with self._lock:
            worker = self._workers.get(camera_id)
            if not worker:
                return False
            return worker.start()

    def start_all(self) -> int:
        """Start all registered workers."""
        started = 0
        with self._lock:
            workers = list(self._workers.values())

        for w in workers:
            if not w.is_running() and w.start():
                started += 1
        return started

    def stop_all(self, timeout: float = 5.0) -> int:
        """Gracefully stop all workers in the pool."""
        stopped = 0
        with self._lock:
            workers = list(self._workers.values())

        for w in workers:
            if w.stop(timeout=timeout):
                stopped += 1
        return stopped

    def auto_recover_failed_workers(self) -> List[str]:
        """Perform health checks and automatically restart crashed workers."""
        recovered = []
        with self._lock:
            workers = list(self._workers.items())

        for camera_id, worker in workers:
            if not worker.is_running():
                self._logger.warning(f"Worker pool detected crashed worker for {camera_id}. Recovering...")
                if worker.restart():
                    recovered.append(camera_id)
                    with self._lock:
                        self._worker_health[camera_id] = True
                else:
                    with self._lock:
                        self._worker_health[camera_id] = False

        return recovered

    def get_pool_health(self) -> Dict[str, Any]:
        """Get pool health status and statistics."""
        with self._lock:
            active_count = sum(1 for w in self._workers.values() if w.is_running())
            return {
                "total_workers": len(self._workers),
                "active_workers": active_count,
                "max_workers": self.max_workers,
                "min_workers": self.min_workers,
                "health_map": self._worker_health.copy(),
            }

    def get_worker(self, camera_id: str) -> Optional[CameraWorker]:
        """Retrieve worker for a camera."""
        with self._lock:
            return self._workers.get(camera_id)
