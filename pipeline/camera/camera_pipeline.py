import time
from threading import Lock
from typing import Any

import numpy as np

from core.logger import setup_logger


class CameraPipeline:
    def __init__(
        self,
        camera_id: str,
        detection_processor: Any | None = None,
        inference_pipeline: Any | None = None,
    ) -> None:
        self.camera_id = camera_id
        self._logger = setup_logger(f"pipeline.{camera_id}")

        self.detection_processor = detection_processor
        self.inference_pipeline = inference_pipeline

        self._lock = Lock()

        self.current_frame: np.ndarray | None = None
        self.current_detections: list[dict] = []
        self.current_tracks: dict = {}
        self.current_gei: np.ndarray | None = None

        self.stats = {
            "frames_processed": 0,
            "detections": 0,
            "tracks": 0,
            "gei_built": False,
            "recognitions": 0,
            "avg_detection_time_ms": 0.0,
            "avg_recognition_time_ms": 0.0,
            "last_update": time.monotonic(),
        }

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        frame_start = time.perf_counter()
        results = {
            "camera_id": self.camera_id,
            "frame": frame,
            "detections": [],
            "tracks": [],
            "gei": None,
            "recognitions": [],
            "latency_ms": 0.0,
        }

        try:
            with self._lock:
                self.current_frame = frame.copy()

            if self.detection_processor is not None:
                det_start = time.perf_counter()
                detections = self.detection_processor.detect(frame)
                det_time = (time.perf_counter() - det_start) * 1000.0

                with self._lock:
                    self.current_detections = detections
                    stats = self.stats
                    stats["detections"] = len(detections)

                    if stats["avg_detection_time_ms"] == 0.0:
                        stats["avg_detection_time_ms"] = det_time
                    else:
                        stats["avg_detection_time_ms"] = 0.9 * stats["avg_detection_time_ms"] + 0.1 * det_time

                results["detections"] = detections

            if self.inference_pipeline is not None and results["detections"]:
                rec_start = time.perf_counter()

                for detection in results["detections"]:
                    try:
                        identity = detection.get("id", "UNKNOWN")
                        score = detection.get("confidence", 0.0)

                        results["recognitions"].append(
                            {
                                "detection_id": detection.get("id"),
                                "identity": identity,
                                "score": score,
                            }
                        )

                    except (KeyError, ValueError, TypeError) as e:
                        self._logger.error(f"Recognition error: {e!s}")

                rec_time = (time.perf_counter() - rec_start) * 1000.0

                with self._lock:
                    stats = self.stats
                    stats["recognitions"] += len(results["recognitions"])

                    if stats["avg_recognition_time_ms"] == 0.0:
                        stats["avg_recognition_time_ms"] = rec_time
                    else:
                        stats["avg_recognition_time_ms"] = 0.9 * stats["avg_recognition_time_ms"] + 0.1 * rec_time

            frame_time = (time.perf_counter() - frame_start) * 1000.0
            results["latency_ms"] = frame_time

            with self._lock:
                stats = self.stats
                stats["frames_processed"] += 1
                stats["last_update"] = time.monotonic()

        except (RuntimeError, ValueError, TypeError, OSError) as e:
            self._logger.error(f"Pipeline error: {e!s}")

        return results

    def get_current_frame(self) -> np.ndarray | None:
        with self._lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_current_detections(self) -> list[dict]:
        with self._lock:
            return self.current_detections.copy()

    def get_current_gei(self) -> np.ndarray | None:
        with self._lock:
            return self.current_gei.copy() if self.current_gei is not None else None

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return self.stats.copy()
