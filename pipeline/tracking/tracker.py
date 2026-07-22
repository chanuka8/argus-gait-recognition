import threading
import time

import numpy as np
import supervision as sv

from monitoring.logging_config import get_logger


class PersonTracker:
    def __init__(self) -> None:
        self.logger = get_logger("detection")
        self.lock = threading.Lock()
        self.tracker = sv.ByteTrack()
        self.last_seen: dict[int, float] = {}

    def update(self, detections_list: list[dict], frame_shape: tuple[int, ...]) -> list[dict]:
        if not detections_list:
            sv_detections = sv.Detections.empty()
        else:
            xyxy = np.array([d["bbox"] for d in detections_list], dtype=np.float32)
            confidence = np.array([d["confidence"] for d in detections_list], dtype=np.float32)
            class_id = np.zeros(len(detections_list), dtype=int)

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

        with self.lock:
            tracked_detections = self.tracker.update_with_detections(sv_detections)

        now = time.monotonic()
        results = []

        if tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections.tracker_id)):
                track_id = int(tracked_detections.tracker_id[i])
                bbox = tracked_detections.xyxy[i].astype(int).tolist()

                self.last_seen[track_id] = now

                results.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "timestamp": now,
                })

        return results

    def cleanup_inactive(self, max_idle_seconds: float = 5.0) -> list[int]:
        now = time.monotonic()
        inactive_ids = []

        with self.lock:
            for track_id, last_ts in list(self.last_seen.items()):
                if now - last_ts > max_idle_seconds:
                    inactive_ids.append(track_id)
                    self.last_seen.pop(track_id, None)

        return inactive_ids
