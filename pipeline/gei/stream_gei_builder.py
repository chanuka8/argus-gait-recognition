import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from monitoring.logging_config import get_logger


class StreamGEIBuilder:
    def __init__(self, config_path: str = "configs/gei.yaml") -> None:
        self.logger = get_logger("detection")
        self.config = self._load_config(config_path)
        self.lock = threading.Lock()

        self.max_frames = int(self.config.get("max_frames", 15))
        self.min_frames = int(self.config.get("min_frames", 10))
        target_size_cfg = self.config.get("target_size", [64, 128])
        self.target_size = (int(target_size_cfg[0]), int(target_size_cfg[1]))

        self.track_buffers: dict[int, list[np.ndarray]] = {}
        self.last_updated: dict[int, float] = {}

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        defaults = {
            "max_frames": 15,
            "min_frames": 10,
            "target_size": [64, 128],
        }

        if not path.exists():
            return defaults

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
                for key, val in defaults.items():
                    data.setdefault(key, val)
                return data
        except Exception:
            return defaults

    def add_silhouette(self, track_id: int, silhouette: np.ndarray | None) -> None:
        if silhouette is None or silhouette.size == 0:
            return

        resized = cv2.resize(silhouette, self.target_size)
        normalized = (resized > 0).astype(np.float32)

        now = time.monotonic()

        with self.lock:
            if track_id not in self.track_buffers:
                self.track_buffers[track_id] = []

            buffer = self.track_buffers[track_id]
            buffer.append(normalized)

            if len(buffer) > self.max_frames:
                buffer.pop(0)

            self.last_updated[track_id] = now

    def is_ready(self, track_id: int) -> bool:
        with self.lock:
            buffer = self.track_buffers.get(track_id, [])
            return len(buffer) >= self.min_frames

    def build_gei(self, track_id: int) -> np.ndarray | None:
        with self.lock:
            buffer = self.track_buffers.get(track_id, [])

            if len(buffer) < self.min_frames:
                return None

            gei_mean = np.mean(buffer, axis=0)
            gei_uint8 = (gei_mean * 255.0).astype(np.uint8)

            return gei_uint8

    def get_frame_count(self, track_id: int) -> int:
        with self.lock:
            return len(self.track_buffers.get(track_id, []))

    def clear_track(self, track_id: int) -> None:
        with self.lock:
            self.track_buffers.pop(track_id, None)
            self.last_updated.pop(track_id, None)

    def cleanup_inactive(self, max_idle_seconds: float = 10.0) -> list[int]:
        now = time.monotonic()
        removed = []

        with self.lock:
            for track_id, last_ts in list(self.last_updated.items()):
                if now - last_ts > max_idle_seconds:
                    self.track_buffers.pop(track_id, None)
                    self.last_updated.pop(track_id, None)
                    removed.append(track_id)

        return removed
