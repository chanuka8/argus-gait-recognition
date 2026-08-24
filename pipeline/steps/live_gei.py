from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


class LiveGEI:
    """
    Gait-Cycle-Aware Live GEI Rolling Buffer.

    Features:
      - Consecutive duplicate frame rejection (via silhouette IoU).
      - Autocorrelation-based gait cycle detection (silhouette width periodicity).
      - Cycle-aware frame slice aggregation when cycle detected.
      - Graceful fallback to standard rolling mean when no cycle is detected.
    """

    def __init__(
        self,
        max_frames: int = 15,
        min_frames: int | None = None,
        size: tuple[int, int] = (64, 128),
        cycle_detection_enabled: bool = False,
        duplicate_filter_enabled: bool | None = None,
        min_cycle_frames: int = 6,
        max_cycle_frames: int = 24,
        cycle_confidence_threshold: float = 0.35,
        duplicate_threshold: float = 0.98,
        config_path: str | None = None,
    ) -> None:
        self.config = self._load_config(Path(config_path)) if config_path else {}

        self.max_frames = int(self.config.get("max_frames", max_frames))
        raw_min = min_frames if min_frames is not None else self.config.get("min_frames", 10)
        self.min_frames = max(1, min(self.max_frames, int(raw_min)))
        self.size = size

        if "cycle_detection_enabled" in self.config:
            self.cycle_detection_enabled = bool(self.config["cycle_detection_enabled"])
        else:
            self.cycle_detection_enabled = cycle_detection_enabled

        if duplicate_filter_enabled is not None:
            self.duplicate_filter_enabled = bool(duplicate_filter_enabled)
        elif "duplicate_filter_enabled" in self.config:
            self.duplicate_filter_enabled = bool(self.config["duplicate_filter_enabled"])
        else:
            self.duplicate_filter_enabled = self.cycle_detection_enabled

        self.min_cycle_frames = int(self.config.get("min_cycle_frames", min_cycle_frames))
        self.max_cycle_frames = int(self.config.get("max_cycle_frames", max_cycle_frames))
        self.cycle_confidence_threshold = float(self.config.get("cycle_confidence_threshold", cycle_confidence_threshold))
        self.duplicate_threshold = float(self.config.get("duplicate_threshold", duplicate_threshold))

        self.cycle_history_window = int(self.config.get("cycle_history_window", 30))
        self.history_capacity = max(self.max_frames, self.cycle_history_window) if self.cycle_detection_enabled else self.max_frames

        self.frames: list[np.ndarray] = []
        self.width_signals: list[float] = []
        self.valid_frames = 0
        self.rejected_frames = 0
        self.duplicate_frames = 0
        self.last_cycle_detected: int | None = None


    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists() or yaml is None:
            return {}
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError, AttributeError):
            return {}

    def _is_duplicate(self, binary_frame: np.ndarray) -> bool:
        if not self.duplicate_filter_enabled or not self.frames:
            return False
        prev = self.frames[-1]
        intersection = np.logical_and(prev > 0, binary_frame > 0).sum()
        union = np.logical_or(prev > 0, binary_frame > 0).sum()
        if union == 0:
            return True
        iou = float(intersection / union)
        return iou >= self.duplicate_threshold

    def add(self, silhouette: np.ndarray | None) -> None:
        if silhouette is None or getattr(silhouette, "size", 0) == 0:
            self.rejected_frames += 1
            return

        frame = cv2.resize(silhouette, self.size)
        binary_frame = (frame > 0).astype(np.float32)

        if self._is_duplicate(binary_frame):
            self.duplicate_frames += 1
            return

        non_zero_cols = np.where(binary_frame.sum(axis=0) > 0)[0]
        width_val = float(non_zero_cols[-1] - non_zero_cols[0] + 1) if len(non_zero_cols) > 0 else 0.0

        self.frames.append(binary_frame)
        self.width_signals.append(width_val)
        self.valid_frames += 1

        if len(self.frames) > self.history_capacity:
            self.frames.pop(0)
            self.width_signals.pop(0)

    def ready(self) -> bool:
        return len(self.frames) >= self.min_frames

    def detect_gait_cycle(self) -> int | None:
        """
        Estimate gait cycle period using normalized autocorrelation on silhouette width signal.
        Returns estimated cycle length in frames, or None if unconfident.
        """
        if not self.cycle_detection_enabled or len(self.width_signals) < self.min_cycle_frames * 2:
            return None

        signal = np.array(self.width_signals, dtype=np.float32)
        signal_mean = np.mean(signal)
        signal_norm = signal - signal_mean

        variance = float(np.sum(signal_norm ** 2))
        if variance < 1e-5:
            return None

        n = len(signal_norm)
        autocorr = []
        max_lag = min(self.max_cycle_frames, n // 2)

        for lag in range(self.min_cycle_frames, max_lag + 1):
            r_lag = float(np.sum(signal_norm[: n - lag] * signal_norm[lag:]))
            norm_r = float(r_lag / (variance + 1e-8))
            autocorr.append((lag, norm_r))

        if not autocorr:
            return None

        best_lag, best_score = max(autocorr, key=lambda item: item[1])
        if best_score >= self.cycle_confidence_threshold:
            self.last_cycle_detected = best_lag
            return best_lag

        return None

    def build(self) -> np.ndarray | None:
        if not self.ready():
            return None

        cycle_len = self.detect_gait_cycle()
        if cycle_len is not None and len(self.frames) >= cycle_len:
            frame_slice = self.frames[-cycle_len:]
        else:
            frame_slice = self.frames[-self.max_frames:]

        gei = np.mean(frame_slice, axis=0)
        return (gei * 255.0).astype(np.uint8)

    def count(self) -> int:
        return len(self.frames)

    def clear(self) -> None:
        self.frames.clear()
        self.width_signals.clear()
        self.valid_frames = 0
        self.rejected_frames = 0
        self.duplicate_frames = 0
        self.last_cycle_detected = None

    def reset(self) -> None:
        self.clear()

