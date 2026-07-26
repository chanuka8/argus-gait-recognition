"""
Stateless Score Normalizer for Gait and ReID Biometric Scores.

Normalizes raw cosine similarity / distance scores into the [0.0, 1.0] range.
"""

from typing import Tuple


class ScoreNormalizer:
    """
    Stateless score normalization module.

    Converts raw similarity scores into normalized probabilities/confidences
    in the range [0.0, 1.0] using configurable min-max bounds or clipping.
    """

    def __init__(
        self,
        gait_min_max: Tuple[float, float] = (0.0, 1.0),
        reid_min_max: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.gait_min, self.gait_max = gait_min_max
        self.reid_min, self.reid_max = reid_min_max

    @staticmethod
    def _min_max_scale(
        score: float,
        min_val: float,
        max_val: float,
    ) -> float:
        if max_val <= min_val:
            return float(score >= min_val)

        clamped = max(min_val, min(max_val, float(score)))
        normalized = (clamped - min_val) / (max_val - min_val)
        return float(max(0.0, min(1.0, normalized)))

    def normalize_gait(
        self,
        score: float | None,
    ) -> float | None:
        """Normalize gait cosine score to [0.0, 1.0]."""
        if score is None:
            return None
        return self._min_max_scale(
            score,
            self.gait_min,
            self.gait_max,
        )

    def normalize_reid(
        self,
        score: float | None,
    ) -> float | None:
        """Normalize ReID cosine score to [0.0, 1.0]."""
        if score is None:
            return None
        return self._min_max_scale(
            score,
            self.reid_min,
            self.reid_max,
        )

    def normalize(
        self,
        score: float | None,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> float | None:
        """Normalize generic score with explicit bounds."""
        if score is None:
            return None
        return self._min_max_scale(
            score,
            min_val,
            max_val,
        )
