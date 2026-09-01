
class ScoreNormalizer:
    def __init__(
        self,
        gait_min_max: tuple[float, float] = (0.0, 1.0),
        reid_min_max: tuple[float, float] = (0.0, 1.0),
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
        if score is None:
            return None
        return self._min_max_scale(
            score,
            min_val,
            max_val,
        )
