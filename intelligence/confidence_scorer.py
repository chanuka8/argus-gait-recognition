class ConfidenceScorer:
    def __init__(
        self,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.75,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def level(self, score: float) -> str:
        if score >= self.high_threshold:
            return "HIGH"

        if score >= self.medium_threshold:
            return "MEDIUM"

        return "LOW"

    def trusted(self, score: float) -> bool:
        return score >= self.medium_threshold

    def evaluate(self, score: float) -> dict:
        return {
            "score": round(score, 4),
            "level": self.level(score),
            "trusted": self.trusted(score),
        }