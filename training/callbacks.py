from pathlib import Path


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0

    def should_stop(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = score > self.best_score if self.mode == "max" else score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class TrainingLogger:
    def __init__(self, log_file: str = "outputs/reports/training_log.txt") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(message + "\n")