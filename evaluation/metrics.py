from collections import defaultdict


class EvaluationMetrics:
    def __init__(self) -> None:
        self.correct = 0
        self.incorrect = 0
        self.total = 0
        self.confusion = defaultdict(lambda: defaultdict(int))

    def update(
        self,
        actual: str,
        predicted: str,
    ) -> None:
        actual = str(actual)
        predicted = str(predicted)

        self.total += 1

        if actual == predicted:
            self.correct += 1
        else:
            self.incorrect += 1

        self.confusion[actual][predicted] += 1

    def confusion_dict(
        self,
    ) -> dict:
        return {
            actual: dict(predictions)
            for actual, predictions in self.confusion.items()
        }

    def summary(
        self,
    ) -> dict:
        accuracy = self.correct / self.total if self.total else 0.0

        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "rank1_accuracy": accuracy,
            "confusion_matrix": self.confusion_dict(),
        }