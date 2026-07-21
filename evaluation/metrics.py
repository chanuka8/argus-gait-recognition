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

    def _compute_precision_recall_f1(self) -> tuple[float, float, float]:
        all_labels = set(self.confusion.keys())
        for preds in self.confusion.values():
            all_labels.update(preds.keys())

        if not all_labels:
            return 0.0, 0.0, 0.0

        precisions = []
        recalls = []
        f1_scores = []

        for label in all_labels:
            tp = self.confusion.get(label, {}).get(label, 0)
            fp = sum(
                self.confusion.get(other, {}).get(label, 0)
                for other in all_labels
                if other != label
            )
            fn = sum(
                count
                for pred, count in self.confusion.get(label, {}).items()
                if pred != label
            )

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)

        macro_p = sum(precisions) / len(precisions)
        macro_r = sum(recalls) / len(recalls)
        macro_f1 = sum(f1_scores) / len(f1_scores)

        return macro_p, macro_r, macro_f1

    def summary(
        self,
    ) -> dict:
        accuracy = self.correct / self.total if self.total else 0.0
        precision, recall, f1_score = self._compute_precision_recall_f1()

        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "rank1_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": self.confusion_dict(),
        }