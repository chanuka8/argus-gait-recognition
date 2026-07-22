import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two 1D feature vectors."""
    v1 = np.asarray(vec1, dtype=np.float32).flatten()
    v2 = np.asarray(vec2, dtype=np.float32).flatten()

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (n1 * n2))


def compute_rank_k_accuracies(
    ranked_predictions: list[list[str]],
    true_labels: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[int, float]:
    """Calculates Rank-k identification accuracies for given k values."""
    if not ranked_predictions or not true_labels or len(ranked_predictions) != len(true_labels):
        return {k: 0.0 for k in ks}

    total = len(true_labels)
    correct_counts = {k: 0 for k in ks}

    for ranked, true_id in zip(ranked_predictions, true_labels):
        true_id_str = str(true_id)
        ranked_str = [str(r) for r in ranked]

        for k in ks:
            if true_id_str in ranked_str[:k]:
                correct_counts[k] += 1

    return {k: correct_counts[k] / total if total > 0 else 0.0 for k in ks}


def compute_cmc_curve(
    ranked_predictions: list[list[str]],
    true_labels: list[str],
    max_k: int = 20,
) -> list[float]:
    """Computes Cumulative Match Characteristic (CMC) curve up to max_k."""
    if not ranked_predictions or not true_labels:
        return [0.0] * max_k

    total = len(true_labels)
    counts = [0] * max_k

    for ranked, true_id in zip(ranked_predictions, true_labels):
        true_id_str = str(true_id)
        ranked_str = [str(r) for r in ranked]

        for k in range(1, max_k + 1):
            if true_id_str in ranked_str[:k]:
                counts[k - 1] += 1

    return [c / total if total > 0 else 0.0 for c in counts]


def compute_biometric_rates(
    scores: list[float] | np.ndarray,
    is_genuine: list[bool] | np.ndarray,
    threshold: float,
) -> dict:
    """
    Computes biometric security rates at a specific threshold:
    - FAR: False Accept Rate (Impostor accepted)
    - FRR: False Reject Rate (Genuine rejected)
    - TAR: True Accept Rate (1 - FRR)
    - TNR: True Reject Rate (1 - FAR)
    - Precision, Recall, F1
    """
    scores_arr = np.asarray(scores, dtype=np.float32)
    is_gen_arr = np.asarray(is_genuine, dtype=bool)

    if len(scores_arr) == 0:
        return {
            "threshold": threshold,
            "FAR": 0.0, "FRR": 0.0, "TAR": 0.0, "TNR": 0.0,
            "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
        }

    accepted = scores_arr >= threshold

    num_genuine = int(np.sum(is_gen_arr))
    num_impostor = len(is_gen_arr) - num_genuine

    tp = int(np.sum(accepted & is_gen_arr))
    fn = num_genuine - tp
    fp = int(np.sum(accepted & (~is_gen_arr)))
    tn = num_impostor - fp

    far = fp / num_impostor if num_impostor > 0 else 0.0
    frr = fn / num_genuine if num_genuine > 0 else 0.0
    tar = tp / num_genuine if num_genuine > 0 else 0.0
    tnr = tn / num_impostor if num_impostor > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tar
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "FAR": far,
        "FRR": frr,
        "TAR": tar,
        "TNR": tnr,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def compute_roc_auc_eer(
    scores: list[float] | np.ndarray,
    is_genuine: list[bool] | np.ndarray,
    num_thresholds: int = 500,
) -> dict:
    """
    Computes ROC curve points, ROC-AUC, and Equal Error Rate (EER) with threshold.
    """
    scores_arr = np.asarray(scores, dtype=np.float32)
    is_gen_arr = np.asarray(is_genuine, dtype=bool)

    if len(scores_arr) == 0 or np.all(is_gen_arr) or not np.any(is_gen_arr):
        return {
            "roc_auc": 0.0,
            "eer": 0.0,
            "eer_threshold": 0.0,
            "thresholds": [],
            "far_list": [],
            "tar_list": [],
        }

    min_s = float(np.min(scores_arr))
    max_s = float(np.max(scores_arr))

    thresholds = np.linspace(max_s + 1e-4, min_s - 1e-4, num_thresholds)

    far_list = []
    tar_list = []
    frr_list = []

    for th in thresholds:
        rates = compute_biometric_rates(scores_arr, is_gen_arr, th)
        far_list.append(rates["FAR"])
        tar_list.append(rates["TAR"])
        frr_list.append(rates["FRR"])

    far_arr = np.asarray(far_list, dtype=np.float32)
    tar_arr = np.asarray(tar_list, dtype=np.float32)
    frr_arr = np.asarray(frr_list, dtype=np.float32)

    # ROC AUC calculation (Trapezoidal integration of TAR vs FAR)
    order = np.argsort(far_arr)
    sorted_far = far_arr[order]
    sorted_tar = tar_arr[order]
    roc_auc = float(np.trapezoid(sorted_tar, sorted_far))

    # EER calculation (where FAR = FRR, min |FAR - FRR|)
    diff = np.abs(far_arr - frr_arr)
    eer_idx = int(np.argmin(diff))
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    return {
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "thresholds": thresholds.tolist(),
        "far_list": far_list,
        "tar_list": tar_list,
        "frr_list": frr_list,
    }


class EvaluationMetrics:
    def __init__(self) -> None:
        self.correct = 0
        self.incorrect = 0
        self.total = 0
        self.confusion = defaultdict(lambda: defaultdict(int))

    def update(self, actual: str, predicted: str) -> None:
        actual = str(actual)
        predicted = str(predicted)
        self.total += 1
        if actual == predicted:
            self.correct += 1
        else:
            self.incorrect += 1
        self.confusion[actual][predicted] += 1

    def confusion_dict(self) -> dict:
        return {actual: dict(predictions) for actual, predictions in self.confusion.items()}

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
            fp = sum(self.confusion.get(other, {}).get(label, 0) for other in all_labels if other != label)
            fn = sum(count for pred, count in self.confusion.get(label, {}).items() if pred != label)

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

    def summary(self) -> dict:
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