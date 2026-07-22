import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    compute_cosine_similarity,
    compute_rank_k_accuracies,
    compute_cmc_curve,
    compute_biometric_rates,
    compute_roc_auc_eer,
    EvaluationMetrics,
)


class TestMetricCorrectness:

    def test_cosine_similarity_perfect(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert pytest.approx(compute_cosine_similarity(v1, v2), 1e-6) == 1.0

    def test_cosine_similarity_orthogonal(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert pytest.approx(compute_cosine_similarity(v1, v2), 1e-6) == 0.0

    def test_cosine_similarity_opposite(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([-1.0, -2.0])
        assert pytest.approx(compute_cosine_similarity(v1, v2), 1e-6) == -1.0

    def test_cosine_similarity_zero_vector(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 2.0])
        assert compute_cosine_similarity(v1, v2) == 0.0

    def test_rank_k_perfect_predictions(self):
        preds = [["A", "B", "C"], ["B", "A", "C"], ["C", "B", "A"]]
        true_labels = ["A", "B", "C"]
        ranks = compute_rank_k_accuracies(preds, true_labels, ks=(1, 5))
        assert ranks[1] == 1.0
        assert ranks[5] == 1.0

    def test_rank_k_all_wrong_rank1_but_in_top5(self):
        preds = [["X", "A", "C"], ["Y", "B", "C"], ["Z", "C", "A"]]
        true_labels = ["A", "B", "C"]
        ranks = compute_rank_k_accuracies(preds, true_labels, ks=(1, 5))
        assert ranks[1] == 0.0
        assert ranks[5] == 1.0

    def test_rank_k_all_wrong(self):
        preds = [["X", "Y", "Z"], ["X", "Y", "Z"], ["X", "Y", "Z"]]
        true_labels = ["A", "B", "C"]
        ranks = compute_rank_k_accuracies(preds, true_labels, ks=(1, 5, 10))
        assert ranks[1] == 0.0
        assert ranks[5] == 0.0
        assert ranks[10] == 0.0

    def test_cmc_curve_calculation(self):
        preds = [["A", "B", "C"], ["X", "B", "C"], ["Y", "Z", "C"]]
        true_labels = ["A", "B", "C"]
        cmc = compute_cmc_curve(preds, true_labels, max_k=3)
        # Sample 1: A in rank 1 -> counts at rank 1, 2, 3
        # Sample 2: B in rank 2 -> counts at rank 2, 3
        # Sample 3: C in rank 3 -> counts at rank 3
        # Accuracies: Rank 1 = 1/3, Rank 2 = 2/3, Rank 3 = 3/3
        assert pytest.approx(cmc[0], 1e-4) == 1 / 3
        assert pytest.approx(cmc[1], 1e-4) == 2 / 3
        assert pytest.approx(cmc[2], 1e-4) == 3 / 3

    def test_biometric_rates_perfect(self):
        scores = [0.9, 0.85, 0.1, 0.15]
        is_genuine = [True, True, False, False]
        rates = compute_biometric_rates(scores, is_genuine, threshold=0.5)
        assert rates["FAR"] == 0.0
        assert rates["FRR"] == 0.0
        assert rates["TAR"] == 1.0
        assert rates["TNR"] == 1.0
        assert rates["precision"] == 1.0
        assert rates["recall"] == 1.0
        assert rates["f1_score"] == 1.0

    def test_biometric_rates_boundary_threshold(self):
        scores = [0.5, 0.49]
        is_genuine = [True, False]
        rates = compute_biometric_rates(scores, is_genuine, threshold=0.5)
        assert rates["tp"] == 1
        assert rates["fp"] == 0
        assert rates["tn"] == 1
        assert rates["fn"] == 0

    def test_roc_auc_eer_synthetic(self):
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        is_genuine = [True, True, True, False, False, False]
        res = compute_roc_auc_eer(scores, is_genuine)
        assert res["roc_auc"] == 1.0
        assert res["eer"] == 0.0

    def test_empty_input_handling(self):
        assert compute_rank_k_accuracies([], [], ks=(1, 5)) == {1: 0.0, 5: 0.0}
        assert compute_cmc_curve([], []) == [0.0] * 20
        res = compute_biometric_rates([], [], threshold=0.5)
        assert res["FAR"] == 0.0
        res_roc = compute_roc_auc_eer([], [])
        assert res_roc["roc_auc"] == 0.0

    def test_evaluation_metrics_class_imbalance(self):
        em = EvaluationMetrics()
        # 10 samples of class A (9 correct, 1 misclassified as B)
        for _ in range(9):
            em.update("A", "A")
        em.update("A", "B")
        # 1 sample of class B (1 correct)
        em.update("B", "B")

        summary = em.summary()
        assert summary["total"] == 11
        assert summary["correct"] == 10
        assert summary["rank1_accuracy"] == 10 / 11
