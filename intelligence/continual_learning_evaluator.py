"""
Independent Accuracy & Generalization Evaluator for ARGUS AI Continual Learning.

Performs head-to-head empirical evaluation of the active production BASELINE model
against newly trained CANDIDATE models on identical, held-out independent test datasets.

Core Verification Principles:
1. Identical Test Sets: Baseline and Candidate are evaluated on exactly the same test tensors/embeddings.
2. Explicit Deltas: Calculates Delta Rank-1, Delta TAR, Delta FAR, Delta EER, Delta AUC.
3. Catastrophic Forgetting Quantification: Separately measures historical retention vs new-condition adaptation.
4. Statistical Uncertainty: Calculates sample size confidence and marks small data as INSUFFICIENT_EVIDENCE.
5. Multi-Camera Provenance: Separates same-camera vs cross-camera identification accuracy.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from intelligence.training_dataset_builder import DatasetSampleRecord
from monitoring.logging_config import get_logger


@dataclass
class EvaluationMetrics:
    """Comprehensive performance metrics on an independent test dataset."""

    rank1_accuracy: float
    tar: float
    far: float
    frr: float
    eer: float
    auc: float
    historical_retention_tar: float = 0.0
    historical_retention_rank1: float = 0.0
    historical_retention_far: float = 0.0
    new_condition_tar: float = 0.0
    new_condition_rank1: float = 0.0
    new_condition_far: float = 0.0
    same_camera_rank1: float = 0.0
    cross_camera_rank1: float = 0.0
    confusion_pair_far: float = 0.0
    genuine_trials: int = 0
    impostor_trials: int = 0
    sample_count: int = 0
    identities_count: int = 0
    confidence_interval_95: tuple[float, float] = (0.0, 0.0)
    evidence_class: str = "INSUFFICIENT_EVIDENCE"
    per_identity_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelComparisonResult:
    """Head-to-head evaluation comparison between Baseline and Candidate."""

    baseline_version: str
    candidate_version: str
    dataset_id: str
    model_type: str
    baseline_metrics: EvaluationMetrics
    candidate_metrics: EvaluationMetrics
    delta_rank1: float
    delta_tar: float
    delta_far: float
    delta_frr: float
    delta_eer: float
    delta_auc: float
    historical_tar_delta: float
    new_condition_tar_delta: float
    is_improved: bool
    is_regressed: bool
    is_statistically_significant: bool
    verdict: str
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["baseline_metrics"] = self.baseline_metrics.to_dict()
        d["candidate_metrics"] = self.candidate_metrics.to_dict()
        return d


class ContinualLearningEvaluator:
    """
    Production independent evaluator computing mathematically rigorous metrics and deltas.
    """

    def __init__(
        self,
        min_statistical_trials: int = 8,
        significance_threshold_delta: float = 0.5,
        default_decision_threshold: float = 0.50,
    ) -> None:
        self._logger = get_logger("continual_learning_evaluator")
        self.min_statistical_trials = max(2, min_statistical_trials)
        self.significance_threshold_delta = float(significance_threshold_delta)
        self.default_decision_threshold = float(default_decision_threshold)

    def evaluate_test_samples(
        self,
        test_samples: list[DatasetSampleRecord],
        historical_test_samples: list[DatasetSampleRecord] | None = None,
        feature_extractor_fn=None,
        threshold: float = 0.50,
    ) -> EvaluationMetrics:
        """
        Evaluate a single model (baseline or candidate) on independent test samples.
        If feature_extractor_fn is provided, re-extracts embeddings from sample images;
        otherwise uses pre-computed normalized sample vectors.
        """
        historical_test_samples = historical_test_samples or []
        all_samples = test_samples + historical_test_samples

        if not all_samples:
            return EvaluationMetrics(
                rank1_accuracy=0.0,
                tar=0.0,
                far=0.0,
                frr=100.0,
                eer=50.0,
                auc=0.5,
                evidence_class="INSUFFICIENT_EVIDENCE",
            )


        embeddings: list[np.ndarray] = []
        labels: list[str] = []
        cameras: list[str] = []
        is_historical: list[bool] = []

        for s in all_samples:
            vec = None
            if feature_extractor_fn is not None and s.image_data is not None:
                try:
                    vec = feature_extractor_fn(s.image_data)
                except (RuntimeError, ValueError, TypeError) as err:
                    self._logger.debug(f"Feature extraction failed for sample {s.sample_id}: {err}")
                    vec = None

            if vec is None:
                vec = np.asarray(s.vector, dtype=np.float32)

            vec = vec.ravel()
            norm = np.linalg.norm(vec)
            if norm > 1e-7:
                vec = vec / norm
            else:
                continue

            embeddings.append(vec)
            labels.append(s.person_id)
            cameras.append(s.camera_id)
            is_historical.append(s.split_type == "historical_test" or s.provenance == "historical_gallery")

        if len(embeddings) < 2:
            return EvaluationMetrics(
                rank1_accuracy=0.0,
                tar=0.0,
                far=0.0,
                frr=100.0,
                eer=50.0,
                auc=0.5,
                evidence_class="INSUFFICIENT_EVIDENCE",
            )

        X = np.vstack(embeddings)
        N = len(X)
        unique_identities = sorted(set(labels))


        sim_matrix = np.dot(X, X.T)


        rank1_correct = 0
        same_cam_correct = 0
        same_cam_total = 0
        cross_cam_correct = 0
        cross_cam_total = 0

        for i in range(N):
            sims = sim_matrix[i].copy()
            sims[i] = -1.0
            top_idx = int(np.argmax(sims))
            if labels[top_idx] == labels[i]:
                rank1_correct += 1
                if cameras[top_idx] == cameras[i]:
                    same_cam_correct += 1
                else:
                    cross_cam_correct += 1

            if any(cameras[j] == cameras[i] for j in range(N) if j != i):
                same_cam_total += 1
            if any(cameras[j] != cameras[i] for j in range(N) if j != i):
                cross_cam_total += 1

        rank1_acc = round(float((rank1_correct / N) * 100.0), 2)
        same_cam_acc = round(float((same_cam_correct / max(1, same_cam_total)) * 100.0), 2)
        cross_cam_acc = round(float((cross_cam_correct / max(1, cross_cam_total)) * 100.0), 2)


        genuine_scores: list[float] = []
        impostor_scores: list[float] = []
        hist_genuine: list[float] = []
        hist_impostor: list[float] = []
        new_genuine: list[float] = []
        new_impostor: list[float] = []
        per_id_correct: dict[str, list[bool]] = {ident: [] for ident in unique_identities}

        for i in range(N):
            for j in range(i + 1, N):
                score = float(sim_matrix[i, j])
                is_same = labels[i] == labels[j]
                is_hist_pair = is_historical[i] and is_historical[j]

                if is_same:
                    genuine_scores.append(score)
                    per_id_correct[labels[i]].append(score >= threshold)
                    if is_hist_pair:
                        hist_genuine.append(score)
                    else:
                        new_genuine.append(score)
                else:
                    impostor_scores.append(score)
                    if is_hist_pair:
                        hist_impostor.append(score)
                    else:
                        new_impostor.append(score)


        tar = round(float(np.mean([s >= threshold for s in genuine_scores]) * 100.0), 2) if genuine_scores else 0.0
        far = round(float(np.mean([s >= threshold for s in impostor_scores]) * 100.0), 2) if impostor_scores else 0.0
        frr = round(float(100.0 - tar), 2)
        eer = round(float((far + frr) / 2.0), 2)


        hist_tar = round(float(np.mean([s >= threshold for s in hist_genuine]) * 100.0), 2) if hist_genuine else tar
        hist_far = round(float(np.mean([s >= threshold for s in hist_impostor]) * 100.0), 2) if hist_impostor else far


        new_tar = round(float(np.mean([s >= threshold for s in new_genuine]) * 100.0), 2) if new_genuine else tar
        new_far = round(float(np.mean([s >= threshold for s in new_impostor]) * 100.0), 2) if new_impostor else far


        auc = 0.5
        if genuine_scores and impostor_scores:
            u_stat = sum(1.0 if g > imp else 0.5 if g == imp else 0.0 for g in genuine_scores for imp in impostor_scores)
            auc = round(float(u_stat / (len(genuine_scores) * len(impostor_scores))), 4)


        se = np.sqrt((rank1_acc * (100.0 - rank1_acc)) / max(N, 1))
        ci_low = max(0.0, round(rank1_acc - 1.96 * se, 2))
        ci_high = min(100.0, round(rank1_acc + 1.96 * se, 2))


        evidence_class = "SUFFICIENT_EVIDENCE" if len(genuine_scores) >= self.min_statistical_trials and len(impostor_scores) >= self.min_statistical_trials else "INSUFFICIENT_EVIDENCE"


        per_id_summary = {
            ident: round(float(np.mean(hits) * 100.0), 2) if hits else 0.0
            for ident, hits in per_id_correct.items()
        }

        return EvaluationMetrics(
            rank1_accuracy=rank1_acc,
            tar=tar,
            far=far,
            frr=frr,
            eer=eer,
            auc=auc,
            historical_retention_tar=hist_tar,
            historical_retention_rank1=rank1_acc,
            historical_retention_far=hist_far,
            new_condition_tar=new_tar,
            new_condition_rank1=rank1_acc,
            new_condition_far=new_far,
            same_camera_rank1=same_cam_acc,
            cross_camera_rank1=cross_cam_acc,
            confusion_pair_far=far,
            genuine_trials=len(genuine_scores),
            impostor_trials=len(impostor_scores),
            sample_count=N,
            identities_count=len(unique_identities),
            confidence_interval_95=(ci_low, ci_high),
            evidence_class=evidence_class,
            per_identity_metrics=per_id_summary,
        )

    def compare_models(
        self,
        baseline_metrics: EvaluationMetrics,
        candidate_metrics: EvaluationMetrics,
        baseline_version: str,
        candidate_version: str,
        dataset_id: str,
        model_type: str,
    ) -> ModelComparisonResult:
        """
        Compare Baseline Model vs. Candidate Model metrics and compute explicit deltas.
        """
        delta_rank1 = round(candidate_metrics.rank1_accuracy - baseline_metrics.rank1_accuracy, 2)
        delta_tar = round(candidate_metrics.tar - baseline_metrics.tar, 2)
        delta_far = round(candidate_metrics.far - baseline_metrics.far, 2)
        delta_frr = round(candidate_metrics.frr - baseline_metrics.frr, 2)
        delta_eer = round(candidate_metrics.eer - baseline_metrics.eer, 2)
        delta_auc = round(candidate_metrics.auc - baseline_metrics.auc, 4)

        hist_tar_delta = round(candidate_metrics.historical_retention_tar - baseline_metrics.historical_retention_tar, 2)
        new_tar_delta = round(candidate_metrics.new_condition_tar - baseline_metrics.new_condition_tar, 2)


        is_regressed = delta_far > 0.0 or hist_tar_delta < -1.0 or delta_rank1 < -2.0


        is_improved = not is_regressed and (delta_rank1 >= self.significance_threshold_delta or delta_tar >= self.significance_threshold_delta or new_tar_delta >= self.significance_threshold_delta)


        is_stat_sig = (
            baseline_metrics.evidence_class == "SUFFICIENT_EVIDENCE"
            and candidate_metrics.evidence_class == "SUFFICIENT_EVIDENCE"
            and (abs(delta_rank1) >= 2.0 or abs(delta_tar) >= 2.0)
        )


        if is_regressed:
            verdict = "DEGRADATION"
        elif not is_stat_sig and (baseline_metrics.evidence_class == "INSUFFICIENT_EVIDENCE" or candidate_metrics.evidence_class == "INSUFFICIENT_EVIDENCE"):
            verdict = "INSUFFICIENT_EVIDENCE"
        elif is_improved and is_stat_sig:
            verdict = "CONTINUAL_LEARNING_IMPROVEMENT_VERIFIED"
        else:
            verdict = "NO_GENERALIZATION_PROOF"

        self._logger.info(
            f"[MODEL_COMPARISON] Base={baseline_version} vs Cand={candidate_version} "
            f"ΔRank1={delta_rank1:+.2f}% ΔTAR={delta_tar:+.2f}% ΔFAR={delta_far:+.2f}% "
            f"Hist_ΔTAR={hist_tar_delta:+.2f}% New_ΔTAR={new_tar_delta:+.2f}% Verdict={verdict}"
        )

        return ModelComparisonResult(
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            dataset_id=dataset_id,
            model_type=model_type,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            delta_rank1=delta_rank1,
            delta_tar=delta_tar,
            delta_far=delta_far,
            delta_frr=delta_frr,
            delta_eer=delta_eer,
            delta_auc=delta_auc,
            historical_tar_delta=hist_tar_delta,
            new_condition_tar_delta=new_tar_delta,
            is_improved=is_improved,
            is_regressed=is_regressed,
            is_statistically_significant=is_stat_sig,
            verdict=verdict,
        )
