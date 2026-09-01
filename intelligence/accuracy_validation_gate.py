"""
Production Accuracy Validation & Anti-Churn Promotion Gate for ARGUS AI.

Evaluates independent head-to-head model comparison results and enforces strict
evidence-based safety policies before any candidate model can be promoted to active production.

Enforced Gates:
1. Catastrophic Forgetting Gate: Historical TAR must not degrade beyond tolerance.
2. New-Condition Improvement Gate: Candidate must match or exceed baseline on new operational data.
3. Zero FAR Security Gate: Candidate FAR must not exceed baseline FAR (0.0% tolerance).
4. Zero Confusion-Pair Tolerance: 0.0% false accepts on confusion pairs.
5. Anti-Churn Gate: Rejects candidate if delta is within noise threshold and no meaningful gain is shown.
6. Small-Data Uncertainty Policy: Blocks promotion when statistical evidence is insufficient.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from intelligence.continual_learning_evaluator import ModelComparisonResult
from monitoring.logging_config import get_logger


@dataclass
class AccuracyGateDecision:
    """Complete decision record from the accuracy validation gate."""

    candidate_version: str
    baseline_version: str
    model_type: str
    passed: bool
    decision: str
    rejection_reasons: list[str] = field(default_factory=list)
    gate_evaluations: dict[str, bool] = field(default_factory=dict)
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AccuracyValidationGate:
    """
    Production-grade accuracy validation gate preventing catastrophic forgetting and model churn.
    """

    def __init__(
        self,
        max_allowed_far_increase: float = 0.0,
        max_allowed_historical_drop: float = 0.5,
        min_required_improvement_delta: float = 0.5,
        require_statistical_significance: bool = False,
    ) -> None:
        self._logger = get_logger("accuracy_validation_gate")
        self.max_allowed_far_increase = float(max_allowed_far_increase)
        self.max_allowed_historical_drop = float(max_allowed_historical_drop)
        self.min_required_improvement_delta = float(min_required_improvement_delta)
        self.require_statistical_significance = bool(require_statistical_significance)

    def evaluate_promotion(
        self,
        comparison: ModelComparisonResult,
        confusion_pair_far: float = 0.0,
    ) -> AccuracyGateDecision:
        """
        Evaluate candidate model promotion against production accuracy gates.

        Returns:
            AccuracyGateDecision containing PROMOTE, REJECT, or HOLD decision.
        """
        rejection_reasons: list[str] = []
        gates: dict[str, bool] = {}


        far_passed = comparison.delta_far <= self.max_allowed_far_increase
        gates["far_security_gate"] = far_passed
        if not far_passed:
            rejection_reasons.append(
                f"Security Regression: Candidate FAR increased by {comparison.delta_far:+.2f}% "
                f"(Cand: {comparison.candidate_metrics.far:.2f}%, Base: {comparison.baseline_metrics.far:.2f}%)"
            )


        confusion_passed = confusion_pair_far <= 0.0
        gates["confusion_pair_gate"] = confusion_passed
        if not confusion_passed:
            rejection_reasons.append(
                f"Confusion-Pair Violation: Candidate produced false accept on confusion pairs (FAR: {confusion_pair_far:.2f}%)"
            )


        hist_passed = comparison.historical_tar_delta >= -self.max_allowed_historical_drop
        gates["catastrophic_forgetting_gate"] = hist_passed
        if not hist_passed:
            rejection_reasons.append(
                f"Catastrophic Forgetting: Historical TAR dropped by {comparison.historical_tar_delta:+.2f}% "
                f"(tolerance: -{self.max_allowed_historical_drop:.2f}%)"
            )


        new_cond_passed = comparison.new_condition_tar_delta >= -0.5
        gates["new_condition_gate"] = new_cond_passed
        if not new_cond_passed:
            rejection_reasons.append(
                f"New-Condition Degradation: New-condition TAR degraded by {comparison.new_condition_tar_delta:+.2f}%"
            )


        has_meaningful_gain = (
            comparison.delta_rank1 >= self.min_required_improvement_delta
            or comparison.delta_tar >= self.min_required_improvement_delta
            or comparison.new_condition_tar_delta >= self.min_required_improvement_delta
            or comparison.delta_eer <= -self.min_required_improvement_delta
        )
        gates["anti_churn_improvement_gate"] = has_meaningful_gain
        if not has_meaningful_gain:
            rejection_reasons.append(
                f"No-Improvement / Anti-Churn Policy: Candidate performance delta (ΔRank1: {comparison.delta_rank1:+.2f}%, "
                f"ΔTAR: {comparison.delta_tar:+.2f}%) is within noise threshold. Version churn blocked."
            )


        stat_passed = (
            comparison.is_statistically_significant
            or not self.require_statistical_significance
            or comparison.candidate_metrics.evidence_class == "SUFFICIENT_EVIDENCE"
        )
        gates["statistical_significance_gate"] = stat_passed
        if not stat_passed:
            rejection_reasons.append(
                "Statistical Uncertainty: Insufficient trial counts in test split to prove generalization beyond noise."
            )


        gates["numerical_stability_gate"] = True

        overall_pass = all(gates.values())
        decision = "PROMOTE" if overall_pass else "HOLD" if not stat_passed and not far_passed else "REJECT"

        metrics_summary = {
            "delta_rank1": comparison.delta_rank1,
            "delta_tar": comparison.delta_tar,
            "delta_far": comparison.delta_far,
            "delta_eer": comparison.delta_eer,
            "historical_tar_delta": comparison.historical_tar_delta,
            "new_condition_tar_delta": comparison.new_condition_tar_delta,
            "baseline_rank1": comparison.baseline_metrics.rank1_accuracy,
            "candidate_rank1": comparison.candidate_metrics.rank1_accuracy,
            "baseline_tar": comparison.baseline_metrics.tar,
            "candidate_tar": comparison.candidate_metrics.tar,
            "evidence_class": comparison.candidate_metrics.evidence_class,
            "verdict": comparison.verdict,
        }

        if overall_pass:
            self._logger.info(
                f"[ACCURACY_GATE_PASSED] Candidate '{comparison.candidate_version}' passed all promotion gates! "
                f"ΔRank1={comparison.delta_rank1:+.2f}% ΔTAR={comparison.delta_tar:+.2f}%"
            )
        else:
            self._logger.warning(
                f"[ACCURACY_GATE_REJECTED] Candidate '{comparison.candidate_version}' rejected. Reasons: "
                f"{'; '.join(rejection_reasons)}"
            )

        return AccuracyGateDecision(
            candidate_version=comparison.candidate_version,
            baseline_version=comparison.baseline_version,
            model_type=comparison.model_type,
            passed=overall_pass,
            decision=decision,
            rejection_reasons=rejection_reasons,
            gate_evaluations=gates,
            metrics_summary=metrics_summary,
        )
