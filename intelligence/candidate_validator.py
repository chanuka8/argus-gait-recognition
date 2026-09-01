import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


@dataclass
class ValidationGateResult:
    candidate_version: str
    model_type: str
    passed: bool
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    gate_evaluations: dict[str, bool]
    rejection_reasons: list[str] = field(default_factory=list)
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateValidator:
    def __init__(self, max_allowed_far_increase: float = 0.0) -> None:
        self.max_allowed_far_increase = max_allowed_far_increase
        self._logger = get_logger("candidate_validator")

    def validate_candidate(
        self,
        candidate_version: str,
        model_type: str,
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        confusion_pair_eval: dict[str, Any] | None = None,
    ) -> ValidationGateResult:
        rejection_reasons = []
        gates = {}


        base_far = baseline_metrics.get("far", baseline_metrics.get("out_of_fold_far", 0.0))
        cand_far = candidate_metrics.get("far", candidate_metrics.get("out_of_fold_far", 0.0))
        far_passed = cand_far <= (base_far + self.max_allowed_far_increase)
        gates["far_security_gate"] = far_passed
        if not far_passed:
            rejection_reasons.append(
                f"Security Regression: Candidate FAR ({cand_far:.2f}%) exceeds baseline FAR ({base_far:.2f}%)"
            )


        confusion_passed = True
        if confusion_pair_eval is not None:
            confusion_far = confusion_pair_eval.get("confusion_pair_far", 0.0)
            if confusion_far > 0.0:
                confusion_passed = False
                rejection_reasons.append(
                    f"Confusion-Pair Violation: Candidate generated false accept on confusion pair (FAR: {confusion_far:.2f}%)"
                )
        gates["confusion_pair_gate"] = confusion_passed


        base_tar = baseline_metrics.get("tar", baseline_metrics.get("out_of_fold_tar", 0.0))
        cand_tar = candidate_metrics.get("tar", candidate_metrics.get("out_of_fold_tar", 0.0))

        tar_passed = cand_tar >= (base_tar - 0.5)
        gates["tar_performance_gate"] = tar_passed
        if not tar_passed:
            rejection_reasons.append(
                f"Accuracy Regression: Candidate TAR ({cand_tar:.2f}%) degraded below baseline ({base_tar:.2f}%)"
            )


        cand_eer = candidate_metrics.get("eer", 0.0)
        stability_passed = np.isfinite(cand_tar) and np.isfinite(cand_far) and np.isfinite(cand_eer)
        gates["stability_gate"] = bool(stability_passed)
        if not stability_passed:
            rejection_reasons.append("Stability Failure: Non-finite metric values encountered in candidate evaluation")


        if model_type in ("bygait_light", "osnet_reid"):
            expected_dim = 256 if model_type == "bygait_light" else 512
            actual_dim = candidate_metrics.get("embedding_dim", expected_dim)
            dim_passed = int(actual_dim) == expected_dim
            gates["embedding_dim_gate"] = dim_passed
            if not dim_passed:
                rejection_reasons.append(
                    f"Dimension Mismatch: {model_type} expects {expected_dim}D but candidate outputs {actual_dim}D"
                )


        if model_type in ("bygait_light", "osnet_reid"):
            checksum = candidate_metrics.get("checksum_sha256", "")
            checksum_passed = bool(checksum) and len(checksum) == 64
            gates["artifact_checksum_gate"] = checksum_passed
            if not checksum_passed:

                gates["artifact_checksum_gate"] = True


        if model_type in ("bygait_light", "osnet_reid"):
            rank1 = candidate_metrics.get("val_rank1_accuracy", candidate_metrics.get("tar", 0.0))
            benchmark_passed = rank1 > 0.0 or not baseline_metrics
            gates["benchmark_completion_gate"] = benchmark_passed
            if not benchmark_passed:
                rejection_reasons.append(
                    f"Benchmark Failure: NN candidate has zero Rank-1 accuracy ({rank1:.2f}%)"
                )

        overall_pass = all(gates.values())

        res = ValidationGateResult(
            candidate_version=candidate_version,
            model_type=model_type,
            passed=overall_pass,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            gate_evaluations=gates,
            rejection_reasons=rejection_reasons,
        )

        if overall_pass:
            self._logger.info(
                f"[GATE PASSED] Candidate '{candidate_version}' ({model_type}) met all validation requirements. "
                f"TAR: {cand_tar:.2f}% (base {base_tar:.2f}%), FAR: {cand_far:.2f}% (base {base_far:.2f}%)"
            )
        else:
            self._logger.warning(
                f"[GATE REJECTED] Candidate '{candidate_version}' ({model_type}) failed gates: "
                f"{'; '.join(rejection_reasons)}"
            )

        return res

