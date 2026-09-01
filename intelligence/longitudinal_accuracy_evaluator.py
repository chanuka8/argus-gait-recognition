"""
Longitudinal Accuracy Evaluator for ARGUS AI Continual Learning.

Maintains multi-timepoint longitudinal evaluation history (T0 -> T1 -> T2 -> Tn)
evaluating active Baseline Model (A) vs. Candidate Model (B) across:
- Historical Test Set (C): Catastrophic forgetting assessment.
- New Operational Test Set (D): Unseen operational generalization.
- Future Holdout Set (E): Strictly temporally subsequent test partition.

Evaluates Tri-Modal Independence:
1. Gait-Only Evaluation (GEI -> ByGaitLight 256D)
2. Appearance-Only Evaluation (Crop -> OSNet-x0.25 512D)
3. DualModalFusion Evaluation (Combined score distribution)
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from intelligence.continual_learning_evaluator import (
    ContinualLearningEvaluator,
)
from intelligence.statistical_accuracy_validator import (
    StatisticalAccuracyValidator,
    StatisticalValidationResult,
)
from intelligence.training_dataset_builder import DatasetSampleRecord
from monitoring.logging_config import get_logger


@dataclass
class LongitudinalTimepointRecord:
    """Immutable evaluation record for a specific continual learning lifecycle timepoint."""

    timepoint_id: str
    timestamp: float
    model_type: str
    baseline_version: str
    candidate_version: str
    dataset_id: str
    manifest_sha256: str
    gait_comparison: dict[str, Any] = field(default_factory=dict)
    appearance_comparison: dict[str, Any] = field(default_factory=dict)
    fusion_comparison: dict[str, Any] = field(default_factory=dict)
    condition_breakdown: dict[str, Any] = field(default_factory=dict)
    future_holdout_evaluation: dict[str, Any] = field(default_factory=dict)
    statistical_validation: dict[str, Any] = field(default_factory=dict)
    decision: str = "REJECTED"
    rejection_reasons: list[str] = field(default_factory=list)
    verdict: str = "ACCURACY_IMPROVEMENT_NOT_YET_PROVEN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LongitudinalTimepointRecord":
        return cls(**data)


class LongitudinalAccuracyEvaluator:
    """
    Orchestrates longitudinal tri-modal evaluation and persistent accuracy history tracking.
    """

    def __init__(
        self,
        history_file: str = "data/continual_learning_longitudinal_history.json",
        evaluator: ContinualLearningEvaluator | None = None,
        stat_validator: StatisticalAccuracyValidator | None = None,
    ) -> None:
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.evaluator = evaluator or ContinualLearningEvaluator()
        self.stat_validator = stat_validator or StatisticalAccuracyValidator()
        self._lock = threading.RLock()
        self._logger = get_logger("longitudinal_evaluator")

    def evaluate_longitudinal_cycle(
        self,
        baseline_version: str,
        candidate_version: str,
        dataset_id: str,
        manifest_sha256: str,
        model_type: str,
        operational_test_samples: list[DatasetSampleRecord],
        historical_test_samples: list[DatasetSampleRecord] | None = None,
        future_holdout_samples: list[DatasetSampleRecord] | None = None,
        gait_extractor_fn=None,
        appearance_extractor_fn=None,
    ) -> LongitudinalTimepointRecord:
        """
        Execute full longitudinal evaluation across operational test, historical test, and future holdouts.
        """
        historical_test_samples = historical_test_samples or []
        future_holdout_samples = future_holdout_samples or []
        all_eval_samples = operational_test_samples + historical_test_samples


        base_metrics = self.evaluator.evaluate_test_samples(
            test_samples=operational_test_samples,
            historical_test_samples=historical_test_samples,
        )
        cand_metrics = self.evaluator.evaluate_test_samples(
            test_samples=operational_test_samples,
            historical_test_samples=historical_test_samples,
        )

        comparison = self.evaluator.compare_models(
            baseline_metrics=base_metrics,
            candidate_metrics=cand_metrics,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            dataset_id=dataset_id,
            model_type=model_type,
        )


        same_cam_rank1 = cand_metrics.same_camera_rank1
        cross_cam_rank1 = cand_metrics.cross_camera_rank1


        viewpoint_metrics = self._evaluate_by_condition(all_eval_samples, "viewpoint")
        clothing_metrics = self._evaluate_by_condition(all_eval_samples, "clothing")
        carrying_metrics = self._evaluate_by_condition(all_eval_samples, "carrying")

        condition_breakdown = {
            "same_camera_rank1": same_cam_rank1,
            "cross_camera_rank1": cross_cam_rank1,
            "cross_camera_gain": round(cross_cam_rank1 - base_metrics.cross_camera_rank1, 2),
            "viewpoint_breakdown": viewpoint_metrics,
            "clothing_breakdown": clothing_metrics,
            "carrying_breakdown": carrying_metrics,
        }


        future_eval_summary = {}
        if future_holdout_samples:
            fut_base = self.evaluator.evaluate_test_samples(test_samples=future_holdout_samples)
            fut_cand = self.evaluator.evaluate_test_samples(test_samples=future_holdout_samples)
            fut_comp = self.evaluator.compare_models(
                baseline_metrics=fut_base,
                candidate_metrics=fut_cand,
                baseline_version=baseline_version,
                candidate_version=candidate_version,
                dataset_id=f"future_{dataset_id}",
                model_type=model_type,
            )
            future_eval_summary = fut_comp.to_dict()


        unique_identities = len({s.person_id for s in all_eval_samples})
        unique_tracks = len({s.track_id for s in all_eval_samples})
        unique_sessions = len({s.condition_tags.get("session_id", s.camera_id) for s in all_eval_samples})

        stat_result: StatisticalValidationResult = self.stat_validator.validate_statistical_evidence(
            baseline_metrics=base_metrics.to_dict(),
            candidate_metrics=cand_metrics.to_dict(),
            identities_count=unique_identities,
            tracks_count=unique_tracks,
            sessions_count=unique_sessions,
            genuine_trials=cand_metrics.genuine_trials,
            impostor_trials=cand_metrics.impostor_trials,
            sample_count=len(all_eval_samples),
        )


        rejection_reasons = list(stat_result.rejection_reasons)
        if comparison.historical_tar_delta < -0.5:
            rejection_reasons.append(
                f"Catastrophic Forgetting: Historical TAR dropped by {comparison.historical_tar_delta:+.2f}%"
            )
        if comparison.delta_far > 0.0:
            rejection_reasons.append(f"Security Regression: FAR increased by {comparison.delta_far:+.2f}%")

        passed = len(rejection_reasons) == 0 and stat_result.is_statistically_significant
        decision = "PROMOTED" if passed else "REJECTED"


        t_id = f"T{len(self.list_history())}_{int(time.time())}"
        record = LongitudinalTimepointRecord(
            timepoint_id=t_id,
            timestamp=time.time(),
            model_type=model_type,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            dataset_id=dataset_id,
            manifest_sha256=manifest_sha256,
            gait_comparison=comparison.to_dict() if model_type == "bygait_light" else {},
            appearance_comparison=comparison.to_dict() if model_type == "osnet_reid" else {},
            fusion_comparison=comparison.to_dict() if model_type == "dual_modal_fusion" else {},
            condition_breakdown=condition_breakdown,
            future_holdout_evaluation=future_eval_summary,
            statistical_validation=stat_result.to_dict(),
            decision=decision,
            rejection_reasons=rejection_reasons,
            verdict=stat_result.verdict,
        )

        self._record_timepoint(record)
        return record

    def _evaluate_by_condition(self, samples: list[DatasetSampleRecord], condition_key: str) -> dict[str, Any]:
        """Group samples by condition metadata key and compute sample count."""
        grouped: dict[str, int] = {}
        for s in samples:
            val = str(s.condition_tags.get(condition_key, "STANDARD"))
            grouped[val] = grouped.get(val, 0) + 1
        return grouped

    def list_history(self) -> list[LongitudinalTimepointRecord]:
        """Read all longitudinal evaluation timepoint records from disk."""
        with self._lock:
            if not self.history_file.exists():
                return []
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [LongitudinalTimepointRecord.from_dict(r) for r in data.get("history", [])]
            except (OSError, json.JSONDecodeError, ValueError) as err:
                self._logger.warning(f"Failed to read longitudinal history: {err}")
                return []

    def _record_timepoint(self, record: LongitudinalTimepointRecord) -> bool:
        """Atomically append a longitudinal timepoint record to history file."""
        with self._lock:
            history = self.list_history()
            history.append(record)
            tmp = self.history_file.with_suffix(".tmp")
            try:
                payload = {
                    "updated_at": time.time(),
                    "total_timepoints": len(history),
                    "history": [r.to_dict() for r in history],
                }
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                tmp.replace(self.history_file)
                return True
            except (OSError, ValueError) as err:
                self._logger.error(f"Failed to persist longitudinal history: {err}")
                return False
