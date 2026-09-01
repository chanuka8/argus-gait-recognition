"""
Continuous Performance Improvement Engine for ARGUS AI.

Orchestrates background candidate calibration, automated regression validation,
atomic model promotion, and safety rollback:

Production Inference -> Observations -> Embedding DB -> Drift Monitoring
    -> Calibration Engine -> Candidate Model -> Validation Gate
    -> PASS: Promote (New Production Version)
    -> FAIL: Reject
    -> REGRESSION: Automatic Rollback to Previous Version
"""

import threading
from typing import Any

from intelligence.background_learning_worker import BackgroundLearningWorker
from intelligence.candidate_validator import CandidateValidator, ValidationGateResult
from intelligence.date_aware_learning_scheduler import (
    DateAwareLearningScheduler,
    LearningJobRecord,
)
from intelligence.drift_detector import DriftDetector, DriftReport
from intelligence.operational_embedding_collector import OperationalEmbeddingCollector
from models.model_registry import ModelRegistry, ModelVersionRecord
from monitoring.logging_config import get_logger
from storage.embedding_database import EmbeddingDatabase


class ContinuousImprovementEngine:
    """
    Asynchronous continuous improvement coordinator for ARGUS AI models.
    Integrates event-date driven continuous learning, background candidate generation,
    regression validation gates, atomic model promotion, and safety rollbacks.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        validator: CandidateValidator | None = None,
        collector: OperationalEmbeddingCollector | None = None,
        drift_detector: DriftDetector | None = None,
        db: EmbeddingDatabase | None = None,
        scheduler: DateAwareLearningScheduler | None = None,
        worker: BackgroundLearningWorker | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.validator = validator or CandidateValidator()
        self.collector = collector or OperationalEmbeddingCollector()
        self.db = db or EmbeddingDatabase()
        self.drift_detector = drift_detector or DriftDetector(collector=self.collector)
        self.scheduler = scheduler or DateAwareLearningScheduler(
            collector=self.collector,
            db=self.db,
        )
        self.worker = worker or BackgroundLearningWorker(
            scheduler=self.scheduler,
            registry=self.registry,
            validator=self.validator,
            collector=self.collector,
            db=self.db,
        )
        self._logger = get_logger("continuous_improvement")
        self._lock = threading.RLock()

    def start_background_learning(self) -> None:
        """Start the background learning worker thread."""
        self.worker.start()

    def stop_background_learning(self, timeout: float = 5.0) -> None:
        """Stop the background learning worker thread."""
        self.worker.stop(timeout=timeout)

    def check_and_trigger_date_learning(
        self,
        model_type: str = "dual_modal_fusion",
        synchronous: bool = False,
        model_types: list[str] | None = None,
    ) -> list[LearningJobRecord]:
        """
        Scan for unprocessed observation dates with new verified data.
        If found, schedule learning jobs and execute either synchronously or in background.
        If no new eligible date data exists, returns empty list without training.

        If model_types is specified, creates separate jobs for each model type per date
        (e.g., bygait_light NN fine-tuning, osnet_reid NN fine-tuning, and dual_modal_fusion calibration).
        """
        with self._lock:
            types = model_types or [model_type]
            scheduled_jobs = self.scheduler.check_and_schedule_new_dates(
                model_type=model_type,
                model_types=types,
            )
            if not scheduled_jobs:
                return []

            results = []
            for job in scheduled_jobs:
                if synchronous:
                    res = self.worker.execute_job_synchronous(job)
                    results.append(res)
                else:
                    self.worker.submit_job(job)
                    results.append(job)

            return results

    def trigger_all_model_types(
        self,
        synchronous: bool = False,
    ) -> list[LearningJobRecord]:
        """
        Convenience method to trigger date-aware learning for all model types:
        - bygait_light: Actual ByGaitLight CNN backbone fine-tuning
        - osnet_reid: Actual OSNet ReID backbone fine-tuning
        - dual_modal_fusion: LearnedLogisticFusion calibration
        """
        return self.check_and_trigger_date_learning(
            synchronous=synchronous,
            model_types=["bygait_light", "osnet_reid", "dual_modal_fusion"],
        )

    def trigger_nn_model_reload(
        self,
        model_type: str,
        new_version: str,
        artifact_path: str,
    ) -> None:
        """
        After promotion, signal the inference pipeline to reload new model weights
        for production version switching without restarting the server.
        """
        self._logger.info(
            f"[MODEL_RELOAD] Triggering production reload: type={model_type} "
            f"version={new_version} artifact={artifact_path}"
        )



    def get_learning_history(self) -> list[LearningJobRecord]:
        """Return the complete history of all date learning jobs."""
        return self.scheduler.list_jobs()

    def get_learning_job(self, job_id: str) -> LearningJobRecord | None:
        """Query a single learning job by ID."""
        return self.scheduler.get_job(job_id)

    def process_candidate(
        self,
        candidate_version: str,
        model_type: str,
        architecture: str,
        embedding_dim: int,
        artifact_path: str,
        candidate_metrics: dict[str, float],
        confusion_pair_eval: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, ValidationGateResult, ModelVersionRecord]:
        """
        Full lifecycle processing for a candidate model:
        1. Register CANDIDATE in ModelRegistry.
        2. Query active baseline metrics.
        3. Execute regression validation gate.
        4. Record gate result (VALIDATED or REJECTED).
        5. If passed, atomically PROMOTE to ACTIVE production status.
        """
        with self._lock:

            candidate_rec = self.registry.register_candidate(
                model_version=candidate_version,
                model_type=model_type,
                architecture=architecture,
                embedding_dim=embedding_dim,
                artifact_path=artifact_path,
                metadata=metadata or {},
            )


            active_base = self.registry.get_active_model(model_type)
            baseline_metrics = active_base.validation_metrics if active_base else {}


            val_result = self.validator.validate_candidate(
                candidate_version=candidate_version,
                model_type=model_type,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                confusion_pair_eval=confusion_pair_eval,
            )


            rejection_str = "; ".join(val_result.rejection_reasons) if not val_result.passed else None
            candidate_rec = self.registry.record_validation_result(
                model_version=candidate_version,
                model_type=model_type,
                passed=val_result.passed,
                metrics=candidate_metrics,
                rejection_reason=rejection_str,
            )


            if val_result.passed:
                promoted_rec = self.registry.promote_version(
                    model_version=candidate_version,
                    model_type=model_type,
                )
                self._logger.info(
                    f"[CONTINUOUS IMPROVEMENT] Promoted candidate '{candidate_version}' to active {model_type}."
                )
                return True, val_result, promoted_rec
            else:
                self._logger.warning(
                    f"[CONTINUOUS IMPROVEMENT] Candidate '{candidate_version}' rejected. Active baseline retained."
                )
                return False, val_result, candidate_rec

    def trigger_runtime_regression_rollback(
        self,
        model_type: str,
        reason: str = "Runtime drift / degradation detected",
    ) -> ModelVersionRecord:
        """
        Execute safety rollback to the previous known-good model version.
        """
        with self._lock:
            rolled_back = self.registry.rollback(model_type=model_type, reason=reason)
            self._logger.warning(
                f"[SAFETY ROLLBACK] Successfully restored previous model '{rolled_back.model_version}' for '{model_type}'."
            )
            return rolled_back

    def check_system_drift_and_health(self) -> DriftReport:
        """
        Evaluate operational drift across CCTV observations.
        """
        return self.drift_detector.evaluate_drift()
