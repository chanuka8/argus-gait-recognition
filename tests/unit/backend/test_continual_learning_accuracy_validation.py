import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from intelligence.accuracy_validation_gate import AccuracyValidationGate
from intelligence.background_learning_worker import BackgroundLearningWorker
from intelligence.continual_learning_audit_trail import (
    ContinualLearningAuditTrail,
)
from intelligence.continual_learning_evaluator import (
    ContinualLearningEvaluator,
    EvaluationMetrics,
    ModelComparisonResult,
)
from intelligence.date_aware_learning_scheduler import (
    DateAwareLearningScheduler,
    LearningJobStatus,
)
from intelligence.operational_embedding_collector import (
    OperationalEmbeddingCollector,
)
from intelligence.training_dataset_builder import (
    DatasetSampleRecord,
    TrainingDatasetBuilder,
)
from models.model_registry import ModelDeploymentStatus, ModelRegistry
from storage.embedding_database import EmbeddingDatabase


@pytest.fixture
def isolated_cl_env():
    temp_dir = tempfile.mkdtemp(prefix="argus_cl_accuracy_test_")
    t_path = Path(temp_dir)

    manifests_dir = t_path / "data" / "dataset_manifests"
    audit_file = t_path / "data" / "continual_learning_audit_trail.json"
    registry_file = t_path / "models" / "model_registry.json"
    jobs_file = t_path / "data" / "learning_jobs.json"
    candidates_dir = t_path / "models" / "candidates"
    obs_dir = t_path / "data" / "operational_observations"
    db_dir = t_path / "data" / "embedding_db"

    manifests_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    yield {
        "root": t_path,
        "manifests_dir": str(manifests_dir),
        "audit_file": str(audit_file),
        "registry_file": str(registry_file),
        "jobs_file": str(jobs_file),
        "candidates_dir": str(candidates_dir),
        "obs_dir": str(obs_dir),
        "db_dir": str(db_dir),
    }

    shutil.rmtree(temp_dir, ignore_errors=True)






class TestTrainingDatasetBuilder:
    def test_strict_train_val_test_isolation(self, isolated_cl_env):
        env = isolated_cl_env
        collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
        db = EmbeddingDatabase(db_dir=env["db_dir"])


        date_str = "2026-08-31"
        for ident in ["Subject_01", "Subject_02", "Subject_03"]:
            for i in range(4):
                vec = np.random.randn(256).astype(np.float32)
                vec /= np.linalg.norm(vec)
                obs = collector.record_observation(
                    camera_id=f"cam_{i % 2 + 1}",
                    track_id=100 + i,
                    vector=vec,
                    predicted_identity=ident,
                    confidence=0.92,
                    modality="gait",
                    observation_date=date_str,
                )
                collector.verify_observation(obs.observation_id, verified_identity=ident)

        builder = TrainingDatasetBuilder(
            collector=collector,
            db=db,
            manifest_dir=env["manifests_dir"],
            test_split_ratio=0.25,
            val_split_ratio=0.25,
        )

        train, val, test, _hist_replay, _hist_test, _future_holdout, manifest = builder.build_dataset_for_date(
            training_date=date_str,
            model_type="bygait_light",
            include_historical=False,
        )

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0


        train_ids = {s.sample_id for s in train}
        val_ids = {s.sample_id for s in val}
        test_ids = {s.sample_id for s in test}

        assert len(train_ids.intersection(test_ids)) == 0, "Train and Test sample IDs intersect!"
        assert len(train_ids.intersection(val_ids)) == 0, "Train and Val sample IDs intersect!"
        assert len(val_ids.intersection(test_ids)) == 0, "Val and Test sample IDs intersect!"


        assert manifest.manifest_sha256 != ""
        assert len(manifest.manifest_sha256) == 64
        manifest_file = Path(env["manifests_dir"]) / f"{manifest.dataset_id}.json"
        assert manifest_file.exists()

    def test_unverified_and_invalid_embeddings_excluded(self, isolated_cl_env):
        env = isolated_cl_env
        collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
        db = EmbeddingDatabase(db_dir=env["db_dir"])
        date_str = "2026-08-31"


        vec1 = np.random.randn(256).astype(np.float32)
        collector.record_observation(
            camera_id="cam_01",
            track_id=1,
            vector=vec1,
            predicted_identity="Subject_Unverified",
            confidence=0.85,
            modality="gait",
            observation_date=date_str,
        )


        vec_nan = np.random.randn(256).astype(np.float32)
        vec_nan[10] = np.nan
        obs_nan = collector.record_observation(
            camera_id="cam_01",
            track_id=2,
            vector=vec_nan,
            predicted_identity="Subject_Corrupt",
            confidence=0.90,
            modality="gait",
            observation_date=date_str,
        )
        collector.verify_observation(obs_nan.observation_id, verified_identity="Subject_Corrupt")

        builder = TrainingDatasetBuilder(collector=collector, db=db, manifest_dir=env["manifests_dir"])
        train, val, test, _, _, _, manifest = builder.build_dataset_for_date(date_str, model_type="bygait_light")

        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0
        assert manifest.total_samples == 0






class TestContinualLearningEvaluator:
    def test_metric_calculation_and_deltas(self):
        evaluator = ContinualLearningEvaluator(min_statistical_trials=4)

        base_metrics = EvaluationMetrics(
            rank1_accuracy=75.0,
            tar=80.0,
            far=1.5,
            frr=20.0,
            eer=10.75,
            auc=0.88,
            historical_retention_tar=80.0,
            new_condition_tar=75.0,
            genuine_trials=12,
            impostor_trials=16,
            sample_count=8,
            identities_count=2,
            evidence_class="SUFFICIENT_EVIDENCE",
        )

        cand_metrics = EvaluationMetrics(
            rank1_accuracy=80.0,
            tar=84.0,
            far=1.5,
            frr=16.0,
            eer=8.75,
            auc=0.92,
            historical_retention_tar=80.0,
            new_condition_tar=80.0,
            genuine_trials=12,
            impostor_trials=16,
            sample_count=8,
            identities_count=2,
            evidence_class="SUFFICIENT_EVIDENCE",
        )

        comparison = evaluator.compare_models(
            baseline_metrics=base_metrics,
            candidate_metrics=cand_metrics,
            baseline_version="v1.0.0",
            candidate_version="v2.0.0-cand",
            dataset_id="ds-test-01",
            model_type="bygait_light",
        )

        assert comparison.delta_rank1 == 5.0
        assert comparison.delta_tar == 4.0
        assert comparison.delta_far == 0.0
        assert comparison.is_improved is True
        assert comparison.is_regressed is False
        assert comparison.verdict in ("CONTINUAL_LEARNING_IMPROVEMENT_VERIFIED", "NO_GENERALIZATION_PROOF")

    def test_small_data_uncertainty_flagged(self):
        evaluator = ContinualLearningEvaluator(min_statistical_trials=20)
        samples = [
            DatasetSampleRecord("s1", "P1", "cam1", 1, time.time(), "2026-08-31", "gait", [1.0] * 256),
            DatasetSampleRecord("s2", "P1", "cam1", 2, time.time(), "2026-08-31", "gait", [1.0] * 256),
        ]
        metrics = evaluator.evaluate_test_samples(samples)
        assert metrics.evidence_class == "INSUFFICIENT_EVIDENCE"






class TestAccuracyValidationGate:
    def test_catastrophic_forgetting_blocks_promotion(self):
        gate = AccuracyValidationGate(max_allowed_historical_drop=0.5)

        base_metrics = EvaluationMetrics(
            rank1_accuracy=90.0, tar=95.0, far=0.5, frr=5.0, eer=2.75, auc=0.98,
            historical_retention_tar=95.0, new_condition_tar=90.0,
            genuine_trials=20, impostor_trials=40, sample_count=10, identities_count=2,
            evidence_class="SUFFICIENT_EVIDENCE",
        )
        cand_metrics = EvaluationMetrics(
            rank1_accuracy=85.0, tar=88.0, far=0.5, frr=12.0, eer=6.25, auc=0.92,
            historical_retention_tar=80.0,
            new_condition_tar=96.0,
            genuine_trials=20, impostor_trials=40, sample_count=10, identities_count=2,
            evidence_class="SUFFICIENT_EVIDENCE",
        )

        comparison = ModelComparisonResult(
            baseline_version="v1.0.0",
            candidate_version="v2.0.0-cand",
            dataset_id="ds-test",
            model_type="bygait_light",
            baseline_metrics=base_metrics,
            candidate_metrics=cand_metrics,
            delta_rank1=-5.0,
            delta_tar=-7.0,
            delta_far=0.0,
            delta_frr=7.0,
            delta_eer=3.5,
            delta_auc=-0.06,
            historical_tar_delta=-15.0,
            new_condition_tar_delta=6.0,
            is_improved=False,
            is_regressed=True,
            is_statistically_significant=True,
            verdict="DEGRADATION",
        )

        decision = gate.evaluate_promotion(comparison)
        assert decision.passed is False
        assert decision.decision == "REJECT"
        assert any("Catastrophic Forgetting" in r for r in decision.rejection_reasons)

    def test_anti_churn_gate_blocks_meaningless_version_growth(self):
        gate = AccuracyValidationGate(min_required_improvement_delta=0.5)

        metrics = EvaluationMetrics(
            rank1_accuracy=88.0, tar=90.0, far=1.0, frr=10.0, eer=5.5, auc=0.95,
            historical_retention_tar=90.0, new_condition_tar=90.0,
            genuine_trials=20, impostor_trials=40, sample_count=10, identities_count=2,
            evidence_class="SUFFICIENT_EVIDENCE",
        )

        comparison = ModelComparisonResult(
            baseline_version="v1.0.0",
            candidate_version="v2.0.0-cand",
            dataset_id="ds-test",
            model_type="bygait_light",
            baseline_metrics=metrics,
            candidate_metrics=metrics,
            delta_rank1=0.0,
            delta_tar=0.0,
            delta_far=0.0,
            delta_frr=0.0,
            delta_eer=0.0,
            delta_auc=0.0,
            historical_tar_delta=0.0,
            new_condition_tar_delta=0.0,
            is_improved=False,
            is_regressed=False,
            is_statistically_significant=False,
            verdict="NO_GENERALIZATION_PROOF",
        )

        decision = gate.evaluate_promotion(comparison)
        assert decision.passed is False
        assert decision.decision == "REJECT"
        assert any("Anti-Churn" in r for r in decision.rejection_reasons)

    def test_legitimate_improvement_promotes(self):
        gate = AccuracyValidationGate(
            max_allowed_far_increase=0.0,
            max_allowed_historical_drop=0.5,
            min_required_improvement_delta=0.5,
        )

        base_metrics = EvaluationMetrics(
            rank1_accuracy=80.0, tar=85.0, far=1.0, frr=15.0, eer=8.0, auc=0.90,
            historical_retention_tar=85.0, new_condition_tar=80.0,
            genuine_trials=20, impostor_trials=40, sample_count=10, identities_count=2,
            evidence_class="SUFFICIENT_EVIDENCE",
        )
        cand_metrics = EvaluationMetrics(
            rank1_accuracy=86.0, tar=90.0, far=0.8, frr=10.0, eer=5.4, auc=0.94,
            historical_retention_tar=86.0,
            new_condition_tar=90.0,
            genuine_trials=20, impostor_trials=40, sample_count=10, identities_count=2,
            evidence_class="SUFFICIENT_EVIDENCE",
        )

        comparison = ModelComparisonResult(
            baseline_version="v1.0.0",
            candidate_version="v2.0.0-cand",
            dataset_id="ds-test",
            model_type="bygait_light",
            baseline_metrics=base_metrics,
            candidate_metrics=cand_metrics,
            delta_rank1=6.0,
            delta_tar=5.0,
            delta_far=-0.2,
            delta_frr=-5.0,
            delta_eer=-2.6,
            delta_auc=0.04,
            historical_tar_delta=1.0,
            new_condition_tar_delta=10.0,
            is_improved=True,
            is_regressed=False,
            is_statistically_significant=True,
            verdict="CONTINUAL_LEARNING_IMPROVEMENT_VERIFIED",
        )

        decision = gate.evaluate_promotion(comparison)
        assert decision.passed is True
        assert decision.decision == "PROMOTE"
        assert len(decision.rejection_reasons) == 0






class TestContinualLearningAuditTrail:
    def test_event_recording_and_restart_recovery(self, isolated_cl_env):
        env = isolated_cl_env
        trail = ContinualLearningAuditTrail(audit_file=env["audit_file"])

        event = trail.create_and_record(
            event_type="EVALUATION_COMPLETED",
            trigger_date="2026-08-31",
            model_type="bygait_light",
            dataset_id="ds-001",
            baseline_version="v1.0.0",
            candidate_version="v2.0.0-cand",
            parameters_changed=23,
            total_parameters=224448,
            metric_deltas={"delta_rank1": 4.5, "delta_tar": 3.0},
            validation_passed=True,
            promotion_status="PROMOTED",
            verdict="CONTINUAL_LEARNING_IMPROVEMENT_VERIFIED",
        )

        assert event.event_id != ""


        trail_reloaded = ContinualLearningAuditTrail(audit_file=env["audit_file"])
        events = trail_reloaded.list_events()
        assert len(events) == 1
        assert events[0].event_id == event.event_id
        assert events[0].parameters_changed == 23
        assert events[0].promotion_status == "PROMOTED"






class TestEndToEndWorkerAccuracyValidation:
    def test_full_worker_cycle_with_dataset_builder_and_evaluation(self, isolated_cl_env):
        env = isolated_cl_env
        collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
        db = EmbeddingDatabase(db_dir=env["db_dir"])
        registry = ModelRegistry(registry_file=env["registry_file"])
        scheduler = DateAwareLearningScheduler(
            jobs_file=env["jobs_file"],
            collector=collector,
            db=db,
            min_training_embeddings=1,
            min_identities=1,
        )

        worker = BackgroundLearningWorker(
            scheduler=scheduler,
            registry=registry,
            collector=collector,
            db=db,
            candidate_artifacts_dir=env["candidates_dir"],
            timeout_seconds=60.0,
        )


        date_str = "2026-08-31"
        for ident in ["Subject_Alpha", "Subject_Beta"]:
            for i in range(4):
                vec = np.random.randn(256).astype(np.float32)
                vec /= np.linalg.norm(vec)
                gei = np.random.randint(0, 255, size=(64, 128), dtype=np.uint8)
                obs = collector.record_observation(
                    camera_id=f"cam_{i % 2 + 1}",
                    track_id=100 + i,
                    vector=vec,
                    predicted_identity=ident,
                    confidence=0.92,
                    modality="gait",
                    observation_date=date_str,
                    media_array=gei,
                )
                collector.verify_observation(obs.observation_id, verified_identity=ident)

        job = scheduler.create_learning_job(training_date=date_str, model_type="bygait_light")
        assert job is not None


        executed_job = worker.execute_job_synchronous(job)
        assert executed_job.status in (LearningJobStatus.PROMOTED, LearningJobStatus.REJECTED)


        cand_models = [m for m in registry.list_models("bygait_light") if m.model_version == executed_job.candidate_version]
        assert len(cand_models) == 1
        cand_rec = cand_models[0]
        assert "dataset_id" in cand_rec.metadata
        assert "manifest_sha256" in cand_rec.metadata


        audit_events = worker.audit_trail.list_events()
        assert len(audit_events) >= 1
        assert audit_events[-1].candidate_version == executed_job.candidate_version

    def test_osnet_worker_cycle_with_dataset_builder_and_evaluation(self, isolated_cl_env):
        env = isolated_cl_env
        collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
        db = EmbeddingDatabase(db_dir=env["db_dir"])
        registry = ModelRegistry(registry_file=env["registry_file"])
        scheduler = DateAwareLearningScheduler(
            jobs_file=env["jobs_file"],
            collector=collector,
            db=db,
            min_training_embeddings=1,
            min_identities=1,
        )

        worker = BackgroundLearningWorker(
            scheduler=scheduler,
            registry=registry,
            collector=collector,
            db=db,
            candidate_artifacts_dir=env["candidates_dir"],
            timeout_seconds=60.0,
        )


        date_str = "2026-08-31"
        for ident in ["Subject_Gamma", "Subject_Delta"]:
            for i in range(4):
                vec = np.random.randn(512).astype(np.float32)
                vec /= np.linalg.norm(vec)
                crop = np.random.randint(0, 255, size=(256, 128, 3), dtype=np.uint8)
                obs = collector.record_observation(
                    camera_id=f"cam_{i % 2 + 1}",
                    track_id=200 + i,
                    vector=vec,
                    predicted_identity=ident,
                    confidence=0.95,
                    modality="appearance",
                    observation_date=date_str,
                    media_array=crop,
                )
                collector.verify_observation(obs.observation_id, verified_identity=ident)

        job = scheduler.create_learning_job(training_date=date_str, model_type="osnet_reid")
        assert job is not None


        executed_job = worker.execute_job_synchronous(job)
        assert executed_job.status in (LearningJobStatus.PROMOTED, LearningJobStatus.REJECTED)


        cand_models = [m for m in registry.list_models("osnet_reid") if m.model_version == executed_job.candidate_version]
        assert len(cand_models) == 1
        assert cand_models[0].embedding_dim == 512

    def test_model_registry_rollback_integrity(self, isolated_cl_env):
        env = isolated_cl_env
        registry = ModelRegistry(registry_file=env["registry_file"])

        active_before = registry.get_active_model("bygait_light")
        assert active_before is not None
        assert active_before.model_version == "v1.0.0"


        cand_path = Path(env["candidates_dir"]) / "bygait_v2.pth"
        cand_path.write_bytes(b"dummy_weights_v2")


        registry.register_candidate(
            model_version="v2.0.0",
            model_type="bygait_light",
            architecture="ByGaitLight",
            embedding_dim=256,
            artifact_path=str(cand_path),
        )
        registry.record_validation_result("v2.0.0", "bygait_light", passed=True, metrics={"tar": 92.0, "far": 0.0})
        registry.promote_version("v2.0.0", "bygait_light")

        active_after = registry.get_active_model("bygait_light")
        assert active_after is not None
        assert active_after.model_version == "v2.0.0"


        rolled_back = registry.rollback("bygait_light")
        assert rolled_back.model_version == "v1.0.0"
        assert rolled_back.deployment_status == ModelDeploymentStatus.ACTIVE


        active_restored = registry.get_active_model("bygait_light")
        assert active_restored is not None
        assert active_restored.model_version == "v1.0.0"
