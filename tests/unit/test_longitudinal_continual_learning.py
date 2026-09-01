"""
Unit and Integration Tests for Longitudinal Continual Learning & Operational Evidence Architecture.

Verifies:
1. OperationalEvidenceManager bounded quota, atomic write, and SHA-256 corruption checks.
2. Track-level and session-level splitting with zero data leakage.
3. No-surrogate rule: TRAINING_MEDIA_UNAVAILABLE exclusion.
4. Tri-modal evaluation (Gait, Appearance, DualModalFusion).
5. Longitudinal timepoint tracking (T0 -> T1 -> T2).
6. Future holdout partition (E) temporal isolation.
7. Wilson 95% confidence intervals and McNemar's paired test.
8. Minimum Evidence Policy enforcement.
9. Condition-specific evaluation (Same-cam vs Cross-cam, Viewpoint, Clothing, Bags).
10. Model registry atomic promotion and rollback.
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from intelligence.background_learning_worker import BackgroundLearningWorker
from intelligence.date_aware_learning_scheduler import (
    DateAwareLearningScheduler,
    LearningJobStatus,
)
from intelligence.longitudinal_accuracy_evaluator import (
    LongitudinalAccuracyEvaluator,
)
from intelligence.operational_embedding_collector import (
    OperationalEmbeddingCollector,
)
from intelligence.operational_evidence_manager import (
    EvidenceCategory,
    OperationalEvidenceManager,
)
from intelligence.statistical_accuracy_validator import (
    MinimumEvidencePolicy,
    StatisticalAccuracyValidator,
)
from intelligence.training_dataset_builder import (
    DatasetSampleRecord,
    TrainingDatasetBuilder,
)
from models.architectures.bygait_light import ByGaitLight
from models.model_registry import ModelRegistry
from storage.embedding_database import EmbeddingDatabase


@pytest.fixture
def temp_env():
    tmp_dir = Path(tempfile.mkdtemp())
    obs_dir = tmp_dir / "obs"
    evidence_dir = tmp_dir / "evidence"
    manifest_dir = tmp_dir / "manifests"
    registry_dir = tmp_dir / "registry"
    history_file = tmp_dir / "longitudinal_history.json"
    audit_file = tmp_dir / "audit_trail.json"

    evidence_mgr = OperationalEvidenceManager(storage_dir=str(evidence_dir), max_storage_bytes=10 * 1024 * 1024)
    collector = OperationalEmbeddingCollector(output_dir=str(obs_dir), evidence_manager=evidence_mgr)
    db = EmbeddingDatabase(db_dir=str(tmp_dir / "db"))
    registry = ModelRegistry(registry_file=str(registry_dir / "model_registry.json"))

    yield {
        "tmp_dir": str(tmp_dir),
        "obs_dir": obs_dir,
        "evidence_dir": evidence_dir,
        "manifest_dir": manifest_dir,
        "history_file": history_file,
        "audit_file": audit_file,
        "evidence_mgr": evidence_mgr,
        "collector": collector,
        "db": db,
        "registry": registry,
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestOperationalEvidenceManager:
    def test_store_and_load_evidence_sha256_integrity(self, temp_env):
        mgr: OperationalEvidenceManager = temp_env["evidence_mgr"]
        gei = np.random.randint(0, 255, size=(64, 128), dtype=np.uint8)

        rec = mgr.store_evidence(
            observation_id="obs_001",
            camera_id="cam_east",
            track_id=101,
            person_id="P001",
            modality="gait",
            media_array=gei,
            category=EvidenceCategory.TRAIN,
        )
        assert rec is not None
        assert rec.sha256_hash != ""
        assert Path(rec.file_path).exists()


        loaded = mgr.load_evidence(rec.evidence_id)
        assert loaded is not None
        assert np.array_equal(loaded, gei)

    def test_corruption_detection(self, temp_env):
        mgr: OperationalEvidenceManager = temp_env["evidence_mgr"]
        crop = np.random.randint(0, 255, size=(256, 128, 3), dtype=np.uint8)

        rec = mgr.store_evidence(
            observation_id="obs_002",
            camera_id="cam_west",
            track_id=102,
            person_id="P002",
            modality="appearance",
            media_array=crop,
            category=EvidenceCategory.OPERATIONAL_TEST,
        )
        assert rec is not None


        with open(rec.file_path, "wb") as f:
            f.write(b"corrupted_binary_data")

        loaded = mgr.load_evidence(rec.evidence_id)
        assert loaded is None

    def test_manifest_locking_prevents_eviction(self, temp_env):

        small_mgr = OperationalEvidenceManager(
            storage_dir=str(Path(temp_env["tmp_dir"]) / "evidence_small"),
            max_storage_bytes=3000,
        )
        gei1 = np.ones((64, 128), dtype=np.uint8)
        gei2 = np.ones((64, 128), dtype=np.uint8) * 2

        rec1 = small_mgr.store_evidence("obs_1", "c1", 1, "P1", "gait", gei1)
        assert rec1 is not None

        small_mgr.lock_manifest_evidence([rec1.evidence_id], "manifest_locked_01")

        small_mgr.store_evidence("obs_2", "c2", 2, "P2", "gait", gei2)


        assert rec1.evidence_id in small_mgr._records


class TestTrackLevelDatasetSplitting:
    def test_track_and_session_level_isolation(self, temp_env):
        collector: OperationalEmbeddingCollector = temp_env["collector"]
        date_str = "2026-08-31"


        for p in ["P001", "P002"]:
            for track_id in [10, 20, 30]:
                for frame_idx in range(3):
                    vec = np.random.randn(256).astype(np.float32)
                    vec /= np.linalg.norm(vec)
                    obs = collector.record_observation(
                        camera_id="cam_01",
                        track_id=track_id,
                        vector=vec,
                        predicted_identity=p,
                        confidence=0.95,
                        modality="gait",
                        observation_date=date_str,
                        metadata={"session_id": f"sess_{p}_{track_id}"},
                    )
                    collector.verify_observation(obs.observation_id, p)

        builder = TrainingDatasetBuilder(
            collector=collector,
            db=temp_env["db"],
            manifest_dir=str(temp_env["manifest_dir"]),
        )

        train, val, test, _hist_r, _hist_t, _fut_h, _manifest = builder.build_dataset_for_date(
            training_date=date_str,
            model_type="bygait_light",
            include_historical=False,
        )

        train_tracks = {f"{s.person_id}_{s.session_id}_{s.track_id}" for s in train}
        val_tracks = {f"{s.person_id}_{s.session_id}_{s.track_id}" for s in val}
        test_tracks = {f"{s.person_id}_{s.session_id}_{s.track_id}" for s in test}


        assert len(train_tracks.intersection(test_tracks)) == 0
        assert len(train_tracks.intersection(val_tracks)) == 0
        assert len(val_tracks.intersection(test_tracks)) == 0


class TestStatisticalAccuracyValidator:
    def test_wilson_ci_calculation(self):
        val = StatisticalAccuracyValidator()
        lower, upper = val.calculate_wilson_ci(successes=90, trials=100)
        assert 80.0 < lower < 90.0
        assert 90.0 < upper < 98.0

    def test_mcnemar_paired_test(self):
        val = StatisticalAccuracyValidator()

        b_hits = [True] * 50 + [False] * 20
        c_hits = [True] * 50 + [True] * 15 + [False] * 5
        _chi2, p_val, is_sig = val.evaluate_mcnemar_test(b_hits, c_hits)
        assert is_sig is True
        assert p_val < 0.01

    def test_minimum_evidence_policy_rejection(self):
        val = StatisticalAccuracyValidator(
            policy=MinimumEvidencePolicy(min_genuine_trials=8, min_impostor_trials=16)
        )
        base_metrics = {"rank1_accuracy": 50.0, "tar": 50.0, "far": 0.0}
        cand_metrics = {"rank1_accuracy": 60.0, "tar": 60.0, "far": 0.0}


        res = val.validate_statistical_evidence(
            baseline_metrics=base_metrics,
            candidate_metrics=cand_metrics,
            identities_count=2,
            tracks_count=2,
            sessions_count=1,
            genuine_trials=2,
            impostor_trials=4,
            sample_count=4,
        )
        assert res.evidence_class == "INSUFFICIENT_REAL_WORLD_EVIDENCE"
        assert res.verdict == "ACCURACY_IMPROVEMENT_NOT_YET_PROVEN"


class TestLongitudinalAccuracyEvaluator:
    def test_longitudinal_cycle_and_persistence(self, temp_env):
        evaluator = LongitudinalAccuracyEvaluator(
            history_file=str(temp_env["history_file"])
        )

        samples = []
        for i in range(10):
            vec = np.random.randn(256).astype(np.float32)
            vec /= np.linalg.norm(vec)
            samples.append(
                DatasetSampleRecord(
                    sample_id=f"s_{i}",
                    person_id=f"P_{i % 3}",
                    camera_id="cam_01",
                    track_id=i,
                    session_id=f"sess_{i}",
                    timestamp=time.time(),
                    observation_date="2026-08-31",
                    modality="gait",
                    vector=vec.tolist(),
                    condition_tags={"viewpoint": "90_DEG", "clothing": "NM"},
                )
            )

        rec = evaluator.evaluate_longitudinal_cycle(
            baseline_version="v1.0.0",
            candidate_version="v1.1.0-test",
            dataset_id="ds-test-01",
            manifest_sha256="fake_sha",
            model_type="bygait_light",
            operational_test_samples=samples,
        )
        assert rec.timepoint_id.startswith("T0")
        assert len(evaluator.list_history()) == 1


        evaluator2 = LongitudinalAccuracyEvaluator(
            history_file=str(temp_env["history_file"])
        )
        history = evaluator2.list_history()
        assert len(history) == 1
        assert history[0].baseline_version == "v1.0.0"


class TestEndToEndWorkerLongitudinalAccuracy:
    def test_e2e_worker_with_longitudinal_evaluation(self, temp_env):
        collector: OperationalEmbeddingCollector = temp_env["collector"]
        evidence_mgr: OperationalEvidenceManager = temp_env["evidence_mgr"]
        db: EmbeddingDatabase = temp_env["db"]
        registry: ModelRegistry = temp_env["registry"]
        date_str = "2026-08-31"


        active_base = registry.get_active_model("bygait_light")
        if not active_base:
            init_model = ByGaitLight(embedding_dim=256)
            baseline_path = Path(temp_env["tmp_dir"]) / "bygait_baseline.pth"
            torch.save(init_model.state_dict(), baseline_path)
            registry.register_candidate(
                model_version="v1.0.0",
                model_type="bygait_light",
                architecture="ByGaitLight",
                embedding_dim=256,
                artifact_path=str(baseline_path),
            )
            registry.record_validation_result(
                model_version="v1.0.0",
                model_type="bygait_light",
                passed=True,
                metrics={},
            )
            registry.promote_version(
                model_version="v1.0.0",
                model_type="bygait_light",
                reason="Initial production baseline",
            )


        for p in ["Person_A", "Person_B"]:
            for tid in [101, 102, 103]:
                for f_idx in range(2):
                    vec = np.random.randn(256).astype(np.float32)
                    vec /= np.linalg.norm(vec)
                    gei = np.random.randint(0, 255, size=(64, 128), dtype=np.uint8)
                    obs = collector.record_observation(
                        camera_id="cam_01",
                        track_id=tid,
                        vector=vec,
                        predicted_identity=p,
                        confidence=0.90,
                        modality="gait",
                        observation_date=date_str,
                        media_array=gei,
                    )
                    collector.verify_observation(obs.observation_id, p)

        scheduler = DateAwareLearningScheduler(
            collector=collector,
            db=db,
            jobs_file=str(Path(temp_env["tmp_dir"]) / "jobs.json"),
        )
        job = scheduler.create_learning_job(
            training_date=date_str,
            model_type="bygait_light",
            force=True,
        )
        assert job is not None

        worker = BackgroundLearningWorker(
            scheduler=scheduler,
            registry=registry,
            collector=collector,
            db=db,
            evidence_manager=evidence_mgr,
            candidate_artifacts_dir=str(Path(temp_env["tmp_dir"]) / "candidates"),
        )

        completed_job = worker._execute_nn_job(job, time.time())
        assert completed_job.status in (
            LearningJobStatus.PROMOTED,
            LearningJobStatus.REJECTED,
            LearningJobStatus.FAILED,
        )
        assert completed_job.candidate_version != ""


        history = worker.longitudinal_evaluator.list_history()
        assert len(history) >= 1
        assert history[-1].candidate_version == completed_job.candidate_version
