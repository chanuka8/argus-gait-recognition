import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from enrollment.enrollment_lifecycle import (
    EnrollmentLifecycleManager,
    EnrollmentStatus,
)
from intelligence.background_learning_worker import BackgroundLearningWorker
from intelligence.candidate_validator import CandidateValidator
from intelligence.continuous_improvement_engine import ContinuousImprovementEngine
from intelligence.date_aware_learning_scheduler import (
    DateAwareLearningScheduler,
    LearningJobStatus,
)
from intelligence.operational_embedding_collector import (
    ObservationState,
    OperationalEmbeddingCollector,
)
from models.model_registry import ModelDeploymentStatus, ModelRegistry
from storage.embedding_database import EmbeddingDatabase


@pytest.fixture
def isolated_env():
    temp_dir = tempfile.mkdtemp(prefix="argus_test_date_aware_")
    t_path = Path(temp_dir)
    db_dir = t_path / "data" / "embedding_db"
    gait_gal = t_path / "models" / "live_gallery"
    app_gal = t_path / "models" / "appearance_gallery"
    reg_file = t_path / "models" / "model_registry.json"
    jobs_file = t_path / "data" / "learning_jobs.json"
    obs_dir = t_path / "data" / "observations"
    cand_dir = t_path / "models" / "candidates"
    input_dir = t_path / "data" / "new_input"

    db_dir.mkdir(parents=True, exist_ok=True)
    gait_gal.mkdir(parents=True, exist_ok=True)
    app_gal.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    yield {
        "root": t_path,
        "db_dir": str(db_dir),
        "gait_gal": str(gait_gal),
        "app_gal": str(app_gal),
        "reg_file": str(reg_file),
        "jobs_file": str(jobs_file),
        "obs_dir": str(obs_dir),
        "cand_dir": str(cand_dir),
        "input_dir": str(input_dir),
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def _seed_verified_observations(
    collector: OperationalEmbeddingCollector,
    date_str: str,
    subject_ids: list[str],
    samples_per_subject: int = 2,
    modality: str = "gait",
    dim: int = 256,
):
    for sid in subject_ids:
        for i in range(samples_per_subject):
            vec = np.random.randn(dim).astype(np.float32)
            obs = collector.record_observation(
                camera_id="cam-test-01",
                track_id=100 + i,
                vector=vec,
                predicted_identity=sid,
                confidence=0.92,
                modality=modality,
                observation_date=date_str,
            )
            collector.verify_observation(obs.observation_id, verified_identity=sid)





def test_1_no_new_embeddings_no_training_job(isolated_env):
    env = isolated_env
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(
        jobs_file=env["jobs_file"],
        collector=collector,
        db=db,
    )

    jobs = scheduler.check_and_schedule_new_dates()
    assert len(jobs) == 0
    assert len(scheduler.list_jobs()) == 0





def test_2_new_embeddings_date_x_creates_one_job(isolated_env):
    env = isolated_env
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(
        jobs_file=env["jobs_file"],
        collector=collector,
        db=db,
        min_training_embeddings=4,
        min_identities=2,
    )


    _seed_verified_observations(
        collector,
        date_str="2026-08-27",
        subject_ids=["SubjectA", "SubjectB"],
        samples_per_subject=2,
    )

    jobs = scheduler.check_and_schedule_new_dates()
    assert len(jobs) == 1
    assert jobs[0].training_date == "2026-08-27"
    assert jobs[0].status == LearningJobStatus.PENDING
    assert jobs[0].new_embeddings_count == 4





def test_3_multiple_embeddings_same_date_still_one_job(isolated_env):
    env = isolated_env
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(
        jobs_file=env["jobs_file"],
        collector=collector,
        db=db,
    )


    _seed_verified_observations(
        collector,
        date_str="2026-08-27",
        subject_ids=["SubjectA", "SubjectB", "SubjectC"],
        samples_per_subject=4,
    )

    jobs = scheduler.check_and_schedule_new_dates()
    assert len(jobs) == 1
    assert jobs[0].training_date == "2026-08-27"


    second_check = scheduler.check_and_schedule_new_dates()
    assert len(second_check) == 0





def test_4_new_embeddings_date_y_separate_job(isolated_env):
    env = isolated_env
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(
        jobs_file=env["jobs_file"],
        collector=collector,
        db=db,
    )


    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)
    jobs_x = scheduler.check_and_schedule_new_dates()
    assert len(jobs_x) == 1


    _seed_verified_observations(collector, "2026-08-29", ["SubC", "SubD"], 2)
    jobs_y = scheduler.check_and_schedule_new_dates()
    assert len(jobs_y) == 1
    assert jobs_y[0].training_date == "2026-08-29"

    all_jobs = scheduler.list_jobs()
    dates = {j.training_date for j in all_jobs}
    assert dates == {"2026-08-27", "2026-08-29"}





def test_5_previously_processed_date_no_duplicate(isolated_env):
    env = isolated_env
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(
        jobs_file=env["jobs_file"],
        collector=collector,
        db=db,
    )

    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)
    job = scheduler.create_learning_job("2026-08-27")
    job.status = LearningJobStatus.PROMOTED
    scheduler.update_job(job)


    new_jobs = scheduler.check_and_schedule_new_dates()
    assert len(new_jobs) == 0





def test_6_invalid_embedding_excluded(isolated_env):
    collector = OperationalEmbeddingCollector(output_dir=isolated_env["obs_dir"])


    nan_vec = np.ones(256, dtype=np.float32)
    nan_vec[10] = np.nan
    obs_nan = collector.record_observation(
        camera_id="cam-01", track_id=1, vector=nan_vec, predicted_identity="SubA", confidence=0.9
    )
    collector.verify_observation(obs_nan.observation_id, "SubA")


    bad_dim_vec = np.ones(128, dtype=np.float32)
    obs_bad = collector.record_observation(
        camera_id="cam-01", track_id=2, vector=bad_dim_vec, predicted_identity="SubB", confidence=0.9
    )
    collector.verify_observation(obs_bad.observation_id, "SubB")

    eligible = collector.get_training_eligible()
    assert len(eligible) == 0, "Invalid embeddings must never become TRAINING_ELIGIBLE"





def test_7_unverified_observation_excluded(isolated_env):
    collector = OperationalEmbeddingCollector(output_dir=isolated_env["obs_dir"])
    obs = collector.record_observation(
        camera_id="cam-01",
        track_id=1,
        vector=np.random.randn(256),
        predicted_identity="SubA",
        confidence=0.95,
    )
    assert obs.state == ObservationState.PREDICTED
    assert len(collector.get_training_eligible()) == 0





def test_8_review_required_low_quality_excluded(isolated_env):
    collector = OperationalEmbeddingCollector(output_dir=isolated_env["obs_dir"])
    obs = collector.record_observation(
        camera_id="cam-01",
        track_id=1,
        vector=np.random.randn(256),
        predicted_identity="SubA",
        confidence=0.75,
        quality_score=0.50,
    )

    collector.verify_observation(obs.observation_id, "SubA")
    assert obs.state == ObservationState.VERIFIED
    assert len(collector.get_training_eligible()) == 0





def test_9_training_eligible_included(isolated_env):
    collector = OperationalEmbeddingCollector(output_dir=isolated_env["obs_dir"])
    obs = collector.record_observation(
        camera_id="cam-01",
        track_id=1,
        vector=np.random.randn(256),
        predicted_identity="SubA",
        confidence=0.92,
        quality_score=0.85,
    )
    collector.verify_observation(obs.observation_id, "SubA")
    assert obs.state == ObservationState.TRAINING_ELIGIBLE
    assert len(collector.get_training_eligible()) == 1





def test_10_training_failure_isolated(isolated_env):
    env = isolated_env
    reg = ModelRegistry(registry_file=env["reg_file"])
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)
    worker = BackgroundLearningWorker(
        scheduler=scheduler,
        registry=reg,
        candidate_artifacts_dir=env["cand_dir"],
    )

    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)
    job = scheduler.create_learning_job("2026-08-27")


    with patch.object(worker, "_train_candidate_model", side_effect=RuntimeError("GPU OOM / Training crash")):
        res = worker.execute_job_synchronous(job)

    assert res.status == LearningJobStatus.FAILED
    assert "GPU OOM" in res.error_message

    active = reg.get_active_model("dual_modal_fusion")
    assert active is not None
    assert active.model_version == "v1.0.0"





def test_11_candidate_validation_failure_preserves_active(isolated_env):
    env = isolated_env
    reg = ModelRegistry(registry_file=env["reg_file"])
    validator = CandidateValidator()
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)
    worker = BackgroundLearningWorker(
        scheduler=scheduler,
        registry=reg,
        validator=validator,
        candidate_artifacts_dir=env["cand_dir"],
    )

    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)
    job = scheduler.create_learning_job("2026-08-27")


    degraded_metrics = {"tar": 50.0, "far": 8.0, "eer": 30.0}
    with patch.object(worker, "_train_candidate_model", return_value=(degraded_metrics, {"confusion_pair_far": 0.0})):
        res = worker.execute_job_synchronous(job)

    assert res.status == LearningJobStatus.REJECTED
    assert reg.get_active_model("dual_modal_fusion").model_version == "v1.0.0"





def test_12_candidate_success_promoted(isolated_env):
    env = isolated_env
    reg = ModelRegistry(registry_file=env["reg_file"])
    validator = CandidateValidator()
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)
    worker = BackgroundLearningWorker(
        scheduler=scheduler,
        registry=reg,
        validator=validator,
        candidate_artifacts_dir=env["cand_dir"],
    )

    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)
    job = scheduler.create_learning_job("2026-08-27")

    res = worker.execute_job_synchronous(job)
    assert res.status == LearningJobStatus.PROMOTED
    assert res.candidate_version is not None
    active = reg.get_active_model("dual_modal_fusion")
    assert active.model_version == res.candidate_version
    assert active.previous_production_version == "v1.0.0"





def test_13_runtime_regression_rollback(isolated_env):
    env = isolated_env
    reg = ModelRegistry(registry_file=env["reg_file"])
    engine = ContinuousImprovementEngine(registry=reg)


    engine.process_candidate(
        candidate_version="v2.0.0",
        model_type="dual_modal_fusion",
        architecture="LearnedLogistic-DualModal",
        embedding_dim=256,
        artifact_path="dummy.json",
        candidate_metrics={"tar": 75.0, "far": 1.5, "eer": 20.0},
    )
    assert reg.get_active_model("dual_modal_fusion").model_version == "v2.0.0"


    rolled_back = engine.trigger_runtime_regression_rollback(
        model_type="dual_modal_fusion",
        reason="Runtime drift detected in Zone 4",
    )
    assert rolled_back.model_version == "v1.0.0"
    assert reg.get_active_model("dual_modal_fusion").model_version == "v1.0.0"





def test_14_restart_during_running_job_safe_recovery(isolated_env):
    env = isolated_env
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler1 = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)

    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)
    job = scheduler1.create_learning_job("2026-08-27")
    job.status = LearningJobStatus.RUNNING
    scheduler1.update_job(job)


    scheduler2 = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)
    reloaded_job = scheduler2.get_job(job.job_id)
    assert reloaded_job.status == LearningJobStatus.INTERRUPTED





def test_15_concurrent_trigger_no_duplicate_jobs(isolated_env):
    env = isolated_env
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)

    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)

    results = []

    def trigger():
        job = scheduler.create_learning_job("2026-08-27")
        if job:
            results.append(job.job_id)

    threads = [threading.Thread(target=trigger) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


    assert len(set(results)) == 1
    assert len(scheduler.list_jobs()) == 1





def test_16_model_version_incompatibility_rejected(isolated_env):
    db = EmbeddingDatabase(db_dir=isolated_env["db_dir"])

    is_compat = db.check_model_compatibility(model_version="v2.0.0", expected_dim=1024, modality="gait")
    assert is_compat is False





def test_17_raw_media_deletion_safety(isolated_env):
    env = isolated_env
    db = EmbeddingDatabase(
        db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"], appearance_gallery_dir=env["app_gal"]
    )

    person_dir = Path(env["input_dir"]) / "SafeSubject"
    person_dir.mkdir(parents=True, exist_ok=True)
    raw_photo = person_dir / "safe.jpg"
    raw_photo.write_bytes(b"photo_data")

    mock_gait = MagicMock()
    mock_app = MagicMock()
    mock_app.extract.return_value = np.ones(512, dtype=np.float32)

    manager = EnrollmentLifecycleManager(
        db=db,
        gait_extractor=mock_gait,
        appearance_extractor=mock_app,
    )

    res = manager.enroll_from_media(
        person_id="SafeSubject",
        photo_paths=[raw_photo],
        auto_delete_raw=True,
    )
    assert res.status == EnrollmentStatus.EMBEDDING_ONLY
    assert not raw_photo.exists()





def test_18_camera_worker_unaffected_by_learning_failure(isolated_env):
    env = isolated_env
    reg = ModelRegistry(registry_file=env["reg_file"])
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)
    worker = BackgroundLearningWorker(
        scheduler=scheduler,
        registry=reg,
        candidate_artifacts_dir=env["cand_dir"],
    )

    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)
    job = scheduler.create_learning_job("2026-08-27")


    with patch.object(worker, "_train_candidate_model", side_effect=Exception("Severe Training Crash")):
        worker.execute_job_synchronous(job)


    active = reg.get_active_model("dual_modal_fusion")
    assert active is not None
    assert active.deployment_status == ModelDeploymentStatus.ACTIVE





def test_19_recognition_worker_responsive_during_learning(isolated_env):
    env = isolated_env
    reg = ModelRegistry(registry_file=env["reg_file"])
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)
    worker = BackgroundLearningWorker(
        scheduler=scheduler,
        registry=reg,
        candidate_artifacts_dir=env["cand_dir"],
    )

    worker.start()

    _seed_verified_observations(collector, "2026-08-27", ["SubA", "SubB"], 2)
    job = scheduler.create_learning_job("2026-08-27")
    worker.submit_job(job)


    inference_start = time.time()
    for _ in range(10):
        active = reg.get_active_model("dual_modal_fusion")
        assert active is not None
        time.sleep(0.01)
    inference_duration = time.time() - inference_start


    assert inference_duration < 1.0

    worker.stop(timeout=2.0)





def test_20_no_new_data_day_zero_resource_consumption(isolated_env):
    env = isolated_env
    collector = OperationalEmbeddingCollector(output_dir=env["obs_dir"])
    db = EmbeddingDatabase(db_dir=env["db_dir"], gait_gallery_dir=env["gait_gal"])
    scheduler = DateAwareLearningScheduler(jobs_file=env["jobs_file"], collector=collector, db=db)
    worker = BackgroundLearningWorker(
        scheduler=scheduler,
        candidate_artifacts_dir=env["cand_dir"],
    )


    with patch.object(worker, "_execute_job") as mock_exec:
        jobs = scheduler.check_and_schedule_new_dates()
        assert len(jobs) == 0
        mock_exec.assert_not_called()
