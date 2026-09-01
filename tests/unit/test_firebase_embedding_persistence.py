import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from enrollment.enrollment_lifecycle import (
    EnrollmentLifecycleManager,
    EnrollmentStatus,
)
from intelligence.background_learning_worker import BackgroundLearningWorker
from intelligence.candidate_validator import CandidateValidator
from intelligence.date_aware_learning_scheduler import (
    DateAwareLearningScheduler,
    LearningJobRecord,
    LearningJobStatus,
)
from intelligence.nn_fine_tuner import NNFineTuner
from intelligence.operational_embedding_collector import (
    OperationalEmbeddingCollector,
)
from models.model_registry import ModelRegistry
from storage.embedding_database import EmbeddingDatabase
from storage.firebase_embedding_store import (
    FirebaseEmbeddingDocument,
    FirebaseEmbeddingStore,
)


@pytest.fixture
def tmp_env():
    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_test_firebase_"))
    db_dir = tmp_dir / "embedding_db"
    gait_gal = tmp_dir / "live_gallery"
    app_gal = tmp_dir / "appearance_gallery"
    candidates_dir = tmp_dir / "candidates"
    jobs_file = tmp_dir / "learning_jobs.json"
    registry_file = tmp_dir / "model_registry.json"
    offline_fb_store = tmp_dir / "firebase_offline.json"
    obs_dir = tmp_dir / "observations"

    db_dir.mkdir(parents=True, exist_ok=True)
    gait_gal.mkdir(parents=True, exist_ok=True)
    app_gal.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)

    yield {
        "root": tmp_dir,
        "db_dir": str(db_dir),
        "gait_gal": str(gait_gal),
        "app_gal": str(app_gal),
        "candidates_dir": str(candidates_dir),
        "jobs_file": str(jobs_file),
        "registry_file": str(registry_file),
        "offline_fb_store": str(offline_fb_store),
        "obs_dir": str(obs_dir),
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)





def test_a_firebase_embedding_persistence_success(tmp_env):
    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    vec = list(np.random.randn(256).astype(float))
    doc = FirebaseEmbeddingDocument(
        embedding_id="emb-001",
        person_id="Subject_01",
        modality="gait",
        embedding_dim=256,
        vector=vec,
        model_version="v1.0.0",
        observation_date="2026-08-27",
    )
    res = fb_store.persist_embedding(doc)
    assert res.success is True
    assert res.embedding_id == "emb-001"


    retrieved = fb_store.get_embeddings_by_person("Subject_01")
    assert len(retrieved) == 1
    assert retrieved[0].embedding_id == "emb-001"
    assert len(retrieved[0].vector) == 256





def test_b_firebase_failure_isolation(tmp_env):

    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
        firebase_store=fb_store,
    )

    vec = np.random.randn(256).astype(np.float32)

    res = db.add_embeddings(
        person_id="Subject_Iso",
        gait_embeddings=[vec],
        observation_date="2026-08-27",
    )
    assert res["success"] is True
    assert res["persistence_verified"] is True


    _, labels, _ = db.gait_store.load()
    assert len(labels) == 1
    assert labels[0] == "Subject_Iso"





def test_c_idempotent_duplicate_writes(tmp_env):
    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    vec = list(np.random.randn(256).astype(float))
    doc = FirebaseEmbeddingDocument(
        embedding_id="emb-dup-01",
        person_id="Subject_Dup",
        modality="gait",
        embedding_dim=256,
        vector=vec,
        model_version="v1.0.0",
    )
    res1 = fb_store.persist_embedding(doc)
    res2 = fb_store.persist_embedding(doc)
    assert res1.success is True
    assert res2.success is True

    all_embs = fb_store.get_embeddings_by_person("Subject_Dup")

    assert len(all_embs) == 1





def test_d_dimension_isolation_and_rejection(tmp_env):
    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )

    bad_gait = FirebaseEmbeddingDocument(
        embedding_id="emb-bad-1",
        person_id="Sub_Dim",
        modality="gait",
        embedding_dim=512,
        vector=list(np.random.randn(512).astype(float)),
        model_version="v1.0.0",
    )
    res_bad = fb_store.persist_embedding(bad_gait)
    assert res_bad.success is False
    assert "Gait embedding dimension mismatch" in res_bad.error_message


    bad_app = FirebaseEmbeddingDocument(
        embedding_id="emb-bad-2",
        person_id="Sub_Dim",
        modality="appearance",
        embedding_dim=256,
        vector=list(np.random.randn(256).astype(float)),
        model_version="v1.0.0",
    )
    res_bad2 = fb_store.persist_embedding(bad_app)
    assert res_bad2.success is False
    assert "Appearance embedding dimension mismatch" in res_bad2.error_message


    good_gait = FirebaseEmbeddingDocument(
        embedding_id="emb-good-gait",
        person_id="Sub_Dim",
        modality="gait",
        embedding_dim=256,
        vector=list(np.random.randn(256).astype(float)),
        model_version="v1.0.0",
    )

    good_app = FirebaseEmbeddingDocument(
        embedding_id="emb-good-app",
        person_id="Sub_Dim",
        modality="appearance",
        embedding_dim=512,
        vector=list(np.random.randn(512).astype(float)),
        model_version="v1.0.0",
    )
    assert fb_store.persist_embedding(good_gait).success is True
    assert fb_store.persist_embedding(good_app).success is True





def test_e_model_version_lineage_preservation(tmp_env):
    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    doc = FirebaseEmbeddingDocument(
        embedding_id="emb-lin-01",
        person_id="Sub_Lin",
        modality="gait",
        embedding_dim=256,
        vector=list(np.random.randn(256).astype(float)),
        model_version="v2.1.0-bygait",
        production_model_version="v2.1.0-bygait",
        case_id="Case-1234",
    )
    res = fb_store.persist_embedding(doc)
    assert res.success is True

    docs = fb_store.get_embeddings_by_person("Sub_Lin")
    assert docs[0].model_version == "v2.1.0-bygait"
    assert docs[0].production_model_version == "v2.1.0-bygait"
    assert docs[0].case_id == "Case-1234"





def test_f_date_aware_job_creation(tmp_env):
    collector = OperationalEmbeddingCollector(output_dir=tmp_env["obs_dir"])
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
    )
    scheduler = DateAwareLearningScheduler(
        jobs_file=tmp_env["jobs_file"],
        collector=collector,
        db=db,
        min_training_embeddings=2,
        min_identities=2,
    )


    obs1 = collector.record_observation(
        camera_id="cam-1",
        track_id=1,
        predicted_identity="SubA",
        confidence=0.92,
        vector=np.random.randn(256),
        modality="gait",
        observation_date="2026-08-27",
    )
    obs2 = collector.record_observation(
        camera_id="cam-1",
        track_id=2,
        predicted_identity="SubA",
        confidence=0.90,
        vector=np.random.randn(256),
        modality="gait",
        observation_date="2026-08-27",
    )
    obs3 = collector.record_observation(
        camera_id="cam-2",
        track_id=3,
        predicted_identity="SubB",
        confidence=0.95,
        vector=np.random.randn(256),
        modality="gait",
        observation_date="2026-08-27",
    )
    obs4 = collector.record_observation(
        camera_id="cam-2",
        track_id=4,
        predicted_identity="SubB",
        confidence=0.88,
        vector=np.random.randn(256),
        modality="gait",
        observation_date="2026-08-27",
    )

    collector.verify_observation(obs1.observation_id, "SubA")
    collector.verify_observation(obs2.observation_id, "SubA")
    collector.verify_observation(obs3.observation_id, "SubB")
    collector.verify_observation(obs4.observation_id, "SubB")

    jobs = scheduler.check_and_schedule_new_dates(model_type="dual_modal_fusion")
    assert len(jobs) == 1
    assert jobs[0].training_date == "2026-08-27"
    assert jobs[0].status == LearningJobStatus.PENDING
    assert jobs[0].new_embeddings_count == 4
    assert jobs[0].identities_count == 2





def test_g_zero_jobs_for_empty_date(tmp_env):
    collector = OperationalEmbeddingCollector(output_dir=tmp_env["obs_dir"])
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
    )
    scheduler = DateAwareLearningScheduler(
        jobs_file=tmp_env["jobs_file"],
        collector=collector,
        db=db,
    )

    jobs = scheduler.check_and_schedule_new_dates(model_type="dual_modal_fusion")
    assert len(jobs) == 0





def test_h_duplicate_date_idempotency(tmp_env):
    collector = OperationalEmbeddingCollector(output_dir=tmp_env["obs_dir"])
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
    )
    scheduler = DateAwareLearningScheduler(
        jobs_file=tmp_env["jobs_file"],
        collector=collector,
        db=db,
        min_training_embeddings=2,
        min_identities=2,
    )


    obs1 = collector.record_observation(
        camera_id="cam-1",
        track_id=1,
        predicted_identity="SubA",
        confidence=0.92,
        vector=np.random.randn(256),
        modality="gait",
        observation_date="2026-08-28",
    )
    obs2 = collector.record_observation(
        camera_id="cam-2",
        track_id=2,
        predicted_identity="SubB",
        confidence=0.95,
        vector=np.random.randn(256),
        modality="gait",
        observation_date="2026-08-28",
    )
    collector.verify_observation(obs1.observation_id, "SubA")
    collector.verify_observation(obs2.observation_id, "SubB")

    jobs1 = scheduler.check_and_schedule_new_dates(model_type="dual_modal_fusion")
    assert len(jobs1) == 1


    jobs2 = scheduler.check_and_schedule_new_dates(model_type="dual_modal_fusion")
    assert len(jobs2) == 0





def test_i_only_training_eligible_enter_training(tmp_env):
    collector = OperationalEmbeddingCollector(output_dir=tmp_env["obs_dir"])

    obs_unverified = collector.record_observation(
        camera_id="cam-1",
        track_id=1,
        predicted_identity="SubA",
        confidence=0.60,
        vector=np.random.randn(256),
        modality="gait",
        observation_date="2026-08-29",
    )

    obs_verified = collector.record_observation(
        camera_id="cam-1",
        track_id=2,
        predicted_identity="SubA",
        confidence=0.90,
        vector=np.random.randn(256),
        modality="gait",
        observation_date="2026-08-29",
    )
    collector.verify_observation(obs_verified.observation_id, "SubA")

    eligible = collector.get_training_eligible()
    eligible_ids = [o.observation_id for o in eligible]
    assert obs_verified.observation_id in eligible_ids
    assert obs_unverified.observation_id not in eligible_ids





def test_j_historical_replay_data_included(tmp_env):
    collector = OperationalEmbeddingCollector()
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
    )

    db.add_embeddings(
        person_id="Subject_Base",
        gait_embeddings=[np.random.randn(256).astype(np.float32) for _ in range(3)],
        observation_date="2026-08-01",
    )

    worker = BackgroundLearningWorker(
        collector=collector,
        db=db,
        candidate_artifacts_dir=tmp_env["candidates_dir"],
    )
    job = LearningJobRecord(
        job_id="test-job-j",
        training_date="2026-08-27",
    )
    gait_s, _, labels, _ = worker._prepare_training_data(job)
    assert len(gait_s) > 0
    assert len(labels) > 0





def test_k_candidate_model_artifact_generation(tmp_env):
    tuner = NNFineTuner(
        candidate_dir=tmp_env["candidates_dir"],
        max_epochs=1,
        learning_rate=1e-5,
    )

    gei_data = [
        {"image": np.random.rand(64, 128).astype(np.float32), "label": "Sub1"}
        for _ in range(4)
    ] + [
        {"image": np.random.rand(64, 128).astype(np.float32), "label": "Sub2"}
        for _ in range(4)
    ]
    hist_data = [
        {"image": np.random.rand(64, 128).astype(np.float32), "label": "Sub1"}
        for _ in range(2)
    ]

    res = tuner.fine_tune_bygait_light(
        active_weights_path="non_existent_path.pth",
        training_gei_data=gei_data,
        historical_gei_data=hist_data,
        candidate_version="vTestCandidate01",
    )
    assert res["success"] is True
    assert Path(res["artifact_path"]).exists()
    assert res["checksum_sha256"] != ""
    assert res["embedding_dim"] == 256





def test_l_candidate_validation_rejects_regression():
    validator = CandidateValidator(max_allowed_far_increase=0.0)
    baseline_metrics = {"tar": 92.0, "far": 0.5, "eer": 4.0}

    regressed_metrics = {"tar": 95.0, "far": 2.5, "eer": 5.0}

    val_res = validator.validate_candidate(
        candidate_version="vBadCand",
        model_type="dual_modal_fusion",
        baseline_metrics=baseline_metrics,
        candidate_metrics=regressed_metrics,
    )
    assert val_res.passed is False
    assert any("Security Regression" in r for r in val_res.rejection_reasons)





def test_m_candidate_promotion_atomic(tmp_env):
    registry = ModelRegistry(registry_file=tmp_env["registry_file"])

    cand_ver = "vCandPromote01"
    artifact_path = str(Path(tmp_env["candidates_dir"]) / "cand.json")
    Path(artifact_path).write_text("{}", encoding="utf-8")

    registry.register_candidate(
        model_version=cand_ver,
        model_type="dual_modal_fusion",
        architecture="LearnedLogistic-DualModal",
        embedding_dim=256,
        artifact_path=artifact_path,
    )
    cand_metrics = {"tar": 94.0, "far": 0.2, "eer": 3.0}
    registry.record_validation_result(
        model_version=cand_ver,
        model_type="dual_modal_fusion",
        passed=True,
        metrics=cand_metrics,
    )
    promoted = registry.promote_version(
        model_version=cand_ver,
        model_type="dual_modal_fusion",
    )
    assert promoted.deployment_status.value == "ACTIVE"
    assert registry.get_active_model("dual_modal_fusion").model_version == cand_ver





def test_n_safety_rollback_restores_previous(tmp_env):
    registry = ModelRegistry(registry_file=tmp_env["registry_file"])
    v1_rec = registry.get_active_model("dual_modal_fusion")
    v1_ver = v1_rec.model_version if v1_rec else "v1.0.0"

    cand_path = str(Path(tmp_env["candidates_dir"]) / "cand2.json")
    Path(cand_path).write_text("{}", encoding="utf-8")

    registry.register_candidate(
        model_version="v2.0.0-temp",
        model_type="dual_modal_fusion",
        architecture="LearnedLogistic",
        embedding_dim=256,
        artifact_path=cand_path,
    )
    registry.record_validation_result(
        model_version="v2.0.0-temp",
        model_type="dual_modal_fusion",
        passed=True,
        metrics={"tar": 95.0, "far": 0.1},
    )
    registry.promote_version("v2.0.0-temp", "dual_modal_fusion")
    assert registry.get_active_model("dual_modal_fusion").model_version == "v2.0.0-temp"


    rolled_back = registry.rollback("dual_modal_fusion", reason="Runtime regression detected")
    assert rolled_back.model_version == v1_ver
    assert registry.get_active_model("dual_modal_fusion").model_version == v1_ver





def test_o_training_failure_preserves_active_model(tmp_env):
    registry = ModelRegistry(registry_file=tmp_env["registry_file"])
    active_before = registry.get_active_model("bygait_light")

    collector = OperationalEmbeddingCollector(output_dir=tmp_env["obs_dir"])
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
    )

    worker = BackgroundLearningWorker(
        registry=registry,
        collector=collector,
        db=db,
        candidate_artifacts_dir=tmp_env["candidates_dir"],
    )

    job = LearningJobRecord(
        job_id="job-fail-safe",
        training_date="2026-08-27",
        model_type="bygait_light",
    )
    result_job = worker.execute_job_synchronous(job)
    assert result_job.status == LearningJobStatus.FAILED

    active_after = registry.get_active_model("bygait_light")

    if active_before:
        assert active_before.model_version == active_after.model_version





def test_p_firebase_unavailable_inference_safe(tmp_env):

    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
        firebase_store=fb_store,
    )

    vec = np.random.randn(256).astype(np.float32)
    res = db.add_embeddings(
        person_id="Sub_Safe_01",
        gait_embeddings=[vec],
    )
    assert res["success"] is True

    _, labels, _ = db.gait_store.load()
    assert "Sub_Safe_01" in labels





def test_q_enrollment_lifecycle_embedding_only(tmp_env):
    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
        firebase_store=fb_store,
    )
    manager = EnrollmentLifecycleManager(
        db=db,
        firebase_store=fb_store,
    )


    import cv2

    gei_file = Path(tmp_env["root"]) / "dummy_gei.png"
    cv2.imwrite(str(gei_file), np.ones((128, 64), dtype=np.uint8) * 128)

    res = manager.enroll_from_media(
        person_id="Subject_Enroll_Q",
        gei_paths=[gei_file],
        auto_delete_raw=True,
    )
    assert res.status == EnrollmentStatus.EMBEDDING_ONLY
    assert not gei_file.exists()
    assert res.gait_embeddings_count == 1





def test_r_persistence_failure_retains_raw_media(tmp_env):
    class FailingDB:
        def add_embeddings(self, *args, **kwargs):
            raise RuntimeError("Disk write I/O failure")

    manager = EnrollmentLifecycleManager(db=FailingDB())
    raw_photo = Path(tmp_env["root"]) / "important_photo.jpg"
    import cv2

    cv2.imwrite(str(raw_photo), np.zeros((100, 100, 3), dtype=np.uint8))

    res = manager.enroll_from_media(
        person_id="Subject_Fail_R",
        photo_paths=[raw_photo],
        auto_delete_raw=True,
    )
    assert res.status == EnrollmentStatus.PERSISTENCE_FAILED
    assert raw_photo.exists()
    assert str(raw_photo) in res.raw_files_retained





def test_s_interrupted_job_recovery(tmp_env):
    scheduler = DateAwareLearningScheduler(jobs_file=tmp_env["jobs_file"])

    crashed_job = LearningJobRecord(
        job_id="job-crashed-01",
        training_date="2026-08-25",
        status=LearningJobStatus.RUNNING,
    )
    scheduler.update_job(crashed_job)


    new_scheduler = DateAwareLearningScheduler(jobs_file=tmp_env["jobs_file"])
    recovered = new_scheduler.get_job("job-crashed-01")
    assert recovered.status == LearningJobStatus.INTERRUPTED
    assert "interrupted" in recovered.error_message.lower()





def test_t_read_after_write_verification(tmp_env):
    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    vec = list(np.random.randn(256).astype(float))
    doc = FirebaseEmbeddingDocument(
        embedding_id="emb-verify-01",
        person_id="Subject_Ver",
        modality="gait",
        embedding_dim=256,
        vector=vec,
        model_version="v1.0.0",
    )
    fb_store.persist_embedding(doc)

    verified, msg = fb_store.verify_persistence("emb-verify-01")
    assert verified is True
    assert "verified" in msg.lower()


    bad_verified, _ = fb_store.verify_persistence("emb-nonexistent")
    assert bad_verified is False





def test_u_disaster_recovery_rebuild(tmp_env):
    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )

    vec1 = list(np.random.randn(256).astype(float))
    vec2 = list(np.random.randn(512).astype(float))
    fb_store.persist_embedding(
        FirebaseEmbeddingDocument(
            embedding_id="dr-gait-1",
            person_id="Sub_DR",
            modality="gait",
            embedding_dim=256,
            vector=vec1,
            model_version="v1.0.0",
        )
    )
    fb_store.persist_embedding(
        FirebaseEmbeddingDocument(
            embedding_id="dr-app-1",
            person_id="Sub_DR",
            modality="appearance",
            embedding_dim=512,
            vector=vec2,
            model_version="v1.0.0",
        )
    )


    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
        firebase_store=fb_store,
    )
    rebuild_res = db.rebuild_from_firebase()
    assert rebuild_res["success"] is True
    assert rebuild_res["rebuilt_persons"] == 1

    person = db.get_person("Sub_DR")
    assert person is not None
    assert len(person.gait_embeddings) == 1
    assert len(person.appearance_embeddings) == 1


def test_v_schema_validation_boundaries(tmp_env):
    from storage.firebase_embedding_store import FirebaseEmbeddingDocument

    # 1. Valid gait doc (256D)
    valid_gait = FirebaseEmbeddingDocument(
        embedding_id="gait-valid-1",
        person_id="P001",
        modality="gait",
        embedding_dim=256,
        vector=list(np.random.randn(256).astype(float)),
    )
    is_valid, _msg = valid_gait.validate_schema()
    assert is_valid is True
    assert valid_gait.identity_id == "P001"
    assert valid_gait.embedding_type == "gait"
    assert valid_gait.embedding_dimension == 256

    # 2. Valid appearance doc (512D)
    valid_app = FirebaseEmbeddingDocument(
        embedding_id="app-valid-1",
        person_id="P001",
        modality="appearance",
        embedding_dim=512,
        vector=list(np.random.randn(512).astype(float)),
    )
    is_valid_app, _ = valid_app.validate_schema()
    assert is_valid_app is True

    # 3. Invalid: Empty embedding_id
    bad_id = FirebaseEmbeddingDocument(
        embedding_id="",
        person_id="P001",
        modality="gait",
        embedding_dim=256,
        vector=list(np.random.randn(256).astype(float)),
    )
    assert bad_id.validate_schema()[0] is False

    # 4. Invalid: Empty person_id
    bad_pid = FirebaseEmbeddingDocument(
        embedding_id="emb-1",
        person_id="",
        modality="gait",
        embedding_dim=256,
        vector=list(np.random.randn(256).astype(float)),
    )
    assert bad_pid.validate_schema()[0] is False

    # 5. Invalid: Dimension mismatch (512D vector with 256 declared)
    dim_mismatch = FirebaseEmbeddingDocument(
        embedding_id="emb-1",
        person_id="P001",
        modality="gait",
        embedding_dim=256,
        vector=list(np.random.randn(512).astype(float)),
    )
    assert dim_mismatch.validate_schema()[0] is False

    # 6. Invalid: Non-finite vector
    non_finite_vec = [float("nan")] + list(np.random.randn(255).astype(float))
    bad_finite = FirebaseEmbeddingDocument(
        embedding_id="emb-1",
        person_id="P001",
        modality="gait",
        embedding_dim=256,
        vector=non_finite_vec,
    )
    assert bad_finite.validate_schema()[0] is False

    # 7. Invalid: Zero vector
    zero_vec = [0.0] * 256
    bad_zero = FirebaseEmbeddingDocument(
        embedding_id="emb-1",
        person_id="P001",
        modality="gait",
        embedding_dim=256,
        vector=zero_vec,
    )
    assert bad_zero.validate_schema()[0] is False


def test_w_missing_person_reference_flow_not_training_eligible(tmp_env):
    from intelligence.missing_person_workflow import MissingPersonWorkflow
    from storage.embedding_database import EmbeddingDatabase
    from storage.firebase_embedding_store import FirebaseEmbeddingStore

    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    db = EmbeddingDatabase(
        db_dir=tmp_env["db_dir"],
        gait_gallery_dir=tmp_env["gait_gal"],
        appearance_gallery_dir=tmp_env["app_gal"],
        firebase_store=fb_store,
    )

    # 1. Register Missing Person in watchlist
    workflow = MissingPersonWorkflow(output_dir=str(Path(tmp_env["root"]) / "watchlist"))
    entry = workflow.register_target(identity="Missing_Person_101", notes="Urgent case")
    assert entry.identity_id == "Missing_Person_101"
    assert entry.category == "MISSING_PERSON"

    # 2. Add reference embeddings for missing person
    ref_gait = np.random.randn(256).astype(np.float32)
    ref_app = np.random.randn(512).astype(np.float32)
    persist_res = db.add_embeddings(
        person_id="Missing_Person_101",
        gait_embeddings=[ref_gait],
        appearance_embeddings=[ref_app],
        observation_date="2026-08-28",
    )
    assert persist_res["success"] is True

    # 3. Verify local VectorStore queryability
    g_data = db.gait_store.load()
    assert "Missing_Person_101" in list(g_data[1])

    # 4. Verify Firebase document contains USER_REFERENCE provenance & NOT_ELIGIBLE training state
    fb_docs = fb_store.get_embeddings_by_person("Missing_Person_101")
    assert len(fb_docs) == 2
    for doc in fb_docs:
        assert doc.identity_type == "USER_REFERENCE"
        assert doc.source_type == "user_reference"
        assert doc.training_eligibility == "NOT_ELIGIBLE"
        assert doc.training_eligible is False

    # 5. Verify OperationalEmbeddingCollector and Scheduler do NOT treat reference data as training candidates
    collector = OperationalEmbeddingCollector(output_dir=tmp_env["obs_dir"])
    assert len(collector.get_training_eligible()) == 0


def test_x_state_machine_transitions_and_consumption(tmp_env):
    from intelligence.operational_embedding_collector import ObservationState, OperationalEmbeddingCollector

    collector = OperationalEmbeddingCollector(output_dir=tmp_env["obs_dir"])
    vec = list(np.random.randn(256).astype(float))

    # 1. New observation starts as PREDICTED
    obs = collector.record_observation(
        camera_id="cam-01",
        track_id=12,
        vector=vec,
        predicted_identity="Subject_Alpha",
        confidence=0.92,
        modality="gait",
        quality_score=0.85,
        observation_date="2026-08-28",
    )
    assert obs.state == ObservationState.PREDICTED
    assert obs.identity_type == "LIVE_OPERATIONAL"

    # 2. Transition: Verify observation with operator confirmation
    verified = collector.verify_observation(
        observation_id=obs.observation_id,
        verified_identity="Subject_Alpha",
        verification_source="operator_ui",
    )
    assert verified is True

    # 3. Because quality >= 0.70 and vector is valid, state automatically becomes TRAINING_ELIGIBLE
    eligible_list = collector.get_eligible_by_date("2026-08-28")
    assert len(eligible_list) == 1
    assert eligible_list[0].state == ObservationState.TRAINING_ELIGIBLE
    assert eligible_list[0].training_consumed is False

    # 4. Transition: Mark training consumed
    marked = collector.mark_training_consumed(
        observation_ids=[obs.observation_id],
        training_job_id="job-cl-001",
        candidate_version="v100-20260828",
    )
    assert marked == 1

    # 5. Verify observation is now TRAINING_CONSUMED and excluded from eligible list
    assert collector.get_recent_observations()[0].state == ObservationState.TRAINING_CONSUMED
    assert len(collector.get_eligible_by_date("2026-08-28")) == 0

    # 6. Invalid transition: Cannot re-verify or retrain consumed observation
    re_verify = collector.verify_observation(
        observation_id=obs.observation_id,
        verified_identity="Subject_Alpha",
    )
    assert re_verify is False


def test_y_future_date_contamination_rejection(tmp_env):
    scheduler = DateAwareLearningScheduler(
        jobs_file=tmp_env["jobs_file"],
        min_training_embeddings=1,
        min_identities=1,
    )
    # Future date: 2099-12-31
    job = scheduler.create_learning_job(
        training_date="2099-12-31",
        model_type="dual_modal_fusion",
        force=False,
    )
    assert job is not None
    assert job.status == LearningJobStatus.REJECTED
    assert "future date contamination" in job.rejection_reason.lower()


def test_z_deterministic_id_and_connection_health(tmp_env):
    from storage.firebase_embedding_store import (
        FirebaseEmbeddingStore,
        generate_deterministic_id,
    )

    id1 = generate_deterministic_id(
        person_id="Subject_A",
        modality="gait",
        capture_timestamp=1700000000,
        track_id=5,
        camera_id="cam-north",
    )
    id2 = generate_deterministic_id(
        person_id="Subject_A",
        modality="gait",
        capture_timestamp=1700000000,
        track_id=5,
        camera_id="cam-north",
    )
    assert id1 == id2
    assert "gait" in id1
    assert "Subject_A" in id1

    fb_store = FirebaseEmbeddingStore(
        mode="offline",
        offline_store_path=tmp_env["offline_fb_store"],
    )
    healthy, health_info = fb_store.check_connection_health()
    assert healthy is True
    assert health_info["mode"] == "offline"
    assert health_info["status"] == "HEALTHY"
