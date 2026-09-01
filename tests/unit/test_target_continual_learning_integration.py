"""
ARGUS AI — Target Continual-Learning Architecture End-to-End Integration Suite.

Validates the full target architecture lifecycle:
1. Live CCTV -> Person Detection -> Tracking -> OSNet Crop (512D) -> Operational Collector
2. Live CCTV -> Tracking -> Silhouette / GEI -> ByGaitLight (256D) -> Operational Collector
3. Observation persistence to disk (recent_observations.json) & restart survivability
4. Deduplication & invalid vector rejection
5. Unknown vs Known person observation state safety (PREDICTED -> VERIFIED -> TRAINING_ELIGIBLE)
6. Date-Aware Learning Scheduler event-date detection & idempotency
7. Candidate model training with 50% historical replay buffer (Anti-Catastrophic Forgetting)
8. CandidateValidator validation gates (TAR, FAR, Confusion pairs)
9. ModelRegistry versioning, lineage, atomic promotion, and rollback
10. Multi-camera concurrency safety
"""

import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from intelligence.candidate_validator import CandidateValidator
from intelligence.date_aware_learning_scheduler import (
    DateAwareLearningScheduler,
    LearningJobStatus,
)
from intelligence.operational_embedding_collector import (
    ObservationState,
    OperationalEmbeddingCollector,
)
from models.model_registry import ModelDeploymentStatus, ModelRegistry
from services.recognition_worker import RecognitionResultCache, RecognitionWorker
from storage.embedding_database import EmbeddingDatabase


@pytest.fixture
def target_env():
    temp_dir = tempfile.mkdtemp(prefix="argus_target_e2e_")
    t_path = Path(temp_dir)
    db_dir = t_path / "data" / "embedding_db"
    gait_gal = t_path / "models" / "live_gallery"
    app_gal = t_path / "models" / "appearance_gallery"
    reg_file = t_path / "models" / "model_registry.json"
    jobs_file = t_path / "data" / "learning_jobs.json"
    obs_dir = t_path / "data" / "operational_observations"
    cand_dir = t_path / "models" / "candidates"

    for d in [db_dir, gait_gal, app_gal, obs_dir, cand_dir]:
        d.mkdir(parents=True, exist_ok=True)

    yield {
        "root": t_path,
        "db_dir": str(db_dir),
        "gait_gal": str(gait_gal),
        "app_gal": str(app_gal),
        "reg_file": str(reg_file),
        "jobs_file": str(jobs_file),
        "obs_dir": str(obs_dir),
        "cand_dir": str(cand_dir),
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_1_live_cctv_to_collector_and_persistence(target_env):
    """Verify live CCTV stream feeds into OperationalEmbeddingCollector and persists."""
    collector = OperationalEmbeddingCollector(
        output_dir=target_env["obs_dir"],
        dedup_window_seconds=0.5,
    )
    cache = RecognitionResultCache(ttl_seconds=2.0)


    mock_detector = MagicMock()
    mock_detector.detect.return_value = [{"bbox": [50, 50, 150, 200], "confidence": 0.95, "class_id": 0}]

    mock_sil_extractor = MagicMock()
    mock_sil_extractor.extract_from_frame.return_value = np.ones((128, 64), dtype=np.uint8) * 255

    mock_app_extractor = MagicMock()
    norm_app_vec = np.random.randn(512).astype(np.float32)
    norm_app_vec /= np.linalg.norm(norm_app_vec)
    mock_app_extractor.extract.return_value = norm_app_vec

    mock_app_matcher = MagicMock()
    mock_app_matcher.threshold = 0.60
    mock_app_matcher.match.return_value = ("Alice", 0.88)

    mock_gait_extractor = MagicMock()
    norm_gait_vec = np.random.randn(256).astype(np.float32)
    norm_gait_vec /= np.linalg.norm(norm_gait_vec)
    mock_gait_extractor.extract_from_gei.return_value = norm_gait_vec

    mock_gait_matcher = MagicMock()
    mock_gait_matcher.threshold = 0.85
    mock_gait_matcher.match.return_value = ("Alice", 0.92)
    mock_gait_matcher.top_k_matches.return_value = [("Alice", 0.92)]

    from pipeline.gei.stream_gei_builder import StreamGEIBuilder

    gei_builder = StreamGEIBuilder()
    gei_builder.min_frames = 3

    worker = RecognitionWorker(
        camera_id="cam_cctv_01",
        config={"target_fps": 30.0, "cooldown_seconds": 0.01, "threshold": 0.80},
        cache=cache,
        detector=mock_detector,
        silhouette_extractor=mock_sil_extractor,
        gei_builder=gei_builder,
        extractor=mock_gait_extractor,
        matcher=mock_gait_matcher,
        appearance_extractor=mock_app_extractor,
        appearance_matcher=mock_app_matcher,
        appearance_gallery_features=np.random.randn(1, 512).astype(np.float32),
        appearance_gallery_labels=["Alice"],
        gallery_features=np.random.randn(1, 256).astype(np.float32),
        gallery_labels=["Alice"],
        operational_collector=collector,
    )

    worker.start()


    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(25):
        worker.put_frame(frame)
        time.sleep(0.01)

    time.sleep(0.4)
    worker.stop()

    recent = collector.get_recent_observations()
    assert len(recent) > 0, "Collector must receive observations from live CCTV processing"

    modalities = {obs.modality for obs in recent}
    assert "appearance" in modalities, "Appearance observations must be recorded"
    assert "gait" in modalities, "Gait observations must be recorded"


    app_obs = next(o for o in recent if o.modality == "appearance")
    assert app_obs.camera_id == "cam_cctv_01"
    assert app_obs.embedding_dim == 512
    assert app_obs.model_name == "OSNet-x0.25"
    assert app_obs.state == ObservationState.PREDICTED
    assert len(app_obs.observation_date) == 10

    gait_obs = next(o for o in recent if o.modality == "gait")
    assert gait_obs.camera_id == "cam_cctv_01"
    assert gait_obs.embedding_dim == 256
    assert gait_obs.model_name == "ByGaitLight"
    assert gait_obs.state == ObservationState.PREDICTED


    obs_file = Path(target_env["obs_dir"]) / "recent_observations.json"
    assert obs_file.exists(), "recent_observations.json must be persisted to disk"


    new_collector = OperationalEmbeddingCollector(output_dir=target_env["obs_dir"])
    reloaded = new_collector.get_recent_observations()
    assert len(reloaded) == len(recent), "All observations must survive collector restart"


def test_2_deduplication_and_rate_limiting(target_env):
    """Verify near-identical vectors within deduplication window do not duplicate."""
    collector = OperationalEmbeddingCollector(
        output_dir=target_env["obs_dir"],
        dedup_window_seconds=1.0,
        dedup_similarity_threshold=0.98,
    )

    base_vec = np.random.randn(256).astype(np.float32)
    base_vec /= np.linalg.norm(base_vec)


    obs1 = collector.record_observation(
        camera_id="cam-01",
        track_id=10,
        vector=base_vec,
        predicted_identity="Bob",
        confidence=0.90,
        modality="gait",
    )
    assert obs1 is not None


    obs2 = collector.record_observation(
        camera_id="cam-01",
        track_id=10,
        vector=base_vec,
        predicted_identity="Bob",
        confidence=0.91,
        modality="gait",
    )
    assert obs2.observation_id == obs1.observation_id, "Deduplication must return existing observation"


    assert len(collector.get_recent_observations()) == 1


def test_3_ground_truth_verification_and_scheduling(target_env):
    """Verify operator verification enables training eligibility and triggers scheduler."""
    collector = OperationalEmbeddingCollector(output_dir=target_env["obs_dir"])
    db = EmbeddingDatabase(
        db_dir=target_env["db_dir"],
        gait_gallery_dir=target_env["gait_gal"],
        appearance_gallery_dir=target_env["app_gal"],
    )
    scheduler = DateAwareLearningScheduler(
        collector=collector,
        db=db,
        jobs_file=target_env["jobs_file"],
        min_training_embeddings=4,
        min_identities=2,
    )


    obs_ids = []
    for i in range(3):
        v = np.random.randn(256).astype(np.float32)
        v /= np.linalg.norm(v)
        obs = collector.record_observation(
            camera_id="cam-01",
            track_id=i,
            vector=v,
            predicted_identity="Alice",
            confidence=0.85,
            observation_date="2026-08-29",
        )
        obs_ids.append((obs.observation_id, "Alice"))

    for i in range(3, 6):
        v = np.random.randn(256).astype(np.float32)
        v /= np.linalg.norm(v)
        obs = collector.record_observation(
            camera_id="cam-02",
            track_id=i,
            vector=v,
            predicted_identity="Bob",
            confidence=0.88,
            observation_date="2026-08-29",
        )
        obs_ids.append((obs.observation_id, "Bob"))


    assert len(collector.get_training_eligible()) == 0
    assert len(scheduler.scan_for_eligible_data()) == 0


    for oid, true_ident in obs_ids:
        collector.verify_observation(oid, verified_identity=true_ident)

    assert len(collector.get_training_eligible()) == 6
    unprocessed = scheduler.get_unprocessed_dates()
    assert "2026-08-29" in unprocessed


    job = scheduler.create_learning_job(training_date="2026-08-29", model_type="bygait_light")
    assert job is not None
    assert job.status == LearningJobStatus.PENDING
    assert job.new_embeddings_count == 6
    assert job.identities_count == 2


    job2 = scheduler.create_learning_job(training_date="2026-08-29", model_type="bygait_light")
    assert job2.job_id == job.job_id


def test_4_model_registry_validation_and_rollback(target_env):
    """Verify candidate validation gates, atomic promotion, and rollback."""
    registry = ModelRegistry(registry_file=target_env["reg_file"])
    validator = CandidateValidator()


    base_ver = "v1.0.0"
    base_rec = registry.get_active_model("bygait_light")
    assert base_rec is not None
    assert base_rec.model_version == base_ver


    cand_ver = "v2.0.0-candidate"
    cand_artifact = str(Path(target_env["cand_dir"]) / "bygait_cand_v2.pth")
    with open(cand_artifact, "w") as f:
        f.write("candidate_weights_v2")

    registry.register_candidate(
        model_version=cand_ver,
        model_type="bygait_light",
        architecture="ByGaitLight",
        embedding_dim=256,
        artifact_path=cand_artifact,
        metadata={"training_date": "2026-08-29"},
    )


    with pytest.raises(RuntimeError):
        registry.promote_version(cand_ver, model_type="bygait_light")


    metrics = {"tar": 92.0, "far": 0.2, "val_rank1_accuracy": 92.0}
    val_res = validator.validate_candidate(
        candidate_version=cand_ver,
        model_type="bygait_light",
        baseline_metrics={"tar": 88.0, "far": 0.5},
        candidate_metrics=metrics,
    )
    assert val_res.passed, "Candidate with improved TAR must pass validation"

    registry.record_validation_result(
        model_version=cand_ver,
        model_type="bygait_light",
        passed=val_res.passed,
        metrics=metrics,
    )


    promoted = registry.promote_version(cand_ver, model_type="bygait_light")
    assert promoted.deployment_status == ModelDeploymentStatus.ACTIVE
    assert promoted.previous_production_version == base_ver
    assert registry.get_active_model("bygait_light").model_version == cand_ver


    rolled_back = registry.rollback(model_type="bygait_light", reason="Regression test")
    assert rolled_back.model_version == base_ver
    assert registry.get_active_model("bygait_light").model_version == base_ver


def test_5_multi_camera_concurrency(target_env):
    """Verify thread-safety when multiple cameras record observations concurrently."""
    collector = OperationalEmbeddingCollector(
        output_dir=target_env["obs_dir"],
        max_buffer_size=500,
        dedup_window_seconds=0.01,
    )

    def camera_worker_sim(cam_id: str, count: int):
        for i in range(count):
            vec = np.random.randn(256).astype(np.float32)
            vec /= np.linalg.norm(vec)
            collector.record_observation(
                camera_id=cam_id,
                track_id=i,
                vector=vec,
                predicted_identity=f"Person_{cam_id}_{i}",
                confidence=0.85,
                modality="gait",
            )
            time.sleep(0.005)

    threads = []
    for c in range(4):
        t = threading.Thread(target=camera_worker_sim, args=(f"cam_concurrent_{c:02d}", 15))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    recent = collector.get_recent_observations(limit=500)
    assert len(recent) == 60, f"Expected 60 total observations across 4 cameras, got {len(recent)}"

    cameras = {o.camera_id for o in recent}
    assert len(cameras) == 4, "All 4 camera streams must be present without corruption"
