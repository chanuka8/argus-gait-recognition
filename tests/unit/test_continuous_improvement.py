"""
Comprehensive Unit and Integration Tests for ARGUS AI Continuous Improvement Architecture,
Safe Enrollment Lifecycle, Automatic Raw Data Cleanup, Model Registry, and Rollback System.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from enrollment.enrollment_lifecycle import (
    EnrollmentLifecycleManager,
    EnrollmentStatus,
)
from intelligence.candidate_validator import CandidateValidator
from intelligence.continuous_improvement_engine import ContinuousImprovementEngine
from intelligence.drift_detector import DriftDetector
from intelligence.operational_embedding_collector import (
    ObservationState,
    OperationalEmbeddingCollector,
)
from models.model_registry import ModelDeploymentStatus, ModelRegistry
from storage.embedding_database import EmbeddingDatabase
from storage.vector_store import VectorStore


@pytest.fixture
def temp_environment():
    """Create isolated temporary workspace directory structure."""
    temp_dir = tempfile.mkdtemp(prefix="argus_ci_test_")
    t_path = Path(temp_dir)
    db_dir = t_path / "data" / "embedding_db"
    gait_gal = t_path / "models" / "live_gallery"
    app_gal = t_path / "models" / "appearance_gallery"
    reg_file = t_path / "models" / "model_registry.json"
    obs_dir = t_path / "data" / "observations"
    input_dir = t_path / "data" / "new_input"

    db_dir.mkdir(parents=True, exist_ok=True)
    gait_gal.mkdir(parents=True, exist_ok=True)
    app_gal.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    yield {
        "root": t_path,
        "db_dir": str(db_dir),
        "gait_gal": str(gait_gal),
        "app_gal": str(app_gal),
        "reg_file": str(reg_file),
        "obs_dir": str(obs_dir),
        "input_dir": str(input_dir),
    }

    shutil.rmtree(temp_dir, ignore_errors=True)





def test_a_successful_enrollment_and_raw_cleanup(temp_environment):
    env = temp_environment
    db = EmbeddingDatabase(
        db_dir=env["db_dir"],
        gait_gallery_dir=env["gait_gal"],
        appearance_gallery_dir=env["app_gal"],
    )


    person_dir = Path(env["input_dir"]) / "Subject01"
    person_dir.mkdir(parents=True, exist_ok=True)
    dummy_photo = person_dir / "photo1.jpg"
    dummy_photo.write_bytes(b"dummy_photo_data")
    dummy_gei = person_dir / "gei1.png"
    dummy_gei.write_bytes(b"dummy_gei_data")


    mock_gait_ext = MagicMock()
    mock_gait_ext.extract.return_value = np.ones((256,), dtype=np.float32)
    mock_app_ext = MagicMock()
    mock_app_ext.extract.return_value = np.ones((512,), dtype=np.float32)

    manager = EnrollmentLifecycleManager(
        db=db,
        gait_extractor=mock_gait_ext,
        appearance_extractor=mock_app_ext,
    )

    result = manager.enroll_from_media(
        person_id="Subject01",
        photo_paths=[dummy_photo],
        gei_paths=[dummy_gei],
        auto_delete_raw=True,
    )


    assert result.status == EnrollmentStatus.EMBEDDING_ONLY
    assert result.gait_embeddings_count == 1
    assert result.appearance_embeddings_count == 1
    assert not dummy_photo.exists(), "Raw photo must be deleted after persistence verification"
    assert not dummy_gei.exists(), "Raw GEI must be deleted after persistence verification"


    person_rec = db.get_person("Subject01")
    assert person_rec is not None
    assert len(person_rec.gait_embeddings) == 1
    assert len(person_rec.appearance_embeddings) == 1

    g_store = VectorStore(gallery_dir=env["gait_gal"])
    _, lbls, _ = g_store.load()
    assert "Subject01" in list(lbls)





def test_b_embedding_generation_failure_preserves_raw_media(temp_environment):
    env = temp_environment
    db = EmbeddingDatabase(
        db_dir=env["db_dir"],
        gait_gallery_dir=env["gait_gal"],
        appearance_gallery_dir=env["app_gal"],
    )

    person_dir = Path(env["input_dir"]) / "Subject02"
    person_dir.mkdir(parents=True, exist_ok=True)
    corrupt_photo = person_dir / "corrupt.jpg"
    corrupt_photo.write_bytes(b"corrupted_media")

    mock_gait_ext = MagicMock()
    mock_gait_ext.extract.return_value = None
    mock_app_ext = MagicMock()
    mock_app_ext.extract.return_value = None

    manager = EnrollmentLifecycleManager(
        db=db,
        gait_extractor=mock_gait_ext,
        appearance_extractor=mock_app_ext,
    )

    result = manager.enroll_from_media(
        person_id="Subject02",
        photo_paths=[corrupt_photo],
        auto_delete_raw=True,
    )


    assert result.status == EnrollmentStatus.PROCESSING_FAILED
    assert corrupt_photo.exists(), "SAFETY INVARIANT: Raw media MUST be preserved on generation failure"
    assert str(corrupt_photo) in result.raw_files_retained





def test_c_storage_failure_preserves_raw_media(temp_environment):
    env = temp_environment
    db = EmbeddingDatabase(
        db_dir=env["db_dir"],
        gait_gallery_dir=env["gait_gal"],
        appearance_gallery_dir=env["app_gal"],
    )

    person_dir = Path(env["input_dir"]) / "Subject03"
    person_dir.mkdir(parents=True, exist_ok=True)
    raw_photo = person_dir / "photo.jpg"
    raw_photo.write_bytes(b"valid_bytes")

    mock_gait_ext = MagicMock()
    mock_gait_ext.extract.return_value = np.ones((256,), dtype=np.float32)
    mock_app_ext = MagicMock()
    mock_app_ext.extract.return_value = np.ones((512,), dtype=np.float32)

    manager = EnrollmentLifecycleManager(
        db=db,
        gait_extractor=mock_gait_ext,
        appearance_extractor=mock_app_ext,
    )


    with patch.object(db, "add_embeddings", side_effect=OSError("Disk write failure")):
        result = manager.enroll_from_media(
            person_id="Subject03",
            photo_paths=[raw_photo],
            auto_delete_raw=True,
        )


    assert result.status == EnrollmentStatus.PERSISTENCE_FAILED
    assert raw_photo.exists(), "SAFETY INVARIANT: Raw media MUST be preserved if persistence fails"





def test_d_deletion_failure_handled_gracefully(temp_environment):
    env = temp_environment
    db = EmbeddingDatabase(
        db_dir=env["db_dir"],
        gait_gallery_dir=env["gait_gal"],
        appearance_gallery_dir=env["app_gal"],
    )

    person_dir = Path(env["input_dir"]) / "Subject04"
    person_dir.mkdir(parents=True, exist_ok=True)
    raw_photo = person_dir / "locked_photo.jpg"
    raw_photo.write_bytes(b"data")

    mock_gait_ext = MagicMock()
    mock_app_ext = MagicMock()
    mock_app_ext.extract.return_value = np.ones((512,), dtype=np.float32)

    manager = EnrollmentLifecycleManager(
        db=db,
        gait_extractor=mock_gait_ext,
        appearance_extractor=mock_app_ext,
    )


    with patch.object(
        EnrollmentLifecycleManager,
        "safe_delete_raw_file",
        return_value=(False, "Permission denied: file in use"),
    ):
        result = manager.enroll_from_media(
            person_id="Subject04",
            photo_paths=[raw_photo],
            auto_delete_raw=True,
        )

    assert result.status == EnrollmentStatus.CLEANUP_FAILED
    assert "Permission denied" in result.error_message

    assert db.get_person("Subject04") is not None





def test_e_duplicate_cleanup_is_idempotent(temp_environment):
    p = Path(temp_environment["input_dir"]) / "already_deleted.jpg"
    assert not p.exists()

    success, err = EnrollmentLifecycleManager.safe_delete_raw_file(p)
    assert success is True
    assert err is None





def test_f_inferior_candidate_rejected(temp_environment):
    reg = ModelRegistry(registry_file=temp_environment["reg_file"])
    validator = CandidateValidator()
    engine = ContinuousImprovementEngine(registry=reg, validator=validator)


    inferior_metrics = {
        "tar": 50.00,
        "far": 8.50,
        "eer": 35.00,
    }

    passed, val_res, rec = engine.process_candidate(
        candidate_version="v1.1.0-inferior",
        model_type="dual_modal_fusion",
        architecture="LinearOptimal-DualModal",
        embedding_dim=256,
        artifact_path="dummy_path.json",
        candidate_metrics=inferior_metrics,
    )

    assert passed is False
    assert val_res.passed is False
    assert rec.deployment_status == ModelDeploymentStatus.REJECTED

    active = reg.get_active_model("dual_modal_fusion")
    assert active.model_version == "v1.0.0"





def test_g_superior_candidate_promoted(temp_environment):
    reg = ModelRegistry(registry_file=temp_environment["reg_file"])
    validator = CandidateValidator()
    engine = ContinuousImprovementEngine(registry=reg, validator=validator)

    superior_metrics = {
        "tar": 75.00,
        "far": 1.50,
        "eer": 20.00,
    }

    passed, val_res, rec = engine.process_candidate(
        candidate_version="v2.0.0-superior",
        model_type="dual_modal_fusion",
        architecture="LearnedLogistic-DualModal",
        embedding_dim=256,
        artifact_path="configs/fusion_profiles/fusion_verification_profile.json",
        candidate_metrics=superior_metrics,
    )

    assert passed is True
    assert val_res.passed is True
    assert rec.deployment_status == ModelDeploymentStatus.ACTIVE
    assert rec.previous_production_version == "v1.0.0"


    active = reg.get_active_model("dual_modal_fusion")
    assert active.model_version == "v2.0.0-superior"





def test_h_confusion_pair_regression_rejected(temp_environment):
    reg = ModelRegistry(registry_file=temp_environment["reg_file"])
    validator = CandidateValidator()
    engine = ContinuousImprovementEngine(registry=reg, validator=validator)

    candidate_metrics = {
        "tar": 80.00,
        "far": 2.00,
        "eer": 18.00,
    }

    confusion_eval = {"confusion_pair_far": 4.50}

    passed, val_res, rec = engine.process_candidate(
        candidate_version="v1.2.0-confusion-fail",
        model_type="dual_modal_fusion",
        architecture="LinearOptimal-DualModal",
        embedding_dim=256,
        artifact_path="dummy.json",
        candidate_metrics=candidate_metrics,
        confusion_pair_eval=confusion_eval,
    )

    assert passed is False
    assert val_res.passed is False
    assert "Confusion-Pair Violation" in str(val_res.rejection_reasons)
    assert rec.deployment_status == ModelDeploymentStatus.REJECTED





def test_i_rollback_restores_previous_active_version(temp_environment):
    reg = ModelRegistry(registry_file=temp_environment["reg_file"])
    validator = CandidateValidator()
    engine = ContinuousImprovementEngine(registry=reg, validator=validator)


    engine.process_candidate(
        candidate_version="v2.0.0",
        model_type="dual_modal_fusion",
        architecture="LearnedLogistic-DualModal",
        embedding_dim=256,
        artifact_path="dummy.json",
        candidate_metrics={"tar": 72.0, "far": 2.0, "eer": 22.0},
    )
    assert reg.get_active_model("dual_modal_fusion").model_version == "v2.0.0"


    restored = engine.trigger_runtime_regression_rollback(
        model_type="dual_modal_fusion",
        reason="Elevated false accept rate in production CCTV zone 3",
    )


    assert restored.model_version == "v1.0.0"
    assert restored.deployment_status == ModelDeploymentStatus.ACTIVE

    rolled_back_v2 = reg.get_model("v2.0.0", "dual_modal_fusion")
    assert rolled_back_v2.deployment_status == ModelDeploymentStatus.ROLLED_BACK





def test_j_operational_observation_lifecycle(temp_environment):
    collector = OperationalEmbeddingCollector(output_dir=temp_environment["obs_dir"])

    dummy_vec = np.random.randn(256).astype(np.float32)
    obs = collector.record_observation(
        camera_id="cam-zone-01",
        track_id=101,
        vector=dummy_vec,
        predicted_identity="SubjectA",
        confidence=0.92,
        modality="gait",
    )

    assert obs.state == ObservationState.PREDICTED
    assert obs.verified_identity is None


    eligible_before = collector.get_training_eligible()
    assert len(eligible_before) == 0


    collector.verify_observation(obs.observation_id, verified_identity="SubjectA")
    eligible_after = collector.get_training_eligible()
    assert len(eligible_after) == 1
    assert eligible_after[0].verified_identity == "SubjectA"





def test_k_drift_detector_flags_degradation(temp_environment):
    collector = OperationalEmbeddingCollector(output_dir=temp_environment["obs_dir"])
    detector = DriftDetector(
        collector=collector,
        gait_gallery_dir=temp_environment["gait_gal"],
        confidence_threshold=0.85,
    )


    for i in range(20):
        collector.record_observation(
            camera_id="cam-drift",
            track_id=i,
            vector=np.random.randn(256),
            predicted_identity="UNKNOWN",
            confidence=0.45,
        )

    report = detector.evaluate_drift(window_size=20)
    assert report.drift_detected is True
    assert report.drift_severity in ("HIGH", "MODERATE")
    assert report.low_confidence_ratio > 0.50
