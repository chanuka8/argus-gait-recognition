"""Unit tests for the biometric embedding lifecycle, quality validation, and atomic gallery updates."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus
from storage.embedding_database import EmbeddingDatabase


@pytest.fixture
def temp_db_env():
    temp_dir = tempfile.mkdtemp(prefix="test_biometric_lifecycle_")
    db_dir = Path(temp_dir) / "embedding_db"
    gait_gal = Path(temp_dir) / "live_gallery"
    app_gal = Path(temp_dir) / "appearance_gallery"
    jobs_dir = Path(temp_dir) / "reference_jobs"

    db = EmbeddingDatabase(
        db_dir=str(db_dir),
        gait_gallery_dir=str(gait_gal),
        appearance_gallery_dir=str(app_gal),
    )

    ReferenceJobManager._instance = None
    job_mgr = ReferenceJobManager(jobs_dir=str(jobs_dir), max_workers=2)

    yield {
        "temp_dir": temp_dir,
        "db": db,
        "job_mgr": job_mgr,
        "gait_gal": gait_gal,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)
    ReferenceJobManager._instance = None


def test_atomic_commit_and_activate_lifecycle(temp_db_env):
    """Test initial activation, followed by atomic update with retirement of previous active embeddings."""
    db: EmbeddingDatabase = temp_db_env["db"]
    person_id = "PERSON_V1"

    # 1. Initial activation: 2 valid 256D gait embeddings
    vec1 = np.random.randn(256).astype(np.float32)
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = np.random.randn(256).astype(np.float32)
    vec2 = vec2 / np.linalg.norm(vec2)

    res1 = db.commit_and_activate_embeddings(
        person_id=person_id,
        gait_embeddings=[vec1, vec2],
        model_version="v1.0.0",
        source_session_id="session_initial",
    )
    assert res1["success"] is True
    assert res1["gait_embeddings_added"] == 2
    assert res1["gait_embeddings_retired"] == 0

    person = db.get_person(person_id)
    assert person is not None
    assert len(person.gait_embeddings) == 2
    assert all(e.status == "ACTIVE" for e in person.gait_embeddings)

    # Verify VectorStore contains only active embeddings
    _, labels, _ = db.gait_store.load()
    active_in_store = int(np.sum(labels == person_id))
    assert active_in_store == 2

    # 2. Update with new embeddings: old embeddings must be RETIRED
    vec3 = np.random.randn(256).astype(np.float32)
    vec3 = vec3 / np.linalg.norm(vec3)

    res2 = db.commit_and_activate_embeddings(
        person_id=person_id,
        gait_embeddings=[vec3],
        model_version="v1.0.0",
        source_session_id="session_replacement",
        retire_previous=True,
    )
    assert res2["success"] is True
    assert res2["gait_embeddings_added"] == 1
    assert res2["gait_embeddings_retired"] == 2  # previous 2 retired

    person_after = db.get_person(person_id)
    assert len(person_after.gait_embeddings) == 3
    retired_records = [e for e in person_after.gait_embeddings if e.status == "RETIRED"]
    active_records = [e for e in person_after.gait_embeddings if e.status == "ACTIVE"]
    assert len(retired_records) == 2
    assert len(active_records) == 1
    assert active_records[0].source_session_id == "session_replacement"

    # Verify VectorStore now contains ONLY the single active embedding
    _, labels2, _ = db.gait_store.load()
    assert int(np.sum(labels2 == person_id)) == 1


def test_validation_rejection_protects_existing_active(temp_db_env):
    """Test that invalid candidate embeddings are rejected before mutation, leaving existing active untouched."""
    db: EmbeddingDatabase = temp_db_env["db"]
    person_id = "PERSON_PROTECTED"

    # Seed with 1 valid active embedding
    valid_vec = np.random.randn(256).astype(np.float32)
    valid_vec = valid_vec / np.linalg.norm(valid_vec)
    db.commit_and_activate_embeddings(
        person_id=person_id,
        gait_embeddings=[valid_vec],
        model_version="v1.0.0",
    )

    # Attempt 1: Dimension mismatch (e.g., 128 instead of 256)
    bad_dim_vec = np.random.randn(128).astype(np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        db.commit_and_activate_embeddings(
            person_id=person_id,
            gait_embeddings=[bad_dim_vec],
        )

    # Attempt 2: Non-finite values (NaN / Inf)
    nan_vec = np.random.randn(256).astype(np.float32)
    nan_vec[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        db.commit_and_activate_embeddings(
            person_id=person_id,
            gait_embeddings=[nan_vec],
        )

    # Attempt 3: Near-zero norm
    zero_vec = np.zeros(256, dtype=np.float32)
    with pytest.raises(ValueError, match="zero norm"):
        db.commit_and_activate_embeddings(
            person_id=person_id,
            gait_embeddings=[zero_vec],
        )

    # Assert existing active embedding was NEVER retired or corrupted
    person = db.get_person(person_id)
    assert len(person.gait_embeddings) == 1
    assert person.gait_embeddings[0].status == "ACTIVE"
    _, labels, _ = db.gait_store.load()
    assert int(np.sum(labels == person_id)) == 1


def test_reference_job_manager_lifecycle_and_metrics(temp_db_env):
    """Test ReferenceJobManager lifecycle stages and media_type awareness."""
    job_mgr: ReferenceJobManager = temp_db_env["job_mgr"]

    job = job_mgr.create_job(
        person_id="CASE_IMAGE_01",
        case_id="CASE_IMAGE_01",
        media_path="data/reference_photos/photo1.jpg",
        media_type="image",
        owner="investigator_01",
    )

    assert job.job_id.startswith("ref_")
    assert job.media_type == "image"
    assert job.status == ReferenceJobStatus.QUEUED.value

    # Transition through explicit biometric stages
    job_mgr.update_progress(job.job_id, stage="VALIDATING", status=ReferenceJobStatus.VALIDATING)
    rec1 = job_mgr.get_job(job.job_id)
    assert rec1.status == "VALIDATING"

    job_mgr.update_progress(
        job.job_id,
        stage="FEATURE_EXTRACTION",
        status=ReferenceJobStatus.FEATURE_EXTRACTION,
        embeddings_generated=2,
    )
    rec2 = job_mgr.get_job(job.job_id)
    assert rec2.status == "FEATURE_EXTRACTION"
    assert rec2.progress.embeddings_generated == 2

    # Complete job as ACTIVE
    job_mgr.complete_job(
        job.job_id,
        result={"embeddings_committed": 2, "gait_embeddings_committed": 2},
        terminal_status=ReferenceJobStatus.ACTIVE,
    )
    rec_final = job_mgr.get_job(job.job_id)
    assert rec_final.status == "ACTIVE"
    assert rec_final.result["embeddings_committed"] == 2


def test_process_reference_photos_async_flow(temp_db_env):
    """Test decoupled async photo processing via MissingPersonVideoProcessor."""
    import cv2

    from services.missing_person_processor import MissingPersonVideoProcessor

    db: EmbeddingDatabase = temp_db_env["db"]
    job_mgr: ReferenceJobManager = temp_db_env["job_mgr"]
    temp_dir = temp_db_env["temp_dir"]

    # 1. Create a dummy image
    img_path = Path(temp_dir) / "test_photo.jpg"
    img_data = np.zeros((100, 100, 3), dtype=np.uint8)
    img_data[20:80, 20:80] = 255
    cv2.imwrite(str(img_path), img_data)

    # 2. Mock extractor and silhouette
    mock_detector = MagicMock()
    mock_extractor = MagicMock()
    fake_vec = np.random.randn(256).astype(np.float32)
    fake_vec = fake_vec / np.linalg.norm(fake_vec)
    mock_extractor.extract_from_gei.return_value = fake_vec
    mock_extractor.backend = None

    mock_silhouette = MagicMock()
    mock_silhouette.extract_from_crop.return_value = np.zeros((128, 64), dtype=np.uint8)

    processor = MissingPersonVideoProcessor(
        detector=mock_detector,
        tracker=MagicMock(),
        extractor=mock_extractor,
        silhouette_step=mock_silhouette,
        store=db.gait_store,
        embedding_db=db,
        job_manager=job_mgr,
    )

    job = job_mgr.create_job(
        person_id="ASYNC_PHOTO_01",
        media_path=str(img_path),
        media_type="image",
    )

    res = processor.process_reference_photos(
        person_id="ASYNC_PHOTO_01",
        photo_paths=[img_path],
        job_id=job.job_id,
        case_id="ASYNC_PHOTO_01",
    )

    assert res["success"] is True
    assert res["embeddings_committed"] == 1

    # Check job terminal status
    finished_job = job_mgr.get_job(job.job_id)
    assert finished_job.status in ("COMPLETED", "ACTIVE")

    # Check database activation
    person = db.get_person("ASYNC_PHOTO_01")
    assert person is not None
    assert len(person.gait_embeddings) == 1
    assert person.gait_embeddings[0].status == "ACTIVE"

