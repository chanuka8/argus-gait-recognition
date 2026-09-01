"""
Unit & Forensic Negative-Path Tests for Continual Learning Data Eligibility & Safety.

Verifies strict rejection of:
1. PREDICTED observation -> training split
2. Unverified identity -> training split
3. Duplicate observation -> deduplicated/rejected
4. Outlier observation -> quality score gate rejection
5. Invalid embedding -> rejection
6. Wrong embedding dimension -> rejection
7. NaN embedding -> rejection
8. Infinite embedding -> rejection
9. Corrupted persisted embedding -> rejection
10. Missing identity label -> rejection
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from intelligence.operational_embedding_collector import (
    ObservationState,
    OperationalEmbeddingCollector,
)
from intelligence.operational_evidence_manager import (
    OperationalEvidenceManager,
)
from intelligence.training_dataset_builder import TrainingDatasetBuilder
from storage.embedding_database import EmbeddingDatabase


@pytest.fixture
def temp_neg_env():
    tmp = Path(tempfile.mkdtemp(prefix="argus_test_neg_"))
    obs_dir = tmp / "obs"
    db_dir = tmp / "db"
    man_dir = tmp / "manifests"
    ev_dir = tmp / "evidence"

    db = EmbeddingDatabase(db_dir=str(db_dir))
    collector = OperationalEmbeddingCollector(output_dir=str(obs_dir))
    ev_mgr = OperationalEvidenceManager(storage_dir=str(ev_dir))
    builder = TrainingDatasetBuilder(collector=collector, db=db, manifest_dir=str(man_dir))

    yield {
        "tmp": tmp,
        "collector": collector,
        "ev_mgr": ev_mgr,
        "db": db,
        "builder": builder,
    }

    shutil.rmtree(tmp, ignore_errors=True)


class TestContinualLearningNegativePaths:
    def test_01_predicted_observation_rejected_from_training(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]
        builder: TrainingDatasetBuilder = temp_neg_env["builder"]

        vec = np.random.randn(256).astype(np.float32)
        vec /= np.linalg.norm(vec)
        obs = col.record_observation(
            camera_id="cam-1",
            track_id=101,
            vector=vec,
            predicted_identity="SubUnverified",
            confidence=0.88,
            modality="gait",
            observation_date="2026-08-31",
        )
        assert obs.state == ObservationState.PREDICTED

        train_s, _, _, _, _, _, m = builder.build_dataset_for_date("2026-08-31", model_type="bygait_light")
        assert len(train_s) == 0
        assert m.total_samples == 0

    def test_02_unverified_identity_rejected(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]
        builder: TrainingDatasetBuilder = temp_neg_env["builder"]

        vec = np.random.randn(256).astype(np.float32)
        vec /= np.linalg.norm(vec)
        obs = col.record_observation(
            camera_id="cam-1",
            track_id=102,
            vector=vec,
            predicted_identity="SubNoVer",
            confidence=0.92,
            modality="gait",
            observation_date="2026-08-31",
        )
        assert obs.verified_identity is None

        train_s, _, _, _, _, _, _ = builder.build_dataset_for_date("2026-08-31", model_type="bygait_light")
        assert len(train_s) == 0

    def test_03_duplicate_observation_deduplicated(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]

        vec = np.random.randn(256).astype(np.float32)
        vec /= np.linalg.norm(vec)

        obs1 = col.record_observation(
            camera_id="cam-1",
            track_id=103,
            vector=vec,
            predicted_identity="SubDup",
            confidence=0.90,
            modality="gait",
        )
        obs2 = col.record_observation(
            camera_id="cam-1",
            track_id=103,
            vector=vec,
            predicted_identity="SubDup",
            confidence=0.90,
            modality="gait",
        )
        assert obs1.observation_id == obs2.observation_id

    def test_04_outlier_low_quality_rejected_from_training_eligible(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]

        vec = np.random.randn(256).astype(np.float32)
        vec /= np.linalg.norm(vec)

        obs = col.record_observation(
            camera_id="cam-2",
            track_id=104,
            vector=vec,
            predicted_identity="SubOutlier",
            confidence=0.35,
            quality_score=0.35,
            modality="gait",
        )
        col.verify_observation(obs.observation_id, verified_identity="SubOutlier")

        assert obs.state == ObservationState.VERIFIED
        assert obs.state != ObservationState.TRAINING_ELIGIBLE

    def test_05_invalid_zero_norm_embedding_rejected(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]

        zero_vec = np.zeros(256, dtype=np.float32)
        obs = col.record_observation(
            camera_id="cam-1",
            track_id=105,
            vector=zero_vec,
            predicted_identity="SubZero",
            confidence=0.90,
            modality="gait",
        )
        col.verify_observation(obs.observation_id, verified_identity="SubZero")
        assert obs.quality_score == 0.0
        assert obs.state != ObservationState.TRAINING_ELIGIBLE

    def test_06_wrong_dimension_embedding_rejected(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]

        vec_64 = np.random.randn(64).astype(np.float32)
        obs = col.record_observation(
            camera_id="cam-1",
            track_id=106,
            vector=vec_64,
            predicted_identity="Sub64",
            confidence=0.90,
            modality="gait",
        )
        col.verify_observation(obs.observation_id, verified_identity="Sub64")
        assert obs.quality_score == 0.0
        assert obs.state != ObservationState.TRAINING_ELIGIBLE

    def test_07_nan_embedding_rejected(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]

        vec_nan = np.random.randn(256).astype(np.float32)
        vec_nan[0] = np.nan
        obs = col.record_observation(
            camera_id="cam-1",
            track_id=107,
            vector=vec_nan,
            predicted_identity="SubNaN",
            confidence=0.90,
            modality="gait",
        )
        col.verify_observation(obs.observation_id, verified_identity="SubNaN")
        assert obs.quality_score == 0.0
        assert obs.state != ObservationState.TRAINING_ELIGIBLE

    def test_08_infinite_embedding_rejected(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]

        vec_inf = np.random.randn(256).astype(np.float32)
        vec_inf[10] = np.inf
        obs = col.record_observation(
            camera_id="cam-1",
            track_id=108,
            vector=vec_inf,
            predicted_identity="SubInf",
            confidence=0.90,
            modality="gait",
        )
        col.verify_observation(obs.observation_id, verified_identity="SubInf")
        assert obs.quality_score == 0.0
        assert obs.state != ObservationState.TRAINING_ELIGIBLE

    def test_09_corrupted_persisted_evidence_rejected(self, temp_neg_env):
        ev_mgr: OperationalEvidenceManager = temp_neg_env["ev_mgr"]

        gei = np.random.randint(0, 255, size=(64, 128), dtype=np.uint8)
        rec = ev_mgr.store_evidence(
            observation_id="obs_test_corrupt",
            camera_id="cam_1",
            track_id=109,
            person_id="SubCorrupt",
            modality="gait",
            media_array=gei,
        )
        assert rec is not None

        with open(rec.file_path, "wb") as f:
            f.write(b"CORRUPTED_BYTES_HERE")

        loaded = ev_mgr.load_evidence(rec.evidence_id)
        assert loaded is None

    def test_10_missing_identity_label_rejected(self, temp_neg_env):
        col: OperationalEmbeddingCollector = temp_neg_env["collector"]
        builder: TrainingDatasetBuilder = temp_neg_env["builder"]

        vec = np.random.randn(256).astype(np.float32)
        vec /= np.linalg.norm(vec)

        obs = col.record_observation(
            camera_id="cam-1",
            track_id=110,
            vector=vec,
            predicted_identity="UNKNOWN",
            confidence=0.10,
            modality="gait",
            observation_date="2026-08-31",
        )
        col.verify_observation(obs.observation_id, verified_identity="")

        train_s, _, _, _, _, _, _ = builder.build_dataset_for_date("2026-08-31", model_type="bygait_light")
        assert len(train_s) == 0
