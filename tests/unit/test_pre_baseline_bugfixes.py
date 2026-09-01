"""
Regression tests for pre-baseline bug fixes in ARGUS AI.

Validates:
1. RecognitionWorker modality state handling under various fusion conditions.
2. Intelligence package complete symbol exports in __all__.
3. OSNetBackbone checkpoint loading integrity.
4. EmbeddingDatabase input validation and NaN/dimension safeguards.
"""

import time
import warnings

import numpy as np
import pytest

import intelligence
from services.recognition_worker import RecognitionResult, RecognitionResultCache


def test_intelligence_package_exports():
    """Verify all critical intelligence components are exported in intelligence.__all__."""
    required_exports = [
        "TrackIdentityAggregator",
        "ScoreCalibrator",
        "ConfusionDetector",
        "NNFineTuner",
        "LearnedFusion",
        "FusionDiagnostics",
        "DualModalFusion",
        "DateAwareLearningScheduler",
        "CandidateValidator",
        "DriftDetector",
        "OperationalEmbeddingCollector",
    ]
    for symbol in required_exports:
        assert symbol in intelligence.__all__, f"{symbol} missing from intelligence.__all__"
        assert hasattr(intelligence, symbol), f"{symbol} not accessible on intelligence module"


def test_recognition_worker_modality_state_safety():
    """Verify RecognitionWorker cache and result state handling."""
    cache = RecognitionResultCache(ttl_seconds=5.0)
    now = time.monotonic()


    res = RecognitionResult(
        camera_id="test_cam",
        track_id=1,
        identity="subject_01",
        similarity=0.92,
        confidence=0.92,
        decision="KNOWN",
        status="CONFIRMED",
        bbox=[10, 10, 100, 200],
        timestamp=now,
        iso_timestamp="2026-08-29T00:00:00Z",
    )
    cache.put(res)
    fetched = cache.get("test_cam", 1)
    assert fetched is not None
    assert fetched.identity == "subject_01"


def test_embedding_database_dimension_and_nan_safeguards(tmp_path):
    """Verify EmbeddingDatabase rejects invalid dimensions, NaNs, and zero norms."""
    from storage.embedding_database import EmbeddingDatabase

    db = EmbeddingDatabase(
        db_dir=str(tmp_path / "embedding_db"),
        gait_gallery_dir=str(tmp_path / "live_gallery"),
        appearance_gallery_dir=str(tmp_path / "app_gallery"),
    )


    with pytest.raises(ValueError, match="Gait embedding dimension mismatch"):
        db.add_embeddings(
            person_id="person_fail",
            gait_embeddings=[np.ones(255, dtype=np.float32)],
        )


    with pytest.raises(ValueError, match="Appearance embedding dimension mismatch"):
        db.add_embeddings(
            person_id="person_fail",
            appearance_embeddings=[np.ones(511, dtype=np.float32)],
        )


    nan_vec = np.ones(256, dtype=np.float32)
    nan_vec[5] = np.nan
    with pytest.raises(ValueError, match="contains non-finite values"):
        db.add_embeddings(
            person_id="person_fail",
            gait_embeddings=[nan_vec],
        )


    valid_gait = np.random.randn(256).astype(np.float32)
    valid_app = np.random.randn(512).astype(np.float32)
    result = db.add_embeddings(
        person_id="person_valid",
        gait_embeddings=[valid_gait],
        appearance_embeddings=[valid_app],
    )
    assert result["gait_embeddings_added"] == 1
    assert result["appearance_embeddings_added"] == 1

    person = db.get_person("person_valid")
    assert person is not None
    assert len(person.gait_embeddings) == 1
    assert len(person.appearance_embeddings) == 1


def test_osnet_backbone_no_future_warnings():
    """Verify OSNetBackbone load does not trigger torch.load weights_only warnings."""
    from models.reid.osnet_backbone import OSNetBackbone

    backbone = OSNetBackbone(model_path="models/weights/osnet_x0_25.pth")

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        model = backbone._ensure_model()
        assert model is not None


        future_warnings = [
            w for w in recorded_warnings
            if issubclass(w.category, FutureWarning) and "torch.load" in str(w.message)
        ]
        assert len(future_warnings) == 0, f"Unexpected FutureWarning: {future_warnings}"
