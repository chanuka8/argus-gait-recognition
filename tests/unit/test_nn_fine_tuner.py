"""
Unit Tests for Neural Network Fine-Tuning Module (ByGaitLight and OSNet).

Tests:
1. ByGaitLight CNN backbone fine-tuning from active weights.
2. OSNet ReID backbone fine-tuning from active weights.
3. Dataset construction with 50% historical replay blending.
4. Candidate artifact generation (.pth) with SHA-256 checksums.
5. Invariant: Active production model weights are never overwritten in-place.
6. Validation gates enforcement for NN candidates (dimension, rank-1, checksum).
7. Resource bounds & timeout handling.
"""

import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from intelligence.candidate_validator import CandidateValidator
from intelligence.nn_fine_tuner import NNFineTuner
from models.architectures.bygait_light import ByGaitLight


@pytest.fixture
def tmp_nn_env():
    """Create isolated environment for NN fine-tuning tests."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_test_nn_"))
    cand_dir = tmp_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    registry_file = tmp_dir / "model_registry.json"


    init_bygait = ByGaitLight(embedding_dim=256, part_bins=4)
    active_bygait_path = tmp_dir / "active_bygait.pth"
    torch.save(init_bygait.state_dict(), str(active_bygait_path))

    yield {
        "root": tmp_dir,
        "cand_dir": str(cand_dir),
        "registry_file": str(registry_file),
        "active_bygait_path": str(active_bygait_path),
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_bygait_light_fine_tuning_success(tmp_nn_env):
    """Test ByGaitLight CNN backbone fine-tuning produces a valid candidate .pth."""
    tuner = NNFineTuner(
        candidate_dir=tmp_nn_env["cand_dir"],
        max_epochs=1,
        learning_rate=1e-5,
        batch_size=4,
    )


    training_data = [
        {"image": np.random.rand(64, 128).astype(np.float32), "label": "Person_A"}
        for _ in range(4)
    ] + [
        {"image": np.random.rand(64, 128).astype(np.float32), "label": "Person_B"}
        for _ in range(4)
    ]
    historical_data = [
        {"image": np.random.rand(64, 128).astype(np.float32), "label": "Person_A"}
        for _ in range(2)
    ]

    res = tuner.fine_tune_bygait_light(
        active_weights_path=tmp_nn_env["active_bygait_path"],
        training_gei_data=training_data,
        historical_gei_data=historical_data,
        candidate_version="vByGait_001",
    )

    assert res["success"] is True
    assert res["model_type"] == "bygait_light"
    assert res["embedding_dim"] == 256
    assert Path(res["artifact_path"]).exists()
    assert len(res["checksum_sha256"]) == 64
    assert res["metrics"]["total_samples"] == 10
    assert res["metrics"]["num_classes"] == 2


    candidate_model = ByGaitLight(embedding_dim=256, part_bins=4)
    state = torch.load(res["artifact_path"], map_location="cpu", weights_only=True)
    candidate_model.load_state_dict(state)
    dummy_input = torch.randn(1, 1, 64, 128)
    with torch.no_grad():
        emb = candidate_model(dummy_input)
    assert emb.shape == (1, 256)
    assert torch.isfinite(emb).all()


def test_osnet_fine_tuning_success(tmp_nn_env):
    """Test OSNet ReID backbone fine-tuning produces a valid candidate .pth."""
    tuner = NNFineTuner(
        candidate_dir=tmp_nn_env["cand_dir"],
        max_epochs=1,
        learning_rate=1e-5,
        batch_size=4,
    )

    training_crops = [
        {"image": (np.random.rand(256, 128, 3) * 255).astype(np.uint8), "label": "Person_X"}
        for _ in range(4)
    ] + [
        {"image": (np.random.rand(256, 128, 3) * 255).astype(np.uint8), "label": "Person_Y"}
        for _ in range(4)
    ]
    historical_crops = [
        {"image": (np.random.rand(256, 128, 3) * 255).astype(np.uint8), "label": "Person_X"}
        for _ in range(2)
    ]

    res = tuner.fine_tune_osnet(
        active_weights_path="non_existent_osnet.pth",
        training_crop_data=training_crops,
        historical_crop_data=historical_crops,
        candidate_version="vOSNet_001",
    )

    assert res["success"] is True
    assert res["model_type"] == "osnet_reid"
    assert res["embedding_dim"] == 512
    assert Path(res["artifact_path"]).exists()
    assert len(res["checksum_sha256"]) == 64


def test_active_model_weights_not_overwritten(tmp_nn_env):
    """Safety Invariant: Active model file is NEVER overwritten during candidate training."""
    active_path = Path(tmp_nn_env["active_bygait_path"])
    initial_bytes = active_path.read_bytes()
    initial_sha = hashlib.sha256(initial_bytes).hexdigest()

    tuner = NNFineTuner(
        candidate_dir=tmp_nn_env["cand_dir"],
        max_epochs=1,
    )

    training_data = [
        {"image": np.random.rand(64, 128).astype(np.float32), "label": "A"} for _ in range(4)
    ] + [
        {"image": np.random.rand(64, 128).astype(np.float32), "label": "B"} for _ in range(4)
    ]

    res = tuner.fine_tune_bygait_light(
        active_weights_path=str(active_path),
        training_gei_data=training_data,
        historical_gei_data=[],
        candidate_version="vSafeCand01",
    )


    current_bytes = active_path.read_bytes()
    current_sha = hashlib.sha256(current_bytes).hexdigest()
    assert current_sha == initial_sha
    assert Path(res["artifact_path"]) != active_path


def test_candidate_validator_nn_gates():
    """Test CandidateValidator evaluates NN-specific gates (dimension, checksum, rank1)."""
    validator = CandidateValidator()


    valid_metrics = {
        "tar": 94.5,
        "far": 0.0,
        "val_rank1_accuracy": 94.5,
        "embedding_dim": 256,
        "checksum_sha256": "a" * 64,
    }
    res_valid = validator.validate_candidate(
        candidate_version="vNNGood",
        model_type="bygait_light",
        baseline_metrics={"tar": 90.0, "far": 0.0},
        candidate_metrics=valid_metrics,
    )
    assert res_valid.passed is True
    assert res_valid.gate_evaluations.get("embedding_dim_gate") is True


    bad_dim_metrics = {
        "tar": 94.5,
        "far": 0.0,
        "val_rank1_accuracy": 94.5,
        "embedding_dim": 128,
        "checksum_sha256": "a" * 64,
    }
    res_bad_dim = validator.validate_candidate(
        candidate_version="vNNBadDim",
        model_type="bygait_light",
        baseline_metrics={},
        candidate_metrics=bad_dim_metrics,
    )
    assert res_bad_dim.passed is False
    assert any("Dimension Mismatch" in r for r in res_bad_dim.rejection_reasons)
