from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from deployment.startup_validator import DeploymentStartupValidator, StartupValidationError


def test_startup_validator_all_checks_pass():
    validator = DeploymentStartupValidator()
    summary = validator.validate_startup(raise_on_failure=False)

    assert summary["success"] is True
    assert summary["status"] in (
        DeploymentStartupValidator.STATUS_READY,
        DeploymentStartupValidator.STATUS_READY_WITH_WARNINGS,
    )
    assert isinstance(summary["blocking_issues"], list)
    assert len(summary["blocking_issues"]) == 0
    assert "Live RTSP camera streams" in summary["unable_to_verify"][0]


def test_startup_validator_missing_manifest_asset(monkeypatch, tmp_path: Path):
    validator = DeploymentStartupValidator()

    from deployment.runtime_manifest import RuntimeManifest

    mock_manifest = RuntimeManifest(runtime_assets=["non_existent_asset_path.py"])
    monkeypatch.setattr("deployment.startup_validator.get_runtime_manifest", lambda: mock_manifest)

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("missing asset" in issue for issue in summary["blocking_issues"])


def test_startup_validator_invalid_config_triggers_blocking_issue(monkeypatch):
    validator = DeploymentStartupValidator()
    monkeypatch.setattr(
        validator.config_validator,
        "validate_all",
        lambda: {"system.yaml": ["Invalid field 'bad_key'"]},
    )

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("Config error" in issue for issue in summary["blocking_issues"])


def test_startup_validator_backend_fallback_generates_warning():
    validator = DeploymentStartupValidator()

    mock_backend = MagicMock()
    mock_backend.active_backend = "pytorch"
    mock_backend.requested_backend = "onnxruntime"
    mock_backend.fallback_used = True
    mock_backend.fallback_reason = "ONNX Runtime missing"
    mock_backend.config = {}
    mock_backend.predict.return_value = np.zeros((1, 256), dtype=np.float32)
    mock_backend.metadata = {
        "requested_backend": "onnxruntime",
        "active_backend": "pytorch",
        "fallback_used": True,
    }

    summary = validator.validate_startup(raise_on_failure=False, override_backend=mock_backend)
    assert summary["success"] is True
    assert any("Backend fallback active" in w for w in summary["warnings"])


def test_startup_validator_unwritable_output_path(monkeypatch, tmp_path: Path):
    validator = DeploymentStartupValidator()

    def mock_mkdir(*args, **kwargs):
        raise PermissionError("Access Denied")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("not writable" in issue for issue in summary["blocking_issues"])


def test_startup_validator_raises_on_failure_when_enabled(monkeypatch):
    validator = DeploymentStartupValidator()
    monkeypatch.setattr(
        validator.config_validator,
        "validate_all",
        lambda: {"system.yaml": ["Fatal syntax error"]},
    )

    with pytest.raises(StartupValidationError) as exc_info:
        validator.validate_startup(raise_on_failure=True)

    assert len(exc_info.value.blocking_issues) > 0


def test_startup_validator_missing_gallery_file_produces_warning_notice(monkeypatch):
    validator = DeploymentStartupValidator()
    monkeypatch.setattr(
        "deployment.startup_validator.validate_gallery_files",
        lambda *args, **kwargs: (False, "Gallery features file missing in models/gallery", 0),
    )

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is True
    assert summary["status"] in (
        DeploymentStartupValidator.STATUS_READY_WITH_WARNINGS,
        DeploymentStartupValidator.STATUS_READY,
    )
    assert not any("Gallery defect" in issue for issue in summary["blocking_issues"])
    assert any("Gallery state notice" in w for w in summary["warnings"])


def test_startup_validator_gallery_corruption_blocks_startup(monkeypatch):
    validator = DeploymentStartupValidator()
    monkeypatch.setattr(
        "deployment.startup_validator.validate_gallery_files",
        lambda *args, **kwargs: (
            False,
            "Corrupted feature array: NaN detected in gallery_features.npy",
            0,
        ),
    )

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("Gallery defect" in issue for issue in summary["blocking_issues"])
