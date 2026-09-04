from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from tools.validation.deployment_smoke_test import run_deployment_smoke_test


def test_smoke_test_complete_success(tmp_path: Path):
    code, report = run_deployment_smoke_test(output_dir=str(tmp_path))

    assert code == 0
    assert report["status"] == "PASSED"
    assert report["checks"]["synthetic_inference"] == "PASSED"
    assert (tmp_path / "deployment_smoke_test.json").exists()
    assert (tmp_path / "deployment_smoke_test.md").exists()


def test_smoke_test_backend_initialization_failure(monkeypatch, tmp_path: Path):
    def mock_validate_startup(*args, **kwargs):
        return {
            "success": False,
            "status": "NOT_READY",
            "backend": None,
            "blocking_issues": ["Failed to load engine"],
        }

    monkeypatch.setattr(
        "tools.validation.deployment_smoke_test.DeploymentStartupValidator.validate_startup",
        mock_validate_startup,
    )

    code, report = run_deployment_smoke_test(output_dir=str(tmp_path))

    assert code == 1
    assert report["status"] == "FAILED"
    assert any("Inference backend failed" in d or "Startup validator" in d for d in report["defects"])


def test_smoke_test_invalid_embedding_output(monkeypatch, tmp_path: Path):
    mock_backend = MagicMock()

    mock_backend.predict.return_value = np.zeros((1, 100), dtype=np.float32)

    def mock_validate_startup(*args, **kwargs):
        return {
            "success": True,
            "status": "READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING",
            "backend": mock_backend,
            "blocking_issues": [],
        }

    monkeypatch.setattr(
        "tools.validation.deployment_smoke_test.DeploymentStartupValidator.validate_startup",
        mock_validate_startup,
    )

    code, report = run_deployment_smoke_test(output_dir=str(tmp_path))

    assert code == 1
    assert report["checks"]["synthetic_inference"] == "FAILED"
    assert any("Synthetic inference assertion failed" in d for d in report["defects"])


def test_smoke_test_report_write_failure(monkeypatch, tmp_path: Path):
    def mock_mkdir(*args, **kwargs):
        raise PermissionError("Write access denied")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    code, report = run_deployment_smoke_test(output_dir=str(tmp_path))

    assert code == 1
    assert any("Failed writing smoke test report" in d for d in report["defects"])
