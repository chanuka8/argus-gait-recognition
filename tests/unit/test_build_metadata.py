from pathlib import Path
from unittest.mock import MagicMock

from deployment.build_metadata import (
    compute_configuration_fingerprint,
    extract_build_metadata,
    get_application_version,
    get_git_branch,
    get_git_commit,
)


def test_git_metadata_extraction_and_fallback(monkeypatch, tmp_path: Path):
    commit = get_git_commit(repo_dir=".")
    branch = get_git_branch(repo_dir=".")

    assert len(commit) >= 4 or commit == "UNKNOWN"
    assert len(branch) >= 1 or branch == "UNKNOWN"

    def mock_subprocess_run(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    assert get_git_commit(repo_dir=str(tmp_path)) == "UNKNOWN"
    assert get_git_branch(repo_dir=str(tmp_path)) == "UNKNOWN"


def test_application_version_reading_and_missing_file(tmp_path: Path):
    v_file = tmp_path / "VERSION"
    v_file.write_text("1.2.3\n", encoding="utf-8")

    assert get_application_version(version_path=str(v_file)) == "1.2.3"
    assert get_application_version(version_path=str(tmp_path / "NON_EXISTENT")) == "0.0.0-dev"


def test_configuration_fingerprint_stability():
    fp1 = compute_configuration_fingerprint(configs_dir="configs")
    fp2 = compute_configuration_fingerprint(configs_dir="configs")

    assert len(fp1) == 12
    assert fp1 == fp2


def test_extract_build_metadata_integration():
    mock_backend = MagicMock()
    mock_backend.requested_backend = "onnxruntime"
    mock_backend.active_backend = "pytorch"
    mock_backend.config = {"model_path": "runs/exp_001/best_model.pth"}

    meta = extract_build_metadata(backend=mock_backend)
    data = meta.to_dict()

    assert data["application_name"] == "ARGUS AI"
    assert data["backend_requested"] == "onnxruntime"
    assert data["backend_active"] == "pytorch"
    assert "C:\\Users" not in str(data)
    assert "/home/" not in str(data)
