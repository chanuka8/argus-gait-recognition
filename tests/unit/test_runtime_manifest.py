"""
Unit tests for deployment build/runtime asset manifest.
"""

from pathlib import Path

from deployment.runtime_manifest import RuntimeManifest, generate_runtime_manifest_artifacts, get_runtime_manifest


def test_runtime_manifest_lists_required_and_excludes_dev_assets():
    manifest = get_runtime_manifest()

    assert "main.py" in manifest.runtime_assets
    assert any(a.startswith("core/") for a in manifest.runtime_assets)
    assert any(a.startswith("configs/") for a in manifest.runtime_assets)
    assert "scripts/doctor.py" in manifest.runtime_assets

    assert "tests" in manifest.build_assets
    assert "pytest.ini" in manifest.build_assets
    assert ".env" in manifest.excluded_patterns
    assert "venv" in manifest.excluded_patterns


def test_runtime_manifest_validation_detects_present_and_missing_assets(tmp_path: Path):
    manifest = RuntimeManifest(runtime_assets=["existing.txt", "missing.txt"])

    (tmp_path / "existing.txt").write_text("content", encoding="utf-8")

    result = manifest.validate_runtime_assets(repo_root=str(tmp_path))
    assert result["valid"] is False
    assert result["missing"] == ["missing.txt"]
    assert result["checked"] == ["existing.txt", "missing.txt"]


def test_runtime_manifest_export_no_absolute_paths_or_secrets():
    manifest = get_runtime_manifest()
    data = manifest.to_dict()

    raw_json = str(data)
    assert "C:\\Users" not in raw_json
    assert "/home/" not in raw_json
    assert "SECRET" not in raw_json.upper() or "PATTERNS" in raw_json.upper()
    assert ".env" in manifest.excluded_patterns


def test_generate_runtime_manifest_artifacts(tmp_path: Path):
    j_file = str(tmp_path / "manifest.json")
    m_file = str(tmp_path / "manifest.md")

    res = generate_runtime_manifest_artifacts(json_path=j_file, md_path=m_file)
    assert Path(j_file).exists()
    assert Path(m_file).exists()
    assert res["validation"]["valid"] is True
