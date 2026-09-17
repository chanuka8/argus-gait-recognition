from pathlib import Path

import pytest

from deployment.doctor import run_doctor


def test_doctor_execution_and_health_report(tmp_path: Path):

    json_path = tmp_path / "health_report.json"
    md_path = tmp_path / "health_report.md"

    exit_code, report_data = run_doctor(
        json_path=str(json_path),
        md_path=str(md_path),
    )

    assert exit_code in (0, 1, 2)
    assert json_path.exists()
    assert md_path.exists()
    assert report_data["overall_status"] in {
        "READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING",
        "READY_WITH_WARNINGS",
        "NOT_READY",
        "UNABLE_TO_VERIFY",
    }
    assert "checks" in report_data
    assert isinstance(report_data["checks"], list)

    probe_file = Path("outputs/reports/.doctor_probe.tmp")
    assert not probe_file.exists()


def test_doctor_internal_exception_returns_exit_code_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_checks(*args, **kwargs):
        raise RuntimeError("Simulated internal doctor error with rtsp://admin:pass@10.0.0.1/live")

    from deployment import doctor

    monkeypatch.setattr(doctor, "_execute_doctor_checks", fake_checks)

    exit_code, report = run_doctor(
        json_path=str(tmp_path / "err.json"),
        md_path=str(tmp_path / "err.md"),
    )

    assert exit_code == 2
    assert report["overall_status"] == "UNABLE_TO_VERIFY"
    assert len(report["blocking_issues"]) == 1
    assert "pass" not in report["blocking_issues"][0]
    assert "admin:***@" in report["blocking_issues"][0]


def test_doctor_non_destructive_guarantee(tmp_path: Path):
    models_dir = Path("models")
    mtime_before = {f: f.stat().st_mtime for f in models_dir.rglob("*") if f.is_file()}

    json_path = tmp_path / "health_report.json"
    md_path = tmp_path / "health_report.md"

    run_doctor(json_path=str(json_path), md_path=str(md_path))

    mtime_after = {f: f.stat().st_mtime for f in models_dir.rglob("*") if f.is_file()}
    assert mtime_before == mtime_after


def test_doctor_missing_gallery_file_produces_warning(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "deployment.doctor.validate_gallery_files",
        lambda *args, **kwargs: (False, "Gallery features file missing in models/gallery", 0),
    )

    json_path = tmp_path / "doctor_missing.json"
    md_path = tmp_path / "doctor_missing.md"

    _exit_code, report = run_doctor(json_path=str(json_path), md_path=str(md_path))

    gallery_checks = [c for c in report.get("checks", []) if c.get("name") == "gallery_integrity"]
    assert len(gallery_checks) == 1
    assert gallery_checks[0]["status"] == "WARN"
    assert any("Gallery state notice" in w for w in report.get("warnings", []))
    assert not any("Gallery defect" in issue for issue in report.get("blocking_issues", []))


def test_doctor_corrupted_gallery_produces_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "deployment.doctor.validate_gallery_files",
        lambda *args, **kwargs: (
            False,
            "Corrupted feature array: NaN values found in gallery_features.npy",
            0,
        ),
    )

    json_path = tmp_path / "doctor_corrupt.json"
    md_path = tmp_path / "doctor_corrupt.md"

    _exit_code, report = run_doctor(json_path=str(json_path), md_path=str(md_path))

    gallery_checks = [c for c in report.get("checks", []) if c.get("name") == "gallery_integrity"]
    assert len(gallery_checks) == 1
    assert gallery_checks[0]["status"] == "FAIL"
    assert "Gallery defect" in gallery_checks[0]["details"]
