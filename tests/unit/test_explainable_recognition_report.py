"""
Unit tests for Explainable Recognition Report Generator.
"""

import csv
import json
from pathlib import Path
import pytest

from intelligence.explainable_recognition_report import (
    ExplainableRecognitionReporter,
    RecognitionEvidence,
)


@pytest.fixture
def reporter_config(tmp_path: Path) -> dict:
    return {
        "enabled": True,
        "output_dir": str(tmp_path / "explainable"),
        "formats": ["json", "csv", "markdown"],
        "report_confirmed": True,
        "report_deferred": True,
        "report_watchlist": True,
        "duplicate_cooldown_seconds": 10.0,
    }


def test_disabled_mode_produces_no_files(tmp_path: Path):
    cfg = {
        "enabled": False,
        "output_dir": str(tmp_path / "explainable"),
        "formats": ["json"],
    }
    reporter = ExplainableRecognitionReporter(config=cfg)
    evidence = RecognitionEvidence(
        camera_id="cam_01",
        local_track_id=101,
        predicted_identity="P001",
        final_identity="P001",
        final_decision="KNOWN",
    )
    result = reporter.generate_report(evidence)
    assert result is None
    assert not (tmp_path / "explainable").exists()


def test_confirmed_recognition_report(reporter_config: dict, tmp_path: Path):
    reporter = ExplainableRecognitionReporter(config=reporter_config)
    evidence = RecognitionEvidence(
        camera_id="cam_01",
        local_track_id=1,
        predicted_identity="P001",
        final_identity="P001",
        final_decision="KNOWN",
        gait_similarity=0.91,
        quality_score=0.85,
    )
    files = reporter.generate_report(evidence)
    assert files is not None
    assert "json" in files
    assert "csv" in files
    assert "markdown" in files

    with open(files["json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["camera_id"] == "cam_01"
        assert data["final_identity"] == "P001"
        assert data["final_decision"] == "KNOWN"

    with open(files["csv"], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = dict(list(reader))
        assert rows["final_identity"] == "P001"

    with open(files["markdown"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "# Explainable Recognition Report" in content
        assert "`P001`" in content


def test_deferred_and_watchlist_reports(reporter_config: dict):
    reporter = ExplainableRecognitionReporter(config=reporter_config)
    deferred_ev = RecognitionEvidence(
        camera_id="cam_02",
        local_track_id=2,
        recognition_deferred=True,
        defer_reason="Occlusion ratio high",
    )
    files_def = reporter.generate_report(deferred_ev)
    assert files_def is not None

    wl_ev = RecognitionEvidence(
        camera_id="cam_03",
        local_track_id=3,
        watchlist_matched=True,
        watchlist_category="CRITICAL",
    )
    files_wl = reporter.generate_report(wl_ev)
    assert files_wl is not None


def test_missing_optional_fields_handled(reporter_config: dict):
    reporter = ExplainableRecognitionReporter(config=reporter_config)
    sparse_ev = RecognitionEvidence(
        camera_id="cam_sparse",
        local_track_id=4,
        final_decision="KNOWN",
    )
    files = reporter.generate_report(sparse_ev)
    assert files is not None
    with open(files["json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["gait_similarity"] is None
        assert data["occlusion_score"] is None


def test_credentials_and_embeddings_excluded(reporter_config: dict):
    reporter = ExplainableRecognitionReporter(config=reporter_config)
    ev = RecognitionEvidence(
        camera_id="rtsp://admin:secret123@192.168.1.100:554/live",
        local_track_id=5,
        final_decision="KNOWN",
        decision_reason="Stream connected via rtsp://user:pass@host/path",
    )
    files = reporter.generate_report(ev)
    assert files is not None
    with open(files["json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "secret123" not in data["camera_id"]
        assert "pass" not in data["decision_reason"]
        assert "rtsp://***:***@" in data["camera_id"]
        assert "raw_embedding" not in data


def test_duplicate_suppression_cooldown(reporter_config: dict):
    reporter = ExplainableRecognitionReporter(config=reporter_config)
    ev = RecognitionEvidence(
        camera_id="cam_01",
        local_track_id=99,
        final_identity="P001",
        final_decision="KNOWN",
    )
    first_gen = reporter.generate_report(ev)
    assert first_gen is not None

    second_gen = reporter.generate_report(ev)
    assert second_gen is None

    forced_gen = reporter.generate_report(ev, force_export=True)
    assert forced_gen is not None
