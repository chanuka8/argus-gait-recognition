"""
Unit tests for Event Timeline Reconstruction Module.
"""

import json
from pathlib import Path
import pytest

from intelligence.event_timeline_reconstructor import (
    EventTimelineReconstructor,
)


@pytest.fixture
def timeline_config(tmp_path: Path) -> dict:
    return {
        "enabled": True,
        "output_dir": str(tmp_path / "timelines"),
        "formats": ["json", "csv", "markdown"],
        "export_on_track_close": True,
        "export_on_watchlist": True,
        "retention_seconds": 3600.0,
        "maximum_events_per_track": 5,
    }


def test_disabled_mode_behavior(tmp_path: Path):
    cfg = {"enabled": False, "output_dir": str(tmp_path / "timelines")}
    reconstructor = EventTimelineReconstructor(config=cfg)
    ev = reconstructor.record_event("TRACK_CREATED", camera_id="cam_1", local_track_id=1)
    assert ev is None


def test_ordered_event_accumulation_and_deduplication(timeline_config: dict):
    reconstructor = EventTimelineReconstructor(config=timeline_config)
    reconstructor.record_event("TRACK_CREATED", camera_id="cam_1", local_track_id=10)
    reconstructor.record_event("IDENTITY_CANDIDATE", camera_id="cam_1", local_track_id=10, identity_id="P001")

    # Duplicate call should be suppressed
    dup = reconstructor.record_event("IDENTITY_CANDIDATE", camera_id="cam_1", local_track_id=10, identity_id="P001")
    assert dup is None

    timeline = reconstructor.get_timeline("track_cam_1_10")
    assert len(timeline) == 2
    assert timeline[0].event_type == "TRACK_CREATED"
    assert timeline[1].event_type == "IDENTITY_CANDIDATE"


def test_camera_local_isolation_and_global_track_continuity(timeline_config: dict):
    reconstructor = EventTimelineReconstructor(config=timeline_config)
    reconstructor.record_event("TRACK_CREATED", camera_id="cam_1", local_track_id=1, global_track_id="GTRACK-001")
    reconstructor.record_event("CAMERA_ENTER", camera_id="cam_2", local_track_id=5, global_track_id="GTRACK-001")

    events = reconstructor.get_timeline("global_GTRACK-001")
    assert len(events) == 2
    assert events[0].camera_id == "cam_1"
    assert events[1].camera_id == "cam_2"


def test_track_close_and_watchlist_export(timeline_config: dict, tmp_path: Path):
    reconstructor = EventTimelineReconstructor(config=timeline_config)
    reconstructor.record_event("TRACK_CREATED", camera_id="cam_1", local_track_id=20)
    files = reconstructor.export_timeline("track_cam_1_20")
    assert files is not None
    assert "json" in files
    assert "csv" in files
    assert "markdown" in files

    with open(files["json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["event_count"] == 1


def test_maximum_event_bound(timeline_config: dict):
    reconstructor = EventTimelineReconstructor(config=timeline_config)
    for i in range(10):
        reconstructor.record_event(f"EVENT_{i}", camera_id="cam_1", local_track_id=30)

    events = reconstructor.get_timeline("track_cam_1_30")
    # Bound is set to 5 in fixture
    assert len(events) == 5
    assert events[-1].event_type == "EVENT_9"


def test_no_secret_or_credential_leakage(timeline_config: dict):
    reconstructor = EventTimelineReconstructor(config=timeline_config)
    reconstructor.record_event(
        "CAMERA_ENTER",
        camera_id="rtsp://admin:supersecret@10.0.0.1:554/stream",
        local_track_id=40,
        reason="Connected to rtsp://usr:pwd@host/path",
    )
    key = reconstructor._get_key(None, "rtsp://admin:supersecret@10.0.0.1:554/stream", 40)
    events = reconstructor.get_timeline(key)
    assert len(events) == 1
    dict_ev = events[0].to_dict()
    assert "supersecret" not in dict_ev["camera_id"]
    assert "pwd" not in dict_ev["reason"]
