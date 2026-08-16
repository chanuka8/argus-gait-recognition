import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.server import app
from services.camera_worker import normalize_camera_source
from services.gait_service import GaitService


def test_normalize_camera_source_webcam_indices():
    """Verify numeric strings and integers are normalized to integer device indices."""
    assert normalize_camera_source("0") == 0
    assert normalize_camera_source("1") == 1
    assert normalize_camera_source(" 2 ") == 2
    assert normalize_camera_source(0) == 0
    assert normalize_camera_source(3) == 3


def test_normalize_camera_source_rtsp_and_http():
    """Verify RTSP, HTTP, HTTPS, and file paths are preserved as unchanged strings."""
    assert normalize_camera_source("rtsp://192.168.1.100:554/stream1") == "rtsp://192.168.1.100:554/stream1"
    assert normalize_camera_source("http://192.168.1.100:8080/video") == "http://192.168.1.100:8080/video"
    assert normalize_camera_source("https://example.com/live/feed.m3u8") == "https://example.com/live/feed.m3u8"
    assert normalize_camera_source("data/test_video.mp4") == "data/test_video.mp4"


def test_camera_start_and_stop_lifecycle():
    """Verify camera start, list, and stop lifecycle through GaitService."""
    service = GaitService()

    # Start worker
    cam_info = service.start_camera(
        camera_id="CCTV-TEST-101",
        source="0",
        location="Western Transit Corridor",
    )
    assert cam_info["camera_id"] == "CCTV-TEST-101"
    assert cam_info["status"] == "ACTIVE"
    assert "CCTV-TEST-101" in service.active_cameras

    # Stop worker
    stopped = service.stop_camera("CCTV-TEST-101")
    assert stopped is True
    assert "CCTV-TEST-101" not in service.active_cameras

    # Stopping non-existent camera returns False
    assert service.stop_camera("NON_EXISTENT") is False


def test_camera_api_endpoints():
    """Verify FastAPI endpoints /api/v1/cameras/start, /stop, and /cameras using lifespan singleton."""
    with TestClient(app) as client:
        # Start Camera via API
        start_resp = client.post(
            "/api/v1/cameras/start",
            json={
                "camera_id": "CCTV-API-99",
                "source": "0",
                "location": "Test Platform",
            },
        )
        assert start_resp.status_code == 200
        data = start_resp.json()
        assert data["camera_id"] == "CCTV-API-99"
        assert data["status"] == "ACTIVE"

        # List Cameras
        list_resp = client.get("/api/v1/cameras")
        assert list_resp.status_code == 200
        cameras = list_resp.json()
        assert any(c["camera_id"] == "CCTV-API-99" for c in cameras)

        # Stop Camera
        stop_resp = client.post(
            "/api/v1/cameras/stop",
            json={"camera_id": "CCTV-API-99"},
        )
        assert stop_resp.status_code == 200
        assert stop_resp.json()["success"] is True

        # Stop non-existent returns 404
        stop_404 = client.post(
            "/api/v1/cameras/stop",
            json={"camera_id": "CCTV-API-99"},
        )
        assert stop_404.status_code == 404
