import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.server import app
from services.camera_source_resolver import CameraSourceResolver
from services.camera_worker import normalize_camera_source
from services.gait_service import GaitService


def test_normalize_camera_source():
    """Verify source normalization for numeric indices and URLs."""
    assert normalize_camera_source("0") == 0
    assert normalize_camera_source(" 2 ") == 2
    assert normalize_camera_source("rtsp://192.168.1.100:554/stream") == "rtsp://192.168.1.100:554/stream"
    assert normalize_camera_source("http://192.168.1.100:8080/video") == "http://192.168.1.100:8080/video"


def test_source_resolver_free_usb_discovery():
    """Verify resolver discovers a free USB webcam."""
    resolver = CameraSourceResolver()

    with patch.object(resolver, "probe_usb_webcam", return_value=True):
        res = resolver.resolve_source(camera_id="CCTV-TEST-1", requested_source="auto")
        assert res["resolved_source_type"] == "usb"
        assert res["resolved_source"] == "0"
        assert res["resolved_source_label"] == "USB Webcam 0"
        assert resolver.is_source_reserved("usb:0") is True

        # Next worker gets next free device index
        res2 = resolver.resolve_source(camera_id="CCTV-TEST-2", requested_source="auto")
        assert res2["resolved_source"] == "1"
        assert res2["resolved_source_label"] == "USB Webcam 1"

        # Release first worker source
        resolver.release_source_by_camera_id("CCTV-TEST-1")
        assert resolver.is_source_reserved("usb:0") is False


def test_source_resolver_skip_unavailable_usb_and_select_rtsp():
    """Verify resolver skips failing USB devices and falls back to registered RTSP."""
    resolver = CameraSourceResolver()
    resolver._registered_cameras = [
        {"id": "camera_01", "name": "Main Gate", "url": "rtsp://192.168.1.100:554/stream1", "enabled": True}
    ]

    # USB fails, RTSP stream probe succeeds
    with patch.object(resolver, "probe_usb_webcam", return_value=False),          patch.object(resolver, "probe_stream", return_value=True):
        res = resolver.resolve_source(camera_id="CCTV-TEST-RTSP", requested_source="auto")
        assert res["resolved_source_type"] == "rtsp"
        assert res["resolved_source"] == "rtsp://192.168.1.100:554/stream1"
        assert "Main Gate" in res["resolved_source_label"]


def test_source_resolver_no_source_available_raises():
    """Verify controlled error when no USB or RTSP sources are reachable."""
    resolver = CameraSourceResolver()
    resolver._registered_cameras = []

    with patch.object(resolver, "probe_usb_webcam", return_value=False):
        with pytest.raises(RuntimeError) as exc_info:
            resolver.resolve_source(camera_id="CCTV-FAIL", requested_source="auto")
            assert "camera source is available" in str(exc_info.value)


def test_gait_service_auto_source_lifecycle():
    """Verify GaitService start_camera and stop_camera with auto source and reservation cleanup."""
    service = GaitService()

    with patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):
        # Start with source: "auto"
        info = service.start_camera(
            camera_id="CCTV-AUTO-1",
            source="auto",
            location="Sector A",
            zone_id="Z01"
        )
        assert info["camera_id"] == "CCTV-AUTO-1"
        assert info["status"] == "ACTIVE"
        assert info["resolved_source_type"] == "usb"
        assert info["zone_id"] == "Z01"
        assert "usb:0" in service.source_resolver._reserved_sources

        # Stop releases reservation
        stopped = service.stop_camera("CCTV-AUTO-1")
        assert stopped is True
        assert "usb:0" not in service.source_resolver._reserved_sources


def test_api_cameras_start_auto_contract():
    """Verify FastAPI /api/v1/cameras/start with source: auto."""
    with TestClient(app) as client:
        # Start worker via API
        resp = client.post(
            "/api/v1/cameras/start",
            json={
                "camera_id": "CCTV-API-AUTO",
                "source": "auto",
                "location": "North Gate",
                "zone_id": "Z02"
            }
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["camera_id"] == "CCTV-API-AUTO"
        assert data["status"] == "ACTIVE"
        assert data["resolved_source_type"] in ["usb", "rtsp"]
        assert data["resolved_source_label"] is not None

        # Verify listed in GET /api/v1/cameras
        list_resp = client.get("/api/v1/cameras")
        assert list_resp.status_code == 200
        cams = list_resp.json()
        matching = [c for c in cams if c["camera_id"] == "CCTV-API-AUTO"]
        assert len(matching) == 1
        assert matching[0]["resolved_source_label"] is not None

        # Stop
        stop_resp = client.post(
            "/api/v1/cameras/stop",
            json={"camera_id": "CCTV-API-AUTO"}
        )
        assert stop_resp.status_code == 200
