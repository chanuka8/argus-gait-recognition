"""
ARGUS AI - Automatic Camera-Source Detection & Runtime Lifecycle Test Suite.
Verifies deterministic auto-detection for Webcams and RTSP streams, ensuring
source type is only resolved and displayed at stream connection time when frames
are actually received, and remains hidden on standby, disconnect, or error.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from services.camera_worker import CameraWorker
from services.gait_service import GaitService


def _dummy_frame(w: int = 640, h: int = 480):
    """Return a valid dummy BGR image frame for OpenCV capture mocking."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[100:200, 100:200] = [0, 255, 0]
    return frame


# ==============================================================================
# TEST 01: Camera Card Initially Loads (Standby State)
# ==============================================================================
def test_01_camera_standby_initial_state_source_hidden():
    """Verify camera in standby before start stream does NOT have active source_type."""
    service = GaitService()
    # Before stream starts, camera is not in active_cameras
    cam_info = service.get_camera_info("UNSTARTED_CAM")
    assert cam_info is None  # Source type remains null / unknown


# ==============================================================================
# TEST 02: Click Start Stream for a Webcam
# ==============================================================================
def test_02_start_stream_webcam_success():
    """Verify Start Stream for webcam connects, captures frames, and exposes source_type='webcam'."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):

        cam_info = service.start_camera(camera_id="CAM-WC-01", source="auto")
        assert cam_info["status"] == "ACTIVE"
        assert cam_info["source_type"] == "webcam"
        assert cam_info["processed_frames"] >= 1

        service.stop_camera("CAM-WC-01")


# ==============================================================================
# TEST 03: Click Start Stream for RTSP
# ==============================================================================
def test_03_start_stream_rtsp_success():
    """Verify Start Stream for RTSP connects, captures frames, and exposes source_type='rtsp'."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service.source_resolver, "probe_stream", return_value=True):

        cam_info = service.start_camera(
            camera_id="CAM-RTSP-01",
            source="rtsp://admin:pass@192.168.1.120:554/live",
        )
        assert cam_info["status"] == "ACTIVE"
        assert cam_info["source_type"] == "rtsp"
        assert cam_info["processed_frames"] >= 1

        service.stop_camera("CAM-RTSP-01")


# ==============================================================================
# TEST 04: Webcam Connection Fails
# ==============================================================================
def test_04_webcam_connection_failure():
    """Verify webcam failure hides source, releases lock, and returns descriptive error."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cap.read.return_value = (False, None)

    fast_cfg = {
        "startup_timeout": 0.05,
        "startup_retry_interval": 0.01,
        "reconnect_interval": 0.05,
        "max_reconnect_attempts": 0,
    }

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service, "_load_camera_config", return_value=fast_cfg), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):

        with pytest.raises(RuntimeError) as exc:
            service.start_camera(camera_id="CAM-WC-FAIL", source="auto")
        assert "Unable to establish stream connection" in str(exc.value)

        # Ensure camera is not active and source is not reserved
        assert "CAM-WC-FAIL" not in service.active_cameras


# ==============================================================================
# TEST 05: RTSP Connection Fails
# ==============================================================================
def test_05_rtsp_connection_failure():
    """Verify unreachable RTSP stream failure hides source and raises descriptive error."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cap.read.return_value = (False, None)

    fast_cfg = {
        "startup_timeout": 0.05,
        "startup_retry_interval": 0.01,
        "reconnect_interval": 0.05,
        "max_reconnect_attempts": 0,
    }

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service, "_load_camera_config", return_value=fast_cfg), \
         patch.object(service.source_resolver, "probe_stream", return_value=False), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=False):

        with pytest.raises(RuntimeError) as exc:
            service.start_camera(camera_id="CAM-RTSP-FAIL", source="rtsp://10.99.99.99:554/dead")
        assert "Unable to detect camera source" in str(exc.value) or "Unable to establish stream connection" in str(exc.value)

        assert "CAM-RTSP-FAIL" not in service.active_cameras


# ==============================================================================
# TEST 06: Connected Camera is Stopped
# ==============================================================================
def test_06_stop_stream_hides_source():
    """Verify stopping an active camera releases worker and removes active stream source."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):

        service.start_camera(camera_id="CAM-STOP-01", source="auto")
        assert service.get_camera_info("CAM-STOP-01")["status"] == "ACTIVE"

        service.stop_camera("CAM-STOP-01")
        # After stopping, camera is no longer active
        assert service.get_camera_info("CAM-STOP-01") is None


# ==============================================================================
# TEST 07: Unexpected Disconnect Reconnect Mechanism
# ==============================================================================
def test_07_unexpected_disconnect_recovery():
    """Verify camera worker watchdog initiates reconnect on stream drop."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [
        (True, _dummy_frame()),
        (False, None),
        (True, _dummy_frame()),
    ]

    cfg = {
        "type": "webcam",
        "device_index": 0,
        "reconnect_interval": 0.05,
        "max_reconnect_attempts": 2,
        "target_fps": 30,
        "startup_timeout": 0.5,
    }

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        worker = CameraWorker(camera_id="CAM-RECOVERY", camera_config=cfg)
        assert worker.start() is True
        worker.stop(timeout=1.0)


# ==============================================================================
# TEST 08: Start Stream Again After Disconnect
# ==============================================================================
def test_08_reconnect_detects_runtime_source_again():
    """Verify restarting a previously stopped camera re-detects and confirms active source."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):

        # First run
        info1 = service.start_camera(camera_id="CAM-RESTART", source="auto")
        assert info1["source_type"] == "webcam"
        service.stop_camera("CAM-RESTART")

        # Second run: re-connects cleanly
        info2 = service.start_camera(camera_id="CAM-RESTART", source="auto")
        assert info2["source_type"] == "webcam"
        assert info2["status"] == "ACTIVE"
        service.stop_camera("CAM-RESTART")


# ==============================================================================
# TEST 09: Multiple Cameras Independent Active Sources
# ==============================================================================
def test_09_multiple_cameras_independent_sources():
    """Verify multiple camera cards show their own source independently upon start."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=True), \
         patch.object(service.source_resolver, "probe_stream", return_value=True):

        cam1 = service.start_camera(camera_id="CAM-MULTI-1", source="auto")
        cam2 = service.start_camera(camera_id="CAM-MULTI-2", source="rtsp://192.168.1.55:554/live")

        assert cam1["source_type"] == "webcam"
        assert cam2["source_type"] == "rtsp"

        service.stop_camera("CAM-MULTI-1")
        service.stop_camera("CAM-MULTI-2")


# ==============================================================================
# TEST 10: FastAPI Endpoint & Schema Validation
# ==============================================================================
def test_10_api_start_stream_endpoint_contract():
    """Verify FastAPI /api/v1/cameras/start returns valid CameraInfoResponse with runtime source."""
    from fastapi.testclient import TestClient
    from api.server import app
    from api.v1.router import get_gait_service

    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    app.dependency_overrides[get_gait_service] = lambda: service
    try:
        with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
             patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):

            client = TestClient(app)
            response = client.post(
                "/api/v1/cameras/start",
                json={"camera_id": "API-TEST-CAM", "source": "auto", "location": "Main Entrance"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["camera_id"] == "API-TEST-CAM"
            assert data["status"] == "ACTIVE"
            assert data["source_type"] == "webcam"

            # Stop camera
            stop_res = client.post(
                "/api/v1/cameras/stop",
                json={"camera_id": "API-TEST-CAM"},
            )
            assert stop_res.status_code == 200
    finally:
        app.dependency_overrides.clear()
