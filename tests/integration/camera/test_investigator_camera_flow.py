from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_session_store
from security_layer.authorization import Permission, Role, has_permission


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers():
    session_store = get_session_store()

    # Create investigator session
    inv_token = session_store.create_session("op_inv", "inv_user", Role.INVESTIGATOR.value)

    # Create admin session
    admin_token = session_store.create_session("op_admin", "admin_user", Role.ADMIN.value)

    # Create unauthorized session (guest/unauthorized role)
    guest_token = session_store.create_session("op_guest", "guest_user", "guest")

    return {
        "investigator": {"Authorization": f"Bearer {inv_token.token}"},
        "admin": {"Authorization": f"Bearer {admin_token.token}"},
        "guest": {"Authorization": f"Bearer {guest_token.token}"},
    }


def test_investigator_has_camera_operational_permissions():
    """Verify investigator possesses camera view/start/stop but NOT camera configuration/management."""
    assert has_permission(Role.INVESTIGATOR.value, Permission.CAMERA_START) is True
    assert has_permission(Role.INVESTIGATOR.value, Permission.CAMERA_STOP) is True
    assert has_permission(Role.INVESTIGATOR.value, Permission.CAMERA_VIEW) is True
    assert has_permission(Role.INVESTIGATOR.value, Permission.CAMERA_STREAM) is True
    assert has_permission(Role.INVESTIGATOR.value, Permission.CAMERA_LIST) is True

    # Destructive/config management must remain forbidden to investigator
    assert has_permission(Role.INVESTIGATOR.value, Permission.CAMERA_MANAGE) is False
    assert has_permission(Role.INVESTIGATOR.value, Permission.CAMERA_CREDENTIAL_MANAGE) is False


def test_investigator_camera_flow_and_rbac(test_client, auth_headers):
    """Verify complete CCTV lifecycle for investigator: GET, START (no 403), STREAM, STOP, RECONNECT."""
    # 1. Investigator can list cameras
    list_resp = test_client.get("/api/v1/cameras", headers=auth_headers["investigator"])
    assert list_resp.status_code == 200
    assert isinstance(list_resp.json(), list)

    # Mock camera hardware capture to guarantee reliable test execution across environments
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    cv2.putText(dummy_frame, "TEST_FEED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640 if prop == cv2.CAP_PROP_FRAME_WIDTH else (480 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 30)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        # 2. Investigator starts camera: MUST SUCCEED (200, NOT 403!)
        start_payload = {
            "camera_id": "cam_investigator_test_01",
            "source": "0",
            "location": "Sector A CCTV",
            "zone_id": "Z01",
        }
        start_resp = test_client.post(
            "/api/v1/cameras/start",
            json=start_payload,
            headers=auth_headers["investigator"],
        )
        assert start_resp.status_code == 200, f"Expected 200 for investigator start, got {start_resp.status_code}: {start_resp.text}"
        cam_info = start_resp.json()
        assert cam_info["camera_id"] == "cam_investigator_test_01"
        assert cam_info["status"] in ("ACTIVE", "connected")

        # 3. Stream delivery: verify stream endpoint delivers multipart MJPEG frames
        stream_resp = test_client.get(
            "/api/v1/cameras/cam_investigator_test_01/stream?max_frames=1",
            headers=auth_headers["investigator"],
        )
        assert stream_resp.status_code == 200
        assert "multipart/x-mixed-replace" in stream_resp.headers.get("content-type", "")
        assert len(stream_resp.content) > 0
        assert b"--frame" in stream_resp.content
        assert b"Content-Type: image/jpeg" in stream_resp.content

        # 4. Snapshot delivery
        snap_resp = test_client.get(
            "/api/v1/cameras/cam_investigator_test_01/snapshot",
            headers=auth_headers["investigator"],
        )
        assert snap_resp.status_code == 200
        assert snap_resp.content.startswith(b"\xff\xd8")  # JPEG magic bytes

        # 5. Investigator stops camera: MUST SUCCEED (200)
        stop_payload = {"camera_id": "cam_investigator_test_01"}
        stop_resp = test_client.post(
            "/api/v1/cameras/stop",
            json=stop_payload,
            headers=auth_headers["investigator"],
        )
        assert stop_resp.status_code == 200
        assert stop_resp.json()["success"] is True

        # 6. Reconnect works cleanly
        reconnect_resp = test_client.post(
            "/api/v1/cameras/start",
            json=start_payload,
            headers=auth_headers["investigator"],
        )
        assert reconnect_resp.status_code == 200

        # Clean up
        test_client.post(
            "/api/v1/cameras/stop",
            json=stop_payload,
            headers=auth_headers["investigator"],
        )


def test_camera_security_and_rejections(test_client, auth_headers):
    """Verify unauthorized roles, expired tokens, and query string auth are rejected."""
    cam_payload = {"camera_id": "cam_security_check", "source": "0"}

    # 1. Unauthorized/guest role must be rejected with 403
    guest_start = test_client.post(
        "/api/v1/cameras/start",
        json=cam_payload,
        headers=auth_headers["guest"],
    )
    assert guest_start.status_code == 403

    # 2. Invalid session token must be rejected with 401
    invalid_token_start = test_client.post(
        "/api/v1/cameras/start",
        json=cam_payload,
        headers={"Authorization": "Bearer invalid_token_12345"},
    )
    assert invalid_token_start.status_code == 401

    # 3. Query string token authentication MUST remain rejected (no auth header = 401)
    query_start = test_client.post(
        "/api/v1/cameras/start?token=some_token",
        json=cam_payload,
    )
    assert query_start.status_code == 401

    # 4. Admin role can perform camera start and stop
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.get.return_value = 640

    with patch("cv2.VideoCapture", return_value=mock_cap):
        admin_start = test_client.post(
            "/api/v1/cameras/start",
            json=cam_payload,
            headers=auth_headers["admin"],
        )
        assert admin_start.status_code == 200

        admin_stop = test_client.post(
            "/api/v1/cameras/stop",
            json={"camera_id": "cam_security_check"},
            headers=auth_headers["admin"],
        )
        assert admin_stop.status_code == 200
