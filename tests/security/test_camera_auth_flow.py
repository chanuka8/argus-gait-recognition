import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.server import app
from api.v1.router import get_gait_service
from security_layer.auth import extract_bearer_token, get_session_store


@pytest.fixture(autouse=True)
def clean_sessions_and_dependencies():
    """Clear session store and dependency overrides before and after each test."""
    get_session_store().clear()
    app.dependency_overrides.clear()
    yield
    get_session_store().clear()
    app.dependency_overrides.clear()


def test_01_unauthenticated_camera_start_rejected_with_401():
    """INVARIANT 1: Unauthenticated camera start request is rejected with 401."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/start",
            json={"camera_id": "cam_node_01", "source": "auto"},
        )
        assert resp.status_code == 401
        assert "Authentication required. Provide Authorization: Bearer <session_token>" in resp.json()["detail"]


def test_02_authenticated_camera_start_with_admin_accepted():
    """INVARIANT 2: Authenticated camera start request with valid admin Bearer token is accepted."""
    store = get_session_store()
    session = store.create_session(
        operator_id="admin_01",
        username="admin_01",
        role="admin",
    )

    mock_service = MagicMock()
    mock_service.start_camera.return_value = {
        "camera_id": "cam_node_01",
        "source": "auto",
        "source_type": "webcam",
        "status": "connected",
        "location": "Sector A",
        "zone_id": "Z01",
        "processed_frames": 1,
        "fps": 15.0,
    }
    app.dependency_overrides[get_gait_service] = lambda: mock_service

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/start",
            json={"camera_id": "cam_node_01", "source": "auto", "location": "Sector A"},
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["camera_id"] == "cam_node_01"
        assert data["status"] == "connected"


def test_03_missing_authorization_header_rejected_with_401():
    """INVARIANT 3: Request without Authorization header is rejected."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/stop",
            json={"camera_id": "cam_node_01"},
        )
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["detail"]


def test_04_invalid_or_expired_bearer_token_rejected_with_401():
    """INVARIANT 4: Invalid or expired token is rejected with 401."""
    with TestClient(app) as client:
        # Invalid token
        resp = client.post(
            "/api/v1/cameras/start",
            json={"camera_id": "cam_node_01"},
            headers={"Authorization": "Bearer non_existent_token_12345"},
        )
        assert resp.status_code == 401
        assert "Invalid or expired session token" in resp.json()["detail"]

    # Expired token
    store = get_session_store()
    session = store.create_session(operator_id="admin_exp", username="admin_exp", role="admin")
    session.expires_at = time.time() - 3600.0

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/start",
            json={"camera_id": "cam_node_01"},
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert resp.status_code == 401
        assert "Invalid or expired session token" in resp.json()["detail"]


def test_05_investigator_role_camera_start_rejected_with_403():
    """INVARIANT 5: Investigator session lacks camera control privileges -> 403 Forbidden."""
    store = get_session_store()
    session = store.create_session(operator_id="inv_01", username="inv_01", role="investigator")

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/start",
            json={"camera_id": "cam_node_01"},
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert resp.status_code == 403
        assert "administrative privileges" in resp.json()["detail"]


def test_06_unauthenticated_camera_stream_rejected_with_401():
    """INVARIANT 6: Camera stream endpoint rejects unauthenticated requests with 401."""
    with TestClient(app) as client:
        resp = client.get("/api/v1/cameras/cam_node_01/stream")
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["detail"]


def test_07_authenticated_camera_stream_with_bearer_header_accepted():
    """INVARIANT 7: Camera stream request with valid Bearer Authorization header is accepted."""
    store = get_session_store()
    session = store.create_session(operator_id="op_stream", username="op_stream", role="investigator")

    mock_worker = MagicMock()
    mock_worker.is_running.return_value = False
    mock_worker.get_latest_jpeg.return_value = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\xff\xd9"

    mock_service = MagicMock()
    mock_service.active_cameras = {"cam_test": "cam_test"}
    mock_service.get_camera_worker.return_value = mock_worker
    app.dependency_overrides[get_gait_service] = lambda: mock_service

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/cameras/cam_test/stream",
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.headers["content-type"]


def test_08_query_parameter_token_rejected_with_401():
    """INVARIANT 8: Camera stream request with token query parameter must NOT authenticate (rejected with 401)."""
    store = get_session_store()
    session = store.create_session(operator_id="op_query", username="op_query", role="investigator")

    mock_worker = MagicMock()
    mock_worker.is_running.return_value = False
    mock_worker.get_latest_jpeg.return_value = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\xff\xd9"

    mock_service = MagicMock()
    mock_service.active_cameras = {"cam_test2": "cam_test2"}
    mock_service.get_camera_worker.return_value = mock_worker
    app.dependency_overrides[get_gait_service] = lambda: mock_service

    with TestClient(app) as client:
        # Query parameter token without Authorization header must be rejected
        resp = client.get(
            f"/api/v1/cameras/cam_test2/stream?token={session.token}",
        )
        assert resp.status_code == 401
        assert "Authentication required. Provide Authorization: Bearer <session_token>" in resp.json()["detail"]


def test_09_extract_bearer_token_rejects_query_parameters():
    """INVARIANT 9: extract_bearer_token strictly extracts from Authorization header and ignores query params."""
    mock_req_valid = MagicMock()
    mock_req_valid.headers = {"Authorization": "Bearer header_token_abc"}
    mock_req_valid.query_params = {"token": "query_token_xyz"}

    token = extract_bearer_token(mock_req_valid)
    assert token == "header_token_abc"

    # Query param only -> None
    mock_req_query_only = MagicMock()
    mock_req_query_only.headers = {}
    mock_req_query_only.query_params = {"token": "query_token_xyz"}

    token = extract_bearer_token(mock_req_query_only)
    assert token is None

    # Empty request -> None
    mock_req_empty = MagicMock()
    mock_req_empty.headers = {}
    mock_req_empty.query_params = {}

    assert extract_bearer_token(mock_req_empty) is None


def test_10_camera_snapshot_authenticated():
    """INVARIANT 10: /cameras/{camera_id}/snapshot requires session token and returns JPEG."""
    store = get_session_store()
    session = store.create_session(operator_id="op_snap", username="op_snap", role="investigator")

    mock_worker = MagicMock()
    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\xff\xd9"
    mock_worker.get_latest_jpeg.return_value = dummy_jpeg

    mock_service = MagicMock()
    mock_service.active_cameras = {"cam_snap": "cam_snap"}
    mock_service.get_camera_worker.return_value = mock_worker
    app.dependency_overrides[get_gait_service] = lambda: mock_service

    with TestClient(app) as client:
        # Unauthenticated -> 401
        resp_unauth = client.get("/api/v1/cameras/cam_snap/snapshot")
        assert resp_unauth.status_code == 401

        # Authenticated -> 200 JPEG
        resp = client.get(
            "/api/v1/cameras/cam_snap/snapshot",
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == dummy_jpeg
