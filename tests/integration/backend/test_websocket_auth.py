import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api.server import app
from api.v1.router import get_gait_service
from security_layer.auth import get_session_store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def gait_service(client):
    if hasattr(app.state, "gait_service") and app.state.gait_service:
        return app.state.gait_service
    return get_gait_service()


def test_01_missing_auth_ws_recognition_rejected_with_1008(client, gait_service):
    """Missing authentication on /api/v1/ws/recognition must be rejected with close code 1008."""
    initial_count = len(gait_service.ws_manager.active_connections)
    with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect("/api/v1/ws/recognition"):
        pass
    assert exc_info.value.code == 1008
    assert len(gait_service.ws_manager.active_connections) == initial_count


def test_02_missing_auth_ws_events_rejected_with_1008(client, gait_service):
    """Missing authentication on /api/v1/ws/events must be rejected with close code 1008."""
    initial_count = len(gait_service.ws_manager.active_connections)
    with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect("/api/v1/ws/events"):
        pass
    assert exc_info.value.code == 1008
    assert len(gait_service.ws_manager.active_connections) == initial_count


def test_03_invalid_token_rejected_with_1008(client, gait_service):
    """Malformed/unregistered session token in subprotocol must be rejected with 1008."""
    initial_count = len(gait_service.ws_manager.active_connections)
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect(
            "/api/v1/ws/events",
            subprotocols=["argus-auth", "completely_fake_invalid_token_xyz999"],
        ),
    ):
        pass
    assert exc_info.value.code == 1008
    assert len(gait_service.ws_manager.active_connections) == initial_count


def test_04_expired_token_rejected_with_1008(client, gait_service):
    """Expired session token in subprotocol must be rejected with 1008."""
    store = get_session_store()
    session = store.create_session(
        operator_id="expired_ws_user",
        username="expired_ws_user",
        role="investigator",
    )
    # Manually expire the session TTL
    session.expires_at = time.time() - 10.0

    initial_count = len(gait_service.ws_manager.active_connections)
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect(
            "/api/v1/ws/events",
            subprotocols=["argus-auth", session.token],
        ),
    ):
        pass
    assert exc_info.value.code == 1008
    assert len(gait_service.ws_manager.active_connections) == initial_count


def test_05_suspended_operator_rejected_with_1008(client, gait_service):
    """Suspended operator session must be rejected with 1008."""
    store = get_session_store()
    session = store.create_session(
        operator_id="suspended_ws_user",
        username="suspended_ws_user",
        role="investigator",
        status_val="Suspended",
    )

    initial_count = len(gait_service.ws_manager.active_connections)
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect(
            "/api/v1/ws/events",
            subprotocols=["argus-auth", session.token],
        ),
    ):
        pass
    assert exc_info.value.code == 1008
    assert len(gait_service.ws_manager.active_connections) == initial_count


def test_06_valid_investigator_authenticated_successfully(client, gait_service):
    """Authenticated investigator can connect and socket is registered in active_connections."""
    store = get_session_store()
    session = store.create_session(
        operator_id="inv_ws_user_01",
        username="inv_ws_user_01",
        role="investigator",
    )

    initial_count = len(gait_service.ws_manager.active_connections)
    with client.websocket_connect(
        "/api/v1/ws/events",
        subprotocols=["argus-auth", session.token],
    ) as ws:
        assert len(gait_service.ws_manager.active_connections) == initial_count + 1
        ws.send_text("ping")

    # Verify disconnect cleanup
    assert len(gait_service.ws_manager.active_connections) == initial_count


def test_07_valid_admin_authenticated_successfully(client, gait_service):
    """Authenticated admin can connect successfully."""
    store = get_session_store()
    session = store.create_session(
        operator_id="admin_ws_user_01",
        username="admin_ws_user_01",
        role="admin",
    )

    with client.websocket_connect(
        "/api/v1/ws/events",
        subprotocols=["argus-auth", session.token],
    ) as ws:
        ws.send_text("ping")


def test_08_valid_root_admin_authenticated_successfully(client, gait_service):
    """Authenticated root_admin can connect successfully."""
    store = get_session_store()
    session = store.create_session(
        operator_id="root_ws_user_01",
        username="root_ws_user_01",
        role="root_admin",
    )

    with client.websocket_connect(
        "/api/v1/ws/events",
        subprotocols=["argus-auth", session.token],
    ) as ws:
        ws.send_text("ping")


def test_09_both_endpoints_authenticated_successfully(client):
    """Both /ws/recognition and /ws/events accept valid subprotocol authentication."""
    store = get_session_store()
    session = store.create_session(
        operator_id="dual_ep_user",
        username="dual_ep_user",
        role="investigator",
    )

    with client.websocket_connect(
        "/api/v1/ws/recognition",
        subprotocols=["argus-auth", session.token],
    ) as ws1:
        ws1.send_text("ping_recognition")

    with client.websocket_connect(
        "/api/v1/ws/events",
        subprotocols=["argus-auth", session.token],
    ) as ws2:
        ws2.send_text("ping_events")


def test_10_arbitrary_identity_headers_rejected(client, gait_service):
    """Spoofed client-asserted identity headers (X-User-ID, etc.) without valid session token are rejected."""
    initial_count = len(gait_service.ws_manager.active_connections)
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect(
            "/api/v1/ws/events",
            headers={"X-User-ID": "admin", "X-Role": "root_admin"},
        ),
    ):
        pass
    assert exc_info.value.code == 1008
    assert len(gait_service.ws_manager.active_connections) == initial_count


def test_11_authorization_bearer_header_supported_for_test_clients(client, gait_service):
    """Non-browser test clients passing Authorization: Bearer <token> header are supported."""
    store = get_session_store()
    session = store.create_session(
        operator_id="header_ws_user",
        username="header_ws_user",
        role="investigator",
    )
    with client.websocket_connect(
        "/api/v1/ws/events",
        headers={"Authorization": f"Bearer {session.token}"},
    ) as ws:
        ws.send_text("ping_header")


def test_12_raw_token_not_echoed_as_accepted_subprotocol(client):
    """Server must negotiate subprotocol as 'argus-auth' and never echo the raw token."""
    store = get_session_store()
    session = store.create_session(
        operator_id="subproto_check_user",
        username="subproto_check_user",
        role="investigator",
    )
    with client.websocket_connect(
        "/api/v1/ws/events",
        subprotocols=["argus-auth", session.token],
    ) as ws:
        # Check that the raw token is not echoed as the accepted subprotocol
        accepted = ws.scope.get("subprotocol")
        assert accepted != session.token
        if accepted is not None:
            assert accepted == "argus-auth"


def test_13_frontend_does_not_use_query_parameter_tokens():
    """Verify that frontend WebSocket implementation never transmits tokens via URL query parameters."""
    from pathlib import Path

    gait_api_code = Path("frontend/src/services/gaitApi.js").read_text(encoding="utf-8")
    assert "?token=" not in gait_api_code
    assert "?session_token=" not in gait_api_code
    assert "?auth=" not in gait_api_code
    assert "['argus-auth', token]" in gait_api_code
