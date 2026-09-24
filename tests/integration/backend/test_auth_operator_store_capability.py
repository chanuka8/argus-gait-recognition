"""Regression tests for the operator-store realtime-monitoring capability.

Background: the frontend used to run a Firestore onSnapshot listener for every
logged-in user to catch admin-initiated suspension/deletion in real time. In
ARGUS_OPERATOR_STORE_MODE=offline (the documented dev/test auth backend), no
operator ever has a Firestore document, so that listener's "empty and
server-confirmed" branch always fired within about a second of login and
force-logged the user out - offline mode was effectively unusable for
anything beyond the very first render.

The fix: SessionToken.to_profile_dict() (returned by both POST /auth/login
and GET /auth/me) now includes capabilities.realtime_operator_status, true
only when the backend's authoritative operator store is Firebase-backed and
therefore has a document for the frontend to watch. The frontend only
registers its Firestore listener when this flag is true.
"""

import time

from fastapi.testclient import TestClient

from app.api.server import app
from app.security_layer.auth import SessionToken, get_session_store


def setup_function():
    get_session_store().clear()


def test_to_profile_dict_reports_offline_mode_capability_false(monkeypatch):
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "offline")
    session = SessionToken(
        token="tok",
        operator_id="op1",
        username="op1",
        role="investigator",
    )
    profile = session.to_profile_dict()
    assert profile["capabilities"]["realtime_operator_status"] is False


def test_to_profile_dict_reports_firebase_mode_capability_true(monkeypatch):
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")
    session = SessionToken(
        token="tok",
        operator_id="op1",
        username="op1",
        role="investigator",
    )
    profile = session.to_profile_dict()
    assert profile["capabilities"]["realtime_operator_status"] is True


def test_auth_me_response_includes_capability_in_offline_mode(monkeypatch):
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "offline")
    session_store = get_session_store()
    session = session_store.create_session(
        operator_id="offline_op",
        username="offline_op",
        role="investigator",
    )

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["capabilities"]["realtime_operator_status"] is False


def test_offline_session_survives_past_ten_seconds(monkeypatch):
    """Offline-mode backend session must remain valid well past the window
    where the old Firestore listener used to force a false-positive logout
    (observed within ~1s of login)."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "offline")
    session_store = get_session_store()
    session = session_store.create_session(
        operator_id="offline_op2",
        username="offline_op2",
        role="investigator",
    )

    with TestClient(app) as client:
        for _ in range(3):
            resp = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {session.token}"},
            )
            assert resp.status_code == 200
            time.sleep(3.5)

    # 3 rounds x 3.5s > 10s total elapsed, and the session is still valid
    # (TTL is 8h, idle timeout 30min - this window changes neither).
    assert session_store.get_session(session.token) is not None
