import time

from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_operator_store, get_session_store
from security_layer.password_hasher import get_password_hasher


def setup_function():
    """Clear in-memory sessions before each test."""
    get_session_store().clear()


def test_missing_auth_header_rejected_with_401():
    with TestClient(app) as client:
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["detail"]


def test_spoofed_x_user_id_header_ignored_and_rejected():
    with TestClient(app) as client:
        # Attacker provides X-User-ID to impersonate root administrator
        resp = client.get(
            "/api/v1/auth/me",
            headers={"X-User-ID": "admin_root"},
        )
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["detail"]


def test_invalid_bearer_token_rejected_with_401():
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer totally_fabricated_token_xyz999"},
        )
        assert resp.status_code == 401
        assert "Invalid or expired session token" in resp.json()["detail"]


def test_expired_session_token_rejected_with_401():
    session_store = get_session_store()
    session = session_store.create_session(
        operator_id="expired_op",
        username="expired_op",
        role="investigator",
    )
    # Manually expire the session
    session.expires_at = time.time() - 10.0

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert resp.status_code == 401
        assert "Invalid or expired session token" in resp.json()["detail"]


def test_suspended_operator_session_rejected_with_403():
    session_store = get_session_store()
    session = session_store.create_session(
        operator_id="suspended_op",
        username="suspended_op",
        role="investigator",
        status_val="Suspended",
    )

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert resp.status_code == 403
        assert "suspended" in resp.json()["detail"].lower()


def test_successful_login_and_me_profile_flow():
    op_store = get_operator_store()
    username = "test_investigator_42"
    password = "SecureInvestigatorPass@2026"

    # Seed operator with Argon2id
    op_store.create_or_update_operator(
        collection_name="investigators",
        doc_id=username,
        username=username,
        password=password,
        role="investigator",
        name="Agent 42",
        nic="199501019999",
    )

    with TestClient(app) as client:
        login_resp = client.post(
            "/api/v1/auth/login",
            json={
                "username": username,
                "password": password,
                "role": "investigator",
            },
        )
        assert login_resp.status_code == 200
        data = login_resp.json()
        assert data["success"] is True
        token = data["token"]
        assert len(token) >= 32
        assert data["operator"]["username"] == username
        assert data["operator"]["role"] == "investigator"
        assert "password" not in str(data)
        assert "password_hash" not in str(data)

        # Access /me
        me_resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_resp.status_code == 200
        me_data = me_resp.json()
        assert me_data["username"] == username
        assert me_data["name"] == "Agent 42"
        assert "password" not in str(me_data)


def test_login_invalid_password_rejected_with_401():
    op_store = get_operator_store()
    username = "test_inv_wrong_pw"
    op_store.create_or_update_operator(
        collection_name="investigators",
        doc_id=username,
        username=username,
        password="CorrectPassword123!",
        role="investigator",
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "username": username,
                "password": "WrongPassword456!",
            },
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]


def test_login_nonexistent_user_rejected_with_401():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "username": "ghost_user_does_not_exist",
                "password": "AnyPassword123!",
            },
        )
        assert resp.status_code == 401
        assert "not found" in resp.json()["detail"].lower()


def test_logout_revokes_session():
    op_store = get_operator_store()
    username = "test_logout_user"
    password = "LogoutPass@2026"
    op_store.create_or_update_operator(
        collection_name="investigators",
        doc_id=username,
        username=username,
        password=password,
        role="investigator",
    )

    with TestClient(app) as client:
        # 1. Login
        login_resp = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
        )
        token = login_resp.json()["token"]

        # 2. Verify /me works
        assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200

        # 3. Logout
        logout_resp = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert logout_resp.status_code == 200
        assert logout_resp.json()["success"] is True

        # 4. Verify /me now fails with 401
        me_retry = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_retry.status_code == 401


def test_change_password_flow_and_session_invalidation():
    op_store = get_operator_store()
    username = "test_change_pw_user"
    old_pw = "OldPassword123!"
    new_pw = "NewSecurePassword456@"
    op_store.create_or_update_operator(
        collection_name="investigators",
        doc_id=username,
        username=username,
        password=old_pw,
        role="investigator",
    )

    with TestClient(app) as client:
        # Login with old password
        login_resp = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": old_pw},
        )
        old_token = login_resp.json()["token"]

        # Change password
        change_resp = client.post(
            "/api/v1/auth/change-password",
            headers={"Authorization": f"Bearer {old_token}"},
            json={
                "current_password": old_pw,
                "new_password": new_pw,
            },
        )
        assert change_resp.status_code == 200
        new_token = change_resp.json()["token"]
        assert new_token != old_token

        # Old token is revoked
        assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {old_token}"}).status_code == 401

        # New token works
        assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {new_token}"}).status_code == 200

        # Login with new password works
        fresh_login = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": new_pw},
        )
        assert fresh_login.status_code == 200


def test_failsafe_argon2_migration_on_login():
    """Verify that a legacy account with plaintext password seamlessly migrates to Argon2id on login."""
    op_store = get_operator_store()
    username = "legacy_plaintext_user"
    plaintext_pw = "LegacyPlaintextSecret999!"

    # Simulate legacy state: write raw plaintext without password_hash
    offline_data = op_store._load_offline_store()
    offline_data.setdefault("investigators", {})[username] = {
        "name": "Legacy Investigator",
        "username": username,
        "password": plaintext_pw,  # Legacy plaintext
        "role": "investigator",
        "status": "Active",
    }
    op_store._save_offline_store(offline_data)

    with TestClient(app) as client:
        # 1. Login with the legacy password
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": plaintext_pw},
        )
        assert resp.status_code == 200
        token = resp.json()["token"]
        assert token is not None

        # 2. Verify stored record was migrated to Argon2id
        migrated_data, _, _ = op_store.get_operator(username)
        assert migrated_data is not None
        assert "password_hash" in migrated_data
        assert migrated_data["password_hash"].startswith("$argon2id")
        assert "password" not in migrated_data  # Plaintext permanently removed
        assert migrated_data.get("password_migrated") is True

        # 3. Verify subsequent login directly uses the Argon2id hash
        hasher = get_password_hasher()
        is_valid, needs_rehash = hasher.verify(plaintext_pw, migrated_data["password_hash"])
        assert is_valid is True
        assert needs_rehash is False
