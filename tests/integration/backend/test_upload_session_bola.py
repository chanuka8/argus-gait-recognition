"""SEC-04: Upload session ownership BOLA/IDOR regression tests.

Verifies that authenticated investigators cannot access, modify, commit,
or cancel upload sessions owned by other investigators.

Tests cover:
  - Owner access (status, chunk, commit, cancel)
  - Cross-user rejection for all 4 operations with state non-mutation
  - Admin and root_admin oversight access
  - Unauthenticated rejection
  - Invalid upload_id → 404
  - Client owner-spoofing does not bypass authorization
  - Ownership persists across session reconnect
"""

import io

import pytest
from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_operator_store, get_session_store
from security_layer.password_hasher import get_password_hasher
from services.upload_session_manager import UploadSessionManager


@pytest.fixture(autouse=True)
def setup_sec04_fixtures(tmp_path):
    """Seed operators and reset upload sessions for each test."""
    get_session_store().clear()
    op_store = get_operator_store()
    hasher = get_password_hasher()

    offline_data = {
        "admins": {
            "root_boss": {
                "name": "Root Boss",
                "username": "root_boss",
                "password_hash": hasher.hash("RootAdmin@2026!"),
                "role": "root_admin",
                "status": "Active",
            },
            "regular_admin": {
                "name": "Regular Admin",
                "username": "regular_admin",
                "password_hash": hasher.hash("RegAdmin@2026!"),
                "role": "admin",
                "status": "Active",
            },
        },
        "investigators": {
            "inv_alice": {
                "name": "Alice Detective",
                "username": "inv_alice",
                "password_hash": hasher.hash("AlicePass@2026!"),
                "role": "investigator",
                "status": "Active",
            },
            "inv_bob": {
                "name": "Bob Detective",
                "username": "inv_bob",
                "password_hash": hasher.hash("BobPass@2026!"),
                "role": "investigator",
                "status": "Active",
            },
        },
    }
    op_store._save_offline_store(offline_data)

    # Reset singleton for clean session dir
    UploadSessionManager._instance = None
    mgr = UploadSessionManager(sessions_dir=str(tmp_path / "upload_sessions"))
    UploadSessionManager._instance = mgr

    yield

    UploadSessionManager._instance = None


def create_token_for(username: str, role: str, status: str = "Active") -> str:
    session = get_session_store().create_session(
        operator_id=username,
        username=username,
        role=role,
        status=status,
    )
    return session.token


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_upload_session(client: TestClient, token: str, person_id: str = "person_alice_01") -> dict:
    """Helper to create an upload session and return the response JSON."""
    resp = client.post(
        "/api/v1/cases/upload-session/init",
        json={
            "person_id": person_id,
            "filename": "test_video.mp4",
            "total_size": 4096,
            "chunk_size": 1024,
            "media_type": "video",
        },
        headers=auth_header(token),
    )
    assert resp.status_code == 201, f"Failed to create upload session: {resp.text}"
    return resp.json()


def upload_chunk(client: TestClient, token: str, upload_id: str, chunk_index: int, data: bytes = b"X" * 1024):
    """Helper to upload a chunk."""
    return client.post(
        f"/api/v1/cases/upload-session/{upload_id}/chunk",
        data={"chunk_index": str(chunk_index)},
        files={"file": ("chunk.bin", io.BytesIO(data), "application/octet-stream")},
        headers=auth_header(token),
    )


def get_status(client: TestClient, token: str, upload_id: str):
    """Helper to get upload session status."""
    return client.get(
        f"/api/v1/cases/upload-session/{upload_id}/status",
        headers=auth_header(token),
    )


def commit_session(client: TestClient, token: str, upload_id: str):
    """Helper to commit an upload session."""
    return client.post(
        f"/api/v1/cases/upload-session/{upload_id}/commit",
        headers=auth_header(token),
    )


def cancel_session(client: TestClient, token: str, upload_id: str):
    """Helper to cancel an upload session."""
    return client.post(
        f"/api/v1/cases/upload-session/{upload_id}/cancel",
        headers=auth_header(token),
    )


# ──────────────────────────────────────────────────────────────────
# 1–4: Owner (Alice) can access her own upload session
# ──────────────────────────────────────────────────────────────────


class TestOwnerAccess:
    """Tests 1–4: Owner can perform all operations on their own upload session."""

    def test_01_owner_can_get_status(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = get_status(client, alice_token, upload_id)
            assert resp.status_code == 200
            assert resp.json()["upload_id"] == upload_id

    def test_02_owner_can_upload_chunk(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = upload_chunk(client, alice_token, upload_id, chunk_index=0)
            assert resp.status_code == 200
            assert resp.json()["chunks_received"] == 1

    def test_03_owner_can_cancel(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = cancel_session(client, alice_token, upload_id)
            assert resp.status_code == 200
            assert resp.json()["status"] == "CANCELLED"

    def test_04_owner_can_commit_after_all_chunks(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]
            total_chunks = session_data["total_chunks"]

            # Upload all chunks
            for i in range(total_chunks):
                resp = upload_chunk(client, alice_token, upload_id, chunk_index=i)
                assert resp.status_code == 200

            resp = commit_session(client, alice_token, upload_id)
            assert resp.status_code == 202


# ──────────────────────────────────────────────────────────────────
# 5–8: Cross-user (Bob) cannot access Alice's upload session
# ──────────────────────────────────────────────────────────────────


class TestCrossUserRejection:
    """Tests 5–8: Another investigator cannot access a different investigator's upload session."""

    def test_05_cross_user_cannot_get_status(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = get_status(client, bob_token, upload_id)
            assert resp.status_code == 403
            assert "owned by another operator" in resp.json()["detail"]

    def test_06_cross_user_cannot_upload_chunk(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            # Capture state before attack
            status_before = get_status(client, alice_token, upload_id).json()

            # Bob attempts chunk upload
            resp = upload_chunk(client, bob_token, upload_id, chunk_index=0)
            assert resp.status_code == 403

            # Verify state unchanged after attack
            status_after = get_status(client, alice_token, upload_id).json()
            assert status_after["chunks_received"] == status_before["chunks_received"]
            assert status_after["bytes_received"] == status_before["bytes_received"]
            assert status_after["status"] == status_before["status"]

    def test_07_cross_user_cannot_commit(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]
            total_chunks = session_data["total_chunks"]

            # Alice uploads all chunks
            for i in range(total_chunks):
                upload_chunk(client, alice_token, upload_id, chunk_index=i)

            # Capture state before attack
            status_before = get_status(client, alice_token, upload_id).json()

            # Bob attempts commit
            resp = commit_session(client, bob_token, upload_id)
            assert resp.status_code == 403

            # Verify state unchanged after attack
            status_after = get_status(client, alice_token, upload_id).json()
            assert status_after["status"] == status_before["status"]
            assert status_after["status"] != "COMMITTED"

    def test_08_cross_user_cannot_cancel(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            # Upload a chunk so there's actual state
            upload_chunk(client, alice_token, upload_id, chunk_index=0)

            # Capture state before attack
            status_before = get_status(client, alice_token, upload_id).json()

            # Bob attempts cancel
            resp = cancel_session(client, bob_token, upload_id)
            assert resp.status_code == 403

            # Verify state unchanged after attack
            status_after = get_status(client, alice_token, upload_id).json()
            assert status_after["status"] == status_before["status"]
            assert status_after["status"] == "UPLOADING"
            assert status_after["chunks_received"] == status_before["chunks_received"]
            assert status_after["bytes_received"] == status_before["bytes_received"]


# ──────────────────────────────────────────────────────────────────
# 9: Invalid upload_id → 404
# ──────────────────────────────────────────────────────────────────


class TestInvalidUploadId:
    """Test 9: Non-existent upload_id returns 404 for all operations."""

    def test_09_invalid_upload_id_status(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            resp = get_status(client, alice_token, "nonexistent_upload_999")
            assert resp.status_code == 404

    def test_09_invalid_upload_id_chunk(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            resp = upload_chunk(client, alice_token, "nonexistent_upload_999", chunk_index=0)
            assert resp.status_code == 404

    def test_09_invalid_upload_id_commit(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            resp = commit_session(client, alice_token, "nonexistent_upload_999")
            assert resp.status_code == 404

    def test_09_invalid_upload_id_cancel(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            resp = cancel_session(client, alice_token, "nonexistent_upload_999")
            assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────
# 10: Unauthenticated access → 401
# ──────────────────────────────────────────────────────────────────


class TestUnauthenticatedAccess:
    """Test 10: Unauthenticated requests are rejected with 401."""

    def test_10_unauthenticated_status(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = client.get(f"/api/v1/cases/upload-session/{upload_id}/status")
            assert resp.status_code == 401

    def test_10_unauthenticated_chunk(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = client.post(
                f"/api/v1/cases/upload-session/{upload_id}/chunk",
                data={"chunk_index": "0"},
                files={"file": ("chunk.bin", io.BytesIO(b"X" * 1024), "application/octet-stream")},
            )
            assert resp.status_code == 401

    def test_10_unauthenticated_commit(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = client.post(f"/api/v1/cases/upload-session/{upload_id}/commit")
            assert resp.status_code == 401

    def test_10_unauthenticated_cancel(self):
        alice_token = create_token_for("inv_alice", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = client.post(f"/api/v1/cases/upload-session/{upload_id}/cancel")
            assert resp.status_code == 401


# ──────────────────────────────────────────────────────────────────
# 11–12: Admin and Root Admin oversight
# ──────────────────────────────────────────────────────────────────


class TestAdminOversight:
    """Tests 11–12: Admin and root_admin can access any investigator's upload session."""

    def test_11_admin_can_access_investigator_session_status(self):
        alice_token = create_token_for("inv_alice", "investigator")
        admin_token = create_token_for("regular_admin", "admin")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = get_status(client, admin_token, upload_id)
            assert resp.status_code == 200
            assert resp.json()["upload_id"] == upload_id

    def test_11_admin_can_cancel_investigator_session(self):
        alice_token = create_token_for("inv_alice", "investigator")
        admin_token = create_token_for("regular_admin", "admin")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = cancel_session(client, admin_token, upload_id)
            assert resp.status_code == 200
            assert resp.json()["status"] == "CANCELLED"

    def test_12_root_admin_can_access_investigator_session_status(self):
        alice_token = create_token_for("inv_alice", "investigator")
        root_token = create_token_for("root_boss", "root_admin")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = get_status(client, root_token, upload_id)
            assert resp.status_code == 200
            assert resp.json()["upload_id"] == upload_id

    def test_12_root_admin_can_upload_chunk_to_investigator_session(self):
        alice_token = create_token_for("inv_alice", "investigator")
        root_token = create_token_for("root_boss", "root_admin")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = upload_chunk(client, root_token, upload_id, chunk_index=0)
            assert resp.status_code == 200
            assert resp.json()["chunks_received"] == 1


# ──────────────────────────────────────────────────────────────────
# 13: Ownership persists across reconnect/new session for same user
# ──────────────────────────────────────────────────────────────────


class TestOwnershipPersistence:
    """Test 13: Ownership remains enforced after reconnect with new session token."""

    def test_13_ownership_persists_across_new_session_token(self):
        alice_token_1 = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token_1)
            upload_id = session_data["upload_id"]

            # Alice reconnects with new session token
            alice_token_2 = create_token_for("inv_alice", "investigator")

            # Alice can still access with new token
            resp = get_status(client, alice_token_2, upload_id)
            assert resp.status_code == 200

            # Bob still cannot access
            resp = get_status(client, bob_token, upload_id)
            assert resp.status_code == 403


# ──────────────────────────────────────────────────────────────────
# 14: Client spoofing of owner/user identity
# ──────────────────────────────────────────────────────────────────


class TestOwnerSpoofing:
    """Test 14: Client-provided owner identifiers do not bypass authorization."""

    def test_14_x_user_id_header_does_not_bypass_ownership(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            # Bob sends spoofed X-User-ID header claiming to be Alice
            headers = auth_header(bob_token)
            headers["X-User-ID"] = "inv_alice"
            resp = client.get(
                f"/api/v1/cases/upload-session/{upload_id}/status",
                headers=headers,
            )
            assert resp.status_code == 403

    def test_14_owner_query_param_does_not_bypass_ownership(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            resp = client.get(
                f"/api/v1/cases/upload-session/{upload_id}/status?owner=inv_alice",
                headers=auth_header(bob_token),
            )
            assert resp.status_code == 403


# ──────────────────────────────────────────────────────────────────
# 15–16: Rejected operations do not mutate state
# ──────────────────────────────────────────────────────────────────


class TestStateMutationProtection:
    """Tests 15–16: Verify that rejected cross-user operations do not alter upload session state."""

    def test_15_rejected_cancel_does_not_mutate_session(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            # Upload a chunk as Alice
            upload_chunk(client, alice_token, upload_id, chunk_index=0)

            # Snapshot state
            before = get_status(client, alice_token, upload_id).json()

            # Bob attempts cancel
            resp = cancel_session(client, bob_token, upload_id)
            assert resp.status_code == 403

            # Verify nothing changed
            after = get_status(client, alice_token, upload_id).json()
            assert after["status"] == before["status"]
            assert after["chunks_received"] == before["chunks_received"]
            assert after["bytes_received"] == before["bytes_received"]

    def test_16_rejected_chunk_upload_does_not_modify_stored_data(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]

            # Snapshot state (0 chunks)
            before = get_status(client, alice_token, upload_id).json()
            assert before["chunks_received"] == []
            assert before["bytes_received"] == 0

            # Bob attempts to upload a chunk with different data
            resp = upload_chunk(client, bob_token, upload_id, chunk_index=0, data=b"EVIL" * 256)
            assert resp.status_code == 403

            # Verify no chunks were written
            after = get_status(client, alice_token, upload_id).json()
            assert after["chunks_received"] == []
            assert after["bytes_received"] == 0
            assert after["status"] == "UPLOADING"

    def test_16_rejected_commit_does_not_change_committed_state(self):
        alice_token = create_token_for("inv_alice", "investigator")
        bob_token = create_token_for("inv_bob", "investigator")
        with TestClient(app) as client:
            session_data = create_upload_session(client, alice_token)
            upload_id = session_data["upload_id"]
            total_chunks = session_data["total_chunks"]

            # Alice uploads all chunks
            for i in range(total_chunks):
                upload_chunk(client, alice_token, upload_id, chunk_index=i)

            # Snapshot
            before = get_status(client, alice_token, upload_id).json()

            # Bob attempts commit
            resp = commit_session(client, bob_token, upload_id)
            assert resp.status_code == 403

            # Verify session is still UPLOADING, not COMMITTED
            after = get_status(client, alice_token, upload_id).json()
            assert after["status"] == before["status"]
            assert after["status"] != "COMMITTED"
