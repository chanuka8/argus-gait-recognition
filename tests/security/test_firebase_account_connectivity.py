import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.server import app
from scripts.migrate_passwords import PasswordMigrator
from security_layer.auth import (
    AuthenticationInfrastructureError,
    get_operator_store,
    get_session_store,
)
from security_layer.password_hasher import get_password_hasher


def test_01_firebase_credentials_missing_fails_closed_503(monkeypatch):
    """In firebase mode, missing credentials must fail closed with HTTP 503 (zero fallback to local JSON)."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    store = get_operator_store()
    store.reset_client()
    monkeypatch.setattr(store, "_get_firestore_client", lambda: None)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "AnyPassword123!", "role": "admin"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "temporarily unavailable" in data["detail"].lower()
        # Verify no credentials or paths leaked
        assert "service_account" not in data["detail"].lower()
        assert "argus-17702" not in data["detail"]


def test_02_firebase_initialization_failure_returns_503(monkeypatch):
    """If Firebase Admin SDK initialization throws an exception, login returns HTTP 503."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")

    store = get_operator_store()
    store.reset_client()

    def mock_broken_client():
        raise AuthenticationInfrastructureError("Simulated Firebase connection failure")

    monkeypatch.setattr(store, "_get_firestore_client", mock_broken_client)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "AnyPassword123!", "role": "admin"},
        )
        assert resp.status_code == 503
        assert "temporarily unavailable" in resp.json()["detail"].lower()


def test_03_firestore_query_exception_returns_503(monkeypatch):
    """If Firestore query throws an error during lookup, it fails closed with HTTP 503."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")

    mock_client = MagicMock()
    mock_client.collection.side_effect = RuntimeError("Firestore network timeout")

    store = get_operator_store()
    store.reset_client()
    monkeypatch.setattr(store, "_get_firestore_client", lambda: mock_client)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "AnyPassword123!", "role": "admin"},
        )
        assert resp.status_code == 503
        assert "network timeout" not in resp.json()["detail"].lower()


def test_04_no_fallback_to_local_json_in_firebase_mode(tmp_path: Path, monkeypatch):
    """In firebase mode, even if a user exists in the local JSON store, the backend MUST NOT authenticate them."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")

    hasher = get_password_hasher()
    pw = "OfflineSecret123!"
    offline_file = tmp_path / "sneaky_offline.json"
    offline_data = {
        "admins": {
            "sneaky_admin": {
                "username": "sneaky_admin",
                "password_hash": hasher.hash(pw),
                "role": "admin",
                "status": "Active",
            }
        }
    }
    offline_file.write_text(json.dumps(offline_data), encoding="utf-8")

    store = get_operator_store()
    store.reset_client()
    store.offline_store_path = offline_file

    # Mock uninitialized firebase
    monkeypatch.setattr(store, "_get_firestore_client", lambda: None)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "sneaky_admin", "password": pw, "role": "admin"},
        )
        # MUST fail closed with 503, NOT succeed via offline JSON
        assert resp.status_code == 503


def test_05_explicit_offline_mode_works(tmp_path: Path, monkeypatch):
    """When ARGUS_OPERATOR_STORE_MODE=offline is explicitly set, local JSON store is utilized."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "offline")

    hasher = get_password_hasher()
    admin_pw = "OfflineAdminPass2026!"
    store_file = tmp_path / "test_store.json"
    data = {
        "admins": {
            "dev_admin": {
                "username": "dev_admin",
                "password_hash": hasher.hash(admin_pw),
                "role": "admin",
                "status": "Active",
            }
        }
    }
    store_file.write_text(json.dumps(data), encoding="utf-8")

    store = get_operator_store()
    store.offline_store_path = store_file

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "dev_admin", "password": admin_pw, "role": "admin"},
        )
        assert resp.status_code == 200
        auth_data = resp.json()
        assert auth_data["operator"]["username"] == "dev_admin"
        assert auth_data["operator"]["role"] == "admin"


def test_06_mock_firebase_valid_admin_login(monkeypatch):
    """Simulate authoritative Firebase Admin SDK with valid admin account (admin -> Root Admin)."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")
    hasher = get_password_hasher()
    admin_pw = "RealAdminPassword@2026"

    mock_doc = MagicMock()
    mock_doc.id = "admin_root"
    mock_doc.to_dict.return_value = {
        "username": "admin",
        "name": "Root Administrator",
        "role": "Root Admin",
        "status": "Active",
        "password_hash": hasher.hash(admin_pw),
        "password_migrated": True,
    }

    mock_query = MagicMock()
    mock_query.stream.return_value = [mock_doc]

    mock_col = MagicMock()
    mock_col.where.return_value.limit.return_value = mock_query

    mock_client = MagicMock()
    mock_client.collection.return_value = mock_col

    store = get_operator_store()
    store.reset_client()
    monkeypatch.setattr(store, "_get_firestore_client", lambda: mock_client)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": admin_pw, "role": "admin"},
        )
        assert resp.status_code == 200
        res = resp.json()
        assert res["operator"]["username"] == "admin"
        assert res["operator"]["role"] == "root_admin"  # Normalized from "Root Admin"
        token = res["token"]

        # Verify /me endpoint
        me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        assert me_resp.json()["role"] == "root_admin"


def test_07_mock_firebase_valid_investigator_login(monkeypatch):
    """Simulate authoritative Firebase Admin SDK with investigator account (john.inv)."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")
    hasher = get_password_hasher()
    inv_pw = "JohnInvestigator@2026"

    mock_doc = MagicMock()
    mock_doc.id = "john.inv"
    mock_doc.to_dict.return_value = {
        "username": "john.inv",
        "name": "John Detective",
        "role": "investigator",
        "status": "Active",
        "password_hash": hasher.hash(inv_pw),
        "password_migrated": True,
    }

    mock_query = MagicMock()
    mock_query.stream.return_value = [mock_doc]

    mock_col = MagicMock()
    mock_col.where.return_value.limit.return_value = mock_query

    mock_client = MagicMock()
    mock_client.collection.return_value = mock_col

    store = get_operator_store()
    store.reset_client()
    monkeypatch.setattr(store, "_get_firestore_client", lambda: mock_client)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "john.inv", "password": inv_pw, "role": "investigator"},
        )
        assert resp.status_code == 200
        res = resp.json()
        assert res["operator"]["username"] == "john.inv"
        assert res["operator"]["role"] == "investigator"
        token = res["token"]

        # Investigator cannot access camera controls (admin required -> 403)
        cam_resp = client.post(
            "/api/v1/cameras/start",
            json={"camera_id": "cam_1"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert cam_resp.status_code == 403


def test_08_mock_firebase_wrong_password_returns_401(monkeypatch):
    """Wrong password in Firebase mode returns HTTP 401."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")
    hasher = get_password_hasher()

    mock_doc = MagicMock()
    mock_doc.id = "admin_root"
    mock_doc.to_dict.return_value = {
        "username": "admin",
        "role": "Root Admin",
        "status": "Active",
        "password_hash": hasher.hash("CorrectPassword@123"),
        "password_migrated": True,
    }

    mock_query = MagicMock()
    mock_query.stream.return_value = [mock_doc]

    mock_col = MagicMock()
    mock_col.where.return_value.limit.return_value = mock_query

    mock_client = MagicMock()
    mock_client.collection.return_value = mock_col

    store = get_operator_store()
    store.reset_client()
    monkeypatch.setattr(store, "_get_firestore_client", lambda: mock_client)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "WrongPassword@999", "role": "admin"},
        )
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()


def test_09_mock_firebase_unknown_user_returns_401(monkeypatch):
    """Unknown user in Firebase mode returns HTTP 401."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")

    mock_query = MagicMock()
    mock_query.stream.return_value = []

    mock_col = MagicMock()
    mock_col.where.return_value.limit.return_value = mock_query

    mock_client = MagicMock()
    mock_client.collection.return_value = mock_col

    store = get_operator_store()
    store.reset_client()
    monkeypatch.setattr(store, "_get_firestore_client", lambda: mock_client)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "ghost_user", "password": "AnyPassword123!", "role": "admin"},
        )
        assert resp.status_code == 401
        detail_low = resp.json()["detail"].lower()
        assert "not found" in detail_low or "invalid" in detail_low


def test_10_mock_firebase_suspended_account_rejected(monkeypatch):
    """Suspended account in Firebase mode is rejected."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "firebase")
    hasher = get_password_hasher()

    mock_doc = MagicMock()
    mock_doc.id = "suspended_user"
    mock_doc.to_dict.return_value = {
        "username": "suspended_user",
        "role": "investigator",
        "status": "Suspended",
        "password_hash": hasher.hash("SomePassword@123"),
        "password_migrated": True,
    }

    mock_query = MagicMock()
    mock_query.stream.return_value = [mock_doc]

    mock_col = MagicMock()
    mock_col.where.return_value.limit.return_value = mock_query

    mock_client = MagicMock()
    mock_client.collection.return_value = mock_col

    store = get_operator_store()
    store.reset_client()
    monkeypatch.setattr(store, "_get_firestore_client", lambda: mock_client)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/login",
            json={"username": "suspended_user", "password": "SomePassword@123", "role": "investigator"},
        )
        assert resp.status_code == 401
        assert "suspended" in resp.json()["detail"].lower()


def test_11_test_execution_cannot_mutate_production_operator_store():
    """Verify that test suite execution does NOT touch or mutate data/operator_store.json."""
    real_store_path = Path("data/operator_store.json")
    if real_store_path.exists():
        initial_hash = hashlib.sha256(real_store_path.read_bytes()).hexdigest()

        # Perform store actions through get_operator_store() (which should be isolated by conftest)
        store = get_operator_store()
        store._save_offline_store({"admins": {"temp": {"username": "temp"}}})

        after_hash = hashlib.sha256(real_store_path.read_bytes()).hexdigest()
        assert initial_hash == after_hash, "Regression: Test mutated data/operator_store.json!"


def test_12_password_migration_dry_run_is_non_destructive(tmp_path: Path):
    """Password migration dry-run must report metadata without altering store data."""
    store_file = tmp_path / "mig_test.json"
    initial_data = {
        "admins": {
            "admin_legacy": {
                "username": "admin",
                "password": "LegacyPassword@123",
                "role": "Root Admin",
                "status": "Active",
            }
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))
    stats = migrator.migrate(dry_run=True)

    assert stats["mode"] == "DRY_RUN"
    assert stats["needs_migration"] == 1
    assert stats["migrated_success"] == 0

    after_data = json.loads(store_file.read_text(encoding="utf-8"))
    assert after_data["admins"]["admin_legacy"]["password"] == "LegacyPassword@123"
    assert "password_hash" not in after_data["admins"]["admin_legacy"]


def test_13_password_migration_apply_verifies_persistence_before_deletion(tmp_path: Path, monkeypatch):
    """Password migration apply persists Argon2id hash and verifies persistence before removing plaintext."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "offline")
    store_file = tmp_path / "mig_test.json"
    initial_data = {
        "investigators": {
            "inv_legacy": {
                "username": "john.inv",
                "password": "LegacyPlaintext@456",
                "role": "investigator",
                "status": "Active",
            }
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))
    stats = migrator.migrate(dry_run=False)

    assert stats["mode"] == "APPLY"
    assert stats["migrated_success"] == 1
    assert stats["migration_failed"] == 0

    after_data = json.loads(store_file.read_text(encoding="utf-8"))
    migrated_record = after_data["investigators"]["inv_legacy"]
    assert "password" not in migrated_record  # Plaintext removed
    assert "password_hash" in migrated_record
    assert migrated_record["password_hash"].startswith("$argon2id")
    assert migrated_record["password_migrated"] is True


def test_14_migration_failure_retains_legacy_credential(tmp_path: Path, monkeypatch):
    """If persistence fails during migration, the legacy password is preserved."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "offline")
    store_file = tmp_path / "mig_fail.json"
    initial_data = {
        "admins": {
            "adm": {
                "username": "admin",
                "password": "ImportantPassword@999",
                "role": "admin",
            }
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))

    # Mock save failure
    monkeypatch.setattr(migrator, "_save_offline_store", lambda data: False)

    stats = migrator.migrate(dry_run=False)
    assert stats["migration_failed"] >= 1

    # Ensure store_file on disk still has the original plaintext
    disk_data = json.loads(store_file.read_text(encoding="utf-8"))
    assert disk_data["admins"]["adm"]["password"] == "ImportantPassword@999"


def test_15_forged_x_user_id_and_client_role_rejected(tmp_path: Path, monkeypatch):
    """Spoofed X-User-ID or request body role cannot override authenticated session."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "offline")
    session_store = get_session_store()
    session_store.clear()

    inv_session = session_store.create_session(
        operator_id="inv_user",
        username="inv_user",
        role="investigator",
    )

    with TestClient(app) as client:
        # Attempt to impersonate root_admin via header and query parameter
        resp = client.get(
            "/api/v1/auth/me",
            headers={
                "Authorization": f"Bearer {inv_session.token}",
                "X-User-ID": "root_boss",
                "X-Operator-Role": "root_admin",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Identity remains investigator from verified server session
        assert data["username"] == "inv_user"
        assert data["role"] == "investigator"
