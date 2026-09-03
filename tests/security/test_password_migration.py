import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

from api.server import app
from scripts.migrate_passwords import PasswordMigrator
from security_layer.auth import get_operator_store, get_session_store
from security_layer.password_hasher import get_password_hasher


def test_cli_dry_run_identifies_legacy_and_migrated(tmp_path: Path):
    store_file = tmp_path / "operator_store.json"
    hasher = get_password_hasher()
    argon_hash = hasher.hash("AlreadyHashed@2026")

    initial_data = {
        "admins": {
            "admin_legacy": {
                "username": "admin_legacy",
                "password": "PlaintextAdminPass!123",
                "role": "admin",
            },
            "admin_modern": {
                "username": "admin_modern",
                "password_hash": argon_hash,
                "password_migrated": True,
                "role": "admin",
            },
        },
        "investigators": {
            "inv_malformed": {
                "username": "inv_malformed",
                "role": "investigator",
            }
        },
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))
    stats = migrator.migrate(dry_run=True)

    assert stats["mode"] == "DRY_RUN"
    assert stats["total_scanned"] == 3
    assert stats["already_migrated"] == 1
    assert stats["needs_migration"] == 1
    assert stats["malformed_records"] == 1
    assert stats["migrated_success"] == 0

    # Ensure store_file was NOT modified in dry-run
    after_data = json.loads(store_file.read_text(encoding="utf-8"))
    assert "password" in after_data["admins"]["admin_legacy"]
    assert "password_hash" not in after_data["admins"]["admin_legacy"]


def test_cli_apply_migrates_plaintext_to_argon2id(tmp_path: Path):
    store_file = tmp_path / "operator_store.json"
    plaintext = "SecureLegacyPass999!"

    initial_data = {
        "investigators": {
            "inv_test": {
                "username": "inv_test",
                "password": plaintext,
                "role": "investigator",
            }
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))
    stats = migrator.migrate(dry_run=False)

    assert stats["mode"] == "APPLY"
    assert stats["total_scanned"] == 1
    assert stats["migrated_success"] == 1
    assert stats["migration_failed"] == 0

    after_data = json.loads(store_file.read_text(encoding="utf-8"))
    migrated_record = after_data["investigators"]["inv_test"]

    # Plaintext MUST be removed
    assert "password" not in migrated_record
    assert "password_hash" in migrated_record
    assert migrated_record["password_hash"].startswith("$argon2id")
    assert migrated_record["password_migrated"] is True

    # Verify password verifies against new hash
    hasher = get_password_hasher()
    is_valid, needs_rehash = hasher.verify(plaintext, migrated_record["password_hash"])
    assert is_valid is True
    assert needs_rehash is False


def test_cli_migration_idempotency(tmp_path: Path):
    store_file = tmp_path / "operator_store.json"
    initial_data = {
        "investigators": {
            "inv_idempotent": {
                "username": "inv_idempotent",
                "password": "PasswordToMigrate1!",
                "role": "investigator",
            }
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))

    # First migration run: converts plaintext
    stats1 = migrator.migrate(dry_run=False)
    assert stats1["migrated_success"] == 1

    # Second migration run: skips already-migrated
    stats2 = migrator.migrate(dry_run=False)
    assert stats2["total_scanned"] == 1
    assert stats2["already_migrated"] == 1
    assert stats2["needs_migration"] == 0
    assert stats2["migrated_success"] == 0


def test_cli_handles_malformed_record_without_crashing(tmp_path: Path):
    store_file = tmp_path / "operator_store.json"
    initial_data = {
        "investigators": {
            "empty_record": {},
            "empty_password": {"username": "user1", "password": ""},
            "whitespace_password": {"username": "user2", "password": "   "},
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))
    stats = migrator.migrate(dry_run=False)

    assert stats["total_scanned"] == 3
    assert stats["malformed_records"] == 3
    assert stats["migrated_success"] == 0
    assert stats["migration_failed"] == 0


def test_cli_failed_persistence_preserves_legacy_password(tmp_path: Path, monkeypatch):
    store_file = tmp_path / "operator_store.json"
    initial_data = {
        "admins": {
            "admin_fail_safe": {
                "username": "admin_fail_safe",
                "password": "ImportantAdminPassword!",
                "role": "admin",
            }
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))

    # Simulate write/disk failure
    def mock_save_failure(self, data):
        return False

    monkeypatch.setattr(PasswordMigrator, "_save_offline_store", mock_save_failure)

    stats = migrator.migrate(dry_run=False)
    assert stats["migration_failed"] == 1
    assert stats["migrated_success"] == 0

    # Ensure disk data was not corrupted
    persisted = json.loads(store_file.read_text(encoding="utf-8"))
    assert persisted["admins"]["admin_fail_safe"]["password"] == "ImportantAdminPassword!"


def test_cli_never_logs_or_prints_passwords(tmp_path: Path, caplog):
    store_file = tmp_path / "operator_store.json"
    secret_pass = "TopSecretStringNeverLogMe12345!"

    initial_data = {
        "investigators": {
            "secret_agent": {
                "username": "secret_agent",
                "password": secret_pass,
                "role": "investigator",
            }
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    migrator = PasswordMigrator(offline_store_path=str(store_file))
    with caplog.at_level(logging.DEBUG):
        migrator.migrate(dry_run=False)

    all_logs = caplog.text
    assert secret_pass not in all_logs


def test_rehash_on_login_and_verify_password_endpoint(tmp_path: Path):
    get_session_store().clear()
    op_store = get_operator_store()
    username = "inv_rehash_test"
    plaintext_pw = "RehashOnLoginSecret@2026"

    # Seed legacy plaintext directly in offline store
    offline_data = op_store._load_offline_store()
    offline_data.setdefault("investigators", {})[username] = {
        "name": "Rehash Investigator",
        "username": username,
        "password": plaintext_pw,
        "role": "investigator",
        "status": "Active",
    }
    op_store._save_offline_store(offline_data)

    with TestClient(app) as client:
        # 1. Login should verify legacy plaintext, compute Argon2id, and update
        login_resp = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": plaintext_pw},
        )
        assert login_resp.status_code == 200
        token = login_resp.json()["token"]

        # 2. Check record was migrated
        record, _, _ = op_store.get_operator(username)
        assert record is not None
        assert "password" not in record
        assert record["password_hash"].startswith("$argon2id")

        # 3. Test verify-password endpoint with correct password
        verify_resp = client.post(
            "/api/v1/auth/verify-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"password": plaintext_pw},
        )
        assert verify_resp.status_code == 200
        assert verify_resp.json()["valid"] is True

        # 4. Test verify-password endpoint with wrong password
        verify_wrong = client.post(
            "/api/v1/auth/verify-password",
            headers={"Authorization": f"Bearer {token}"},
            json={"password": "IncorrectPassword!"},
        )
        assert verify_wrong.status_code == 200
        assert verify_wrong.json()["valid"] is False
