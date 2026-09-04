import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_operator_store, get_session_store
from security_layer.password_hasher import get_password_hasher
from tools.migration.bootstrap_admin import AdminBootstrapper, validate_password_complexity


def test_password_complexity_rules():
    # Less than 12 chars
    valid, _ = validate_password_complexity("Short1!")
    assert valid is False

    # Missing uppercase
    valid, _ = validate_password_complexity("lowercase12345!")
    assert valid is False

    # Missing lowercase
    valid, _ = validate_password_complexity("UPPERCASE12345!")
    assert valid is False

    # Missing number
    valid, _ = validate_password_complexity("NoNumbersSpecial!")
    assert valid is False

    # Missing special char
    valid, _ = validate_password_complexity("NoSpecialChar12345")
    assert valid is False

    # Valid enterprise password
    valid, _ = validate_password_complexity("StrongP@ssw0rd!2026")
    assert valid is True


def test_bootstrap_succeeds_when_no_admin_exists(tmp_path: Path):
    store_file = tmp_path / "operator_store.json"
    bootstrapper = AdminBootstrapper(offline_store_path=str(store_file))

    assert bootstrapper.admin_exists() is False

    success = bootstrapper.bootstrap(
        username="admin_root",
        password="ValidRootPassword@2026",
        name="Chief Admin",
    )
    assert success is True
    assert bootstrapper.admin_exists() is True

    # Inspect stored record
    data = json.loads(store_file.read_text(encoding="utf-8"))
    admin_record = data["admins"]["admin_root"]

    assert "password" not in admin_record
    assert admin_record["password_hash"].startswith("$argon2id")
    assert admin_record["role"] == "root_admin"


def test_bootstrap_aborts_safely_when_admin_exists(tmp_path: Path):
    store_file = tmp_path / "operator_store.json"
    initial_data = {
        "admins": {
            "existing_admin": {
                "username": "existing_admin",
                "password_hash": "$argon2id$v=19$mockhash",
                "role": "admin",
            }
        }
    }
    store_file.write_text(json.dumps(initial_data), encoding="utf-8")

    bootstrapper = AdminBootstrapper(offline_store_path=str(store_file))
    assert bootstrapper.admin_exists() is True

    # Attempting to bootstrap must safely fail
    success = bootstrapper.bootstrap(
        username="admin_intruder",
        password="ValidRootPassword@2026",
    )
    assert success is False

    # Ensure existing admin was untouched
    data = json.loads(store_file.read_text(encoding="utf-8"))
    assert "admin_intruder" not in data["admins"]
    assert "existing_admin" in data["admins"]


def test_bootstrap_rejects_weak_password(tmp_path: Path):
    store_file = tmp_path / "operator_store.json"
    bootstrapper = AdminBootstrapper(offline_store_path=str(store_file))

    success = bootstrapper.bootstrap(
        username="admin_root",
        password="weak",
    )
    assert success is False
    assert bootstrapper.admin_exists() is False


def test_bootstrap_never_leaks_password(tmp_path: Path, caplog):
    store_file = tmp_path / "operator_store.json"
    bootstrapper = AdminBootstrapper(offline_store_path=str(store_file))
    secret_pw = "UltraSecretAdminPass#2026!"

    with caplog.at_level(logging.DEBUG):
        bootstrapper.bootstrap(
            username="admin_root",
            password=secret_pw,
        )

    all_logs = caplog.text
    assert secret_pw not in all_logs


def test_admin_provisioning_endpoints_and_rbac():
    get_session_store().clear()
    op_store = get_operator_store()
    hasher = get_password_hasher()

    # Seed an admin and an investigator
    admin_pw = "RootPass#123456!"
    inv_pw = "InvestigatorPass#123!"

    offline_data = op_store._load_offline_store()
    offline_data["admins"] = {
        "admin_sec": {
            "name": "Sec Admin",
            "username": "admin_sec",
            "password_hash": hasher.hash(admin_pw),
            "role": "root_admin",
            "status": "Active",
        }
    }
    offline_data["investigators"] = {
        "inv_sec": {
            "name": "Sec Investigator",
            "username": "inv_sec",
            "password_hash": hasher.hash(inv_pw),
            "role": "investigator",
            "status": "Active",
        }
    }
    op_store._save_offline_store(offline_data)

    with TestClient(app) as client:
        # 1. Login as investigator
        inv_login = client.post(
            "/api/v1/auth/login",
            json={"username": "inv_sec", "password": inv_pw},
        )
        inv_token = inv_login.json()["token"]

        # 2. Login as admin
        admin_login = client.post(
            "/api/v1/auth/login",
            json={"username": "admin_sec", "password": admin_pw},
        )
        admin_token = admin_login.json()["token"]

        # 3. Investigator tries to create operator (Role escalation / RBAC violation) -> 403 Forbidden
        escalation_resp = client.post(
            "/api/v1/auth/admin/users",
            headers={"Authorization": f"Bearer {inv_token}"},
            json={
                "username": "rogue_admin",
                "password": "RoguePassword123!",
                "role": "admin",
            },
        )
        assert escalation_resp.status_code == 403

        # 4. Admin creates operator -> 201 Created
        create_resp = client.post(
            "/api/v1/auth/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "username": "inv_new",
                "password": "NewOperatorPass!123",
                "role": "investigator",
                "name": "New Agent",
                "nic": "123456789V",
            },
        )
        assert create_resp.status_code == 201

        # Check new operator has Argon2id hash and NO plaintext password
        new_record, _, _ = op_store.get_operator("inv_new")
        assert new_record is not None
        assert "password" not in new_record
        assert new_record["password_hash"].startswith("$argon2id")

        # 5. Admin updates operator profile
        update_resp = client.put(
            "/api/v1/auth/admin/users/inv_new",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "Updated Agent Name", "status": "Active"},
        )
        assert update_resp.status_code == 200

        # 6. Admin deletes operator
        del_resp = client.delete(
            "/api/v1/auth/admin/users/inv_new",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert del_resp.status_code == 200

        deleted_record, _, _ = op_store.get_operator("inv_new")
        assert deleted_record is None
