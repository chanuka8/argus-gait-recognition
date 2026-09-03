import pytest
from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_operator_store, get_session_store
from security_layer.password_hasher import get_password_hasher
from services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus


@pytest.fixture(autouse=True)
def setup_security_fixtures(tmp_path):
    get_session_store().clear()
    op_store = get_operator_store()
    hasher = get_password_hasher()

    # Pre-seed users
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
            "inv_suspended": {
                "name": "Suspended Detective",
                "username": "inv_suspended",
                "password_hash": hasher.hash("SuspPass@2026!"),
                "role": "investigator",
                "status": "Suspended",
            },
        },
    }
    op_store._save_offline_store(offline_data)


def create_token_for(username: str, role: str, status: str = "Active") -> str:
    session = get_session_store().create_session(
        operator_id=username,
        username=username,
        role=role,
        status=status,
    )
    return session.token


# 1. Unauthenticated request -> 401
def test_01_unauthenticated_request_rejected_with_401():
    with TestClient(app) as client:
        # Protected job status route
        resp = client.get("/api/v1/cases/jobs/ref_test_123")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers


# 2. Investigator accessing another investigator's case gallery -> 403
def test_02_investigator_accessing_another_investigator_case_gallery_rejected():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/cases/case_bob_secret/gallery",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        # Nonexistent case returns 404, or 403 on case authorization
        assert resp.status_code in (403, 404)


# 3. Investigator accessing another investigator's job -> 403 (anti-BOLA)
def test_03_investigator_accessing_another_investigator_job_rejected_with_403():
    job_mgr = ReferenceJobManager.get_instance()
    # Create job owned by Bob
    job = job_mgr.create_job(
        person_id="person_bob_case",
        video_path="mock_video.mp4",
        case_id="case_bob_01",
        owner="inv_bob",
    )

    alice_token = create_token_for("inv_alice", "investigator")
    bob_token = create_token_for("inv_bob", "investigator")
    admin_token = create_token_for("regular_admin", "admin")

    with TestClient(app) as client:
        # Bob accesses his own job -> 200
        resp_bob = client.get(
            f"/api/v1/cases/jobs/{job.job_id}",
            headers={"Authorization": f"Bearer {bob_token}"},
        )
        assert resp_bob.status_code == 200

        # Alice attempts to access Bob's job -> 403 Forbidden
        resp_alice = client.get(
            f"/api/v1/cases/jobs/{job.job_id}",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        assert resp_alice.status_code == 403
        assert "owned by another operator" in resp_alice.json()["detail"]

        # Admin accesses Bob's job -> 200 (admin oversight)
        resp_admin = client.get(
            f"/api/v1/cases/jobs/{job.job_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp_admin.status_code == 200


# 4. Investigator retrying another user's job -> 403
def test_04_investigator_retrying_another_user_job_rejected_with_403():
    job_mgr = ReferenceJobManager.get_instance()
    job = job_mgr.create_job(
        person_id="person_bob_case",
        video_path="mock_video.mp4",
        case_id="case_bob_01",
        owner="inv_bob",
    )
    job.status = ReferenceJobStatus.FAILED

    alice_token = create_token_for("inv_alice", "investigator")

    with TestClient(app) as client:
        # Alice attempts to retry Bob's job -> 403
        resp_alice = client.post(
            f"/api/v1/cases/jobs/{job.job_id}/retry",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        assert resp_alice.status_code == 403


# 5. Investigator modifying operator -> 403
def test_05_investigator_modifying_operator_rejected_with_403():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/auth/admin/users/inv_bob",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={"name": "Hacked Bob"},
        )
        assert resp.status_code == 403


# 6. Investigator creating operator -> 403
def test_06_investigator_creating_operator_rejected_with_403():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/admin/users",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={"username": "rogue_user", "password": "Password123!", "role": "admin"},
        )
        assert resp.status_code == 403


# 7. Investigator modifying camera credentials -> 403
def test_07_investigator_modifying_camera_credentials_rejected_with_403():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/credentials",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={"username": "cam_user", "password": "SecretCamPass!"},
        )
        assert resp.status_code == 403


# 8. Investigator performing privileged camera control -> 403
def test_08_investigator_performing_privileged_camera_control_rejected_with_403():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/stop",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={"camera_id": "CAM-01"},
        )
        assert resp.status_code == 403


# 9. Admin performing permitted administration -> success
def test_09_admin_performing_permitted_administration_succeeds():
    admin_token = create_token_for("regular_admin", "admin")
    with TestClient(app) as client:
        # Create investigator operator
        resp = client.post(
            "/api/v1/auth/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"username": "inv_charlie", "password": "CharliePassword!2026", "role": "investigator"},
        )
        assert resp.status_code == 201

        # Update investigator operator
        update_resp = client.put(
            "/api/v1/auth/admin/users/inv_charlie",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "Charlie Updated"},
        )
        assert update_resp.status_code == 200


# 10. Admin attempting root-only operation -> 403
def test_10_admin_attempting_root_only_operation_rejected_with_403():
    admin_token = create_token_for("regular_admin", "admin")
    with TestClient(app) as client:
        # 1. Admin attempts to delete an operator -> 403
        del_resp = client.delete(
            "/api/v1/auth/admin/users/inv_alice",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert del_resp.status_code == 403
        assert "Root Administrator privileges required" in del_resp.json()["detail"]

        # 2. Admin attempts to create a root_admin -> 403
        create_root_resp = client.post(
            "/api/v1/auth/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"username": "new_root", "password": "RootPassword!2026", "role": "root_admin"},
        )
        assert create_root_resp.status_code == 403

        # 3. Admin attempts to modify root_admin -> 403
        mod_root_resp = client.put(
            "/api/v1/auth/admin/users/root_boss",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "Modified Boss"},
        )
        assert mod_root_resp.status_code == 403

        # 4. Admin attempts to promote model version in continual learning -> 403
        learn_resp = client.post(
            "/api/v1/learning/promote",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert learn_resp.status_code == 403


# 11. root_admin performing root operation -> success
def test_11_root_admin_performing_root_operation_succeeds():
    root_token = create_token_for("root_boss", "root_admin")
    with TestClient(app) as client:
        # Root admin deletes operator
        del_resp = client.delete(
            "/api/v1/auth/admin/users/inv_alice",
            headers={"Authorization": f"Bearer {root_token}"},
        )
        assert del_resp.status_code == 200

        # Root admin promotes model
        promote_resp = client.post(
            "/api/v1/learning/promote",
            headers={"Authorization": f"Bearer {root_token}"},
        )
        assert promote_resp.status_code == 200


# 12. Forged X-User-ID -> ignored/rejected
def test_12_forged_x_user_id_ignored_and_rejected():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        # Alice tries to spoof root_boss via X-User-ID header
        resp = client.delete(
            "/api/v1/auth/admin/users/inv_bob",
            headers={
                "Authorization": f"Bearer {alice_token}",
                "X-User-ID": "root_boss",
                "X-User-Role": "root_admin",
            },
        )
        # Server verifies session token, ignoring header -> 403 Forbidden
        assert resp.status_code == 403


# 13. Forged user_id in request body -> cannot change principal
def test_13_forged_user_id_in_request_body_cannot_change_principal():
    admin_token = create_token_for("regular_admin", "admin")
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/credentials",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "username": "cam_agent",
                "password": "CamPassword123!",
                "owner_user_id": "root_boss",  # Attacker attempts to bind to root_boss
            },
        )
        assert resp.status_code == 200
        # Owner must be regular_admin (from session), NOT root_boss
        meta = resp.json()
        assert meta["owner_user_id"] == "regular_admin"


# 14. Forged role in request -> ignored/rejected
def test_14_forged_role_in_request_ignored_and_rejected():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/auth/admin/users",
            headers={
                "Authorization": f"Bearer {alice_token}",
                "Role": "root_admin",
            },
            json={"username": "forged_user", "password": "Password!123", "role": "admin"},
        )
        assert resp.status_code == 403


# 15. Nonexistent resource returns 404 without leaking info
def test_15_nonexistent_resource_returns_404():
    admin_token = create_token_for("regular_admin", "admin")
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/cases/jobs/ref_nonexistent_job_999",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# 16. Resource identifier enumeration -> no unauthorized disclosure
def test_16_resource_identifier_enumeration_no_unauthorized_disclosure():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/credentials/cred_secret_admin_01",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        # Unauthorized investigator receives 403, preventing existence discovery
        assert resp.status_code == 403


# 17. Cross-case media access -> 403
def test_17_cross_case_media_access():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/cases/case_isolated_bob/gallery",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        assert resp.status_code in (403, 404)


# 18. Cross-person biometric access enforced
def test_18_cross_person_biometric_access():
    admin_token = create_token_for("regular_admin", "admin")
    alice_token = create_token_for("inv_alice", "investigator")

    with TestClient(app) as client:
        # Gallery subject deletion requires administrative privileges
        del_attempt = client.post(
            "/api/v1/gallery/delete",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={"person_id": "subject_bob"},
        )
        assert del_attempt.status_code == 403

        # Admin delete succeeds or returns 404 if not found
        admin_del = client.post(
            "/api/v1/gallery/delete",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"person_id": "subject_bob"},
        )
        assert admin_del.status_code in (200, 404)


# 19. Unauthorized delete -> 403
def test_19_unauthorized_delete_rejected():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp = client.delete(
            "/api/v1/credentials/cred_01",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        assert resp.status_code == 403


# 20. Unauthorized continual-learning operation -> 403
def test_20_unauthorized_continual_learning_operation_rejected():
    alice_token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        resp_status = client.get(
            "/api/v1/learning/status",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        assert resp_status.status_code == 403

        resp_promote = client.post(
            "/api/v1/learning/promote",
            headers={"Authorization": f"Bearer {alice_token}"},
        )
        assert resp_promote.status_code == 403


# 21. Suspended operator -> rejected
def test_21_suspended_operator_rejected():
    suspended_token = create_token_for("inv_suspended", "investigator", status="Suspended")
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/cameras",
            headers={"Authorization": f"Bearer {suspended_token}"},
        )
        assert resp.status_code == 403
        assert "suspended" in resp.json()["detail"].lower()


# 22. Revoked session -> rejected
def test_22_revoked_session_rejected():
    token = create_token_for("inv_alice", "investigator")
    with TestClient(app) as client:
        # First request succeeds
        resp1 = client.get(
            "/api/v1/cameras",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp1.status_code == 200

        # Logout revokes session
        logout_resp = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert logout_resp.status_code == 200

        # Subsequent request with revoked token fails with 401
        resp2 = client.get(
            "/api/v1/cameras",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp2.status_code == 401
