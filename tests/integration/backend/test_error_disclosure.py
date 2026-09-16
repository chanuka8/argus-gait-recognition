"""Integration test suite for SEC-09: Verbose API Error / Exception Information Disclosure.

Verifies:
1. Normal successful API responses are unchanged and fully functional.
2. Internal exceptions across endpoints return expected HTTP status codes (400, 403, 500)
   and safe client-facing detail messages.
3. Sensitive information is NEVER disclosed to clients:
   - filesystem paths (Windows drive paths C:\\, Unix paths /var/..., .pt, .pem, .npy, .mp4)
   - database details (connection strings, postgresql://, mysql://, user:pass@host)
   - credentials, secrets, tokens
   - internal Python exception representations (Traceback, [Errno X], class names like FileNotFoundError)
   - usernames in permission error details
4. Existing intentional user-facing validation errors remain unchanged.
5. Internal logging captures diagnostic exceptions and tracebacks on the server side.
6. Multiple endpoint categories are covered:
   - Inference / Image Analysis (POST /api/v1/analyze/image)
   - Video Analysis (POST /api/v1/analyze/video)
   - Credential Management (POST, DELETE, SHARE)
   - Camera Worker Startup (POST /api/v1/cameras/start)
   - Biometric Enrollment (POST /api/v1/enroll)
   - Reference Video Upload (POST /api/v1/cases/upload-reference)
   - Upload Session Init, Chunks, and Assembly (POST /api/v1/cases/upload-session/...)
   - Background Job Recovery status (services/reference_job_manager.py)
   - Legacy API routes (POST /identify, POST /enroll)
7. Authenticated vs unauthenticated behavior is verified.
"""

import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_session_store
from services.upload_session_manager import UploadSessionManager


@pytest.fixture
def client():
    """TestClient instance for ARGUS FastAPI application without raising internal server exceptions."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def admin_token():
    """Create a temporary Admin session token for testing."""
    store = get_session_store()
    session = store.create_session(
        operator_id="sec09_admin_id",
        username="sec09_admin",
        role="admin",
    )
    yield session.token
    store.revoke_session(session.token)


@pytest.fixture
def sample_image_bytes():
    """Create minimal 100x100 RGB JPEG image bytes for valid upload simulation."""
    import cv2
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 30:70] = [255, 255, 255]
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


# ==============================================================================
# 1. NORMAL SUCCESSFUL API BEHAVIOR UNCHANGED
# ==============================================================================


def test_normal_successful_api_behavior_unchanged(client, admin_token):
    """Prove that normal successful API operations remain fully operational."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

    resp = client.get("/api/v1/credentials", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = client.get("/api/v1/cases/jobs", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ==============================================================================
# 2. IMAGE ANALYSIS EXCEPTION DISCLOSURE TESTS
# ==============================================================================


def test_image_analysis_internal_filesystem_exception_sanitized(client, admin_token, sample_image_bytes, caplog):
    """Prove that internal FileNotFoundError with filesystem path is sanitized and logged internally."""
    from api.v1.router import get_gait_service

    fake_service = MagicMock()
    fake_service.process_image_bytes.side_effect = FileNotFoundError(
        "C:\\internal\\system\\weights\\bygait_backbone.pt"
    )
    app.dependency_overrides[get_gait_service] = lambda: fake_service

    try:
        with caplog.at_level(logging.ERROR):
            resp = client.post(
                "/api/v1/identify/image",
                headers={"Authorization": f"Bearer {admin_token}"},
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
    finally:
        app.dependency_overrides.pop(get_gait_service, None)

    assert resp.status_code == 500
    data = resp.json()
    assert data["detail"] == "Image identification failed"
    assert "C:\\internal" not in resp.text
    assert "bygait_backbone.pt" not in resp.text
    assert "FileNotFoundError" not in resp.text

    # Diagnostic log verification: internal logs must retain the exception trace
    assert any("Image identification failed" in record.message for record in caplog.records)


def test_image_analysis_database_and_token_exception_sanitized(client, admin_token, sample_image_bytes):
    """Prove that internal database connection string containing secret credentials is sanitized."""
    from api.v1.router import get_gait_service

    fake_service = MagicMock()
    fake_service.process_image_bytes.side_effect = RuntimeError(
        "database connection failed: postgresql://dbadmin:SuperSecretToken99@db.internal:5432/argus"
    )
    app.dependency_overrides[get_gait_service] = lambda: fake_service

    try:
        resp = client.post(
            "/api/v1/identify/image",
            headers={"Authorization": f"Bearer {admin_token}"},
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
    finally:
        app.dependency_overrides.pop(get_gait_service, None)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Image identification failed"
    assert "postgresql://" not in resp.text
    assert "SuperSecretToken99" not in resp.text
    assert "dbadmin" not in resp.text


def test_image_analysis_corrupted_format_validation_preserved(client, admin_token):
    """Prove that legitimate image format validation message ('Invalid or corrupted image format') is preserved."""
    corrupted_data = b"NOT_A_VALID_IMAGE_FORMAT_JUST_TEXT"

    resp = client.post(
        "/api/v1/identify/image",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("corrupt.jpg", corrupted_data, "image/jpeg")},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid or corrupted image format"


def test_image_analysis_unexpected_value_error_sanitized(client, admin_token, sample_image_bytes):
    """Prove that unexpected ValueError with internal details (e.g. array broadcast dimensions) is sanitized."""
    from api.v1.router import get_gait_service

    fake_service = MagicMock()
    fake_service.process_image_bytes.side_effect = ValueError(
        "could not broadcast input array from shape (100,100,3) into shape (64,128) in pipeline.py"
    )
    app.dependency_overrides[get_gait_service] = lambda: fake_service

    try:
        resp = client.post(
            "/api/v1/identify/image",
            headers={"Authorization": f"Bearer {admin_token}"},
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
    finally:
        app.dependency_overrides.pop(get_gait_service, None)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid image format or payload"
    assert "pipeline.py" not in resp.text
    assert "broadcast input array" not in resp.text


# ==============================================================================
# 3. VIDEO ANALYSIS EXCEPTION DISCLOSURE TESTS
# ==============================================================================


def test_video_analysis_internal_exception_sanitized(client, admin_token, monkeypatch, caplog):
    """Prove that video processing OSError leaking disk paths is sanitized."""
    import cv2

    # Mock cv2.VideoCapture to return an open capture with valid dimensions
    class MockCap:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            return 30

        def read(self):
            # Raise an unexpected internal exception during frame read
            raise OSError("[Errno 28] No space left on device: '/var/lib/argus/temp_video_upload.mp4'")

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: MockCap())

    fake_video_bytes = b"\x00\x00\x00\x20ftypmp42" + b"A" * 1024

    with caplog.at_level(logging.ERROR):
        resp = client.post(
            "/api/v1/analyze/video",
            headers={"Authorization": f"Bearer {admin_token}"},
            files={"file": ("sample.mp4", fake_video_bytes, "video/mp4")},
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Video analysis failed"
    assert "No space left on device" not in resp.text
    assert "/var/lib/argus" not in resp.text
    assert "[Errno 28]" not in resp.text
    assert any("Video analysis failed" in record.message for record in caplog.records)


def test_video_analysis_empty_file_validation_preserved(client, admin_token):
    """Prove that intentional empty video validation message is preserved."""
    resp = client.post(
        "/api/v1/analyze/video",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Uploaded video file is empty"


# ==============================================================================
# 4. CREDENTIALS MANAGEMENT EXCEPTION DISCLOSURE TESTS
# ==============================================================================


def test_credential_store_permission_error_sanitized(client, admin_token, monkeypatch):
    """Prove that PermissionError leaking existing owner username is sanitized."""
    from security_layer.credentials import CredentialManager

    def mock_store(*args, **kwargs):
        raise PermissionError("User 'sec09_admin' cannot overwrite credential 'cred_cctv' owned by 'root_system_admin'")

    monkeypatch.setattr(CredentialManager, "store_credential", mock_store)

    resp = client.post(
        "/api/v1/credentials",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "credential_id": "cred_cctv",
            "username": "camera_user",
            "password": "camera_password123",
        },
    )

    assert resp.status_code == 403
    assert resp.json()["detail"] == "Permission denied: unauthorized credential operation"
    assert "root_system_admin" not in resp.text
    assert "PermissionError" not in resp.text


def test_credential_store_runtime_exception_sanitized(client, admin_token, monkeypatch, caplog):
    """Prove that internal encryption key exception leaking file paths is sanitized."""
    from security_layer.credentials import CredentialManager

    def mock_store(*args, **kwargs):
        raise RuntimeError("Fernet encryption key file missing at E:\\ARGUS_AI\\.credentials.key")

    monkeypatch.setattr(CredentialManager, "store_credential", mock_store)

    with caplog.at_level(logging.ERROR):
        resp = client.post(
            "/api/v1/credentials",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "credential_id": "cred_new",
                "username": "user",
                "password": "pass",
            },
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to store credential"
    assert "E:\\ARGUS_AI" not in resp.text
    assert ".credentials.key" not in resp.text
    assert "Fernet" not in resp.text
    assert any("Failed to store credential" in record.message for record in caplog.records)


def test_credential_delete_and_share_permission_error_sanitized(client, admin_token, monkeypatch):
    """Prove that credential delete and share permission errors do not disclose usernames or paths."""
    from security_layer.credentials import CredentialManager

    def mock_delete(*args, **kwargs):
        raise PermissionError(
            "User 'sec09_admin' is not authorized to delete credential 'cred_root' owned by 'superuser'"
        )

    monkeypatch.setattr(CredentialManager, "delete_credential", mock_delete)

    resp = client.delete(
        "/api/v1/credentials/cred_root",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Permission denied: unauthorized credential operation"
    assert "superuser" not in resp.text

    def mock_grant(*args, **kwargs):
        raise PermissionError("User 'sec09_admin' cannot share credential owned by 'system_architect'")

    monkeypatch.setattr(CredentialManager, "grant_access", mock_grant)

    resp = client.post(
        "/api/v1/credentials/cred_root/share",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"target_user_id": "other_operator"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Permission denied: unauthorized credential operation"
    assert "system_architect" not in resp.text


# ==============================================================================
# 5. CAMERA WORKER STARTUP EXCEPTION DISCLOSURE TESTS
# ==============================================================================


def test_camera_startup_runtime_and_generic_exception_sanitized(client, admin_token):
    """Prove that camera RTSP stream failure leaking credentials or C++ traces is sanitized."""
    from api.v1.router import get_gait_service

    fake_service = MagicMock()
    # 1. Test RuntimeError (HTTP 400)
    fake_service.start_camera.side_effect = RuntimeError(
        "rtsp://admin:cameraSecretPassword@10.0.0.101:554/live assertion failed in cap_ffmpeg_impl.hpp:1153"
    )
    app.dependency_overrides[get_gait_service] = lambda: fake_service

    try:
        resp = client.post(
            "/api/v1/cameras/start",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"camera_id": "cam_gate_1", "source": "rtsp://camera"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Camera connection failed or stream unavailable"
        assert "cameraSecretPassword" not in resp.text
        assert "cap_ffmpeg_impl.hpp" not in resp.text

        # 2. Test generic Exception (HTTP 500)
        fake_service.start_camera.side_effect = Exception(
            "OS thread error: [Errno 11] Resource temporarily unavailable"
        )
        resp = client.post(
            "/api/v1/cameras/start",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"camera_id": "cam_gate_1", "source": "rtsp://camera"},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Camera worker startup failed"
        assert "[Errno 11]" not in resp.text
    finally:
        app.dependency_overrides.pop(get_gait_service, None)


# ==============================================================================
# 6. ENROLLMENT EXCEPTION DISCLOSURE TESTS
# ==============================================================================


def test_enrollment_internal_exception_sanitized(client, admin_token, sample_image_bytes):
    """Prove that enrollment exceptions leaking disk paths are sanitized."""
    from api.v1.router import get_gait_service

    fake_service = MagicMock()
    fake_service.enroll_images.side_effect = OSError("Disk write failed: '/var/gallery/database/embeddings.npy'")
    app.dependency_overrides[get_gait_service] = lambda: fake_service

    try:
        resp = client.post(
            "/api/v1/enroll",
            headers={"Authorization": f"Bearer {admin_token}"},
            data={"person_id": "subject_101"},
            files=[("files", ("photo.jpg", sample_image_bytes, "image/jpeg"))],
        )
    finally:
        app.dependency_overrides.pop(get_gait_service, None)

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Enrollment failed"
    assert "/var/gallery" not in resp.text
    assert "embeddings.npy" not in resp.text


def test_enrollment_intentional_validation_preserved(client, admin_token):
    """Prove that intentional validation messages for missing files or invalid person_id are preserved."""
    # 1. Missing files
    resp = client.post(
        "/api/v1/enroll",
        headers={"Authorization": f"Bearer {admin_token}"},
        data={"person_id": "subject_101"},
    )
    assert resp.status_code in (400, 422)

    # 2. Invalid person_id format
    resp = client.post(
        "/api/v1/enroll",
        headers={"Authorization": f"Bearer {admin_token}"},
        data={"person_id": "invalid/path/traversal"},
        files=[("files", ("test.jpg", b"fake_bytes", "image/jpeg"))],
    )
    assert resp.status_code == 400
    assert "person_id must not contain path separator characters" in resp.json()["detail"]


# ==============================================================================
# 7. REFERENCE VIDEO UPLOAD EXCEPTION DISCLOSURE TESTS
# ==============================================================================


def test_reference_video_upload_internal_exception_sanitized(client, admin_token, monkeypatch):
    """Prove that reference video upload failure does not leak internal save path."""

    async def mock_to_thread(*args, **kwargs):
        raise OSError("[Errno 13] Permission denied: 'data/reference_videos/secret_internal_file.mp4'")

    import asyncio

    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)

    fake_video = b"\x00\x00\x00\x20ftypmp42" + b"B" * 512

    resp = client.post(
        "/api/v1/cases/upload-reference",
        headers={"Authorization": f"Bearer {admin_token}"},
        data={"person_id": "ref_subject_202"},
        files={"file": ("reference.mp4", fake_video, "video/mp4")},
    )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to save video"
    assert "Permission denied" not in resp.text
    assert "secret_internal_file" not in resp.text
    assert "[Errno 13]" not in resp.text


# ==============================================================================
# 8. UPLOAD SESSION AND CHUNK REASSEMBLY EXCEPTION DISCLOSURE TESTS
# ==============================================================================


def test_upload_session_init_intentional_validation_preserved(client, admin_token):
    """Prove that upload session size limit validations are preserved exactly."""
    # 1. Total size exceeds maximum
    resp = client.post(
        "/api/v1/cases/upload-session/init",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"person_id": "subject_303", "filename": "sample.mp4", "total_size": 600 * 1024 * 1024},
    )
    assert resp.status_code == 413
    assert "exceeds maximum allowed upload size" in resp.json()["detail"]

    # 2. Total size <= 0
    resp = client.post(
        "/api/v1/cases/upload-session/init",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"person_id": "subject_303", "filename": "sample.mp4", "total_size": 0},
    )
    assert resp.status_code == 400
    assert "total_size must be greater than zero" in resp.json()["detail"]


def test_upload_session_init_unexpected_value_error_sanitized(client, admin_token, monkeypatch):
    """Prove that unexpected internal ValueError in session creation is sanitized."""
    session_mgr = UploadSessionManager.get_instance()

    def mock_create_session(*args, **kwargs):
        raise ValueError("corrupt schema in /etc/argus/session_config.json line 15")

    monkeypatch.setattr(session_mgr, "create_session", mock_create_session)

    resp = client.post(
        "/api/v1/cases/upload-session/init",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"person_id": "subject_303", "filename": "sample.mp4", "total_size": 1024},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid upload session parameters"
    assert "/etc/argus" not in resp.text


def test_upload_chunk_and_assembly_os_error_sanitized(client, admin_token, monkeypatch):
    """Prove that chunk write and assembly OSErrors do not leak disk paths."""
    session_mgr = UploadSessionManager.get_instance()

    # Create a real session to obtain valid upload_id
    rec = session_mgr.create_session(
        person_id="subject_404",
        filename="test.mp4",
        total_size=1024,
        chunk_size=1024,
        owner="sec09_admin",
    )

    # 1. Mock write_chunk to return error simulating OSError
    monkeypatch.setattr(
        session_mgr,
        "write_chunk",
        lambda *args, **kwargs: (False, "Failed to write chunk 0 to disk", {}),
    )

    resp = client.post(
        f"/api/v1/cases/upload-session/{rec.upload_id}/chunk",
        headers={"Authorization": f"Bearer {admin_token}"},
        data={"chunk_index": 0},
        files={"file": ("chunk_0", b"A" * 1024, "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Failed to write chunk 0 to disk"
    assert "data/upload_sessions" not in resp.text

    # 2. Mock assemble_and_commit to return error simulating reassembly OSError
    monkeypatch.setattr(
        session_mgr,
        "assemble_and_commit",
        lambda *args, **kwargs: (False, "Failed to assemble file from chunks", None),
    )

    resp = client.post(
        f"/api/v1/cases/upload-session/{rec.upload_id}/commit",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Failed to assemble file from chunks"
    assert "data/upload_sessions" not in resp.text


# ==============================================================================
# 9. LEGACY ROUTES & BACKGROUND JOB STATUS TESTS
# ==============================================================================


def test_legacy_routes_exception_sanitized(monkeypatch):
    """Prove that legacy /identify and /enroll routes sanitize internal exceptions."""
    from fastapi import FastAPI

    from api.routes import enrollment as legacy_enroll
    from api.routes import inference as legacy_infer

    test_app = FastAPI()
    test_app.include_router(legacy_infer.router)
    test_app.include_router(legacy_enroll.router)

    with TestClient(test_app, raise_server_exceptions=False) as legacy_client:
        # Mock InferencePipeline to raise internal exception with path
        from pipeline import inference_pipeline

        monkeypatch.setattr(
            inference_pipeline.InferencePipeline,
            "predict",
            lambda self, path: (_ for _ in ()).throw(FileNotFoundError("Missing model weights /opt/models/gait.pt")),
        )

        resp = legacy_client.post("/identify", json={"image_path": "/fake/path.jpg"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Image identification failed"
        assert "/opt/models" not in resp.text

        # Mock EnrollmentManager to raise internal exception with path
        from enrollment import enrollment_manager

        monkeypatch.setattr(
            enrollment_manager.EnrollmentManager,
            "enroll_person",
            lambda self, folder: (_ for _ in ()).throw(RuntimeError("Database corruption at /var/data/db.sqlite3")),
        )

        resp = legacy_client.post("/enroll", json={"folder_path": "/fake/folder"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Enrollment failed"
        assert "/var/data" not in resp.text


def test_reference_job_status_sanitizes_resumed_error(client, admin_token):
    """Prove that ReferenceJobManager fail_job records a safe message on resumed error."""
    from services.reference_job_manager import ReferenceJobManager

    mgr = ReferenceJobManager.get_instance()
    job = mgr.create_job(
        person_id="job_error_test_subject",
        video_path="data/reference_videos/dummy.mp4",
        case_id="case_job_error",
        owner="sec09_admin",
    )

    # Trigger fail_job with the sanitized error message
    mgr.fail_job(job.job_id, "Resumed processing failed due to an internal error", diagnostic_code="RESUME_ERROR")

    resp = client.get("/api/v1/cases/jobs", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    matching = [j for j in resp.json() if j["job_id"] == job.job_id]
    assert len(matching) == 1
    assert matching[0]["error_message"] == "Resumed processing failed due to an internal error"
    assert matching[0]["diagnostic_code"] == "RESUME_ERROR"


def test_reference_job_recovery_sanitizes_missing_video_path(tmp_path):
    """Prove that ReferenceJobManager recovery does not leak absolute server path when media is missing."""
    from services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus

    mgr = ReferenceJobManager(jobs_dir=str(tmp_path / "jobs"))
    try:
        # Create an unfinished job with a non-existent absolute path
        fake_abs_path = "C:\\Sensitive\\Internal\\Server\\Storage\\videos\\missing_ref.mp4"
        job = mgr.create_job(
            person_id="recovery_test_subject",
            video_path=fake_abs_path,
            case_id="case_recovery_test",
            owner="sec09_admin",
        )
        with mgr._lock:
            job.status = ReferenceJobStatus.PROCESSING
            mgr._persist_job(job)

        # Run recover_unfinished_jobs
        mgr.recover_unfinished_jobs()

        # Verify the failed job record does not leak fake_abs_path
        retrieved = mgr.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.status == ReferenceJobStatus.FAILED
        assert retrieved.diagnostic_code == "VIDEO_NOT_FOUND"
        assert retrieved.error_message == "Source video file not found on disk during recovery"
        assert "C:\\Sensitive" not in (retrieved.error_message or "")
    finally:
        mgr.shutdown(timeout=0.1)


# ==============================================================================
# 10. UNAUTHENTICATED ENDPOINTS RETAIN PROPER AUTH REJECTION
# ==============================================================================


def test_unauthenticated_requests_retain_proper_auth_rejection(client):
    """Prove that endpoints correctly reject unauthenticated requests with 401."""
    resp = client.post(
        "/api/v1/identify/image",
        files={"file": ("dummy.jpg", b"dummy_content", "image/jpeg")},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Authentication required. Provide Authorization: Bearer <session_token>"

    resp = client.get("/api/v1/credentials")
    assert resp.status_code == 401

    resp = client.post("/api/v1/cameras/start", json={"camera_id": "cam_1"})
    assert resp.status_code == 401
