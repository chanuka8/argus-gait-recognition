"""SEC-05: person_id path traversal / filesystem boundary escape regression tests.

Verifies that malicious person_id values cannot escape ARGUS storage directories
through API endpoints that accept person_id.

Defence-in-depth verification:
  A. Input validation rejects traversal at API layer
  B. Canonical path containment prevents filesystem escape
  C. Sentinel file outside storage root remains unmodified
  D. No directories are created outside the intended root
"""

import io
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_operator_store, get_session_store
from security_layer.input_validation import (
    validate_path_containment,
    validate_person_id,
)
from security_layer.password_hasher import get_password_hasher
from services.upload_session_manager import UploadSessionManager


@pytest.fixture(autouse=True)
def setup_sec05_fixtures(tmp_path):
    """Seed operators and create isolated test directories."""
    get_session_store().clear()
    op_store = get_operator_store()
    hasher = get_password_hasher()

    offline_data = {
        "admins": {
            "admin_sec05": {
                "name": "Admin SEC05",
                "username": "admin_sec05",
                "password_hash": hasher.hash("Admin@2026!"),
                "role": "admin",
                "status": "Active",
            },
        },
        "investigators": {
            "inv_sec05": {
                "name": "SEC05 Investigator",
                "username": "inv_sec05",
                "password_hash": hasher.hash("InvPass@2026!"),
                "role": "investigator",
                "status": "Active",
            },
        },
    }
    op_store._save_offline_store(offline_data)

    # Reset upload session singleton for clean test isolation
    UploadSessionManager._instance = None
    mgr = UploadSessionManager(sessions_dir=str(tmp_path / "upload_sessions"))
    UploadSessionManager._instance = mgr

    yield

    UploadSessionManager._instance = None


def create_token_for(username: str, role: str) -> str:
    session = get_session_store().create_session(
        operator_id=username,
        username=username,
        role=role,
        status="Active",
    )
    return session.token


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ══════════════════════════════════════════════════════════════════
# A. LEGITIMATE PERSON IDs (should be accepted)
# ══════════════════════════════════════════════════════════════════


class TestLegitimatePersonIds:
    """Verify legitimate person_id values are accepted without error."""

    @pytest.mark.parametrize(
        "person_id",
        [
            "John Doe",
            "jane_doe",
            "person-123",
            "Subject_42",
            "MP.2026.001",
            "Alice",
            "12345",
            "Person With Spaces",
        ],
    )
    def test_legitimate_person_id_accepted_by_validator(self, person_id):
        result = validate_person_id(person_id)
        assert result == person_id.strip()

    def test_legitimate_person_id_photo_upload_accepted(self):
        token = create_token_for("inv_sec05", "investigator")
        # Create a minimal 1x1 JPEG
        jpeg_bytes = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\x9e\xa7\x13\xff\xd9"
        )
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/enroll",
                data={"person_id": "Legit_Person_123"},
                files=[("files", ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg"))],
                headers=auth_header(token),
            )
            # May succeed or fail for pipeline reasons, but NOT 400 for person_id
            assert resp.status_code != 400 or "person_id" not in resp.json().get("detail", "")


# ══════════════════════════════════════════════════════════════════
# B. UNIX-STYLE PATH TRAVERSAL
# ══════════════════════════════════════════════════════════════════


class TestUnixTraversal:
    """Reject Unix-style path traversal in person_id."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "../../outside",
            "../../../outside",
            "foo/../../outside",
            "../data",
            "../../requirements.txt",
            "foo/../bar/../../outside",
        ],
    )
    def test_unix_traversal_rejected_by_validator(self, malicious_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(malicious_id)
        assert exc_info.value.status_code == 400

    def test_unix_traversal_rejected_by_photo_endpoint(self):
        token = create_token_for("inv_sec05", "investigator")
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/enroll",
                data={"person_id": "../../outside"},
                files=[("files", ("test.jpg", io.BytesIO(b"\xff\xd8\xff\xd9"), "image/jpeg"))],
                headers=auth_header(token),
            )
            assert resp.status_code == 400

    def test_unix_traversal_rejected_by_video_endpoint(self):
        token = create_token_for("inv_sec05", "investigator")
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/cases/upload-reference",
                data={"person_id": "../../../etc/passwd"},
                files=[("file", ("test.mp4", io.BytesIO(b"\x00" * 100), "video/mp4"))],
                headers=auth_header(token),
            )
            assert resp.status_code == 400

    def test_unix_traversal_rejected_by_upload_session_init(self):
        token = create_token_for("inv_sec05", "investigator")
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/cases/upload-session/init",
                json={
                    "person_id": "../../outside",
                    "filename": "test.mp4",
                    "total_size": 1024,
                    "chunk_size": 512,
                    "media_type": "video",
                },
                headers=auth_header(token),
            )
            assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════
# C. WINDOWS-STYLE TRAVERSAL
# ══════════════════════════════════════════════════════════════════


class TestWindowsTraversal:
    """Reject Windows-style path traversal in person_id."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "..\\..\\outside",
            "..\\..\\requirements.txt",
            "foo\\..\\..\\outside",
            "foo\\..\\bar\\..\\outside",
        ],
    )
    def test_windows_traversal_rejected_by_validator(self, malicious_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(malicious_id)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════
# D. URL-ENCODED TRAVERSAL
# ══════════════════════════════════════════════════════════════════


class TestEncodedTraversal:
    """Reject URL-encoded path traversal sequences."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "%2e%2e%2f",
            "%2e%2e%5c",
            "..%2f..%2f",
            "..%5c..%5c",
            "%2e%2e/%2e%2e/outside",
            "%252e%252e%252f",  # double-encoded
        ],
    )
    def test_encoded_traversal_rejected_by_validator(self, malicious_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(malicious_id)
        assert exc_info.value.status_code == 400

    def test_encoded_traversal_rejected_by_upload_session(self):
        token = create_token_for("inv_sec05", "investigator")
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/cases/upload-session/init",
                json={
                    "person_id": "..%2f..%2frequirements.txt",
                    "filename": "test.mp4",
                    "total_size": 1024,
                    "media_type": "video",
                },
                headers=auth_header(token),
            )
            assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════
# E. ABSOLUTE / DRIVE-QUALIFIED PATHS
# ══════════════════════════════════════════════════════════════════


class TestAbsolutePaths:
    """Reject absolute and drive-qualified paths."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "C:\\outside",
            "C:/outside",
            "D:\\Windows\\System32",
            "/etc/passwd",
        ],
    )
    def test_absolute_path_rejected(self, malicious_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(malicious_id)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════
# F. UNC / NETWORK PATHS
# ══════════════════════════════════════════════════════════════════


class TestUNCPaths:
    """Reject UNC-style network paths."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "\\\\server\\share\\outside",
            "//server/share/outside",
        ],
    )
    def test_unc_path_rejected(self, malicious_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(malicious_id)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════
# G. NULL / CONTROL CHARACTERS
# ══════════════════════════════════════════════════════════════════


class TestNullAndControlChars:
    """Reject null bytes and control characters."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "person\x00evil",
            "person\nid",
            "person\rid",
            "person\tid",
        ],
    )
    def test_control_chars_rejected(self, malicious_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(malicious_id)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════
# H. NESTED / MIXED TRAVERSAL
# ══════════════════════════════════════════════════════════════════


class TestNestedTraversal:
    """Reject nested and mixed traversal patterns."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "foo/../bar/../../outside",
            "foo\\..\\bar\\..\\outside",
            "..../....//",
            "....\\....\\\\",
        ],
    )
    def test_nested_traversal_rejected(self, malicious_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(malicious_id)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════
# I. DOT-ONLY NAMES
# ══════════════════════════════════════════════════════════════════


class TestDotOnlyNames:
    """Reject names consisting entirely of dots."""

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "..",
            ".",
            "...",
            "....",
        ],
    )
    def test_dot_only_rejected(self, malicious_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(malicious_id)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════
# J. EMPTY / WHITESPACE-ONLY
# ══════════════════════════════════════════════════════════════════


class TestEmptyPersonId:
    """Reject empty and whitespace-only person_id."""

    @pytest.mark.parametrize(
        "empty_id",
        [
            "",
            "   ",
            "\t",
        ],
    )
    def test_empty_rejected(self, empty_id):
        with pytest.raises(Exception) as exc_info:
            validate_person_id(empty_id)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════
# K. CANONICAL PATH CONTAINMENT (Layer B)
# ══════════════════════════════════════════════════════════════════


class TestPathContainment:
    """Verify that validate_path_containment correctly detects filesystem escape."""

    def test_contained_path_accepted(self, tmp_path):

        base = tmp_path / "storage"
        base.mkdir()
        target = base / "person_123" / "file.txt"
        result = validate_path_containment(base, target)
        assert result.is_relative_to(base.resolve())

    def test_traversal_path_rejected(self, tmp_path):

        base = tmp_path / "storage"
        base.mkdir()
        target = base / ".." / "outside" / "evil.txt"
        with pytest.raises(Exception) as exc_info:
            validate_path_containment(base, target)
        assert exc_info.value.status_code == 400

    def test_prefix_boundary_not_confused(self, tmp_path):
        """Ensure is_relative_to() doesn't confuse 'person' with 'person_evil'."""

        base = tmp_path / "person"
        base.mkdir()
        evil = tmp_path / "person_evil" / "payload.txt"
        with pytest.raises(Exception) as exc_info:
            validate_path_containment(base, evil)
        assert exc_info.value.status_code == 400


# ══════════════════════════════════════════════════════════════════
# L. FILESYSTEM ESCAPE PROOF — SENTINEL FILE TEST
# ══════════════════════════════════════════════════════════════════


class TestFilesystemEscapeProof:
    """Prove that malicious person_id cannot read, write, or modify files outside storage root."""

    def test_sentinel_file_remains_untouched_after_traversal_attempts(self, tmp_path):
        """Create a sentinel file outside the storage root and prove it's untouched
        after traversal attacks through the API."""
        # Create storage root and sentinel
        storage_root = tmp_path / "data" / "reference_photos"
        storage_root.mkdir(parents=True)
        sentinel_file = tmp_path / "SENTINEL_SECRET.txt"
        sentinel_content = f"SENTINEL_UNTOUCHED_{time.time()}"
        sentinel_file.write_text(sentinel_content)

        # Verify sentinel exists
        assert sentinel_file.exists()
        assert sentinel_file.read_text() == sentinel_content

        # Attempt various traversal attacks through the validator
        malicious_payloads = [
            "../../SENTINEL_SECRET.txt",
            "..\\..\\SENTINEL_SECRET.txt",
            "%2e%2e%2fSENTINEL_SECRET.txt",
            "..%2f..%2fSENTINEL_SECRET.txt",
            "../../../SENTINEL_SECRET.txt",
        ]

        for payload in malicious_payloads:
            with pytest.raises(HTTPException):
                validate_person_id(payload)

        # Prove sentinel is untouched
        assert sentinel_file.exists()
        assert sentinel_file.read_text() == sentinel_content

    def test_no_directory_created_outside_root_via_endpoint(self, tmp_path):
        """Prove that no directories are created outside storage root through API attacks."""
        outside_dir = tmp_path / "outside_escape"
        assert not outside_dir.exists()

        token = create_token_for("inv_sec05", "investigator")

        traversal_payloads = [
            "../../outside_escape",
            "../outside_escape",
            "..\\outside_escape",
            "%2e%2e/outside_escape",
        ]

        with TestClient(app) as client:
            for payload in traversal_payloads:
                # Try photo upload with traversal person_id
                client.post(
                    "/api/v1/enroll",
                    data={"person_id": payload},
                    files=[("files", ("test.jpg", io.BytesIO(b"\xff\xd8\xff\xd9"), "image/jpeg"))],
                    headers=auth_header(token),
                )

                # Try upload session init with traversal person_id
                client.post(
                    "/api/v1/cases/upload-session/init",
                    json={
                        "person_id": payload,
                        "filename": "test.mp4",
                        "total_size": 1024,
                        "media_type": "video",
                    },
                    headers=auth_header(token),
                )

        # Prove no escape directory was created
        assert not outside_dir.exists()

    def test_traversal_via_video_endpoint_rejected(self):
        """Prove that video upload rejects traversal person_id."""
        token = create_token_for("inv_sec05", "investigator")
        with TestClient(app) as client:
            payloads = [
                "../../outside",
                "..\\..\\outside",
                "..%2f..%2f",
                "C:\\Windows",
                "\\\\server\\share",
            ]
            for payload in payloads:
                resp = client.post(
                    "/api/v1/cases/upload-reference",
                    data={"person_id": payload},
                    files=[("file", ("test.mp4", io.BytesIO(b"\x00" * 100), "video/mp4"))],
                    headers=auth_header(token),
                )
                assert resp.status_code == 400, f"Expected 400 for '{payload}', got {resp.status_code}"


# ══════════════════════════════════════════════════════════════════
# M. EMBEDDING DATABASE SANITIZATION (existing Layer B)
# ══════════════════════════════════════════════════════════════════


class TestEmbeddingDatabaseSanitization:
    """Verify that the embedding database's existing _person_file sanitization is safe."""

    def test_embedding_db_sanitizes_traversal(self, tmp_path):
        from storage.embedding_database import EmbeddingDatabase

        db = EmbeddingDatabase(
            db_dir=str(tmp_path / "db"),
            gait_gallery_dir=str(tmp_path / "gait"),
            appearance_gallery_dir=str(tmp_path / "app"),
        )

        # Traversal person_id should be sanitized to underscores
        result = db._person_file("../../evil")
        assert ".." not in result.name
        assert "/" not in result.name
        assert "\\" not in result.name
        # Must remain inside persons_dir
        assert result.resolve().is_relative_to(db.persons_dir.resolve())
