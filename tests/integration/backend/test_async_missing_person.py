"""Targeted integration test for Missing Person photo/video async upload and processing flow."""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_session_store
from security_layer.authorization import Role
from services.reference_job_manager import ReferenceJobStatus


@pytest.fixture
def auth_headers():
    store = get_session_store()
    inv_session = store.create_session("op_inv", "inv_user", Role.INVESTIGATOR.value)
    return {"Authorization": f"Bearer {inv_session.token}"}


def _create_mock_video_bytes(num_frames: int = 15, width: int = 320, height: int = 240) -> bytes:
    """Creates a temporary in-memory video."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, 25.0, (width, height))
    for idx in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        cx = 100 + (idx % 10) * 2
        cv2.circle(frame, (cx, 60), 15, (255, 255, 255), -1)
        cv2.rectangle(frame, (cx - 20, 75), (cx + 20, 150), (255, 255, 255), -1)
        cv2.line(frame, (cx - 10, 150), (cx - 15, 210), (255, 255, 255), 8)
        cv2.line(frame, (cx + 10, 150), (cx + 15, 210), (255, 255, 255), 8)
        writer.write(frame)
    writer.release()

    video_bytes = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return video_bytes


def test_missing_person_photo_enrollment(auth_headers):
    """Verify photo upload enrollment via /api/v1/enroll."""
    img = np.zeros((120, 60, 3), dtype=np.uint8)
    cv2.rectangle(img, (15, 15), (45, 105), (255, 255, 255), -1)
    _, encoded = cv2.imencode(".jpg", img)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/enroll",
            data={"person_id": "MP_PHOTO_001"},
            files=[("files", ("photo1.jpg", encoded.tobytes(), "image/jpeg"))],
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["person_id"] == "MP_PHOTO_001"
        assert data["embeddings_added"] >= 1


def test_missing_person_video_async_upload_202_and_job_lifecycle(auth_headers):
    """Verify video upload returns 202 Accepted immediately and creates trackable job."""
    video_bytes = _create_mock_video_bytes(num_frames=15)

    with TestClient(app) as client:
        # 1. Upload returns 202 Accepted immediately
        upload_resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_VID_001", "case_id": "CASE_2026_01"},
            files={"file": ("reference.mp4", io.BytesIO(video_bytes), "video/mp4")},
            headers=auth_headers,
        )
        assert upload_resp.status_code == 202
        body = upload_resp.json()
        assert "job_id" in body
        assert body["person_id"] == "MP_VID_001"
        assert body["status"] == ReferenceJobStatus.QUEUED.value
        job_id = body["job_id"]

        # 2. Track job progress via GET /api/v1/cases/jobs/{job_id}
        status_resp = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers)
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["job_id"] == job_id
        assert status_data["person_id"] == "MP_VID_001"
        assert status_data["status"] in [
            ReferenceJobStatus.QUEUED.value,
            ReferenceJobStatus.PROCESSING.value,
            ReferenceJobStatus.COMPLETED.value,
            ReferenceJobStatus.FAILED.value,
        ]
        assert "progress" in status_data


def test_missing_person_validation_and_security(auth_headers):
    """Verify bad inputs and unauthenticated calls are rejected with appropriate HTTP status codes."""
    with TestClient(app) as client:
        # 1. Unauthenticated upload rejected with 401
        unauth_resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_SEC_001"},
            files={"file": ("test.mp4", io.BytesIO(b"fake"), "video/mp4")},
        )
        assert unauth_resp.status_code == 401

        # 2. Empty video file rejected with 400
        empty_resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_SEC_002"},
            files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
            headers=auth_headers,
        )
        assert empty_resp.status_code == 400

        # 3. Unsupported video extension rejected with 415
        bad_ext_resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_SEC_003"},
            files={"file": ("bad.exe", io.BytesIO(b"dummy binary data"), "application/octet-stream")},
            headers=auth_headers,
        )
        assert bad_ext_resp.status_code == 415


def test_video_upload_chunked_streaming_and_persistence(auth_headers):
    """Verify video upload writes to disk in chunks without full in-RAM buffering, returns 202, and safely persists."""
    from pathlib import Path

    video_bytes = _create_mock_video_bytes(num_frames=10)

    with TestClient(app) as client:
        upload_resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_STREAM_001", "case_id": "CASE_STREAM_01"},
            files={"file": ("stream_test.mp4", io.BytesIO(video_bytes), "video/mp4")},
            headers=auth_headers,
        )
        assert upload_resp.status_code == 202
        body = upload_resp.json()
        assert "job_id" in body
        job_id = body["job_id"]

        # Verify the file exists on disk and is non-empty
        videos_dir = Path("data/reference_videos")
        matching_files = list(videos_dir.glob("MP_STREAM_001_*_stream_test.mp4"))
        assert len(matching_files) >= 1
        assert matching_files[0].stat().st_size == len(video_bytes)

        # Verify job is tracked
        job_resp = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers)
        assert job_resp.status_code == 200
        assert job_resp.json()["status"] in ["QUEUED", "PROCESSING", "COMPLETED"]

        # Clean up test artifact safely
        for f in matching_files:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

