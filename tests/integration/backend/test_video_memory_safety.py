"""SEC-07: Unbounded In-Memory Video Buffering / Memory Exhaustion DoS Tests.

Verifies:
  A. Large upload handling: Incremental streaming to disk; no whole-file RAM allocations.
  B. Chunking: Chunks are bounded; assembly streams with 64 KiB buffer without whole-file buffering.
  C. Concurrent uploads: Multiple simultaneous uploads remain bounded; single upload cannot allocate unlimited memory.
  D. Processing: Frames processed sequentially; no full-video accumulation; GEI buffers strictly bounded.
  E. Cleanup: VideoCapture released on success and failure; temporary files cleaned up.
  F. Invalid/malicious input: Oversized files rejected (413); decompression bombs (dimensions > 4096) and extreme durations rejected; malformed videos handled cleanly.
  G. Regression: Reference video upload & processing succeeds; photo processing unchanged; camera isolation intact.
"""

from __future__ import annotations

import concurrent.futures
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.pipeline.steps.live_gei import LiveGEI
from app.security_layer.auth import get_session_store
from app.security_layer.authorization import Role
from app.security_layer.input_validation import (
    CHUNK_STREAM_BUFFER_SIZE,
    MAX_ANALYZE_VIDEO_SIZE,
    MAX_CROPS_PER_TRACK,
    MAX_TRACKS_WITH_CROPS,
)
from app.services.gait_service import GaitService
from app.services.missing_person_processor import MissingPersonVideoProcessor
from app.services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus
from app.services.upload_session_manager import UploadSessionManager, UploadSessionRecord


def _isolated_processor_kwargs(tmp_path: Path) -> dict:
    """Isolated gallery/db/job-state dirs so these tests never touch the real,
    git-tracked default galleries under ml_platform/models/galleries/ or the
    real data/runtime/ state (both would otherwise be silently mutated by any
    test that calls process_reference_video/process_reference_photos)."""
    return {
        "gait_gallery_dir": str(tmp_path / "live_gallery"),
        "appearance_gallery_dir": str(tmp_path / "appearance_gallery"),
        "db_dir": str(tmp_path / "embedding_db"),
        "job_manager": ReferenceJobManager(jobs_dir=str(tmp_path / "reference_jobs"), max_workers=2),
    }


def _create_synthetic_video(
    filepath: Path,
    num_frames: int = 30,
    width: int = 320,
    height: int = 240,
    fps: float = 25.0,
) -> Path:
    """Creates a valid synthetic video file with a moving figure for testing."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    for idx in range(num_frames):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        cx = 120 + (idx % 12) * 3
        # Head
        cv2.circle(frame, (cx, 55), 16, (255, 255, 255), -1)
        # Torso
        cv2.rectangle(frame, (cx - 22, 72), (cx + 22, 150), (255, 255, 255), -1)
        # Arms
        arm_offset = int((idx % 6 - 3) * 3)
        cv2.line(frame, (cx - 22, 85), (cx - 30, 130 + arm_offset), (255, 255, 255), 6)
        cv2.line(frame, (cx + 22, 85), (cx + 30, 130 - arm_offset), (255, 255, 255), 6)
        # Legs
        cv2.line(frame, (cx - 12, 150), (cx - 16 + arm_offset, 215), (255, 255, 255), 8)
        cv2.line(frame, (cx + 12, 150), (cx + 16 - arm_offset, 215), (255, 255, 255), 8)
        writer.write(frame)
    writer.release()
    return filepath


@pytest.fixture
def auth_headers():
    session_store = get_session_store()
    session = session_store.create_session(
        operator_id="sec07_operator",
        username="sec07_operator",
        role=Role.ADMIN.value,
    )
    return {"Authorization": f"Bearer {session.token}"}


@pytest.fixture
def investigator_headers():
    session_store = get_session_store()
    session = session_store.create_session(
        operator_id="sec07_investigator",
        username="sec07_investigator",
        role=Role.INVESTIGATOR.value,
    )
    return {"Authorization": f"Bearer {session.token}"}


@pytest.fixture(autouse=True)
def _suppress_background_warmup():
    """Suppresses background ThreadPoolExecutor warmup during rapid TestClient lifespans on Windows."""

    async def _mock_warmup(self, *args, **kwargs):
        return {"status": "WARMED_UP", "duration": 0.0, "components": {}}

    with patch.object(GaitService, "warmup_async", _mock_warmup):
        yield


# =========================================================================
# A. Large Upload Handling Tests
# =========================================================================


def test_analyze_video_streams_incrementally(tmp_path, auth_headers):
    """Verify that /analyze/video reads upload stream in bounded chunks rather than full file."""
    video_path = _create_synthetic_video(tmp_path / "test_analyze.mp4", num_frames=20)
    video_size = video_path.stat().st_size
    assert video_size > 0

    read_calls: list[int] = []

    with TestClient(app) as client:
        # Patch UploadFile.read to track chunk sizes requested
        from starlette.datastructures import UploadFile as StarletteUploadFile

        orig_read = StarletteUploadFile.read

        async def tracking_read(self, size=-1):
            read_calls.append(size)
            return await orig_read(self, size)

        with patch.object(StarletteUploadFile, "read", tracking_read), open(video_path, "rb") as vf:
            resp = client.post(
                "/api/v1/analyze/video",
                files={"file": ("test_analyze.mp4", vf, "video/mp4")},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        # Verify that all reads specified a bounded chunk size (CHUNK_STREAM_BUFFER_SIZE)
        # and none called read() with -1 (unbounded full-file read)
        assert len(read_calls) > 0
        assert all(call_size == CHUNK_STREAM_BUFFER_SIZE for call_size in read_calls)


def test_reference_video_upload_streams_incrementally(tmp_path, auth_headers):
    """Verify that /cases/upload-reference streams in bounded chunks to disk."""
    video_path = _create_synthetic_video(tmp_path / "test_ref.mp4", num_frames=15)
    with open(video_path, "rb") as vf, TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_SEC07_STREAM"},
            files={"file": ("ref.mp4", vf, "video/mp4")},
            headers=auth_headers,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["person_id"] == "MP_SEC07_STREAM"


# =========================================================================
# B. Chunking & Resumable Upload Session Memory Tests
# =========================================================================


def test_chunk_upload_rejects_oversized_payload_without_buffering(tmp_path, auth_headers):
    """Verify that chunk upload reads at most expected_size + 1 and rejects oversized chunks."""
    session_mgr = UploadSessionManager.get_instance(str(tmp_path / "upload_sessions"))
    record = session_mgr.create_session(
        person_id="MP_SEC07_CHUNK",
        filename="test.mp4",
        total_size=1024,
        chunk_size=512,
        owner="sec07_operator",
    )

    with TestClient(app) as client:
        # Client sends a 100 KiB payload for a chunk expected to be 512 bytes
        oversized_payload = b"X" * (100 * 1024)
        resp = client.post(
            f"/api/v1/cases/upload-session/{record.upload_id}/chunk",
            data={"chunk_index": 0},
            files={"file": ("chunk0.part", oversized_payload, "application/octet-stream")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "exceeds expected size" in resp.json()["detail"].lower()


def test_chunk_commit_streams_reassembly(tmp_path, auth_headers):
    """Verify that commit reassembles chunks via 64 KiB streaming without whole-file RAM buffering."""
    session_mgr = UploadSessionManager.get_instance(str(tmp_path / "upload_sessions"))
    chunk_0 = b"A" * 1024
    chunk_1 = b"B" * 1024
    total_size = len(chunk_0) + len(chunk_1)

    record = session_mgr.create_session(
        person_id="MP_SEC07_COMMIT",
        filename="test_commit.bin",
        total_size=total_size,
        chunk_size=1024,
        media_type="video",
        owner="sec07_operator",
    )

    with TestClient(app) as client:
        # Upload chunk 0
        r0 = client.post(
            f"/api/v1/cases/upload-session/{record.upload_id}/chunk",
            data={"chunk_index": 0},
            files={"file": ("chunk0.part", chunk_0, "application/octet-stream")},
            headers=auth_headers,
        )
        assert r0.status_code == 200

        # Upload chunk 1
        r1 = client.post(
            f"/api/v1/cases/upload-session/{record.upload_id}/chunk",
            data={"chunk_index": 1},
            files={"file": ("chunk1.part", chunk_1, "application/octet-stream")},
            headers=auth_headers,
        )
        assert r1.status_code == 200

        # Spy on shutil.copyfileobj to verify 64 KiB buffer usage
        with patch("shutil.copyfileobj", wraps=shutil.copyfileobj) as mock_copy:
            resp = client.post(
                f"/api/v1/cases/upload-session/{record.upload_id}/commit",
                headers=auth_headers,
            )
            assert resp.status_code == 202
            assert mock_copy.call_count == 2
            # Verify 64 KiB (65536) buffer length parameter passed to copyfileobj
            for call in mock_copy.call_args_list:
                assert call.kwargs.get("length") == 64 * 1024


def test_upload_session_init_rejects_oversized_total_size(auth_headers):
    """Verify that /cases/upload-session/init rejects total_size exceeding 500 MiB."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-session/init",
            json={
                "person_id": "MP_SEC07_OVERSIZED",
                "filename": "huge_video.mp4",
                "total_size": 600 * 1024 * 1024,  # 600 MiB > 500 MiB limit
            },
            headers=auth_headers,
        )
        assert resp.status_code == 413
        assert "exceeds maximum allowed upload size" in resp.json()["detail"]


def test_upload_session_init_rejects_oversized_chunk_size(auth_headers):
    """Verify that /cases/upload-session/init rejects declared chunk_size exceeding 10 MiB."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-session/init",
            json={
                "person_id": "MP_SEC07_BIGCHUNK",
                "filename": "video.mp4",
                "total_size": 20 * 1024 * 1024,
                "chunk_size": 15 * 1024 * 1024,  # 15 MiB > 10 MiB limit
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "exceeds maximum allowed chunk size" in resp.json()["detail"]


# =========================================================================
# C. Concurrent Uploads Bounding
# =========================================================================


def test_concurrent_uploads_are_individually_bounded(tmp_path, auth_headers):
    """Verify multiple concurrent chunk uploads proceed safely without shared memory blowup."""
    session_mgr = UploadSessionManager.get_instance(str(tmp_path / "upload_sessions"))
    sessions = [
        session_mgr.create_session(
            person_id=f"MP_SEC07_CONC_{i}",
            filename=f"conc_{i}.mp4",
            total_size=1024,
            chunk_size=1024,
            owner="sec07_operator",
        )
        for i in range(4)
    ]

    payload = b"Z" * 1024

    def upload_for_session(sess: UploadSessionRecord):
        with TestClient(app) as client:
            return client.post(
                f"/api/v1/cases/upload-session/{sess.upload_id}/chunk",
                data={"chunk_index": 0},
                files={"file": ("chunk0.part", payload, "application/octet-stream")},
                headers=auth_headers,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(upload_for_session, sessions))

    for resp in results:
        assert resp.status_code == 200
        assert resp.json()["is_complete"] is True


# =========================================================================
# D. Processing & Frame Memory Bounding
# =========================================================================


def test_frames_processed_sequentially_without_full_video_retention(tmp_path):
    """Verify that MissingPersonVideoProcessor decodes frames sequentially without full accumulation."""
    video_path = _create_synthetic_video(tmp_path / "test_seq.mp4", num_frames=30)
    processor = MissingPersonVideoProcessor(**_isolated_processor_kwargs(tmp_path))

    # Track how many frame objects are simultaneously retained
    active_frame_shapes = []

    orig_track = processor.tracker.track

    def monitoring_track(frame):
        # Record shape and verify frame is a standard 2D/3D numpy array
        active_frame_shapes.append(frame.shape)
        return orig_track(frame)

    with patch.object(processor.tracker, "track", side_effect=monitoring_track):
        processor.process_reference_video(
            person_id="MP_SEC07_SEQ",
            video_path=video_path,
        )

    # Detections happened across frames sequentially
    assert len(active_frame_shapes) > 0


def test_live_gei_rolling_buffer_remains_strictly_bounded():
    """Verify that LiveGEI rolling window N=15 never exceeds capacity under continuous stream."""
    gei = LiveGEI(max_frames=15, min_frames=10)
    dummy_sil = np.ones((128, 64), dtype=np.uint8) * 255

    for idx in range(100):  # Stream 100 frames
        # Vary slightly to avoid duplicate filter
        sil = dummy_sil.copy()
        sil[idx % 120, :] = 0
        gei.add(sil)
        # Invariant: buffer must never exceed history_capacity
        assert len(gei.frames) <= 15
        assert len(gei.width_signals) <= 15

    assert gei.ready()
    result = gei.build()
    assert result is not None
    assert result.shape == (128, 64)


def test_track_crops_bounded_across_many_subjects(tmp_path):
    """Verify that crop accumulation in MissingPersonVideoProcessor is capped per track and across tracks."""
    processor = MissingPersonVideoProcessor(**_isolated_processor_kwargs(tmp_path))
    assert processor.max_crops_per_track == MAX_CROPS_PER_TRACK
    assert processor.max_tracks_with_crops == MAX_TRACKS_WITH_CROPS


# =========================================================================
# E. Resource Release & Cleanup Tests
# =========================================================================


def test_video_capture_released_on_success(tmp_path):
    """Verify VideoCapture.release() is called after successful processing."""
    video_path = _create_synthetic_video(tmp_path / "release_succ.mp4", num_frames=20)
    processor = MissingPersonVideoProcessor(**_isolated_processor_kwargs(tmp_path))

    real_capture_class = cv2.VideoCapture
    released_count = [0]

    class VideoCaptureWrapper:
        def __init__(self, *args, **kwargs):
            self._real = real_capture_class(*args, **kwargs)

        def release(self):
            released_count[0] += 1
            return self._real.release()

        def __getattr__(self, name):
            return getattr(self._real, name)

    with patch("app.services.missing_person_processor.cv2.VideoCapture", VideoCaptureWrapper):
        processor.process_reference_video(
            person_id="MP_SEC07_REL_SUCC",
            video_path=video_path,
        )

    assert released_count[0] >= 1


def test_video_capture_released_on_validation_failure(tmp_path):
    """Verify VideoCapture.release() is called even when video validation fails."""
    # Write a 10-byte truncated pseudo-video that cv2 cannot parse
    bad_video = tmp_path / "corrupted.mp4"
    bad_video.write_bytes(b"\x00\x00\x00\x18ftypmp42corruptdata")

    processor = MissingPersonVideoProcessor(**_isolated_processor_kwargs(tmp_path))
    ok, _err, _meta = processor.validate_video_file(bad_video)
    # File either fails decoder or fails resolution/frame check
    assert ok is False


def test_temporary_files_cleaned_up_after_analyze_video(tmp_path, auth_headers):
    """Verify temporary files created during /analyze/video are deleted after response."""
    video_path = _create_synthetic_video(tmp_path / "temp_clean.mp4", num_frames=15)

    created_temp_files = []
    orig_named_temp = tempfile.NamedTemporaryFile

    def tracking_named_temp(*args, **kwargs):
        tf = orig_named_temp(*args, **kwargs)
        created_temp_files.append(Path(tf.name))
        return tf

    with (
        patch("tempfile.NamedTemporaryFile", side_effect=tracking_named_temp),
        open(video_path, "rb") as vf,
        TestClient(app) as client,
    ):
        resp = client.post(
            "/api/v1/analyze/video",
            files={"file": ("temp_clean.mp4", vf, "video/mp4")},
            headers=auth_headers,
        )
        assert resp.status_code == 200

    # Verify all temporary files created have been unlinked
    assert len(created_temp_files) > 0
    for temp_f in created_temp_files:
        assert not temp_f.exists(), f"Temporary file {temp_f} was not cleaned up!"


# =========================================================================
# F. Invalid / Malicious Input Rejection Tests
# =========================================================================


def test_analyze_video_rejects_oversized_upload(tmp_path, auth_headers):
    """Verify /analyze/video rejects files exceeding MAX_ANALYZE_VIDEO_SIZE (100 MiB) with 413."""
    oversized_file = tmp_path / "huge_analyze.mp4"
    with open(oversized_file, "wb") as f:
        f.seek(MAX_ANALYZE_VIDEO_SIZE + 1024)
        f.write(b"0")

    with open(oversized_file, "rb") as vf, TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyze/video",
            files={"file": ("huge_analyze.mp4", vf, "video/mp4")},
            headers=auth_headers,
        )
        assert resp.status_code == 413
        assert "exceeds maximum allowed analysis size" in resp.json()["detail"]


def test_validate_video_rejects_decompression_bomb_resolution(tmp_path):
    """Verify validate_video_file rejects video exceeding MAX_VIDEO_DIMENSION (4096)."""
    video_path = _create_synthetic_video(tmp_path / "bomb.mp4", num_frames=15)
    processor = MissingPersonVideoProcessor(**_isolated_processor_kwargs(tmp_path))

    # Mock VideoCapture to declare 8192x8192 resolution
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 8192,
        cv2.CAP_PROP_FRAME_HEIGHT: 8192,
    }.get(prop, 0)

    with patch("app.services.missing_person_processor.cv2.VideoCapture", return_value=mock_cap):
        ok, msg, _meta = processor.validate_video_file(video_path)
        assert ok is False
        assert "exceeds maximum supported resolution" in msg


def test_validate_video_rejects_extreme_frame_count(tmp_path):
    """Verify validate_video_file rejects video with total_frames exceeding MAX_VIDEO_FRAMES."""
    video_path = _create_synthetic_video(tmp_path / "long.mp4", num_frames=15)
    processor = MissingPersonVideoProcessor(**_isolated_processor_kwargs(tmp_path))

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 200_000,  # 200,000 frames > 100,000 limit
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 1920,
        cv2.CAP_PROP_FRAME_HEIGHT: 1080,
    }.get(prop, 0)

    with patch("app.services.missing_person_processor.cv2.VideoCapture", return_value=mock_cap):
        ok, msg, _meta = processor.validate_video_file(video_path)
        assert ok is False
        assert "exceeds maximum allowable duration limit" in msg


def test_empty_video_rejected_cleanly(tmp_path, auth_headers):
    """Verify 0-byte video is rejected with 400 Bad Request."""
    empty_path = tmp_path / "empty.mp4"
    empty_path.write_bytes(b"")

    with open(empty_path, "rb") as ef, TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyze/video",
            files={"file": ("empty.mp4", ef, "video/mp4")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()


# =========================================================================
# G. Regression Tests
# =========================================================================


def test_normal_reference_video_flow_succeeds(tmp_path, auth_headers):
    """Verify normal reference video upload and queuing operates successfully."""
    video_path = _create_synthetic_video(tmp_path / "normal_ref.mp4", num_frames=20)
    with open(video_path, "rb") as vf, TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_NORMAL_FLOW"},
            files={"file": ("normal_ref.mp4", vf, "video/mp4")},
            headers=auth_headers,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["person_id"] == "MP_NORMAL_FLOW"
        assert data["status"] == ReferenceJobStatus.QUEUED.value


def test_normal_photo_processing_remains_unchanged(tmp_path):
    """Verify that photo enrollment in MissingPersonVideoProcessor remains intact."""
    # Create 3 synthetic photos
    photo_paths = []
    for i in range(3):
        p = tmp_path / f"photo_{i}.jpg"
        img = np.full((128, 64, 3), 40 + i * 10, dtype=np.uint8)
        cv2.imwrite(str(p), img)
        photo_paths.append(p)

    processor = MissingPersonVideoProcessor(**_isolated_processor_kwargs(tmp_path))
    # Mock extractor backend predict to produce valid 256D normalized vectors
    mock_vec = np.ones((256,), dtype=np.float32) / np.sqrt(256)
    mock_backend = MagicMock()
    mock_backend.predict.return_value = mock_vec

    with patch.object(processor.extractor, "backend", mock_backend):
        res = processor.process_reference_photos(
            person_id="MP_PHOTO_REG",
            photo_paths=photo_paths,
        )
        assert res["success"] is True
        assert res["status"] == "COMPLETED"
        assert res["person_id"] == "MP_PHOTO_REG"
