"""Unit tests for the resumable chunked UploadSessionManager."""

from __future__ import annotations

import shutil
import tempfile
import time

import pytest

from services.upload_session_manager import UploadSessionManager


@pytest.fixture
def temp_session_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_upload_sessions_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def session_mgr(temp_session_dir):
    # Reset singleton
    UploadSessionManager._instance = None
    mgr = UploadSessionManager(sessions_dir=temp_session_dir, default_chunk_size=1024 * 1024, ttl_seconds=3600.0)
    yield mgr
    UploadSessionManager._instance = None


def test_session_initialization(session_mgr):
    """Test session creation with calculated chunk counts."""
    record = session_mgr.create_session(
        person_id="MP-101",
        filename="reference_walk.mp4",
        total_size=5 * 1024 * 1024 + 500,  # ~5.0005 MiB
        chunk_size=1024 * 1024,  # 1 MiB
        media_type="video",
        case_id="Case-101",
    )
    assert record.upload_id is not None
    assert record.person_id == "MP-101"
    assert record.total_chunks == 6
    assert record.status == "UPLOADING"
    assert len(record.chunks_received) == 0


def test_chunk_upload_and_idempotency(session_mgr):
    """Test chunk writing, bytes counting, and duplicate upload idempotency."""
    total_size = 2048
    chunk_size = 1024
    payload_chunk0 = b"A" * 1024
    payload_chunk1 = b"B" * 1024

    record = session_mgr.create_session(
        person_id="MP-102",
        filename="face_crop.jpg",
        total_size=total_size,
        chunk_size=chunk_size,
        media_type="image",
    )
    upload_id = record.upload_id

    # 1. Upload chunk 0
    ok, _, data = session_mgr.write_chunk(upload_id, 0, payload_chunk0)
    assert ok is True
    assert data["bytes_received"] == 1024
    assert data["chunks_received"] == 1
    assert data["is_complete"] is False

    # 2. Idempotent re-upload of chunk 0 (network retry simulation)
    ok2, msg2, data2 = session_mgr.write_chunk(upload_id, 0, payload_chunk0)
    assert ok2 is True
    assert "already received" in msg2.lower()
    assert data2["bytes_received"] == 1024
    assert data2["chunks_received"] == 1

    # 3. Upload chunk 1
    ok3, _, data3 = session_mgr.write_chunk(upload_id, 1, payload_chunk1)
    assert ok3 is True
    assert data3["bytes_received"] == 2048
    assert data3["chunks_received"] == 2
    assert data3["is_complete"] is True


def test_out_of_order_chunk_arrival_and_commit(session_mgr):
    """Test chunks arriving out-of-order and successful streamed assembly."""
    content = b"CHUNK000_" + b"CHUNK001_" + b"CHUNK002_"
    chunk_size = 9
    total_size = len(content)  # 27 bytes

    record = session_mgr.create_session(
        person_id="MP-103",
        filename="test_stream.bin",
        total_size=total_size,
        chunk_size=chunk_size,
        media_type="video",
    )
    upload_id = record.upload_id

    # Upload out of order: chunk 2, chunk 0, chunk 1
    session_mgr.write_chunk(upload_id, 2, content[18:27])
    session_mgr.write_chunk(upload_id, 0, content[0:9])
    session_mgr.write_chunk(upload_id, 1, content[9:18])

    # Commit and assemble
    success, _, final_path = session_mgr.assemble_and_commit(upload_id)
    assert success is True
    assert final_path is not None
    assert final_path.exists()

    with open(final_path, "rb") as f:
        assembled_data = f.read()
    assert assembled_data == content


def test_network_interruption_and_resumption(session_mgr):
    """Test querying status to resume from missing chunks."""
    total_size = 3000
    chunk_size = 1000

    record = session_mgr.create_session(
        person_id="MP-104",
        filename="interrupted.mp4",
        total_size=total_size,
        chunk_size=chunk_size,
        media_type="video",
    )
    upload_id = record.upload_id

    # Client uploads chunk 0 and then connection drops
    session_mgr.write_chunk(upload_id, 0, b"X" * 1000)

    # Client reconnects and queries status
    reconnected_record = session_mgr.get_session(upload_id)
    assert reconnected_record is not None
    assert 0 in reconnected_record.chunks_received
    assert 1 not in reconnected_record.chunks_received
    assert 2 not in reconnected_record.chunks_received

    # Client resumes by sending only chunk 1 and 2
    session_mgr.write_chunk(upload_id, 1, b"Y" * 1000)
    session_mgr.write_chunk(upload_id, 2, b"Z" * 1000)

    success, _, final_path = session_mgr.assemble_and_commit(upload_id)
    assert success is True
    assert final_path.stat().st_size == 3000


def test_size_mismatch_integrity_rejection(session_mgr):
    """Test rejection when chunk lengths or assembled size does not match manifest."""
    total_size = 2000
    chunk_size = 1000

    record = session_mgr.create_session(
        person_id="MP-105",
        filename="corrupted.mp4",
        total_size=total_size,
        chunk_size=chunk_size,
        media_type="video",
    )
    upload_id = record.upload_id

    # Chunk 0 has wrong byte count (500 instead of 1000)
    ok, msg, _ = session_mgr.write_chunk(upload_id, 0, b"X" * 500)
    assert ok is False
    assert "mismatch" in msg.lower()


def test_incomplete_commit_rejection(session_mgr):
    """Test that commit fails when chunks are missing."""
    record = session_mgr.create_session(
        person_id="MP-106",
        filename="incomplete.mp4",
        total_size=3000,
        chunk_size=1000,
        media_type="video",
    )
    upload_id = record.upload_id
    session_mgr.write_chunk(upload_id, 0, b"A" * 1000)
    # chunk 1 and 2 missing

    success, msg, final_path = session_mgr.assemble_and_commit(upload_id)
    assert success is False
    assert "missing" in msg.lower()
    assert final_path is None


def test_cancel_session_cleans_disk(session_mgr):
    """Test cancelling an upload session deletes disk artifacts."""
    record = session_mgr.create_session(
        person_id="MP-107",
        filename="cancelled.mp4",
        total_size=2000,
        chunk_size=1000,
        media_type="video",
    )
    upload_id = record.upload_id
    session_mgr.write_chunk(upload_id, 0, b"A" * 1000)

    s_dir = session_mgr._session_dir(upload_id)
    assert s_dir.exists()

    cancelled = session_mgr.cancel_session(upload_id)
    assert cancelled is True
    assert not s_dir.exists()
    cancelled_rec = session_mgr.get_session(upload_id)
    assert cancelled_rec is not None
    assert cancelled_rec.status == "CANCELLED"


def test_ttl_cleanup_removes_expired_sessions(session_mgr):
    """Test TTL background sweep cleans expired sessions."""
    record = session_mgr.create_session(
        person_id="MP-108",
        filename="expired.mp4",
        total_size=1000,
        chunk_size=1000,
        media_type="video",
    )
    upload_id = record.upload_id
    session_mgr.write_chunk(upload_id, 0, b"A" * 1000)

    # Force expiration timestamp
    record.expires_at = time.time() - 10.0
    session_mgr._persist_session(record)

    # Run cleanup sweep
    purged_count = session_mgr.cleanup_expired_sessions()
    assert purged_count >= 1
    assert session_mgr.get_session(upload_id) is None
