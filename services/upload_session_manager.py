from __future__ import annotations

import json
import math
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from monitoring.logging_config import get_logger


@dataclass
class UploadSessionRecord:
    upload_id: str
    person_id: str
    case_id: str
    media_type: str  # "video" | "image"
    filename: str
    total_size: int
    chunk_size: int
    total_chunks: int
    chunks_received: set[int] = field(default_factory=set)
    bytes_received: int = 0
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 7200.0)  # 2 hour TTL
    status: str = "UPLOADING"  # "UPLOADING" | "COMMITTED" | "CANCELLED" | "EXPIRED"
    owner: str = ""
    final_path: str | None = None
    job_id: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["chunks_received"] = sorted(self.chunks_received)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UploadSessionRecord:
        return cls(
            upload_id=str(data["upload_id"]),
            person_id=str(data["person_id"]),
            case_id=str(data.get("case_id", "")),
            media_type=str(data.get("media_type", "video")),
            filename=str(data.get("filename", "")),
            total_size=int(data["total_size"]),
            chunk_size=int(data["chunk_size"]),
            total_chunks=int(data["total_chunks"]),
            chunks_received={int(c) for c in data.get("chunks_received", [])},
            bytes_received=int(data.get("bytes_received", 0)),
            created_at=float(data.get("created_at", time.time())),
            expires_at=float(data.get("expires_at", time.time() + 7200.0)),
            status=str(data.get("status", "UPLOADING")),
            owner=str(data.get("owner", "")),
            final_path=data.get("final_path"),
            job_id=data.get("job_id"),
            error_message=data.get("error_message"),
        )


class UploadSessionManager:
    """Manages resumable, chunked, memory-bounded upload sessions on disk.

    Guarantees:
      1. Zero full-file RAM retention (streamed persistence).
      2. Chunk-level idempotency and out-of-order arrival tolerance.
      3. Resumption from last confirmed chunk upon network interruption.
      4. Assembly integrity checks (exact size match, file extension safety).
      5. Configurable TTL sweep for incomplete/abandoned sessions.
    """

    _instance: UploadSessionManager | None = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        sessions_dir: str = "data/upload_sessions",
        default_chunk_size: int = 2 * 1024 * 1024,  # 2 MiB default
        ttl_seconds: float = 7200.0,  # 2 hours
    ) -> None:
        self.logger = get_logger("upload_session_manager")
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.default_chunk_size = default_chunk_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._sessions: dict[str, UploadSessionRecord] = {}
        self._load_active_sessions()

    @classmethod
    def get_instance(cls, sessions_dir: str = "data/upload_sessions") -> UploadSessionManager:
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls(sessions_dir=sessions_dir)
        return cls._instance

    def _session_dir(self, upload_id: str) -> Path:
        return self.sessions_dir / upload_id

    def _persist_session(self, session: UploadSessionRecord) -> None:
        s_dir = self._session_dir(session.upload_id)
        s_dir.mkdir(parents=True, exist_ok=True)
        target = s_dir / "session.json"
        tmp_target = s_dir / "session.json.tmp"
        try:
            with open(tmp_target, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
            tmp_target.replace(target)
        except OSError as e:
            self.logger.error(f"Failed to persist upload session {session.upload_id}: {e}")

    def _load_active_sessions(self) -> None:
        now = time.time()
        loaded = 0
        for s_dir in self.sessions_dir.iterdir():
            if not s_dir.is_dir():
                continue
            meta_file = s_dir / "session.json"
            if not meta_file.exists():
                continue
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                record = UploadSessionRecord.from_dict(data)
                if record.status == "UPLOADING" and now > record.expires_at:
                    record.status = "EXPIRED"
                    self._persist_session(record)
                self._sessions[record.upload_id] = record
                loaded += 1
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as err:
                self.logger.warning(f"Could not load upload session from {s_dir.name}: {err}")
        self.logger.info(f"Loaded {loaded} upload sessions from {self.sessions_dir}")

    def create_session(
        self,
        person_id: str,
        filename: str,
        total_size: int,
        chunk_size: int | None = None,
        media_type: str = "video",
        case_id: str = "",
        owner: str = "",
    ) -> UploadSessionRecord:
        normalized_person_id = person_id.strip()
        if not normalized_person_id:
            raise ValueError("person_id is required")
        if total_size <= 0:
            raise ValueError("total_size must be greater than zero")

        eff_chunk_size = chunk_size if (chunk_size and chunk_size > 0) else self.default_chunk_size
        total_chunks = max(1, math.ceil(total_size / eff_chunk_size))
        timestamp = int(time.time())
        unique_suffix = uuid.uuid4().hex[:8]
        upload_id = f"upsess_{normalized_person_id}_{timestamp}_{unique_suffix}"

        safe_name = "".join(c for c in Path(filename).name if c.isalnum() or c in "._-") or f"media_{timestamp}"

        record = UploadSessionRecord(
            upload_id=upload_id,
            person_id=normalized_person_id,
            case_id=case_id or normalized_person_id,
            media_type=media_type.lower(),
            filename=safe_name,
            total_size=total_size,
            chunk_size=eff_chunk_size,
            total_chunks=total_chunks,
            expires_at=time.time() + self.ttl_seconds,
            owner=owner,
        )

        with self._lock:
            self._sessions[upload_id] = record
            self._persist_session(record)

        self.logger.info(
            f"Created upload session '{upload_id}' for '{normalized_person_id}' "
            f"({record.media_type}, {record.total_size} bytes in {record.total_chunks} chunks)"
        )
        return record

    def get_session(self, upload_id: str) -> UploadSessionRecord | None:
        with self._lock:
            return self._sessions.get(upload_id)

    def write_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        chunk_bytes: bytes,
    ) -> tuple[bool, str, dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(upload_id)
            if not session:
                return False, f"Upload session '{upload_id}' not found", {}

            if session.status != "UPLOADING":
                return False, f"Session '{upload_id}' is in terminal state '{session.status}'", {}

            if time.time() > session.expires_at:
                session.status = "EXPIRED"
                self._persist_session(session)
                return False, f"Session '{upload_id}' has expired", {}

            if chunk_index < 0 or chunk_index >= session.total_chunks:
                return False, f"Invalid chunk index {chunk_index}; session requires 0..{session.total_chunks - 1}", {}

            # Expected size for this chunk
            if chunk_index == session.total_chunks - 1:
                expected_len = session.total_size - (session.chunk_size * (session.total_chunks - 1))
            else:
                expected_len = session.chunk_size

            actual_len = len(chunk_bytes)
            if actual_len != expected_len:
                return (
                    False,
                    f"Chunk size mismatch for chunk {chunk_index}: expected {expected_len} bytes, received {actual_len} bytes",
                    {},
                )

            s_dir = self._session_dir(upload_id)
            chunk_file = s_dir / f"chunk_{chunk_index:06d}.part"

            # Idempotency check: if chunk already exists with expected size, acknowledge without error
            if chunk_file.exists() and chunk_file.stat().st_size == expected_len:
                session.chunks_received.add(chunk_index)
                session.bytes_received = sum(
                    session.chunk_size if idx < session.total_chunks - 1 else (session.total_size - (session.chunk_size * (session.total_chunks - 1)))
                    for idx in session.chunks_received
                )
                self._persist_session(session)
                return True, "Chunk already received (idempotent)", {
                    "chunk_index": chunk_index,
                    "chunks_received": len(session.chunks_received),
                    "total_chunks": session.total_chunks,
                    "bytes_received": session.bytes_received,
                    "is_complete": len(session.chunks_received) == session.total_chunks,
                }

            # Atomically write chunk to disk
            tmp_chunk = s_dir / f"chunk_{chunk_index:06d}.tmp"
            try:
                with open(tmp_chunk, "wb") as f:
                    f.write(chunk_bytes)
                tmp_chunk.replace(chunk_file)
            except OSError as e:
                if tmp_chunk.exists():
                    tmp_chunk.unlink(missing_ok=True)
                return False, f"Failed to write chunk {chunk_index} to disk: {e}", {}

            session.chunks_received.add(chunk_index)
            session.bytes_received = sum(
                session.chunk_size if idx < session.total_chunks - 1 else (session.total_size - (session.chunk_size * (session.total_chunks - 1)))
                for idx in session.chunks_received
            )
            self._persist_session(session)

            is_complete = len(session.chunks_received) == session.total_chunks
            return True, "Chunk stored successfully", {
                "chunk_index": chunk_index,
                "chunks_received": len(session.chunks_received),
                "total_chunks": session.total_chunks,
                "bytes_received": session.bytes_received,
                "is_complete": is_complete,
            }

    def assemble_and_commit(
        self,
        upload_id: str,
        destination_dir: str | Path | None = None,
    ) -> tuple[bool, str, Path | None]:
        with self._lock:
            session = self._sessions.get(upload_id)
            if not session:
                return False, f"Upload session '{upload_id}' not found", None

            if session.status == "COMMITTED" and session.final_path:
                return True, "Session already committed", Path(session.final_path)

            if session.status != "UPLOADING":
                return False, f"Cannot commit session in state '{session.status}'", None

            missing_chunks = [i for i in range(session.total_chunks) if i not in session.chunks_received]
            if missing_chunks:
                return (
                    False,
                    f"Cannot commit: missing {len(missing_chunks)} chunks (e.g., indices {missing_chunks[:5]})",
                    None,
                )

            s_dir = self._session_dir(upload_id)
            if destination_dir is None:
                if session.media_type == "video":
                    dst_dir = Path("data/reference_videos")
                else:
                    dst_dir = Path("data/reference_photos")
            else:
                dst_dir = Path(destination_dir)

            dst_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(session.created_at)
            out_filename = f"{session.person_id}_{timestamp}_{session.filename}"
            final_path = dst_dir / out_filename
            tmp_final_path = dst_dir / f"{out_filename}.tmp"

            try:
                # Stream reassembly across chunks with 64 KiB buffer (zero multi-MB RAM heap retention)
                with open(tmp_final_path, "wb") as f_out:
                    for chunk_idx in range(session.total_chunks):
                        chunk_part = s_dir / f"chunk_{chunk_idx:06d}.part"
                        if not chunk_part.exists():
                            raise FileNotFoundError(f"Missing expected chunk file on disk: {chunk_part.name}")
                        with open(chunk_part, "rb") as f_in:
                            shutil.copyfileobj(f_in, f_out, length=64 * 1024)

                assembled_size = tmp_final_path.stat().st_size
                if assembled_size != session.total_size:
                    tmp_final_path.unlink(missing_ok=True)
                    return (
                        False,
                        f"Integrity check failed: assembled size ({assembled_size}) != expected size ({session.total_size})",
                        None,
                    )

                tmp_final_path.replace(final_path)
            except OSError as e:
                if tmp_final_path.exists():
                    tmp_final_path.unlink(missing_ok=True)
                return False, f"Failed to assemble file from chunks: {e}", None

            # Mark committed and record final destination
            session.status = "COMMITTED"
            session.final_path = str(final_path)
            self._persist_session(session)

            # Clean up individual chunk parts to reclaim disk space immediately
            for chunk_file in s_dir.glob("chunk_*.part"):
                chunk_file.unlink(missing_ok=True)

            self.logger.info(
                f"Successfully assembled and committed upload session '{upload_id}' -> '{final_path}' ({assembled_size} bytes)"
            )
            return True, "File assembled and committed successfully", final_path

    def cancel_session(self, upload_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(upload_id)
            if not session:
                return False

            session.status = "CANCELLED"
            self._persist_session(session)

            s_dir = self._session_dir(upload_id)
            if s_dir.exists():
                shutil.rmtree(s_dir, ignore_errors=True)

            self.logger.info(f"Cancelled upload session '{upload_id}' and cleaned up disk artifacts.")
            return True

    def cleanup_expired_sessions(self) -> int:
        now = time.time()
        cleaned = 0
        with self._lock:
            for upload_id, session in list(self._sessions.items()):
                if (session.status in ("COMMITTED", "CANCELLED") and (now - session.created_at > 86400)) or (
                    session.status == "UPLOADING" and now > session.expires_at
                ):
                    session.status = "EXPIRED"
                    s_dir = self._session_dir(upload_id)
                    if s_dir.exists():
                        shutil.rmtree(s_dir, ignore_errors=True)
                    del self._sessions[upload_id]
                    cleaned += 1
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} expired/completed upload sessions.")
        return cleaned
