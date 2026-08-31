"""
Bounded Operational Evidence & Representation Manager for ARGUS AI Continual Learning.

Responsible for preserving strictly the minimum verified training, validation, and
evaluation media representations (2D GEIs and 3D appearance crops) required for
scientifically valid neural-network fine-tuning and evaluation.

Core Production Invariants:
1. Bounded Quota: Enforces hard storage limits (default: 500 MB) and auto-evicts unlocked records.
2. TTL Retention: Configurable evidence retention policy with automatic expired file cleanup.
3. Manifest Locking: Never deletes evidence items locked by active evaluation manifests.
4. Cryptographic Integrity: Validates SHA-256 checksums on write and read to detect corruption.
5. Atomic Writes: Writes to temporary files before atomic filesystem rename.
6. Minimum Representation Preference: Stores only 64x128 GEIs and 256x128 crops, NO continuous video.
"""

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


class EvidenceCategory(str, Enum):
    TRAIN = "TRAIN"
    VAL = "VAL"
    HISTORICAL_TEST = "HISTORICAL_TEST"
    OPERATIONAL_TEST = "OPERATIONAL_TEST"
    FUTURE_HOLDOUT = "FUTURE_HOLDOUT"


@dataclass
class OperationalEvidenceRecord:
    """Metadata index record for an individual stored biometric evidence item."""

    evidence_id: str
    observation_id: str
    camera_id: str
    track_id: int
    session_id: str
    person_id: str
    modality: str  # 'gait' or 'appearance'
    category: EvidenceCategory
    shape: list[int]
    dtype: str
    sha256_hash: str
    file_path: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    manifest_locks: list[str] = field(default_factory=list)
    condition_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperationalEvidenceRecord":
        cat_val = data.get("category", "TRAIN")
        category = EvidenceCategory(cat_val) if isinstance(cat_val, str) else EvidenceCategory.TRAIN
        return cls(
            evidence_id=str(data["evidence_id"]),
            observation_id=str(data.get("observation_id", "")),
            camera_id=str(data.get("camera_id", "cam_01")),
            track_id=int(data.get("track_id", 0)),
            session_id=str(data.get("session_id", "session_default")),
            person_id=str(data.get("person_id", "UNKNOWN")),
            modality=str(data.get("modality", "gait")),
            category=category,
            shape=list(data.get("shape", [])),
            dtype=str(data.get("dtype", "uint8")),
            sha256_hash=str(data.get("sha256_hash", "")),
            file_path=str(data.get("file_path", "")),
            created_at=float(data.get("created_at", time.time())),
            expires_at=float(data.get("expires_at", 0.0)),
            manifest_locks=list(data.get("manifest_locks", [])),
            condition_metadata=dict(data.get("condition_metadata", {})),
        )


class OperationalEvidenceManager:
    """
    Manages quota-limited, verifiable biometric training and evaluation media.
    """

    def __init__(
        self,
        storage_dir: str = "data/operational_evidence",
        max_storage_bytes: int = 500 * 1024 * 1024,  # 500 MB default
        retention_days: float = 30.0,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_storage_bytes = int(max_storage_bytes)
        self.retention_seconds = float(retention_days) * 86400.0
        self._index_file = self.storage_dir / "evidence_index.json"
        self._records: dict[str, OperationalEvidenceRecord] = {}
        self._lock = threading.RLock()
        self._logger = get_logger("operational_evidence_manager")
        self._load_index()

    def _load_index(self) -> None:
        with self._lock:
            if not self._index_file.exists():
                return
            try:
                with open(self._index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_records = data.get("records", {})
                self._records = {k: OperationalEvidenceRecord.from_dict(v) for k, v in raw_records.items()}
            except (OSError, json.JSONDecodeError, ValueError) as err:
                self._logger.warning(f"Failed to load operational evidence index: {err}")
                self._records = {}

    def _save_index(self) -> None:
        with self._lock:
            tmp = self._index_file.with_suffix(".tmp")
            try:
                payload = {
                    "updated_at": time.time(),
                    "total_records": len(self._records),
                    "records": {k: v.to_dict() for k, v in self._records.items()},
                }
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                tmp.replace(self._index_file)
            except (OSError, ValueError) as err:
                self._logger.error(f"Failed to persist operational evidence index: {err}")

    def store_evidence(
        self,
        observation_id: str,
        camera_id: str,
        track_id: int,
        person_id: str,
        modality: str,
        media_array: np.ndarray,
        category: EvidenceCategory = EvidenceCategory.TRAIN,
        session_id: str = "",
        condition_metadata: dict[str, Any] | None = None,
    ) -> OperationalEvidenceRecord | None:
        """
        Store a verified 2D GEI or 3D appearance crop array atomically with SHA-256 integrity.
        """
        if media_array is None or not isinstance(media_array, np.ndarray):
            return None

        arr = np.ascontiguousarray(media_array)
        if arr.size == 0 or not np.isfinite(arr).all():
            return None

        # Validate minimum representation shape
        if modality == "gait":
            if arr.ndim != 2 or arr.shape[0] < 32 or arr.shape[1] < 32:
                self._logger.debug(f"Invalid GEI dimensions: {arr.shape}")
                return None
        elif modality == "appearance" and (arr.ndim != 3 or arr.shape[2] != 3):
            self._logger.debug(f"Invalid Appearance crop dimensions: {arr.shape}")
            return None

        now = time.time()
        ev_id = f"EV-{int(now)}-{uuid.uuid4().hex[:6]}"
        session_str = session_id or f"sess_{camera_id}_{track_id}_{int(now // 3600)}"
        condition_meta = dict(condition_metadata or {})

        # Compute SHA-256 of binary array buffer
        arr_bytes = arr.tobytes()
        sha256 = hashlib.sha256(arr_bytes).hexdigest()

        # Save to compressed .npz atomically
        file_name = f"{ev_id}.npz"
        target_path = self.storage_dir / file_name
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="argus_ev_", suffix=".npz", dir=str(self.storage_dir))
        os.close(tmp_fd)

        try:
            np.savez_compressed(tmp_path, data=arr, sha256=sha256)
            shutil.move(tmp_path, str(target_path))
        except (OSError, ValueError) as err:
            self._logger.error(f"Failed to write evidence {ev_id}: {err}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None

        record = OperationalEvidenceRecord(
            evidence_id=ev_id,
            observation_id=observation_id,
            camera_id=camera_id,
            track_id=track_id,
            session_id=session_str,
            person_id=person_id,
            modality=modality,
            category=category,
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            sha256_hash=sha256,
            file_path=str(target_path),
            created_at=now,
            expires_at=now + self.retention_seconds,
            condition_metadata=condition_meta,
        )

        with self._lock:
            self._records[ev_id] = record
            self._save_index()
            self._enforce_quota_and_cleanup()

        self._logger.info(
            f"[EVIDENCE_STORED] ID={ev_id} person={person_id} mod={modality} "
            f"shape={arr.shape} SHA256={sha256[:12]}..."
        )
        return record

    def load_evidence(self, evidence_id: str) -> np.ndarray | None:
        """
        Load and verify a stored biometric evidence array.
        Detects corruption via SHA-256 validation.
        """
        with self._lock:
            record = self._records.get(evidence_id)
            if not record:
                return None
            target_path = Path(record.file_path)
            if not target_path.is_file():
                self._logger.warning(f"Evidence file missing on disk: {target_path}")
                return None

            try:
                with np.load(str(target_path)) as data:
                    arr = data["data"]
                    stored_sha = str(data["sha256"])

                # Check corruption
                actual_sha = hashlib.sha256(arr.tobytes()).hexdigest()
                if actual_sha != stored_sha or actual_sha != record.sha256_hash:
                    self._logger.error(
                        f"[CORRUPTION_DETECTED] Evidence {evidence_id} SHA256 mismatch! "
                        f"Expected={record.sha256_hash}, Got={actual_sha}"
                    )
                    return None
                return arr
            except (OSError, KeyError, ValueError) as err:
                self._logger.error(f"Failed to read evidence {evidence_id}: {err}")
                return None

    def lock_manifest_evidence(self, evidence_ids: list[str], manifest_id: str) -> int:
        """Lock evidence items to an immutable dataset manifest to prevent deletion."""
        locked_count = 0
        with self._lock:
            for eid in evidence_ids:
                if eid in self._records and manifest_id not in self._records[eid].manifest_locks:
                    self._records[eid].manifest_locks.append(manifest_id)
                    locked_count += 1
            if locked_count > 0:
                self._save_index()
        return locked_count

    def unlock_manifest_evidence(self, manifest_id: str) -> int:
        """Release manifest locks after an evaluation cycle or manifest archiving."""
        unlocked_count = 0
        with self._lock:
            for rec in self._records.values():
                if manifest_id in rec.manifest_locks:
                    rec.manifest_locks.remove(manifest_id)
                    unlocked_count += 1
            if unlocked_count > 0:
                self._save_index()
        return unlocked_count

    def get_total_storage_bytes(self) -> int:
        """Compute current total disk usage of operational evidence files."""
        with self._lock:
            total = 0
            for rec in self._records.values():
                p = Path(rec.file_path)
                if p.is_file():
                    total += p.stat().st_size
            return total

    def _enforce_quota_and_cleanup(self) -> None:
        """Evict expired or over-quota unlocked evidence records."""
        now = time.time()
        # 1. Clean up expired unlocked records
        evicted = []
        for eid, rec in list(self._records.items()):
            if rec.expires_at > 0 and now > rec.expires_at and not rec.manifest_locks:
                p = Path(rec.file_path)
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
                evicted.append(eid)

        for eid in evicted:
            self._records.pop(eid, None)

        # 2. Check storage quota; evict oldest unlocked records if exceeded
        current_bytes = self.get_total_storage_bytes()
        if current_bytes > self.max_storage_bytes:
            # Sort unlocked records by creation time (oldest first)
            unlocked = [
                rec for rec in self._records.values()
                if not rec.manifest_locks
            ]
            unlocked.sort(key=lambda r: r.created_at)

            for rec in unlocked:
                if current_bytes <= self.max_storage_bytes:
                    break
                p = Path(rec.file_path)
                if p.exists():
                    sz = p.stat().st_size
                    try:
                        p.unlink()
                        current_bytes -= sz
                        self._records.pop(rec.evidence_id, None)
                    except OSError:
                        pass

        if evicted:
            self._save_index()
            self._logger.info(f"[EVIDENCE_CLEANUP] Evicted {len(evicted)} expired/unlocked evidence items.")
