import csv
import hashlib
import hmac
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger("ARGUS.SecurityLogger")

GENESIS_HASH = "0" * 64
LEGACY_HEADER_COLUMNS = 7
SIGNED_HEADER_COLUMNS = 9

SIGNED_CSV_HEADER = [
    "timestamp",
    "track_id",
    "identity",
    "score",
    "severity",
    "decision",
    "camera_id",
    "prev_hash",
    "hmac",
]

LEGACY_CSV_HEADER = [
    "timestamp",
    "track_id",
    "identity",
    "score",
    "severity",
    "decision",
    "camera_id",
]


def parse_and_validate_hmac_key(key: str | bytes | None) -> bytes | None:
    """Validate and normalize HMAC-SHA256 secret key.

    Requires at least 256 bits (32 bytes or 64 hex characters) of entropy.
    Never exposes the raw key in exceptions or error logs.
    """
    if key is None:
        return None
    if isinstance(key, str):
        cleaned = key.strip()
        if not cleaned:
            return None
        if len(cleaned) == 64:
            try:
                return bytes.fromhex(cleaned)
            except ValueError:
                pass
        raw_bytes = cleaned.encode("utf-8")
        if len(raw_bytes) < 32:
            raise ValueError(
                "ARGUS_AUDIT_HMAC_KEY must provide at least 256 bits of entropy "
                f"(minimum 32 bytes or 64 hex characters; provided {len(raw_bytes)} bytes)."
            )
        return raw_bytes
    elif isinstance(key, (bytes, bytearray)):
        if len(key) < 32:
            raise ValueError(
                "ARGUS_AUDIT_HMAC_KEY must provide at least 256 bits of entropy "
                f"(minimum 32 bytes; provided {len(key)} bytes)."
            )
        return bytes(key)
    else:
        raise TypeError(f"Invalid key type: {type(key).__name__}")


def canonicalize_event(
    prev_hash: str,
    timestamp: str,
    track_id: int | str,
    identity: str,
    score: float | str,
    severity: str,
    decision: str,
    camera_id: str,
) -> bytes:
    """Deterministically serialize audit event fields for HMAC authentication.

    Format: Deterministic JSON serialization with sorted keys and compact separators.
    - Explicit field ordering
    - Float scores formatted deterministically to 4 decimal places
    - UTF-8 encoding
    - Excludes the resulting HMAC column itself
    """
    try:
        norm_score = f"{float(score):.4f}"
    except (ValueError, TypeError):
        norm_score = "0.0000"

    payload = {
        "camera_id": str(camera_id),
        "decision": str(decision),
        "identity": str(identity),
        "prev_hash": str(prev_hash).strip().lower(),
        "score": norm_score,
        "severity": str(severity),
        "timestamp": str(timestamp),
        "track_id": str(track_id),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_event_hmac(key: bytes, canonical_bytes: bytes) -> str:
    """Compute HMAC-SHA256 hex digest for canonical audit payload."""
    return hmac.new(key, canonical_bytes, hashlib.sha256).hexdigest()


def get_last_signed_hmac(log_file: Path) -> str:
    """Inspect log file from disk and return the HMAC of the final signed record.

    If the file does not exist, has no records, or contains only legacy unsigned records,
    returns GENESIS_HASH (64 zeros).
    """
    if not log_file.exists():
        return GENESIS_HASH
    try:
        with open(log_file, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            last_signed_hmac: str | None = None
            for row in reader:
                if not row or (row[0] and row[0].strip().lower() == "timestamp"):
                    continue
                if len(row) >= SIGNED_HEADER_COLUMNS and row[8].strip():
                    last_signed_hmac = row[8].strip().lower()
            if last_signed_hmac:
                return last_signed_hmac
    except OSError:
        pass
    return GENESIS_HASH


class SecurityLogger:
    """Thread-safe and process-safe audit logger for ARGUS surveillance recognition events.

    Provides chained HMAC-SHA256 cryptographic tamper evidence (Finding U2)
    when configured with a valid 256-bit key.

    Backward compatibility:
      - Reads existing legacy unsigned records without breaking.
      - First signed record establishes genesis boundary linked to GENESIS_HASH.
      - Preserves public `log()` caller API for all recognition pipelines.
    """

    def __init__(
        self,
        log_file="outputs/logs/security/security_events.csv",
        hmac_key: str | bytes | None = None,
        strict_integrity: bool | None = None,
    ):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.log_file.with_name(f".{self.log_file.name}.lock")
        self._file_lock = FileLock(str(self._lock_file), timeout=10.0)
        self._thread_lock = threading.Lock()

        # Determine strict mode
        if strict_integrity is not None:
            self._strict_mode = strict_integrity
        else:
            env_strict = os.environ.get("ARGUS_AUDIT_STRICT_INTEGRITY", "").lower() in ("true", "1", "yes")
            env_prod = os.environ.get("ARGUS_MODE", "").lower() == "production"
            self._strict_mode = env_strict or env_prod

        # Key resolution
        raw_key = hmac_key or os.environ.get("ARGUS_AUDIT_HMAC_KEY")
        if not raw_key:
            key_file = Path(".audit_hmac.key")
            if key_file.exists():
                try:
                    raw_key = key_file.read_text(encoding="utf-8").strip()
                except OSError:
                    raw_key = None

        self._hmac_key: bytes | None = None
        if raw_key:
            self._hmac_key = parse_and_validate_hmac_key(raw_key)

        if self._hmac_key is not None:
            self.integrity_enabled = True
        else:
            if self._strict_mode:
                raise RuntimeError(
                    "ARGUS audit log cryptographic integrity is mandatory in production/strict mode, "
                    "but ARGUS_AUDIT_HMAC_KEY is missing or invalid."
                )
            self.integrity_enabled = False
            logger.warning("ARGUS audit logger is running in UNSIGNED mode (ARGUS_AUDIT_HMAC_KEY not configured).")

        # Header creation if file does not exist
        if not self.log_file.exists():
            with self._thread_lock, self._file_lock:
                if not self.log_file.exists():
                    with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if self.integrity_enabled:
                            writer.writerow(SIGNED_CSV_HEADER)
                        else:
                            writer.writerow(LEGACY_CSV_HEADER)
                        f.flush()

    def log(
        self,
        track_id,
        identity,
        score,
        severity,
        decision,
        camera_id="default",
    ):
        """Log a recognition event with chained HMAC-SHA256 tamper evidence.

        Preserves exact public signature and backward compatibility with all pipeline callers.
        """
        iso_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            norm_score_str = f"{float(score):.4f}"
            norm_score_val = round(float(score), 4)
        except (ValueError, TypeError):
            norm_score_str = "0.0000"
            norm_score_val = 0.0

        with self._thread_lock, self._file_lock:
            if self.integrity_enabled and self._hmac_key is not None:
                prev_hash = get_last_signed_hmac(self.log_file)
                canonical_bytes = canonicalize_event(
                    prev_hash=prev_hash,
                    timestamp=iso_timestamp,
                    track_id=track_id,
                    identity=identity,
                    score=norm_score_str,
                    severity=severity,
                    decision=decision,
                    camera_id=camera_id,
                )
                row_hmac = compute_event_hmac(self._hmac_key, canonical_bytes)
                row = [
                    iso_timestamp,
                    track_id,
                    identity,
                    norm_score_str,
                    severity,
                    decision,
                    camera_id,
                    prev_hash,
                    row_hmac,
                ]
            else:
                row = [
                    iso_timestamp,
                    track_id,
                    identity,
                    norm_score_val,
                    severity,
                    decision,
                    camera_id,
                ]

            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()

        return row[8] if len(row) >= 9 else None
