"""U2: Audit Log Cryptographic Integrity & Tamper Evidence Verifier.

Provides offline verification for ARGUS AI security audit logs
(outputs/logs/security/security_events.csv) using chained HMAC-SHA256.

Detects:
  - Modified field values (timestamps, identities, scores, decisions, cameras)
  - Tampered HMAC signatures
  - Deleted records within the signed chain
  - Inserted unauthenticated records
  - Reordered records
  - Corrupted rows and broken previous-hash continuity

LIMITATIONS:
Without an external trusted checkpoint or remote ledger, chained HMAC stored
solely in the log file CANNOT independently detect:
  - Tail truncation (deletion of the most recent signed rows)
  - Whole-file rollback to an older valid complete copy
"""

import csv
import hmac
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from security_layer.security_logger import (
    GENESIS_HASH,
    SIGNED_HEADER_COLUMNS,
    canonicalize_event,
    compute_event_hmac,
    parse_and_validate_hmac_key,
)


@dataclass
class AuditVerificationResult:
    is_valid: bool
    status: str
    total_rows: int
    verified_count: int
    legacy_count: int
    discrepancies: list[dict[str, Any]] = field(default_factory=list)
    first_tampered_line: int | None = None
    genesis_line: int | None = None
    last_verified_hmac: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "total_rows": self.total_rows,
            "verified_count": self.verified_count,
            "legacy_count": self.legacy_count,
            "discrepancies_count": len(self.discrepancies),
            "first_tampered_line": self.first_tampered_line,
            "genesis_line": self.genesis_line,
            "last_verified_hmac": self.last_verified_hmac,
        }


class AuditLogVerifier:
    """Offline verifier for chained HMAC-SHA256 audit logs."""

    def __init__(self, hmac_key: str | bytes | None = None):
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

    def verify_file(self, log_path: str | Path) -> AuditVerificationResult:
        """Verify an audit log CSV file on disk."""
        path = Path(log_path)
        if not path.exists():
            return AuditVerificationResult(
                is_valid=False,
                status="FILE_NOT_FOUND",
                total_rows=0,
                verified_count=0,
                legacy_count=0,
                discrepancies=[{"line": 0, "status": "FILE_NOT_FOUND", "detail": f"File not found: {path.name}"}],
            )

        if self._hmac_key is None:
            return AuditVerificationResult(
                is_valid=False,
                status="CONFIGURATION_ERROR",
                total_rows=0,
                verified_count=0,
                legacy_count=0,
                discrepancies=[
                    {
                        "line": 0,
                        "status": "CONFIGURATION_ERROR",
                        "detail": "HMAC key missing or invalid. Set ARGUS_AUDIT_HMAC_KEY to verify signed records.",
                    }
                ],
            )

        try:
            with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except OSError as e:
            return AuditVerificationResult(
                is_valid=False,
                status="IO_ERROR",
                total_rows=0,
                verified_count=0,
                legacy_count=0,
                discrepancies=[{"line": 0, "status": "IO_ERROR", "detail": f"Failed to read file: {e}"}],
            )

        return self.verify_rows(rows)

    def verify_rows(self, rows: list[list[str]]) -> AuditVerificationResult:
        """Verify an in-memory list of CSV row tokens."""
        if not rows:
            return AuditVerificationResult(
                is_valid=True,
                status="EMPTY",
                total_rows=0,
                verified_count=0,
                legacy_count=0,
            )

        if self._hmac_key is None:
            return AuditVerificationResult(
                is_valid=False,
                status="CONFIGURATION_ERROR",
                total_rows=len(rows),
                verified_count=0,
                legacy_count=0,
                discrepancies=[
                    {
                        "line": 0,
                        "status": "CONFIGURATION_ERROR",
                        "detail": "HMAC key missing. Cannot verify signed records.",
                    }
                ],
            )

        total_rows = 0
        legacy_count = 0
        verified_count = 0
        discrepancies: list[dict[str, Any]] = []
        genesis_line: int | None = None
        expected_prev_hash = GENESIS_HASH
        last_verified_hmac: str | None = None

        start_index = 0
        # Check if row 0 is header
        if rows and len(rows[0]) > 0 and rows[0][0].strip().lower() == "timestamp":
            start_index = 1

        for idx in range(start_index, len(rows)):
            line_no = idx + 1
            row = rows[idx]
            if not row or (len(row) == 1 and not row[0].strip()):
                continue  # Skip trailing empty lines

            total_rows += 1

            # Legacy rows: 6 or 7 columns (only valid before signed chain starts)
            if len(row) in (6, 7):
                if genesis_line is not None:
                    discrepancies.append(
                        {
                            "line": line_no,
                            "status": "MALFORMED_RECORD",
                            "detail": f"Unsigned legacy record found at line {line_no} after signed chain has begun.",
                        }
                    )
                    break
                legacy_count += 1
                continue

            # Signed rows: at least 9 columns
            if len(row) >= SIGNED_HEADER_COLUMNS:
                ts, track_id, identity, score, severity, decision, camera_id, prev_hash, stored_hmac = row[:9]
                prev_hash_clean = prev_hash.strip().lower()
                stored_hmac_clean = stored_hmac.strip().lower()

                if genesis_line is None:
                    genesis_line = line_no

                # 1. Chain continuity check
                if prev_hash_clean != expected_prev_hash:
                    discrepancies.append(
                        {
                            "line": line_no,
                            "status": "BROKEN_CHAIN",
                            "detail": f"Chain broken at line {line_no}: prev_hash does not match predecessor.",
                        }
                    )
                    break

                # 2. Recompute canonical payload and HMAC
                canonical_bytes = canonicalize_event(
                    prev_hash=prev_hash_clean,
                    timestamp=ts,
                    track_id=track_id,
                    identity=identity,
                    score=score,
                    severity=severity,
                    decision=decision,
                    camera_id=camera_id,
                )
                computed_hmac = compute_event_hmac(self._hmac_key, canonical_bytes)

                # 3. Constant-time HMAC comparison
                if not hmac.compare_digest(stored_hmac_clean, computed_hmac):
                    discrepancies.append(
                        {
                            "line": line_no,
                            "status": "INVALID_HMAC",
                            "detail": f"HMAC signature mismatch at line {line_no}.",
                        }
                    )
                    break

                # Row valid
                verified_count += 1
                expected_prev_hash = stored_hmac_clean
                last_verified_hmac = stored_hmac_clean
            else:
                discrepancies.append(
                    {
                        "line": line_no,
                        "status": "MALFORMED_RECORD",
                        "detail": f"Malformed record at line {line_no}: expected 6, 7, or 9 columns; found {len(row)}.",
                    }
                )
                break

        if discrepancies:
            return AuditVerificationResult(
                is_valid=False,
                status=discrepancies[0]["status"],
                total_rows=total_rows,
                verified_count=verified_count,
                legacy_count=legacy_count,
                discrepancies=discrepancies,
                first_tampered_line=discrepancies[0]["line"],
                genesis_line=genesis_line,
                last_verified_hmac=last_verified_hmac,
            )

        if verified_count > 0:
            final_status = "VALID_SIGNED"
        elif legacy_count > 0:
            final_status = "LEGACY_UNSIGNED"
        else:
            final_status = "EMPTY"

        return AuditVerificationResult(
            is_valid=True,
            status=final_status,
            total_rows=total_rows,
            verified_count=verified_count,
            legacy_count=legacy_count,
            genesis_line=genesis_line,
            last_verified_hmac=last_verified_hmac,
        )


def main():
    """CLI entrypoint for offline audit log verification."""
    default_log = Path("outputs/logs/security/security_events.csv")
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_log

    key = os.environ.get("ARGUS_AUDIT_HMAC_KEY")
    if not key:
        print("[ERROR] ARGUS_AUDIT_HMAC_KEY environment variable is not set.", file=sys.stderr)
        print("Usage: python -m security_layer.audit_verifier [path/to/security_events.csv]", file=sys.stderr)
        sys.exit(2)

    verifier = AuditLogVerifier(hmac_key=key)
    result = verifier.verify_file(log_path)

    print(f"Audit Log Verification Report for: {log_path}")
    print(f"Status: {result.status}")
    print(f"Total Records Inspected: {result.total_rows}")
    print(f"Legacy Unsigned Records: {result.legacy_count}")
    print(f"Cryptographically Verified Records: {result.verified_count}")

    if not result.is_valid:
        print(f"[FAIL] Verification failed at line {result.first_tampered_line}:")
        for disc in result.discrepancies:
            print(f"  - Line {disc['line']}: {disc['status']} — {disc['detail']}")
        sys.exit(1)

    print("[SUCCESS] Audit log cryptographic integrity verified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
