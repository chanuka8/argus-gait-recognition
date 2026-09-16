"""U2: Audit Log Cryptographic Integrity & Tamper Evidence Test Suite.

Verifies:
  - Chained HMAC-SHA256 creation and verification
  - Deterministic canonicalization
  - Tamper detection (modification, deletion, insertion, reordering, corruption)
  - Key management validation and missing key modes
  - Legacy 6-column and 7-column backward compatibility
  - Multi-process write safety with inter-process file locking
  - Tail-truncation limitation boundary
"""

import csv
import multiprocessing
from pathlib import Path

import pytest

from security_layer.audit_verifier import AuditLogVerifier
from security_layer.security_logger import (
    GENESIS_HASH,
    SecurityLogger,
    canonicalize_event,
    parse_and_validate_hmac_key,
)

TEST_HMAC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


@pytest.fixture
def temp_log(tmp_path):
    """Provide temporary audit log path isolated from production outputs."""
    return tmp_path / "security_events.csv"


# 1. Normal signed event creation and verification
def test_normal_signed_event_creation_and_verification(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(track_id=1, identity="person_01", score=0.92, severity="INFO", decision="ALLOW", camera_id="cam_01")
    logger.log(
        track_id=2, identity="UNKNOWN", score=0.35, severity="HIGH", decision="SECURITY_ALERT", camera_id="cam_02"
    )

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)

    assert result.is_valid is True
    assert result.status == "VALID_SIGNED"
    assert result.total_rows == 2
    assert result.verified_count == 2
    assert result.legacy_count == 0
    assert len(result.discrepancies) == 0


# 2. Deterministic canonicalization
def test_deterministic_canonicalization():
    b1 = canonicalize_event(
        prev_hash=GENESIS_HASH,
        timestamp="2026-09-16T10:00:00+00:00",
        track_id=1,
        identity="subject_001",
        score=0.92,
        severity="INFO",
        decision="ALLOW",
        camera_id="cam_01",
    )
    b2 = canonicalize_event(
        prev_hash=GENESIS_HASH,
        timestamp="2026-09-16T10:00:00+00:00",
        track_id="1",
        identity="subject_001",
        score="0.92",
        severity="INFO",
        decision="ALLOW",
        camera_id="cam_01",
    )
    assert b1 == b2
    assert b'score":"0.9200' in b1
    assert b'prev_hash":"' + GENESIS_HASH.encode("utf-8") in b1


# 3. Modified timestamp detection
def test_modified_timestamp_detection(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "027", 0.95, "INFO", "ALLOW", "cam_01")

    # Tamper with timestamp
    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[1][0] = "2020-01-01T00:00:00+00:00"
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "INVALID_HMAC"
    assert result.first_tampered_line == 2


# 4. Modified identity detection
def test_modified_identity_detection(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "UNKNOWN", 0.30, "HIGH", "SECURITY_ALERT", "cam_01")

    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[1][2] = "VIP_PERSON"
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "INVALID_HMAC"


# 5. Modified score detection
def test_modified_score_detection(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "027", 0.65, "MEDIUM", "REVIEW_REQUIRED", "cam_01")

    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[1][3] = "0.9999"
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "INVALID_HMAC"


# 6. Modified decision detection
def test_modified_decision_detection(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "UNKNOWN", 0.30, "HIGH", "SECURITY_ALERT", "cam_01")

    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[1][5] = "ALLOW"
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "INVALID_HMAC"


# 7. Modified HMAC signature detection
def test_modified_hmac_detection(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "027", 0.95, "INFO", "ALLOW", "cam_01")

    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[1][8] = "a" * 64
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "INVALID_HMAC"


# 8. Middle-row deletion detection
def test_middle_row_deletion_detection(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "001", 0.90, "INFO", "ALLOW", "cam_01")
    logger.log(2, "UNKNOWN", 0.20, "HIGH", "SECURITY_ALERT", "cam_01")  # Incriminating alert row
    logger.log(3, "002", 0.88, "INFO", "ALLOW", "cam_01")

    # Attacker deletes row 2
    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows_after_deletion = [rows[0], rows[1], rows[3]]  # Header, row 1, row 3
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_after_deletion)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "BROKEN_CHAIN"
    assert result.first_tampered_line == 3  # Line 3 now fails because its prev_hash pointed to deleted row


# 9. Inserted signed-row chain failure
def test_inserted_row_breaks_chain(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "001", 0.90, "INFO", "ALLOW", "cam_01")
    logger.log(2, "002", 0.88, "INFO", "ALLOW", "cam_01")

    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Attacker injects a fake forged row between row 1 and 2
    fake_row = ["2026-09-16T10:05:00", "99", "FAKE_PERSON", "0.99", "INFO", "ALLOW", "cam_01", GENESIS_HASH, "f" * 64]
    rows.insert(2, fake_row)
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status in ("BROKEN_CHAIN", "INVALID_HMAC")


# 10. Reordered-row detection
def test_reordered_rows_detected(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "001", 0.90, "INFO", "ALLOW", "cam_01")
    logger.log(2, "002", 0.88, "INFO", "ALLOW", "cam_01")

    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # Swap row 1 and row 2
    rows[1], rows[2] = rows[2], rows[1]
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "BROKEN_CHAIN"


# 11. Corrupted previous-HMAC detection
def test_corrupted_prev_hash_detected(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "001", 0.90, "INFO", "ALLOW", "cam_01")
    logger.log(2, "002", 0.88, "INFO", "ALLOW", "cam_01")

    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[2][7] = "0" * 64  # Corrupt prev_hash of second row
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "BROKEN_CHAIN"


# 12. Malformed column count handling
def test_malformed_columns_detected(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "001", 0.90, "INFO", "ALLOW", "cam_01")

    with open(temp_log, "a", newline="", encoding="utf-8") as f:
        f.write("corrupted,line,with,only,four,columns\n")

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is False
    assert result.status == "MALFORMED_RECORD"


# 13. Legacy 6-column row compatibility
def test_legacy_6_column_compatibility(temp_log):
    # Create file with legacy 6-column rows
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "identity", "score", "severity", "decision"])
        writer.writerow(["2026-06-11T09:34:21.803517", "1", "027", "0.92", "INFO", "ALLOW"])
        writer.writerow(["2026-06-11T09:34:21.804516", "2", "UNKNOWN", "0.31", "HIGH", "SECURITY_ALERT"])

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.status == "LEGACY_UNSIGNED"
    assert result.legacy_count == 2
    assert result.verified_count == 0


# 14. Legacy 7-column row compatibility
def test_legacy_7_column_compatibility(temp_log):
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "identity", "score", "severity", "decision", "camera_id"])
        writer.writerow(["2026-06-14T22:24:09.167753", "1", "UNKNOWN", "0.0", "HIGH", "SECURITY_ALERT", "cam_01"])

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.status == "LEGACY_UNSIGNED"
    assert result.legacy_count == 1
    assert result.verified_count == 0


# 15. Transition from legacy rows to signed rows
def test_transition_from_legacy_to_signed_rows(temp_log):
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "identity", "score", "severity", "decision", "camera_id"])
        writer.writerow(["2026-06-14T22:24:09.167753", "1", "027", "0.92", "INFO", "ALLOW", "cam_01"])

    # Now append signed rows with SecurityLogger
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(2, "057", 0.88, "INFO", "ALLOW", "cam_02")
    logger.log(3, "051", 0.91, "INFO", "ALLOW", "cam_02")

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.status == "VALID_SIGNED"
    assert result.legacy_count == 1
    assert result.verified_count == 2


# 16. Unicode identity and camera values
def test_unicode_identity_and_camera(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "subject_田中_001", 0.92, "INFO", "ALLOW", "camera_中央ゲート")
    logger.log(2, "person_ñ_á", 0.85, "INFO", "ALLOW", "cam_français")

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.verified_count == 2


# 17. Special characters: commas, quotes, backslashes, pipes
def test_special_characters_handling(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, 'person,"with",quotes', 0.80, "INFO", "ALLOW", "cam|pipe|01")
    logger.log(2, "person\\with\\backslash", 0.75, "MEDIUM", "REVIEW_REQUIRED", "cam_02")

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.verified_count == 2


# 18. Embedded newlines in fields
def test_embedded_newlines_handled(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "person\nwith\nnewline", 0.90, "INFO", "ALLOW", "cam\r01")

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.verified_count == 1


# 19. Missing key in strict mode raises error
def test_missing_key_in_strict_mode_raises(temp_log, monkeypatch):
    monkeypatch.delenv("ARGUS_AUDIT_HMAC_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ARGUS audit log cryptographic integrity is mandatory"):
        SecurityLogger(log_file=temp_log, hmac_key=None, strict_integrity=True)


# 20. Unsigned mode when key not configured in development mode
def test_unsigned_mode_in_dev_mode(temp_log, monkeypatch):
    monkeypatch.delenv("ARGUS_AUDIT_HMAC_KEY", raising=False)
    logger = SecurityLogger(log_file=temp_log, hmac_key=None, strict_integrity=False)
    assert logger.integrity_enabled is False
    res = logger.log(1, "027", 0.92, "INFO", "ALLOW", "cam_01")
    assert res is None  # Unsigned mode returns None

    # File should have 7 columns
    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows[1]) == 7


# 21. Short key rejected
def test_short_key_rejected():
    with pytest.raises(ValueError, match="at least 256 bits"):
        parse_and_validate_hmac_key("short_secret_key_too_few_bytes")


# 22. Valid 256-bit keys accepted (both hex and raw)
def test_valid_keys_accepted():
    # 64-char hex string
    k1 = parse_and_validate_hmac_key("a" * 64)
    assert k1 is not None and len(k1) == 32

    # 32+ character UTF-8 string
    k2 = parse_and_validate_hmac_key("this_is_a_very_long_secret_key_exceeding_32_bytes")
    assert k2 is not None and len(k2) >= 32


# 23. Logger caller backward compatibility (positional and default kwargs)
def test_logger_caller_backward_compatibility(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)

    # Call with positional args (like pipeline)
    logger.log(1, "027", 0.92, "INFO", "ALLOW")  # camera_id defaults to "default"

    # Call with keyword args
    logger.log(track_id=2, identity="057", score=0.88, severity="INFO", decision="ALLOW", camera_id="cam_02")

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.verified_count == 2


# 24. Multiple sequential writes form a continuous chain
def test_multiple_sequential_writes_form_continuous_chain(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    hmacs = []
    for i in range(1, 11):
        h = logger.log(i, f"person_{i:03d}", 0.85 + (i * 0.01), "INFO", "ALLOW", f"cam_{i % 3}")
        hmacs.append(h)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.verified_count == 10
    assert result.last_verified_hmac == hmacs[-1]


# 25. Restart / reopen appends to existing chain correctly
def test_restart_reopen_appends_to_existing_chain(temp_log):
    # Session 1
    logger1 = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger1.log(1, "p1", 0.90, "INFO", "ALLOW", "cam_01")
    logger1.log(2, "p2", 0.91, "INFO", "ALLOW", "cam_01")

    # Session 2 (fresh logger instance, simulating process restart)
    logger2 = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger2.log(3, "p3", 0.92, "INFO", "ALLOW", "cam_01")
    logger2.log(4, "p4", 0.93, "INFO", "ALLOW", "cam_01")

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.verified_count == 4


# 26. Verifier never prints or discloses HMAC key
def test_verifier_never_discloses_hmac_key(temp_log):
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "027", 0.95, "INFO", "ALLOW", "cam_01")

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    result_dict = result.to_dict()

    # Confirm key never appears in dict representation
    assert TEST_HMAC_KEY not in str(result_dict)
    assert "hmac_key" not in result_dict


# 27. Tail truncation limitation documentation test
def test_tail_truncation_boundary_behavior(temp_log):
    """Document that tail truncation cannot be detected purely from file internal chain."""
    logger = SecurityLogger(log_file=temp_log, hmac_key=TEST_HMAC_KEY)
    logger.log(1, "001", 0.90, "INFO", "ALLOW", "cam_01")
    logger.log(2, "002", 0.91, "INFO", "ALLOW", "cam_01")
    logger.log(3, "003", 0.92, "INFO", "ALLOW", "cam_01")

    # Attacker removes the final row 3
    with open(temp_log, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    truncated_rows = rows[:-1]
    with open(temp_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(truncated_rows)

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)

    # Surviving prefix chain remains internally consistent
    # As explicitly documented: tail truncation cannot be detected without external state
    assert result.is_valid is True
    assert result.verified_count == 2


# Helper for multi-process writer test
def _worker_writer(log_path_str: str, key_str: str, worker_id: int, count: int):
    logger = SecurityLogger(log_file=log_path_str, hmac_key=key_str)
    for i in range(count):
        logger.log(
            track_id=worker_id * 1000 + i,
            identity=f"worker_{worker_id}_person_{i}",
            score=0.85,
            severity="INFO",
            decision="ALLOW",
            camera_id=f"cam_worker_{worker_id}",
        )


# 28. Real multi-process concurrency test
def test_multi_process_concurrent_writes_produce_valid_linear_chain(tmp_path):
    """Launch multiple concurrent OS processes writing to the same log file."""
    log_path = tmp_path / "concurrent_security_events.csv"
    num_workers = 4
    records_per_worker = 10

    processes = []
    for wid in range(num_workers):
        p = multiprocessing.Process(
            target=_worker_writer,
            args=(str(log_path), TEST_HMAC_KEY, wid, records_per_worker),
        )
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join(timeout=30.0)

    # Verify every process terminated successfully
    for p in processes:
        assert p.exitcode == 0

    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(log_path)

    # Under inter-process FileLock, all rows MUST form a single valid unbroken chain
    expected_total = num_workers * records_per_worker
    assert result.total_rows == expected_total
    assert result.verified_count == expected_total
    assert result.is_valid is True
    assert result.status == "VALID_SIGNED"
    assert len(result.discrepancies) == 0


# 29. Empty log verification
def test_empty_log_verification(temp_log):
    temp_log.touch()
    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(temp_log)
    assert result.is_valid is True
    assert result.status == "EMPTY"


# 30. File not found verification
def test_file_not_found_verification(tmp_path):
    non_existent = tmp_path / "does_not_exist.csv"
    verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
    result = verifier.verify_file(non_existent)
    assert result.is_valid is False
    assert result.status == "FILE_NOT_FOUND"


# 31. Verify real production security_events.csv baseline
def test_real_security_events_csv_baseline():
    """Verify that existing repository security_events.csv evaluates cleanly as LEGACY_UNSIGNED."""
    real_path = Path("outputs/logs/security/security_events.csv")
    if real_path.exists():
        verifier = AuditLogVerifier(hmac_key=TEST_HMAC_KEY)
        result = verifier.verify_file(real_path)
        assert result.is_valid is True
        assert result.status == "LEGACY_UNSIGNED"
        assert result.legacy_count > 0
        assert len(result.discrepancies) == 0
