# Security Implementation Report: Biometric Template Encryption at Rest (Finding U3)

**Finding Identifier**: Historical Finding U3 (Biometric Template Encryption at Rest)
**Security Status**: **U3 PASS — CLOSED**
**Numbering Designation**: **U3 — Biometric Template Encryption at Rest** (NOT SEC-10; SEC-10 Remains Undefined)
**Date**: 2026-09-16
**Repository**: `ARGUS_AI` (`chanuka8/argus-gait-recognition`)
**Branch**: `main`

---

## 1. Historical U3 Finding
In the foundational ARGUS AI thesis security and privacy audit (`docs/thesis_audit/08_security_and_privacy.md`, commit `b18e69f^`), historical finding **U3** was formally recorded:
> *"Template Encryption: Gallery stored as plaintext `.npy` | Not implemented | Templates extractable by anyone with file access | Encrypt gallery files at rest"*
> *"Asset: Gait Embeddings (256-dim) | HIGH | Gallery `.npy` files, in-memory"*
> *"STRIDE Threat: Extract biometric templates (Information Disclosure, HIGH)"*

Limitation **L6** in `docs/thesis_audit/13_limitations_and_gaps.md` and item 13 in `docs/FUTURE_ENGINEERING_GAP_AND_REMEDIATION_REPORT.md` further mandated encrypting gallery template files on disk.

---

## 2. Approved Security Scope
The approved implementation scope protects all persistent biometric vectors across ARGUS AI:
1. **VectorStore Gallery Containers**:
   - `models/gallery/gallery_features.enc` (Baseline gait gallery)
   - `models/live_gallery/gallery_features.enc` (Live enrolled gait gallery)
   - `models/appearance_gallery/gallery_features.enc` (ReID appearance gallery)
2. **EmbeddingDatabase Person Records**:
   - `data/embedding_db/persons/*.json` (Granular field-level AEAD vector encryption)

---

## 3. Storage Inventory
All persistent storage artifacts containing biometric vectors:

| Storage Location | Artifact Type | Modality & Dimension | Protection Mechanism | AAD Binding |
|---|---|---|---|---|
| `models/gallery/` | Binary Container | Gait (256D `float32`) | AES-256-GCM (`gallery_features.enc`) | `gallery_type`, `dimension`, `dtype`, `num_records`, `labels_sha256` |
| `models/live_gallery/` | Binary Container | Gait (256D `float32`) | AES-256-GCM (`gallery_features.enc`) | `gallery_type`, `dimension`, `dtype`, `num_records`, `labels_sha256` |
| `models/appearance_gallery/` | Binary Container | Appearance (512D `float32`) | AES-256-GCM (`gallery_features.enc`) | `gallery_type`, `dimension`, `dtype`, `num_records`, `labels_sha256` |
| `data/embedding_db/persons/` | JSON Record | Gait (256D) & Appearance (512D) | Field-level AEAD (`vector_encryption`) | `person_id`, `modality`, `dimension`, `dtype`, `model_version` |

---

## 4. Original Weakness
Prior to U3 implementation:
- `VectorStore.save()` wrote uncompressed, unencrypted IEEE-754 `float32` arrays via `np.save()`. Anyone with local filesystem read permissions could load `gallery_features.npy` via `np.load(allow_pickle=False)` and extract biometric templates.
- `EmbeddingDatabase.save_person()` persisted raw floating-point numbers in `"vector": [0.0119, 0.0363, ...]` within `data/embedding_db/persons/{person_id}.json`.

---

## 5. Implemented Cryptography
- **Primitive**: `cryptography.hazmat.primitives.ciphers.aead.AESGCM` (NIST SP 800-38D).
- **Cipher Suite**: AES-256-GCM (256-bit symmetric key).
- **Authentication**: AES-GCM uses GHASH as part of its authenticated-encryption construction and produces a 128-bit authentication tag appended to the ciphertext.
- **Nonce Generation**: 96-bit (12-byte) fresh random nonces generated exclusively via operating system CSPRNG (`os.urandom(12)`). Deterministic nonces, counters, and timestamp-derived nonces are strictly prohibited.
- **Serialization**: NumPy arrays serialized safely via `np.save(io.BytesIO(), array, allow_pickle=False)`. Pickling is strictly forbidden.

---

## 6. VectorStore Binary Container Format (`.enc`)
Container layout using big-endian network byte order (`!4sHH12sI`):
```text
+-------------------------------------------------------------------------------+
| Magic Bytes (4B): b"AGFE" (ARGUS Gallery Features Encrypted)                  |
+-------------------------------------------------------------------------------+
| Format Version (2B): uint16 big-endian (0x0001)                               |
+-------------------------------------------------------------------------------+
| Cipher Suite (2B): uint16 big-endian (0x0001 = AES-256-GCM)                   |
+-------------------------------------------------------------------------------+
| Nonce (12B): 96-bit CSPRNG bytes from os.urandom(12)                          |
+-------------------------------------------------------------------------------+
| AAD Length (4B): uint32 big-endian (max 65,536 bytes)                         |
+-------------------------------------------------------------------------------+
| Associated Authenticated Data (AAD): Canonical sorted UTF-8 JSON              |
+-------------------------------------------------------------------------------+
| Payload: AES-256-GCM Ciphertext (length identical to plaintext)               |
+-------------------------------------------------------------------------------+
| Authentication Tag (16B): 128-bit authentication tag                          |
+-------------------------------------------------------------------------------+
```

---

## 7. EmbeddingDatabase Encrypted Format
In `data/embedding_db/persons/*.json`, raw vectors are removed from disk and replaced by:
```json
{
  "embedding_id": "gait_person_01_1789383713_fecb7f",
  "person_id": "person_01",
  "modality": "gait",
  "embedding_dim": 256,
  "model_version": "v1.0.0",
  "status": "ACTIVE",
  "vector_encryption": {
    "version": 1,
    "cipher": "AES-256-GCM",
    "nonce": "<base64_encoded_12_bytes>",
    "ciphertext": "<base64_encoded_ciphertext_and_tag>",
    "dimension": 256,
    "modality": "gait",
    "model_version": "v1.0.0",
    "key_id": "argus_v1"
  }
}
```
In protected records, the plaintext `"vector"` list is deleted from the JSON serialization.

> [!NOTE]
> **Model Version Provenance**: The `model_version` field in `EmbeddingRecord` and `vector_encryption` is an application record metadata field and default version marker (`"v1.0.0"`), not an independent cryptographic model-provenance authority or hardware ledger. Authentication of `model_version` is provided via AEAD Associated Authenticated Data (AAD) binding during AES-GCM encryption.

---

## 8. Associated Authenticated Data (AAD) Design
AAD binds ciphertexts to contextual metadata using deterministic UTF-8 JSON serialization (`sort_keys=True, separators=(",", ":"), ensure_ascii=False`):
- **VectorStore AAD**:
  ```json
  {"dimension":256,"dtype":"float32","gallery_type":"baseline_gait","labels_sha256":"<hex>","num_records":10}
  ```
- **EmbeddingDatabase AAD**:
  ```json
  {"dimension":256,"dtype":"float32","modality":"gait","model_version":"v1.0.0","person_id":"person_01"}
  ```
- **Guarantees**:
  - Prevents swapping appearance templates (512D) into gait galleries (256D).
  - Prevents transferring an encrypted vector from one `person_id` record into another.
  - Prevents assigning a gait vector to the appearance modality.

---

## 9. Identity Labels Confidentiality Boundary
- **Boundary**: Subject identifiers (`person_id`) remain readable at rest in `gallery_labels.npy` and `data/embedding_db/persons/*.json`.
- **Binding**: `gallery_features.enc` computes `labels_sha256 = hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()`. Any alteration, substitution, or reordering of identity labels causes AEAD authentication failure (`BiometricDecryptionError`).
- **Privacy Notice**: Biometric feature vectors are encrypted; subject identifiers remain readable at rest. Full identity privacy requires future authorization.

---

## 10. Key Management
- **Primary Configuration**: `ARGUS_BIOMETRIC_ENCRYPTION_KEY` environment variable.
- **Format**: Exactly 32 decoded bytes (256 bits), supported as 64 hexadecimal characters or 32 raw bytes.
- **Validation**: Rejects short keys, malformed hex, or oversized keys without silent truncation or padding. Secrets are never exposed in exception messages or logs.
- **Keyfile Fallback**: Optional `.biometric_encryption.key` (must be administrator-provisioned; ignored by git via `.gitignore:27:*.key`). Auto-creation of keys or keyfiles is prohibited.

---

## 11. Key Separation (U2 vs U3)
- **Domain Separation**: `ARGUS_AUDIT_HMAC_KEY` (U2) and `ARGUS_BIOMETRIC_ENCRYPTION_KEY` (U3) must represent separate cryptographic secrets.
- **Enforcement**: If both keys resolve to identical bytes, `BiometricEncryptor` immediately raises `ConfigurationError("Cryptographic domain separation violation...")` without exposing secret bytes.

---

## 12. Nonce Handling
- Fresh 96-bit random nonce generated for every operation via `os.urandom(12)`.
- Empirically verified: 0 duplicate nonces across 1,000 synthetic operations.

---

## 13. Downgrade Prevention
1. **Mandatory Encrypted Loading**: If `gallery_features.enc` exists, decryption is mandatory. Any failure (`InvalidTag`, bad key, corrupt ciphertext, tampered labels) raises an exception and **strictly NEVER falls back to `.npy`**.
2. **Coexistence Precedence**: If both `.enc` and `.npy` exist, `.enc` takes precedence and a high-priority security warning is logged.
3. **Strict Mode Rule**: In strict mode (`ARGUS_STRICT_BIOMETRIC_ENCRYPTION=true`), unencrypted `.npy` files or plaintext JSON `"vector"` fields raise `RuntimeError`. Plaintext loading is permitted only in development mode when `.enc` does not exist.

---

## 14. Atomic Writes & Durability Boundary
All gallery and database writes execute under atomic temporary swap:
1. Acquire inter-process `FileLock` (`.{gallery}.lock`, timeout=10.0s).
2. Serialize and encrypt in memory (`io.BytesIO`).
3. Write ciphertext to temporary file on same filesystem (`.tmp_{uuid}`).
4. Flush Python buffers: `f.flush()`.
5. Flush OS file cache: `os.fsync(f.fileno())`.
6. Atomic rename: `os.replace(tmp_file, target_file)`.
7. Release `FileLock`.
- **Durability Limitation**: Directory-entry metadata durability after `os.replace()` depends on OS/filesystem journaling semantics.

---

## 15. Migration Tool
- **Utility**: [`tools/security/migrate_gallery_encryption.py`](file:///e:/ARGUS_AI/tools/security/migrate_gallery_encryption.py).
- **Execution Policy**: Standalone CLI utility; **never runs automatically on application startup**.
- **Supported Modes**:
  - `--dry-run`: Scans and reports without modifying disk.
  - `--migrate`: Encrypts `.npy` galleries and JSON records with test decryption and bit-for-bit equivalence verification before atomic replacement.
  - `--verify`: Verifies existing ciphertexts against key.
  - `--purge-plaintext`: Explicit operator action to remove legacy plaintext files (requires `--migrate` and verified existing encrypted container).
- **Safety**: Tested strictly on synthetic fixtures; real repository biometric files remained untouched in clean-state verification.

---

## 16. Focused Test Result
- **Test File**: [`tests/integration/backend/test_biometric_template_encryption.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_biometric_template_encryption.py)
- **Execution Command**:
  ```powershell
  .venv\Scripts\python.exe -m pytest tests/integration/backend/test_biometric_template_encryption.py -v
  ```
- **Result**:
  - **Passed**: 51
  - **Failed**: 0
  - **Skipped**: 0
  - **Duration**: 1.30s
  - **Exit Code**: 0

---

## 17. Full Backend Regression Result
- **Execution Command**:
  ```powershell
  .venv\Scripts\python.exe -m pytest tests/integration/backend/ -q
  ```
- **Result**:
  - **Passed**: 325 (274 baseline + 51 U3 tests)
  - **Failed**: 0
  - **Skipped**: 0
  - **Duration**: 425.51s (0:07:05)
  - **Exit Code**: 0

---

## 18. Performance Reassessment
- Single 256D vector (encrypt + decrypt): **0.0035 ms**
- Live gallery (264 vectors, encrypt + decrypt): **0.31 ms**
- Full baseline gallery (13,544 vectors, encrypt + decrypt): **15.72 ms**
- **Steady-State Recognition Impact**: **No per-frame storage decryption occurs during steady-state recognition after successful gallery preload**. Preloaded vectors reside in memory (`self.gallery_features`) for cosine similarity matching.

---

## 19. Recognition Equivalence Result
Verified in `test_recognition_equivalence`:
$$\text{DecryptedFeatures} \equiv \text{OriginalFeatures}$$
Cosine similarity dot products between probes and original vs decrypted templates are identical ($\text{atol} < 10^{-7}$), producing identical rank-1 recognition matches.

---

## 20. Plaintext Persistence Audit
Exhaustive search for persistent biometric vectors:
- `np.save`: Restricted to in-memory `io.BytesIO()` encryption buffers, labels persistence, and development mode fallback with strict mode gating.
- `"vector":`: Removed from disk serialization in protected `EmbeddingRecord` JSONs.
- `data/embedding_db/`: All newly protected records serialize only `"vector_encryption"`.

---

## 21. Files Modified & Created

### Created:
1. [`security_layer/biometric_encryption.py`](file:///e:/ARGUS_AI/security_layer/biometric_encryption.py): Core AES-256-GCM engine, AAD canonicalization, container packing/unpacking, key separation.
2. [`tools/security/migrate_gallery_encryption.py`](file:///e:/ARGUS_AI/tools/security/migrate_gallery_encryption.py): Standalone operator migration CLI with purge safeguards and validation checks.
3. [`tests/integration/backend/test_biometric_template_encryption.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_biometric_template_encryption.py): 51 integration, security, and purge safeguard test cases including pre/post storage fingerprint guard fixture.
4. [`docs/security/u3_implementation_report.md`](file:///e:/ARGUS_AI/docs/security/u3_implementation_report.md): Implementation security report.

### Modified:
1. [`storage/vector_store.py`](file:///e:/ARGUS_AI/storage/vector_store.py): Encrypted container save/load, downgrade prevention, atomic temp swap, `validate_gallery_files()`.
2. [`storage/embedding_database.py`](file:///e:/ARGUS_AI/storage/embedding_database.py): Field-level AEAD vector encryption, atomic writes, queryable metadata.
3. [`cli.py`](file:///e:/ARGUS_AI/cli.py): Routed `check-docs` gallery check to `validate_gallery_files()`.
4. [`tests/conftest.py`](file:///e:/ARGUS_AI/tests/conftest.py): Registered `test_biometric_template_encryption.py` in `security_test_modules`.

---

## 22. Test Isolation Incident and Resolution

### Historical Transient Side Effect:
Earlier U3 testing temporarily created encrypted artifacts in real gallery directories:
- `models/live_gallery/gallery_features.enc`
- `models/appearance_gallery/gallery_features.enc`

### Root Cause:
`EmbeddingDatabase.__init__` defaults to `gait_gallery_dir="models/live_gallery"` and `appearance_gallery_dir="models/appearance_gallery"`. In initial test executions (specifically tests 24, 25, 28, 29, 33, 47), `EmbeddingDatabase` was instantiated passing only `db_dir=str(tmp_path)` and `encryptor=encryptor`, without explicitly overriding `gait_gallery_dir` and `appearance_gallery_dir`. When `db.add_embeddings()` was executed, `self._sync_vector_stores()` invoked `self.gait_store.save()` and `self.appearance_store.save()`, persisting `.enc` containers into the production gallery directories.

### Corrective Changes:
1. **Isolated Test Paths**: All `EmbeddingDatabase` initializations in `tests/integration/backend/test_biometric_template_encryption.py` were corrected to pass isolated temporary directories: `gait_gallery_dir=str(tmp_path / "live_gal")` and `appearance_gallery_dir=str(tmp_path / "app_gal")`.
2. **Hard Safety Guard**: Added `real_biometric_storage_guard` fixture (`autouse=True`, `scope="module"`) that captures the cryptographic SHA-256 fingerprint and file sizes of all files across `models/gallery`, `models/live_gallery`, `models/appearance_gallery`, and `data/embedding_db` before test execution and asserts bit-for-bit equality after test execution.
3. **Artifact Removal**: The transient `.enc` files in `models/live_gallery/` and `models/appearance_gallery/` were removed.
4. **Strengthened Migration Purge Safety**: Modified `tools/security/migrate_gallery_encryption.py` to enforce that `--purge-plaintext` is strictly rejected unless `--migrate` is active, and verifies the written encrypted container with `verify_vector_store` prior to unlinking any plaintext `.npy` file. Added tests 49, 50, 51 covering purge requirements and failure resistance.

### Clean-State Verification Statement:
> Earlier U3 testing temporarily created encrypted artifacts in real gallery directories. The responsible test/configuration path was corrected. A subsequent clean-state full test run produced zero writes to real biometric directories, and no persistent biometric data modifications remain.

---

## 23. Residual Security Limitations
- **Key Storage**: Security depends on protection of `ARGUS_BIOMETRIC_ENCRYPTION_KEY`. An attacker who steals the key can decrypt all templates.
- **Physical Drive Shredding**: Overwriting filesystem files does not guarantee physical block erasure on SSD wear-leveling controllers.

---

## 24. Rollback Limitation
- **Freshness**: AES-256-GCM guarantees confidentiality and authenticity under the key, but does **NOT** provide freshness.
- **Official Finding**: **encrypted biometric rollback independently detectable: NO** (without external monotonic counters or secure ledger anchoring).

---

## 25. Memory Security Boundary
- **Process Memory**: Encryption at rest does **NOT** protect biometric vectors once decrypted in application memory (`self.gallery_features`), against process compromise, or against root/kernel memory dumping (`/proc/{pid}/mem`).
- **Zeroization**: Python and NumPy garbage collection does not perform cryptographic zeroization upon array deallocation.

---

## 26. SEC Numbering Status
- Historical finding U3 remains designated as **`U3 — Biometric Template Encryption at Rest`**.
- **SEC-10 remains undefined** in the repository. U3 was NOT renamed to SEC-10.

---

## 27. Final Verdict

# `U3 PASS — CLOSED`
