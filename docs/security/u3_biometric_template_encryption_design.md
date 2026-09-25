# Architectural Security Audit & Design: Biometric Template Encryption at Rest (Finding U3)

**Finding Identifier**: Historical Finding U3 (Biometric Template Encryption at Rest)
**Current Status**: **AUDIT & DESIGN CORRECTION ONLY** (Implementation NOT Authorized)
**Security Classification**: Historical Finding U3 (NOT SEC-10; SEC-10 Remains Undefined)
**Date**: 2026-09-16
**Repository**: `ARGUS_AI` (`chanuka8/argus-gait-recognition`)
**Branch**: `main`

---

## 1. Executive Summary & Authoritative Historical Evidence

### 1.1 Historical Authoritative Evidence
In the foundational ARGUS AI thesis security and privacy audit (`docs/thesis_audit/08_security_and_privacy.md`, commit `b18e69f^`), historical finding **U3** was formally recorded under §8.1 (Audit Matrix) and §8.2 (STRIDE Analysis):
> *"Template Encryption: Gallery stored as plaintext `.npy` | Not implemented | Templates extractable by anyone with file access | Encrypt gallery files at rest"*
> *"Template Theft: Raw embeddings in plaintext files | Not implemented | Embeddings extractable | Implement biometric template protection"*
> *"Asset: Gait Embeddings (256-dim) | HIGH | Gallery `.npy` files, in-memory"*
> *"Asset: Gallery Templates | HIGH | `models/gallery/` and `models/live_gallery/`"*
> *"STRIDE Threat: Extract biometric templates (Information Disclosure, HIGH)"*

Furthermore, in `docs/thesis_audit/13_limitations_and_gaps.md` (commit `b18e69f^`), limitation **L6** and research gap **G1** stated:
> *"L6: No template encryption (Security, HIGH) — Gallery stored as plaintext NumPy files"*
> *"G1: No biometric template protection (HIGH) — Raw embeddings stored without any protection scheme"*

In the engineering remediation roadmap (`docs/FUTURE_ENGINEERING_GAP_AND_REMEDIATION_REPORT.md`, commit `14a3be1`), items 11 and 13 prioritized:
> *"11. Implement atomic temp file swapping for gallery saves."*
> *"13. Encrypt `.npy` gallery template files on disk."*

### 1.2 Current Source-Code Confirmation & Scope Gap Discovery
Initial audits focused exclusively on `storage/vector_store.py` (`models/*gallery*`). However, an exhaustive repository audit revealed a **critical scope gap**:
- **Granular Person Database**: [`storage/embedding_database.py`](file:///e:/ARGUS_AI/storage/embedding_database.py) persists individual person records to `data/embedding_db/persons/{person_id}.json`. These JSON records store **unencrypted raw vector arrays** (`"vector": [0.0119, 0.0363, ...]`) for every enrolled subject.
- **Authoritative Status**: `EmbeddingDatabase` is an active production component. It is the authoritative persistence layer for continual learning and the missing person workflow (`MissingPersonVideoProcessor`), and it actively populates `VectorStore` via `_sync_vector_stores()`.
- **Finding**: Encrypting only `VectorStore`'s `.npy` files while leaving `data/embedding_db/persons/*.json` unencrypted would leave all enrolled biometric signatures extractable from disk. **U3 must cover all persistent biometric vector stores across both `VectorStore` and `EmbeddingDatabase`**.

---

## 2. Complete Persistent Biometric Storage Inventory

The table below documents every persistent storage location containing biometric data across the repository:

| Storage Location | File Format | Contains Raw Biometric Vectors? | Modality & Dimension | Production Status | Writer Component | Reader Component | In U3 Scope? |
|---|---|---|---|---|---|---|---|
| `models/gallery/` | `.npy` (binary) | **YES** | Gait (256D `float32`, 13,544 rows) | Active (Baseline gallery) | `VectorStore.save()` / `build_gallery.py` | `VectorStore.load()`, `GaitService`, Pipelines | **YES (Primary)** |
| `models/live_gallery/` | `.npy` (binary) | **YES** | Gait (256D `float32`, 264 rows) | Active (Live enrolled) | `VectorStore.save()` / `EmbeddingDatabase` | `VectorStore.load()`, `GaitService` | **YES (Primary)** |
| `models/appearance_gallery/` | `.npy` (binary) | **YES** | Appearance (512D `float32`, 206 rows) | Active (ReID gallery) | `VectorStore.save()` / `EmbeddingDatabase` | `VectorStore.load()`, `GaitService` | **YES (Primary)** |
| `models/*/gallery_labels.npy` | `.npy` (string) | NO (Identities) | Subject ID strings (`person_id`) | Active (Identities) | `VectorStore.save()` | `VectorStore.load()`, `GaitService` | **Bound via AAD** |
| `models/*/gallery_metadata.json`| `.json` (text) | NO (Metadata) | Counts, status (`ACTIVE`), timestamps | Active (Metadata) | `VectorStore.save()` | `VectorStore.load()`, `MatchingStep` | Optional Container |
| `data/embedding_db/persons/` | `.json` (text) | **YES** | Gait (256D) & Appearance (512D) | Active (Authoritative DB)| `EmbeddingDatabase.save_person()` | `EmbeddingDatabase`, `api/v1/router.py` | **YES (Mandatory)** |
| `outputs/evidence/` | `.npz` (compressed) | NO (Crops/GEI) | Visual silhouettes / GEI images | Active (Audit evidence) | `OperationalEvidenceManager` | Evidence inspection tools | Out of Scope (Evidence) |
| `outputs/logs/security_events.csv`| `.csv` (text) | NO (Scores/IDs) | Similarity scores & decisions | Active (Audit log) | `SecurityLogger.log()` | `AuditLogVerifier` | Out of Scope (U2 HMAC) |
| `models/weights/` | `.pth` / `.pt` | NO (CNN weights)| Neural network layer weights | Active (Model weights) | Training scripts | PyTorch / ONNX backends | Out of Scope (Weights) |

---

## 3. Actual U3 Security Boundary & Data Scope

Based on authoritative historical evidence and active system architecture, the correct U3 scope is **Scope D: Every persistent copy of biometric vectors**:
1. **Gait Biometric Templates**: All 256D floating-point embeddings stored in `models/gallery/gallery_features.npy`, `models/live_gallery/gallery_features.npy`, and `data/embedding_db/persons/*.json`.
2. **Appearance Biometric Templates**: All 512D floating-point embeddings stored in `models/appearance_gallery/gallery_features.npy` and `data/embedding_db/persons/*.json`.
3. **Identity Labels (`person_id`) Confidentiality Boundary**:
   - Historical U3 focuses strictly on **biometric template protection** (preventing theft of immutable biological characteristics).
   - In the proposed baseline design: **Biometric vectors are encrypted; subject identifiers (`person_id`) remain readable at rest**, but are cryptographically bound to the vector ciphertext via AEAD Associated Data (AAD) to prevent template substitution.
   - If subject identifier confidentiality is also required, full gallery container bundling (`gallery_bundle.enc`) and full person record encryption (`{person_id}.enc`) may be enabled as an option.

---

## 4. Current Vulnerability Confirmation

Direct verification of physical storage:
- **`models/gallery/gallery_features.npy`**: 13,869,184 bytes of uncompressed IEEE-754 `float32` binary data. Loading requires zero keys:
  ```python
  import numpy as np
  feats = np.load("models/gallery/gallery_features.npy", allow_pickle=False)  # Extracts 13,544 biometric templates
  ```
- **`data/embedding_db/persons/Legit_Person_123.json`**: Plaintext JSON containing raw floating-point arrays:
  ```json
  "vector": [0.01199, 0.03639, -0.00401, 0.02977, 0.03872, ...]
  ```
  An adversary with read access to `data/embedding_db/` can extract vectors directly without touching `models/`.

---

## 5. Threat Model Analysis

| Threat ID | Threat Scenario | Current Mitigation | Does Encryption At Rest Mitigate? | Residual Exposure / Limitations |
|---|---|---|---|---|
| **T-1** | **Local Filesystem Read**: Unprivileged process or local account reads `.npy` or `.json` files. | OS File Permissions | **YES** (Attacker obtains only ciphertext) | Does not protect if attacker steals key or reads process RAM. |
| **T-2** | **Stolen Disk / Copied VM Image**: Physical drive or backup volume copied. | None | **YES** (Files unreadable without encryption key) | Key must not be stored on the same unencrypted volume. |
| **T-3** | **Backup / Archive Disclosure**: Unencrypted backups leaked or exposed in cloud buckets. | None | **YES** (Backed-up templates are encrypted ciphertexts) | Key backup must be separated from data backups. |
| **T-4** | **Malicious Insider**: System operator with file read access dumps biometric database. | Access logging | **YES** (Cannot dump raw biometric vectors without key) | Operator with root access can inspect process memory/env. |
| **T-5** | **Path Traversal Escape**: Traversal vulnerability reads gallery or database files. | Path validation | **YES** (Defense-in-depth: traversed files yield ciphertext) | Path traversal reaching the keyfile defeats this defense. |
| **T-6** | **Accidental Git Inclusion**: Gallery file committed to public repository. | `.gitignore` rules | **YES** (Ciphertext cannot be converted to vectors) | Labels or metadata could leak if not bundled. |
| **T-7** | **Cross-Gallery Substitution**: Attacker swaps appearance gallery into gait gallery path. | None | **YES** (AEAD AAD check detects dimension/type mismatch) | Attacker with valid key can forge valid ciphertexts. |
| **T-8** | **Label Desynchronization Swap**: Attacker replaces `gallery_labels.npy` with different IDs. | None | **YES** (AEAD AAD includes `labels_sha256`; verification fails) | Attacker with valid key can re-sign tampered labels. |
| **T-9** | **Replay / Rollback of Older Gallery**: Attacker replaces active gallery with older valid one. | None | **NO** (Ciphertext authenticates under key; no freshness check)| **Encrypted-gallery rollback independently detectable: NO**. |
| **T-10**| **Memory Scraping**: Attacker inspects `/proc/{pid}/mem` or attaches debugger to Python. | None | **NO** (Templates must be decrypted in RAM for cosine math) | Homomorphic encryption or SGX enclaves required (out of scope). |
| **T-11**| **Full Process Compromise (RCE)**: Attacker executes arbitrary code within ARGUS server. | Input validation | **NO** (Process has the environment key and memory space) | OS-level sandboxing required (out of scope for U3). |
| **T-12**| **Ransomware / File Deletion**: Attacker deletes gallery files to deny service. | None | **NO** (Encryption does not prevent file deletion) | Requires offline disaster-recovery backups. |

---

## 6. Verification of Existing Access Controls

Existing security controls in ARGUS provide layer-4 and layer-7 protection, but do NOT provide data-at-rest protection:
1. **API Authentication & RBAC (SEC-02)**: Restricts API access, but provides zero defense against local filesystem access.
2. **Session Ownership (SEC-04)**: Restricts upload session modifications, but does not protect static gallery directories.
3. **Path Traversal Boundary Validation (SEC-01, SEC-05)**: Restricts HTTP path traversal.
4. **Audit Log HMAC Chaining (U2)**: Protects event log integrity, but does not encrypt biometric templates.

**Conclusion**: Defense-in-depth at the storage tier via authenticated encryption at rest is strictly required.

---

## 7. Cryptographic Architecture Evaluation

### 7.1 Cipher Primitive Comparison
- **Option A: Fernet (`cryptography.fernet.Fernet`)**:
  - Primitive: AES-128 in CBC mode with PKCS#7 padding + HMAC-SHA256.
  - Evaluation: Proven and simple, but lacks native AEAD associated data support, adds 33% base64 memory overhead, and exhibits 17.3x slower decryption in micro-benchmarks.
- **Option B: AES-256-GCM (`cryptography.hazmat.primitives.ciphers.aead.AESGCM`)**:
  - Primitive: **AES in Galois/Counter Mode with 256-bit symmetric key and 128-bit authentication tag**.
  - Authentication Function: **GHASH** (universal hashing over GF($2^{128}$)). *(Note: POLYVAL is not used in standard AES-GCM; POLYVAL is used in AES-GCM-SIV / XTS)*.
  - Evaluation: Industry standard for biometric templates. Hardware-accelerated via AES-NI and CLMUL instructions. Native support for Associated Authenticated Data (AAD) enables binding identities and metadata to the ciphertext.

### 7.2 Cryptographic Recommendation
**AES-256-GCM** is recommended for `VectorStore` gallery containers and `EmbeddingDatabase` vector fields.

---

## 8. Encrypted Container Format & Strict Parser Specification

### 8.1 Binary Container Specification (`.enc`)
The container utilizes big-endian network byte order (`!4sHH12sI`):
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
| AAD Header Length (4B): uint32 big-endian (max 65,536 bytes)                  |
+-------------------------------------------------------------------------------+
| Associated Data (AAD): Canonical sorted UTF-8 JSON string                     |
+-------------------------------------------------------------------------------+
| Encrypted Payload: AES-256-GCM Ciphertext (length identical to plaintext)     |
+-------------------------------------------------------------------------------+
| Authentication Tag (16B): 128-bit GHASH authentication tag                    |
+-------------------------------------------------------------------------------+
```

### 8.2 Strict Parser & Validation Rules
1. **Minimum Length Check**: Total file length must be $\ge 24 \text{ (header)} + \text{aad\_len} + 16 \text{ (tag)}$. Files $< 40$ bytes immediately raise `ValueError("Corrupted container: file too short")`.
2. **Magic Validation**: Bytes `0..4` must exactly match `b"AGFE"`. Mismatches raise `ValueError("Invalid container magic")`.
3. **Version Validation**: Version must be `1`. Values $> 1$ raise `ValueError("Unsupported container version")`.
4. **Cipher Suite Validation**: Suite must be `1`. Other values raise `ValueError("Unsupported cipher suite")`.
5. **Nonce Validation**: Nonce is strictly 12 bytes (`raw_bytes[8:20]`).
6. **AAD Length Bounds Check**: `aad_len` unpacked from bytes `20..24`. Must be $\le 65,536$ bytes (64 KiB). Total file length must be $\ge 24 + \text{aad\_len} + 16$.
7. **Decryption Invocation**: Ciphertext + 16-byte tag extracted from `raw_bytes[24 + aad_len:]`. Passed with `nonce` and `aad_bytes` to `aesgcm.decrypt()`.
8. **Tamper Rejection**: Any alteration to magic, header, AAD, ciphertext, or tag triggers `cryptography.exceptions.InvalidTag`.

### 8.3 Exact Container Size Overhead
Unlike claims of "0% expansion", the authenticated container introduces header, AAD, and tag overhead. Because AES-GCM operates in Counter (CTR) mode, ciphertext length equals plaintext length. The exact overhead consists of:
- Fixed header: 24 bytes
- AAD JSON payload: ~170–176 bytes
- GHASH authentication tag: 16 bytes
- **Total Fixed Overhead**: $\approx 210–216$ bytes per container.

| Target Storage Artifact | Plaintext Size | Encrypted Container Size | Absolute Overhead | Percentage Expansion |
|---|---|---|---|---|
| **Single 256D Vector (`.npy`)** | 1,152 bytes | 1,360 bytes | **+208 bytes** | **+18.06%** |
| **Current Gait Gallery (`13544 x 256`)** | 13,869,184 bytes (13.23 MiB)| 13,869,396 bytes (13.23 MiB)| **+212 bytes** | **+0.0015%** |
| **Current Live Gallery (`264 x 256`)** | 270,464 bytes (264.1 KiB) | 270,674 bytes (264.3 KiB) | **+210 bytes** | **+0.078%** |
| **Current Appearance Gallery (`206 x 512`)**| 422,016 bytes (412.1 KiB) | 422,232 bytes (412.3 KiB) | **+216 bytes** | **+0.051%** |

---

## 9. Nonce Safety & Collision Resistance

- **Nonce Source**: 96-bit (12-byte) random nonces generated exclusively via operating system CSPRNG (`os.urandom(12)`).
- **Deterministic Nonce Prohibition**: Nonces must **NEVER** be derived from timestamps, subject IDs, or counters, avoiding reuse upon clock rollback or duplicate enrollments.
- **NIST SP 800-38D Compliance**: Under §8.2.2, a 96-bit random nonce is permitted for up to $2^{32}$ encryptions under a single key. For 1,000,000 gallery write operations, the collision probability is $P \approx \frac{10^{12}}{2^{97}} \approx 6.3 \times 10^{-18}$ (negligible).
- **Inter-Process Serialization**: All gallery writes in `VectorStore.save()` are serialized using inter-process `FileLock`, preventing race conditions across concurrent processes.

---

## 10. Associated Authenticated Data (AAD) Specification

### 10.1 AAD Fields & Canonical Serialization
To guarantee reproducible byte-for-byte AAD calculation during decryption:
```python
aad_payload = {
    "dimension": int(features.shape[1]),
    "dtype": str(features.dtype),
    "gallery_type": str(gallery_type),  # "gait" or "appearance"
    "labels_sha256": hashlib.sha256(canonical_labels_bytes).hexdigest(),
    "num_records": int(len(features)),
}
aad_bytes = json.dumps(aad_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
```
- **Labels Canonicalization**: `canonical_labels_bytes` is computed by encoding each label as UTF-8 separated by newline (`\n`), ensuring platform-independent hashing.
- **Model Version Policy**: `model_version` is **omitted** from `VectorStore` AAD because `VectorStore` is a storage abstraction with no authoritative model version state. In `EmbeddingDatabase`, `model_version` is included per-record.

### 10.2 What AAD Actually Protects
- **Cross-Gallery Swap**: Swapping appearance gallery (512D) into gait gallery (256D) is rejected (dimension/type mismatch).
- **Label Substitution**: Modifying or swapping `gallery_labels.npy` independently causes `labels_sha256` verification to fail.
- **Cross-Site Copying**: Adding a deployment identifier into AAD prevents copying ciphertexts between sites.

---

## 11. Rollback & Replay Limitation

> [!WARNING]
> **Rollback Vulnerability Boundary**: AES-256-GCM proves that ciphertext was generated by a holder of the key and has not been altered. It does **NOT** prove freshness. An attacker with filesystem write access can replace `gallery_features.enc` with a valid older version of `gallery_features.enc` from last week. Authentication will succeed.
>
> **Official Finding**: **Encrypted-gallery rollback independently detectable: NO** (without external trusted version state, monotonic sequence hardware, or remote ledger anchoring).

---

## 12. Plaintext Downgrade Prevention Rules

To prevent downgrade attacks where an attacker forces the system into unencrypted mode:
1. **Encrypted Presence Precedence**: If `gallery_features.enc` exists, decryption is attempted. If decryption fails (`InvalidTag`, bad key, corrupt file), the application **MUST RAISE AN EXCEPTION**. It is strictly forbidden to fall back to `.npy` or `.bak`.
2. **Coexistence Alert**: If both `.enc` and `.npy` exist on disk, `.enc` is loaded. A security alert is logged requiring the administrator to delete the plaintext copy.
3. **Strict Mode Rule**: In production mode (`ARGUS_MODE=production` or `ARGUS_STRICT_BIOMETRIC_ENCRYPTION=true`), loading unencrypted `.npy` files raises `RuntimeError`. Plaintext fallback is permitted **only** in development mode when `.enc` does not exist.

---

## 13. Plaintext Backup Security & Migration Safeguards

The initial migration concept proposed creating `gallery_features.npy.bak` beside the encrypted file. This creates an unencrypted biometric vulnerability on disk.

### Corrected Migration Safeguard Policy:
1. **No In-Place Plaintext Backups**: The migration CLI tool shall **NOT** store unencrypted `.bak` files inside the production `models/` directory.
2. **External Operator Backup**: The migration tool mandates that the administrator create an external encrypted or permission-restricted backup directory outside the webroot prior to running migration.
3. **Atomic Replacement with Verification**:
   - Encrypt `.npy` to temporary file `gallery_features.enc.tmp`.
   - Perform test decryption and verify bit-for-bit equality against original.
   - Atomically replace: `os.replace("gallery_features.enc.tmp", "gallery_features.enc")`.
   - Automatically shred or securely delete the original plaintext file once confirmed, or prompt the operator for immediate removal.
4. **Strict Mode Scanner**: `validate_gallery_files()` in production scans for any `.bak` or `.npy` files and raises an alert if unencrypted templates remain.

---

## 14. Atomic Write Architecture & Durability Boundary

Currently, `VectorStore.save()` writes directly to target paths, risking corruption during process termination.

### Proposed Atomic Write Lifecycle:
```text
1. Acquire Inter-Process FileLock (.{gallery_dir.name}.lock, timeout=10.0s)
    ↓
2. Serialize NumPy arrays to in-memory bytes buffer (io.BytesIO)
    ↓
3. Encrypt buffer with AES-256-GCM + CSPRNG nonce + AAD metadata
    ↓
4. Write ciphertext to temporary file on SAME filesystem: {target}.tmp_{uuid}
    ↓
5. Flush Python stream buffers: file.flush()
    ↓
6. Flush operating system file cache: os.fsync(file.fileno())
    ↓
7. Atomic filesystem swap: os.replace(tmp_path, target_path)
    ↓
8. Release FileLock
```
- **Durability Semantics**: `os.replace` is atomic on POSIX and Windows NTFS (on the same volume). Calling `os.fsync()` commits dirty blocks to storage before renaming. On POSIX, directory entry durability strictly requires parent directory `fsync`, which is documented as a residual limitation.

---

## 15. Key Management & Cryptographic Key Separation

### 15.1 Configuration & Resolution
1. Constructor parameter: `encryption_key: str | bytes | None`
2. Environment variable: `ARGUS_BIOMETRIC_ENCRYPTION_KEY` (64 hex characters or 32 raw bytes, providing 256 bits of entropy).
3. Fallback file: `.biometric_encryption.key` (read-only; ignored via `.gitignore:27:*.key`).

### 15.2 Mandatory Key Separation
> [!IMPORTANT]
> **Key Separation Requirement**: The U2 audit log HMAC key (`ARGUS_AUDIT_HMAC_KEY`) and the U3 biometric encryption key (`ARGUS_BIOMETRIC_ENCRYPTION_KEY`) serve fundamentally distinct security domains. **They must never use the same secret key**.
> The initialization routine must enforce:
> $$\text{ARGUS\_AUDIT\_HMAC\_KEY} \ne \text{ARGUS\_BIOMETRIC\_ENCRYPTION\_KEY}$$
> Key reuse triggers an immediate `ConfigurationError`.

---

## 16. Granular JSON Store Design (`EmbeddingDatabase`)

Because `data/embedding_db/persons/{person_id}.json` is an active authoritative store containing raw vector float arrays, U3 must protect it.

### Evaluated Architectural Options:
- **Option A (Whole-File Record Encryption)**: Encrypt the entire `{person_id}.json` document as `{person_id}.enc`.
  - Disadvantage: Precludes querying person metadata, status, or timestamps without decrypting the entire file.
- **Option B (Field-Level Vector AEAD Encryption — RECOMMENDED)**:
  - Keep high-level identity metadata (`person_id`, `status`, `created_at`) in structured JSON.
  - Encrypt only the `"vector"` field of each `EmbeddingRecord`:
    ```json
    {
      "embedding_id": "gait_Legit_Person_123_1789383713_fecb7f",
      "person_id": "Legit_Person_123",
      "modality": "gait",
      "embedding_dim": 256,
      "encrypted_vector": "4f8a...<hex_ciphertext_and_tag>",
      "nonce": "a1b2c3d4e5f60718293a4b5c",
      "status": "ACTIVE"
    }
    ```
  - AAD binds `person_id`, `embedding_id`, and `modality` to prevent vector swapping between records.
  - Allows lightweight metadata operations (`list_all_persons()`, status checks) without decrypting vectors, while completely denying vector disclosure to unauthenticated disk readers.

---

## 17. Performance & Latency Reassessment

Synthetic micro-benchmarks on the ARGUS environment (Intel CPU, Windows 64-bit, Python 3.11):
- Single 256D vector (enc+dec): **0.0035 ms**
- Live gallery (264 vectors, enc+dec): **0.31 ms**
- Full baseline gallery (13,544 vectors, enc+dec): **15.72 ms**

### Steady-State Frame Recognition Conclusion
- Previous reports claimed: *"Live Recognition Frame Rate Impact = 0.00 ms"*.
- **Corrected Evidence-Based Conclusion**: Because `GaitService` decrypts gallery files once upon startup into process memory (`self.gallery_features`), **no per-frame storage decryption is performed during steady-state recognition after successful preload**. Steady-state cosine matching executes entirely in RAM.

---

## 18. Memory Security Boundary

In accordance with strict evidence-based reporting policies:
1. **Decrypted RAM Boundary**: Encryption at rest protects physical non-volatile storage. It does **NOT** protect biometric vectors once decrypted in Python process heap memory.
2. **Process Inspection**: An adversary with root/kernel access or ability to read `/proc/{pid}/mem` can dump in-memory vectors.
3. **Garbage Collection**: Python and NumPy do not guarantee immediate cryptographic memory zeroization upon object deallocation.

---

## 19. Locked Model Pipeline Compatibility

The locked gait recognition pipeline remains **100% UNTOUCHED**:
$$\text{Person Detection} \to \text{ByteTrack} \to \text{UNet Silhouette} \to \text{GEI} \to \text{ByGaitLight CNN} \to \text{256D} \to \text{Cosine}$$

Decryption occurs strictly at the storage boundary inside `VectorStore.load()` and `EmbeddingDatabase`. The returned arrays have identical types (`np.float32`), dimensions, and numerical values:
$$\text{OriginalArray} == \text{AESGCM.decrypt}(\text{AESGCM.encrypt}(\text{OriginalArray}))$$
Cosine similarity matching and recognition thresholds operate without modification.

---

## 20. Corrected Proposed File Set

### Production Files:
1. [`security_layer/biometric_encryption.py`](file:///e:/ARGUS_AI/security_layer/biometric_encryption.py) [NEW, MANDATORY]: Core `BiometricEncryptor` implementing AES-256-GCM container packing, unpacking, field-level vector encryption, AAD validation, CSPRNG nonce management, and key separation checks.
2. [`storage/vector_store.py`](file:///e:/ARGUS_AI/storage/vector_store.py) [MODIFY, MANDATORY]: Integrate container encryption, atomic temporary file replacement, `FileLock`, and downgrade prevention.
3. [`storage/embedding_database.py`](file:///e:/ARGUS_AI/storage/embedding_database.py) [MODIFY, MANDATORY]: Integrate field-level AEAD vector encryption for `data/embedding_db/persons/*.json`.
4. [`cli.py`](file:///e:/ARGUS_AI/cli.py) [MODIFY, MINOR]: Update `argus check-docs` around line 680 to call `validate_gallery_files()`.

### Migration Tool:
1. [`tools/security/migrate_gallery_encryption.py`](file:///e:/ARGUS_AI/tools/security/migrate_gallery_encryption.py) [NEW, MANDATORY]: CLI tool to migrate both `VectorStore` gallery files and `EmbeddingDatabase` JSON records.

### Test Files:
1. [`tests/integration/backend/test_biometric_template_encryption.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_biometric_template_encryption.py) [NEW, MANDATORY]: Comprehensive unit and integration test suite.
2. [`tests/conftest.py`](file:///e:/ARGUS_AI/tests/conftest.py) [MODIFY, MANDATORY]: Register test module in `security_test_modules`.

---

## 21. Corrected Comprehensive Test Plan

The future test plan shall include 32 focused test cases:
1. `test_encrypt_decrypt_valid_256d_template`: Bit-for-bit round-trip array equality.
2. `test_encrypt_decrypt_valid_512d_appearance_template`: Bit-for-bit equality for appearance vectors.
3. `test_full_gallery_round_trip_preserves_dtype_and_shape`: Preserves `(N, 256)` `float32`.
4. `test_wrong_key_fails_decryption`: Fails with `InvalidTag` on incorrect key.
5. `test_tampered_ciphertext_fails`: Bit-flip in ciphertext triggers `InvalidTag`.
6. `test_tampered_auth_tag_fails`: Modifying 16-byte GHASH tag triggers `InvalidTag`.
7. `test_tampered_aad_metadata_fails`: Changing `dimension` or `gallery_type` triggers `InvalidTag`.
8. `test_swapped_labels_sha256_binding_fails`: Substituting labels array causes AAD mismatch.
9. `test_corrupted_magic_bytes_rejected`: Header validation rejects invalid magic bytes.
10. `test_unsupported_version_rejected`: Rejects version $> 1$.
11. `test_nonce_uniqueness_across_many_writes`: Verifies fresh 12-byte CSPRNG nonces per write.
12. `test_key_validation_rejects_short_keys`: Keys $< 256$ bits raise `ValueError`.
13. `test_key_validation_accepts_valid_hex_and_bytes`: 64 hex characters or 32 raw bytes accepted.
14. `test_missing_key_in_strict_mode_raises`: `strict_mode=True` raises `RuntimeError`.
15. `test_missing_key_in_dev_mode_falls_back_with_warning`: Dev mode logs warning.
16. `test_vector_store_save_atomic_replacement`: Verifies temporary file swap via `os.replace`.
17. `test_vector_store_inter_process_locking`: Multi-process concurrent write safety with `FileLock`.
18. `test_downgrade_prevention_corrupted_enc_never_falls_back_to_npy`: Rejects fallback if `.enc` exists.
19. `test_coexistence_warning_when_both_enc_and_npy_exist`: Emits security warning.
20. `test_embedding_db_field_level_vector_encryption`: Verifies `"encrypted_vector"` in JSON.
21. `test_embedding_db_metadata_remains_queryable_without_decryption`: `list_all_persons()` works cleanly.
22. `test_embedding_db_tampered_vector_fails`: Tag mismatch on altered JSON vector string.
23. `test_migration_tool_dry_run`: Verifies zero disk mutation under `--dry-run`.
24. `test_migration_tool_full_migration_and_verification`: Verifies migration of galleries and DB.
25. `test_migration_tool_rejects_corrupted_legacy_files`: Detects truncated or non-finite `.npy`.
26. `test_cryptographic_key_separation_enforced`: Identical audit and biometric keys raise `ConfigurationError`.
27. `test_recognition_pipeline_output_equivalence`: End-to-end matching produces identical similarity scores.
28. `test_multibyte_unicode_identity_binding`: UTF-8 subject IDs handled cleanly in AAD.
29. `test_zero_norm_and_non_finite_vector_rejection`: Prevents encrypting poisoned vectors.
30. `test_encryption_key_never_logged`: Secret keys absent from logs, exceptions, and reports.
31. `test_ciphertext_does_not_contain_plaintext_floats`: Verifies raw floats absent from disk.
32. `test_full_backend_security_regression_preservation`: Confirms SEC-01 through SEC-09 and U2 pass.

---

## 22. Baseline Regression Status

All baseline quality and regression suites were verified cleanly:
- **Full Backend Regression**: `274 passed in 554.14s` (100% pass rate).
- **Focused U2 Suite**: `31 passed in 1.30s`.
- **Ruff Linter**: `All checks passed!` (0 errors).
- **Ruff Formatter**: `177 files already formatted` (0 errors).
- **Python Compilation**: Clean across all 21 packages.
- **Git Diff Whitespace**: Clean (0 conflict or whitespace errors).

---

## 23. Numbering Assessment & Formal Designation

1. **Is historical U3 real?**: **YES**. Documented in `docs/thesis_audit/08_security_and_privacy.md` §8.1 & §8.2, `13_limitations_and_gaps.md` L6, and engineering gap report item 13.
2. **Is U3 currently unresolved?**: **YES**. Plaintext biometric vectors exist in `.npy` and `.json` files.
3. **Does the repository authoritatively define U3 as SEC-10?**: **NO**.
4. **Does the repository authoritatively define SEC-11?**: **NO**.
5. **Formal Designation**:
   > *“U3 is a verified historical security finding, but the repository does not authoritatively define it as SEC-10.”*
   > It must remain designated as **`Finding U3 — Biometric Template Encryption at Rest`**.

---

## 24. Go / No-Go Recommendation

### Revised Recommendation: **CONDITIONAL GO — READY FOR IMPLEMENTATION AUTHORIZATION**
- **Viability**: The architecture is fully viable, centralized, and backed by concrete performance micro-benchmarks.
- **Conditions for Authorization**:
  1. Maintainer approval of the expanded storage scope covering both `storage/vector_store.py` (`.enc` containers) and `storage/embedding_database.py` (field-level AEAD vector encryption).
  2. Maintainer decision on identity label confidentiality policy (whether subject IDs remain readable at rest or must be fully bundled).
- **Action**: STOP. Await explicit maintainer authorization before writing implementation code.
