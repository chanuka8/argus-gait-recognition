# Security Implementation Report: Model Confidentiality and Encryption at Rest (Finding U5 — Phase 2)

**Finding Identifier**: Historical Finding U5 (Model Checkpoint / AI Model Asset Protection — Confidentiality at Rest)
**Security Status**: **U5 PHASE 2 CODE-SIDE CONFIDENTIALITY GATE PASS**
- **U5 Phase 2 Code-Side Confidentiality Gate**: `PASS`
- **U5 Phase 2 Active-Model Deployment**: `PENDING` (Pending operational key rollout and production asset encryption)
- **U5 Phase 2 Candidate/Training Confidentiality**: `PENDING` (Phase 2 addresses inference deployment at rest; training pipeline exports pending Phase 3)
- **U5 FULLY CLOSED**: `NO`
**Date**: 2026-09-21
**Repository**: `ARGUS_AI` (`chanuka8/argus-gait-recognition`)
**Authoritative Baseline**: `16d333741d61877c2682e30ccc88470b9f8b09ac`
**Branch**: `main`

---

## 1. Executive Summary & Root Security Design

### 1.1 Context and Scope
U5 Phase 1 hardened **cryptographic authenticity, integrity, and safe deserialization** via Ed25519 signatures, canonical SHA-256 manifests, role-binding, and `weights_only=True`.
U5 Phase 2 establishes **production-grade model confidentiality at rest** for proprietary ARGUS checkpoints, protecting intellectual property against exfiltration, unauthorized copying, and side-channel tampering when physical or disk access is compromised.

```
Encrypted Container on Disk (.enc)
               │
               ▼
Authorized Key Acquisition (Exact key_id match via Env / File / Static providers)
               │
               ▼
Authenticated Decryption (AES-256-GCM AEAD with AAD Header Binding)
               │
               ▼
In-Memory Model Bytes (Decrypted strictly in RAM)
               │
               ▼
U5 Phase-1 Strict Integrity & Authenticity Verification
(Ed25519 Signature, SHA-256 Digest, Logical Filename, Role & Model ID Binding)
               │
               ▼
In-Memory Framework Runtime Deserialization
(PyTorch: io.BytesIO | ONNX: bytes | TensorRT: bytes)
               │
               ▼
No Application-Generated Decrypted Plaintext Model Files on Disk
```

### 1.2 Precise Verified Guarantees & Explicit Caveats
To prevent overclaiming, the confidentiality protection boundary is defined strictly as follows:
- **Verified Runtime Guarantee**: Encrypted production runtime loading created no application-generated decrypted plaintext model file within the isolated runtime/temp filesystem used by the test.
- **Process Memory**: Python plaintext model bytes exist temporarily in process RAM during and after decryption for deserialization by the deep learning frameworks (PyTorch, ONNX Runtime, TensorRT).
- **No RAM Zeroization Guarantee**: Python runtime garbage collection manages in-memory byte buffers; no cryptographic memory wiping or explicit zeroization guarantee is claimed.
- **No OS Swap/Pagefile Guarantee**: Protection against OS-level paging of process memory to disk swap/pagefiles is out of application scope and is not claimed.
- **Training and Export Scope**: Offline model training, export, and candidate promotion scripts may still produce or consume plaintext model artifacts during research workflows. Candidate/training confidentiality remains **PENDING** (targeted for Phase 3).
- **Status Designation**: U5 FULLY CLOSED remains **NO**. The overall repository is not classified as generic "production ready"; U5 Phase 2 satisfies only the code-side confidentiality gate.

---

## 2. Core Cryptographic Architecture

### 2.1 AES-256-GCM Authenticated Container Format
Model artifacts are encrypted into an authenticated binary container with explicit versioning and cryptographic binding:

```
+-------------------------------------------------------------------------------+
| Offset | Length | Field                | Description                          |
+--------+--------+----------------------+--------------------------------------+
| 0      | 4      | Magic Header         | b"ARGUSENC" (8 bytes: magic)         |
| 8      | 2      | Container Version    | 0x0001 (big-endian uint16)           |
| 10     | 2      | Cipher Suite ID      | 0x0001 = AES-256-GCM (uint16)        |
| 12     | 2      | Key ID Length (K)    | Length of key_id string (uint16)     |
| 14     | K      | Key ID               | UTF-8 string matching [A-Za-z0-9._-] |
| 14+K   | 2      | Nonce Length (N=12)  | Length of nonce (uint16)             |
| 16+K   | 12     | Nonce (IV)           | Cryptographically random 12 bytes    |
| 28+K   | 16     | Authentication Tag   | GCM Tag (16 bytes)                   |
| 44+K   | M      | Ciphertext Payload   | Encrypted model binary (M bytes)     |
+-------------------------------------------------------------------------------+
```

- **Header Authentication**: The authenticated data (AAD) passed into AES-GCM includes all container header bytes up to the tag (`Magic || Version || CipherSuite || KeyIdLen || KeyId || NonceLen || Nonce`). Any tampering with header fields, key IDs, or nonces fails authentication before ciphertext processing.
- **Nonce Generation**: Generated strictly via `secrets.token_bytes(12)` (`os.urandom`) per encryption operation, ensuring 96-bit collision-resistant uniqueness.
- **Generic Decryption Failure**: To prevent side-channel padding/oracle leakage, any decryption, authentication, or corrupted header failure unconditionally raises generic `ModelDecryptionError("Authenticated model decryption failed.")`.

### 2.2 Collision-Free Key ID to Environment Variable Mapping
To prevent normalization collisions (e.g. `key-v1`, `key.v1`, and `key_v1` collapsing to the same variable), key IDs are mapped bijectively using hexadecimal encoding of their exact ASCII bytes:

$$\text{EnvVar} = \text{"ARGUS\_MODEL\_KEY\_HEX\_"} + \text{key\_id.encode('utf-8').hex().upper()}$$

- Example: `argus-gait-2026-v1` $\rightarrow$ `ARGUS_MODEL_KEY_HEX_61726775732D676169742D323032362D7631`
- Fully reversible and guaranteed 1-to-1 bijection.
- Explicit key ID lookups never fall back to default keys. Legacy/fallback environment variable `ARGUS_MODEL_ENCRYPTION_KEY` is checked strictly when `key_id == "default"`.

### 2.3 Non-Downgradable Production Policy
In `security_layer/model_confidentiality.py`, production encryption enforcement is absolute:

```python
if env == "production":
    return True
```

No environment variable, config setting, or call-site parameter can disable encryption for protected models when `ARGUS_ENV=production` or `ENVIRONMENT=production` is active.

### 2.4 Artifact Classification & Public Upstream Allowlist
Model classification is strictly artifact-aware:
1. **Default Policy**: All artifacts default to `ArtifactConfidentiality.PROTECTED`.
2. **Anti-Rename Rule**: File paths, extensions, or directory names can **never** downgrade an artifact to `PUBLIC_UPSTREAM`.
3. **Canonical Public Allowlist**: Only explicit trusted call-site declarations or the repository's authoritative public model identities are recognized as `PUBLIC_UPSTREAM`:
   - `(ROLE_APPEARANCE_EMBEDDING, "osnet_x0_25_msmt17")`
   - `(ROLE_PERSON_DETECTOR, "yolov8n")`
4. Any synthetic, unrecognized, or proprietary model ID defaults to `PROTECTED`.

### 2.5 Separation of Physical Container vs Signed Logical Identity
Physical encrypted containers (`best_model.pth.enc`) and Phase-1 signed logical identities (`best_model.pth`) are strictly delineated:
- `logical_filename = path.name` is prohibited for encrypted containers.
- For protected production call-sites, an explicit non-empty `logical_filename` must be provided (e.g., `best_model.pth` or `silhouette_segmenter.onnx`).
- Renaming physical container files on disk cannot alter the signed logical identity or bypass manifest digest validation.
- Missing logical identities in production trigger fail-closed rejection.

---

## 3. In-Memory Runtime Deserialization (Zero Application-Generated Plaintext Model Files)

Decryption occurs strictly into RAM buffers. No plaintext decrypted model files, intermediate temporary files, or named pipes are created on disk:

1. **PyTorch Backend** (`models/inference/pytorch_backend.py`, `models/reid/osnet_backbone.py`):
   ```python
   model_bytes = load_verified_model_bytes(
       model_path=self.weights_path,
       expected_role=ROLE_GAIT_EMBEDDING,
       logical_filename="best_model.pth",
       confidentiality=ArtifactConfidentiality.PROTECTED,
   )
   checkpoint = torch.load(io.BytesIO(model_bytes), map_location=self.device, weights_only=True)
   ```
2. **ONNX Runtime Backend** (`models/inference/onnx_backend.py`):
   ```python
   model_bytes = load_verified_model_bytes(
       model_path=self.onnx_path,
       expected_role=ROLE_SILHOUETTE_SEGMENTER,
       logical_filename="silhouette_segmenter.onnx",
       confidentiality=ArtifactConfidentiality.PROTECTED,
   )
   self.session = ort.InferenceSession(model_bytes, sess_options=opts, providers=self.providers)
   ```
3. **TensorRT Backend** (`models/inference/tensorrt_backend.py`):
   ```python
   model_bytes = load_verified_model_bytes(
       model_path=self.engine_path,
       expected_role=ROLE_GAIT_EMBEDDING,
       logical_filename="bygait_light_fp16.engine",
       confidentiality=ArtifactConfidentiality.PROTECTED,
   )
   engine = runtime.deserialize_cuda_engine(model_bytes)
   ```

---

## 4. Provisioning Tool Safety & Concurrency

### 4.1 CLI: `tools/security/encrypt_model_artifact.py`
The model provisioning CLI ensures robust write durability and race-free operation:
1. **Destination Lock**: Acquires an exclusive lock file (`.<destination>.lock`) via `os.open(..., os.O_CREAT | os.O_EXCL | os.O_WRONLY)` to prevent race conditions during artifact creation.
2. **No Automatic Stale-Lock Deletion**: Lock conflicts immediately raise `ProvisioningLockError`. Existing locks are never automatically deleted based merely on age.
3. **Atomic Temp File Generation**: Writes encrypted containers to a unique temporary file (`.<destination>.tmp_<hex>.enc`) in the same directory to guarantee filesystem-atomic renaming.
4. **Flush and Sync**:
   ```python
   with os.fdopen(temp_fd, "wb", closefd=True) as fh:
       written = fh.write(container_bytes)
       if written != len(container_bytes):
           raise ModelProvisioningError(...)
       fh.flush()
       os.fsync(fh.fileno())
   ```
5. **Post-Write In-Memory Self-Verification**: Decrypts the written container and verifies its SHA-256 digest against the source model before replacing the destination.
6. **Atomic Rename**: Replaces target destination via `os.replace`. Old destination remains unchanged on any prior failure.
7. **Exception-Safe Cleanup**: Destination lock and temp files are cleaned up in a `finally` block even if writing or verification fails.

---

## 5. Deployment Diagnostics & Startup Validation

1. **Startup Validator** (`deployment/startup_validator.py`):
   - Added `_validate_model_confidentiality()` pre-flight check.
   - Enforces encrypted artifacts for protected roles in production.
   - Executes test decryption and Phase-1 verification in RAM to verify key availability and artifact authenticity before starting worker threads.
2. **Doctor Utility** (`deployment/doctor.py`):
   - Added `model_confidentiality` health check in the `security` category.
   - Reports pass/fail status and flags unencrypted protected models in production environments.

---

## 6. Verification and Regression Suite

| Suite | Component | Scope | Result |
| :--- | :--- | :--- | :--- |
| `test_model_confidentiality.py` | U5 Phase 2 Confidentiality | AEAD roundtrip, nonce uniqueness, tamper rejection, key-ID bijection, exact routing, artifact resolution, anti-rename, physical vs logical separation, non-downgrade, strict Phase 1, zero residue, no secrets in logs, provisioning concurrency, write safety, startup & doctor | **14 / 14 PASS** |
| `test_model_checkpoint_security.py` | U5 Phase 1 Integrity | Manifest digests, Ed25519 signatures, role-binding, weights_only=True, safe loading | **46 / 46 PASS** |
| `test_camera_transport_security.py` & `test_multi_stream_engine_security.py` | U4 RTSP Transport | Secure RTSP enforcement, credential redaction, multi-stream isolation | **42 / 42 PASS** |
| `test_runtime_paths.py` | Core Path Resolution | CWD-independence, app-root resolution | **20 / 20 PASS** |
| `test_operational_collector_shutdown_durability.py` | Observation Durability | Shutdown persistence, flush idempotence | **13 / 13 PASS** |
| `test_sync_folder_readmes.py` | Documentation Integrity | Folder README synchronization | **45 / 45 PASS** |
| **Full Repository Test Suite** | All Subsystems | Comprehensive regression verification | **1460 / 1460 PASS** |

### Linter, Formatting, and README Compliance
- `ruff check`: **0 errors** (all checks passed).
- `ruff format --check`: **0 errors** (all task files already formatted).
- `sync_folder_readmes.py --check`: **0 errors** (all folder READMEs up to date).
- `git diff --check`: **0 whitespace/conflict warnings**.
- `git diff --cached --name-only`: **Empty (0 staged files)**.

---

## 7. Task-Owned Inventory & Pre-Existing Local Files

### 7.1 Task-Owned Tracked Files (Modified by U5 Phase 2)
1. `.env.example`
2. `deployment/doctor.py`
3. `deployment/startup_validator.py`
4. `models/inference/onnx_backend.py`
5. `models/inference/pytorch_backend.py`
6. `models/inference/tensorrt_backend.py`
7. `models/reid/osnet_backbone.py`
8. `pipeline/live_recognition.py`
9. `pipeline/multi_camera_recognition.py`
10. `pipeline/steps/feature_extraction.py`
11. `pipeline/steps/silhouette_step.py`
12. `pipeline/video_recognition.py`
13. `security_layer/README.md`
14. `security_layer/__init__.py`
15. `security_layer/model_integrity.py`

### 7.2 Task-Owned Untracked Files (Created by U5 Phase 2)
1. `security_layer/model_confidentiality.py`
2. `tools/security/encrypt_model_artifact.py`
3. `tests/integration/backend/test_model_confidentiality.py`
4. `docs/security/u5_phase2_implementation_report.md`

### 7.3 Protected Local Files (Pre-Existing / Unrelated / Unstaged)
The following 8 files were modified prior to U5 Phase 2 work and remain strictly untouched, unstaged, unreset, and undiscarded:
1. `docs/firebase_architecture.md`
2. `docs/firebase_data_model.md`
3. `docs/firebase_integration_audit.md`
4. `frontend/src/components/CaseDossierModal.css`
5. `models/appearance_gallery/gallery_features.npy`
6. `models/appearance_gallery/gallery_labels.npy`
7. `models/appearance_gallery/gallery_metadata.json`
8. `pipeline/steps/tracking.py`

---

## 8. Status Classification & Closure Boundaries

| Category | Status | Rationale |
| :--- | :--- | :--- |
| **U5 Phase 2 Code-Side Confidentiality Gate** | **PASS** | AES-256-GCM container format, exact key providers, bijective key-ID mapping, non-downgradable production policy, physical vs logical identity separation, and in-memory loaders fully implemented, integrated, and verified with zero test regressions. |
| **U5 Phase 2 Active-Model Deployment** | **PENDING** | Actual production deployment requires encrypting production weights and provisioning deployment keys in the production environment. Repository code does not fabricate mock production credentials or commit dummy `.enc` models. |
| **U5 Phase 2 Candidate/Training Confidentiality** | **PENDING** | Phase 2 hardens inference runtime checkpoints at rest. Training pipeline candidate export encryption is designated for Phase 3. |
| **U5 FULLY CLOSED** | **NO** | Active deployment rollout and training candidate export protection remain pending. |
