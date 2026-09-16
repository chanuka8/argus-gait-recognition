# Security Roadmap Discovery & Next Security Item Identification Report

**Date**: 2026-09-16
**Auditor**: Antigravity Security Agent
**Repository**: `ARGUS_AI` (`chanuka8/argus-gait-recognition`)
**Branch**: `main`
**Classification**: **CATEGORY B / C — UNNUMBERED FINDINGS ONLY; NO AUTHORITATIVE NEXT NUMBERED SEC ITEM**

---

## 1. Purpose

The objective of this discovery audit is to examine all repository sources, git version control history, documentation, and historical security audits to determine whether an authoritative next numbered security-hardening item (such as `SEC-10` or `SEC-11`) exists in the ARGUS AI project roadmap, or whether only unnumbered historical findings remain.

In compliance with the **Zero False Positive Evidence-Based Reporting Policy** ([AGENTS.md](file:///e:/ARGUS_AI/.agents/AGENTS.md)):
- No security identifier is invented.
- No unnumbered finding (e.g., U2, U3) is promoted to a numbered SEC identifier without explicit maintainer directive.
- Zero production code is modified or implemented.
- The established state of SEC-01 through SEC-09 is strictly preserved.

---

## 2. Sources Searched

An exhaustive search was conducted across all potential repositories of architectural decisions and security definitions:

### 2.1 Current Working Tree
- **Paths Inspected**: `docs/`, `security/`, `security_layer/`, `reports/`, `tests/`, `.agents/`, `README.md`, `configs/`, `storage/`, `services/`, `api/`.
- **Search Query**: Pattern matching for `SEC-10`, `SEC10`, `SEC-11`, `SEC11`, `SEC-`, `security roadmap`, `security backlog`.
- **Findings**:
  - Only completed security items `SEC-01` through `SEC-09` are defined in the active codebase.
  - Zero forward-looking security roadmaps or pending numbered items exist in active files.

### 2.2 Git History & Version Control Objects
- **Branches Inspected**: `main`, `backup-before-security-fix`, `backup/*`, `friend/*`, `origin/*`.
- **Tags & Stashes**: Tag `backup-main-before-rtsp-5f73391`; stashes `stash@{0}` through `stash@{5}`.
- **Commit Log & Pickaxe Search**:
  ```powershell
  git log --all --grep="SEC-" -i
  git log --all --grep="SEC-10" -i
  git log --all --grep="SEC-11" -i
  git log --all -S "SEC-10" -i
  git log --all -S "SEC-11" -i
  git log --all -G "SEC-10" -i
  git log --all -G "SEC-11" -i
  git log --all --grep="roadmap" -i
  git log --all --grep="hardening" -i
  ```
- **Findings**:
  - Zero commits reference `SEC-10`, `SEC-11`, or any future numbered security items.
  - Zero git diffs contain additions or removals of `SEC-10` or `SEC-11`.

### 2.3 Historical & Deleted Audit Documentation
Analyzed documents deleted in commit `b18e69f` (`chore(repo): remove obsolete markdown documentation`):
- **`docs/thesis_audit/08_security_and_privacy.md`**:
  - The comprehensive foundational thesis security assessment (STRIDE analysis, 20+ controls).
  - **Does NOT use a `SEC-XX` numbering schema**.
- **`docs/thesis_audit/13_limitations_and_gaps.md`**:
  - Lists limitations L1 through L18 (L6 template encryption, L7 auth, L8 adversarial robustness, L18 liveness).
  - **Does NOT use a `SEC-XX` numbering schema**.
- **`docs/FUTURE_ENGINEERING_GAP_AND_REMEDIATION_REPORT.md`**:
  - Contained an early 3-item list:
    - `SEC-01`: Unauthenticated API Endpoints
    - `SEC-02`: Unsigned Security Audit Logs (early 3-item proposal)
    - `SEC-03`: Gallery Template Plaintext Storage (early 3-item proposal)
  - This early 3-item list was superseded by the modern September 2026 hardening sequence (SEC-01 through SEC-09). It contained no `SEC-04` through `SEC-10` or beyond.
- **`SECURITY.md`**, **`docs/SECURITY_ALLOW_PICKLE_FIX_REPORT.md`**, **`docs/reports/SECURITY_INTEGRITY_REPORT.md`**:
  - Contain zero future security sequences.

### 2.4 Security Regression Test Suite
Inspected all 14 integration test files under `tests/integration/backend/`:
- `test_spa_path_traversal.py` (SEC-01)
- `test_auth_bypass.py` (SEC-02)
- `test_websocket_auth.py` (SEC-03)
- `test_upload_session_bola.py` (SEC-04)
- `test_person_id_path_traversal.py` (SEC-05)
- `test_login_rate_limit.py` (SEC-06)
- `test_video_memory_safety.py` (SEC-07)
- `test_security_headers.py` (SEC-08)
- `test_error_disclosure.py` (SEC-09)

---

## 3. Cross-Reference of Current Security State (SEC-01 through SEC-09)

The table below summarizes all verified security controls currently operating in the codebase:

| Identifier | Finding / Hardening Scope | Current Status | Primary Source Location | Evidence / Test Suite |
|---|---|---|---|---|
| **SEC-01** | SPA Static Asset Path Traversal Prevention | **PASS** | `api/server.py` | [`tests/integration/backend/test_spa_path_traversal.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_spa_path_traversal.py) (8/8 passed) |
| **SEC-02** | Unauthenticated Access & Auth Bypass Prevention | **PASS** | `security_layer/auth.py`, `api/v1/router.py` | [`tests/integration/backend/test_auth_bypass.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_auth_bypass.py) (12/12 passed) |
| **SEC-03** | WebSocket Connection Authentication Enforcement | **PASS** | `api/v1/router.py`, `security_layer/auth.py` | [`tests/integration/backend/test_websocket_auth.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_websocket_auth.py) (13/13 passed) |
| **SEC-04** | Upload Session Ownership BOLA / IDOR Protection | **PASS** | `services/upload_session_manager.py`, `api/v1/router.py` | [`tests/integration/backend/test_upload_session_bola.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_upload_session_bola.py) (30/30 passed) |
| **SEC-05** | `person_id` Path Traversal & Filesystem Boundary Escape | **PASS** | `security_layer/input_validation.py`, `api/v1/router.py` | [`tests/integration/backend/test_person_id_path_traversal.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_person_id_path_traversal.py) (25/25 passed) |
| **SEC-06** | Login Rate Limiting & Brute-Force Protection | **PASS** | `security_layer/rate_limiter.py`, `api/v1/auth_router.py` | [`tests/integration/backend/test_login_rate_limit.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_login_rate_limit.py) (15/15 passed) |
| **SEC-07** | Unbounded In-Memory Video Buffering DoS Prevention | **PASS** | `security_layer/input_validation.py`, `services/upload_session_manager.py` | [`tests/integration/backend/test_video_memory_safety.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_video_memory_safety.py) (19/19 passed) |
| **SEC-08** | Standard HTTP Security Headers Middleware | **PASS** | `security_layer/security_headers.py`, `api/server.py` | [`tests/integration/backend/test_security_headers.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_security_headers.py) (21/21 passed) |
| **SEC-09** | Verbose API Error / Exception Information Disclosure | **PASS** | `security_layer/error_sanitizer.py`, `api/v1/router.py` | [`tests/integration/backend/test_error_disclosure.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_error_disclosure.py) (21/21 passed) |
| **Next Authoritative Item** | **NOT DEFINED** | **UNDEFINED** | N/A | Exhaustive grep & git pickaxe: 0 matches |

---

## 4. Authoritative Roadmap Evidence

- **Authoritative Future Sequence**: **NON-EXISTENT**.
- Neither the repository root, `docs/`, `.agents/`, nor git history specifies a predetermined roadmap following `SEC-09`.
- The historical document `FUTURE_ENGINEERING_GAP_AND_REMEDIATION_REPORT.md` (written in July 2026, deleted in commit `b18e69f`) was an early preliminary proposal that only reached item 3 under an obsolete indexing scheme, and has no continuity with the current SEC-01 through SEC-09 architecture.
- **Classification**: **NOT DEFINED**.

---

## 5. Historical Unnumbered Findings (U-Series)

From the authoritative thesis security audit ([docs/thesis_audit/08_security_and_privacy.md](file:///e:/ARGUS_AI) from commit `b18e69f^`), the complete inventory of unnumbered security findings is documented below, preserving their original identifiers:

| Identifier | Finding Title & Threat Category | Audit Source | Current Status in Codebase | Technical Details |
|---|---|---|---|---|
| **U1** | Verbose API Error / Exception Information Disclosure (*Information Disclosure*) | Thesis audit & source inspection | **REMEDIATED** as `SEC-09` | Exception sanitization helper active across all endpoints and services. |
| **U2** | Audit Log Integrity / Non-repudiation (*STRIDE: Repudiation / Tampering*) | `08_security_and_privacy.md` §8.1 & §8.2 | **UNRESOLVED** | `security_layer/security_logger.py` writes plaintext CSV records to `outputs/logs/security/security_events.csv` without cryptographic hash chaining, HMAC-SHA256, or tamper-evident sealing. |
| **U3** | Biometric Template Plaintext Storage (*Information Disclosure / Data at Rest*) | `08_security_and_privacy.md` §8.1, `13_limitations_and_gaps.md` L6 | **UNRESOLVED** | 256-D gait feature embeddings are persisted as unencrypted `.npy` files (`models/appearance_gallery/`, `models/gallery/`). If disk access is compromised, biometric vectors are extractable. |
| **U4** | Unencrypted RTSP Stream Transport (*Information Disclosure / Transit*) | `08_security_and_privacy.md` §8.1 | **UNRESOLVED** | Network transport protocol configuration. RTSP streams from external IP cameras may traverse local networks unencrypted unless encapsulated via RTSPS/VPN. |
| **U5** | Model Checkpoint Protection (*Asset Protection*) | `08_security_and_privacy.md` §8.1 | **UNRESOLVED** | PyTorch model weight files (`runs/exp_001/best_model.pth`, ByGaitLight checkpoints) are stored unencrypted on disk. |
| **U6** | Sensitive Data / PII in Audit Logs (*Privacy Risk*) | `08_security_and_privacy.md` §8.1 | **UNRESOLVED** | `security_events.csv` logs identity labels (`person_id`) and matching confidence scores in plain text. |
| **U7** | Absence of Adversarial Robustness (*Algorithm Security*) | `13_limitations_and_gaps.md` L8 | **UNRESOLVED** | Academic research limitation: CNN silhouette gait encoders do not incorporate adversarial perturbation defenses. |
| **U8** | Absence of Biometric Liveness / Replay Detection (*Algorithm Security*) | `13_limitations_and_gaps.md` L18 | **UNRESOLVED** | Academic research limitation: System evaluates pre-recorded or live streams without anti-spoofing / liveness validation. |
| **U9** | Gallery Poisoning via Unauthorized Ingestion (*Integrity*) | `08_security_and_privacy.md` §8.1 | **PARTIALLY MITIGATED** | Mitigated by SEC-02 (JWT/RBAC) and SEC-05 (input validation), though lack of dual-authorization on enrollment leaves insider risk. |

---

## 6. Candidate Next Findings (Evidence-Based Assessment)

If the repository maintainer chooses to formalize the next hardening phase, the candidate findings rank by technical evidence and implementation feasibility as follows:

### Candidate 1: Finding U2 — Audit Log Cryptographic Integrity & Tamper-Proof Sealing
- **Distinction**: **HISTORICAL UNNUMBERED FINDING** (NOT an authoritative SEC item).
- **Direct Evidence**:
  - `docs/thesis_audit/08_security_and_privacy.md`: *"Logs are plaintext CSV; no tamper protection -> Add cryptographic hashing"*.
  - `docs/FUTURE_ENGINEERING_GAP_AND_REMEDIATION_REPORT.md` §22: *"Implement HMAC-SHA256 log signing in SecurityLogger"*.
- **Architectural Scope**:
  - Confined strictly to `security_layer/security_logger.py` and dedicated verification utilities.
  - Zero interference with the locked gait recognition pipeline (`Person Detection → ByteTrack → UNet Silhouette → GEI/LiveGEI → ByGaitLight CNN → 256D → Cosine`).
  - Highly bounded risk profile.

### Candidate 2: Finding U3 — Biometric Template Encryption at Rest
- **Distinction**: **HISTORICAL UNNUMBERED FINDING** (NOT an authoritative SEC item).
- **Direct Evidence**:
  - `docs/thesis_audit/08_security_and_privacy.md`: *"Gallery stored as plaintext .npy -> Encrypt gallery files at rest"*.
  - `docs/thesis_audit/13_limitations_and_gaps.md` L6: *"No template encryption"*.
- **Architectural Scope**:
  - Broad architectural impact across `storage/vector_store.py`, `storage/firebase_embedding_store.py`, gallery indexing, model startup, and CLI inspection tools.
  - High risk of performance overhead or startup latency without formal key management architecture.

### Candidate 3: Finding U6 — PII Pseudonymization in Security Logs
- **Distinction**: **HISTORICAL UNNUMBERED FINDING** (NOT an authoritative SEC item).
- **Direct Evidence**:
  - `docs/thesis_audit/08_security_and_privacy.md` §8.1: *"Security logs contain identity IDs and scores -> Pseudonymize identity IDs in logs"*.
- **Architectural Scope**:
  - Moderate scope affecting `security_logger.py` and downstream incident response reporting.

---

## 7. Whether Any Numbered Next SEC Item is Authoritative

### Determination: **NO**

- **Authoritative Status**: **NOT DEFINED**.
- **Outcome Classification**: **Outcome B / C** (Unnumbered historical findings exist; no authoritative numbered next security item exists).
- Neither `SEC-10`, `SEC-11`, nor any other subsequent numbered item has been defined in repository code, active documentation, or git history.

---

## 8. Evidence for the Conclusion

1. **Repository-Wide Ripgrep**: `ripgrep` for `SEC-10`, `SEC10`, `SEC-11`, `SEC11` across all files produced **0 results**.
2. **Git Commit History**: `git log --all --grep="SEC-10" -i`, `git log --all --grep="SEC-11" -i` produced **0 results**.
3. **Git Pickaxe & Diff Grep**: `git log --all -S "SEC-10"`, `git log --all -S "SEC-11"`, `git log --all -G "SEC-10"`, `git log --all -G "SEC-11"` produced **0 results**.
4. **Deleted Files Inspection**: Commit `b18e69f^` recovered documents confirmed no numbered sequences beyond the early 3-item draft from July 2026.
5. **Transcript Records**: Review of past agent transcripts showed the only prior references to `SEC-10` were explicit negative constraints (`DO NOT implement SEC-10`).

---

## 9. Current Regression Baseline

The entire repository security baseline was audited to confirm stability:

| Verification Suite | Command | Output / Status | Errors |
|---|---|---|---|
| **Ruff Linter** | `.venv\Scripts\ruff.exe check api/ services/ security_layer/ tests/` | `All checks passed!` | **0** |
| **Ruff Formatter** | `.venv\Scripts\ruff.exe format --check api/ services/ security_layer/ tests/` | `175 files already formatted` | **0** |
| **Bytecode Compilation** | `.venv\Scripts\python.exe -m compileall api/ services/ security_layer/ tests/` | Clean compilation across all 21 packages | **0** |
| **Git Diff Whitespace** | `git diff --check` | Clean (0 conflict or whitespace errors) | **0** |
| **Backend Integration Suite** | `.venv\Scripts\python.exe -m pytest tests/integration/backend/ -q` | **243 passed in 439.24s** | **0** |

All 9 completed security controls (SEC-01 through SEC-09) and the underlying backend APIs are completely passing and healthy.

---

## 10. Recommended Next Action

1. **Awaiting Maintainer Direction**:
   The maintainer should review the historical unnumbered findings (U2 through U9) and formally decide:
   - Whether to formalize **Finding U2** (*Audit Log Cryptographic Integrity / HMAC-SHA256 Signing*) or **Finding U3** (*Biometric Template Encryption at Rest*) as the next security hardening target.
   - Whether to assign a specific numerical identifier (e.g., authoritatively declaring U2 as `SEC-10`).
2. **Implementation Gate**:
   Until explicit maintainer instruction is provided, no production files should be modified, no speculative code authored, and no numerical security identifiers invented.
