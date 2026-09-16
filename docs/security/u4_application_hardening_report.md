# ARGUS AI Security Hardening — U4 Camera / RTSP Stream Transport Security
## Application-Layer Hardening & Verification Report

**Target Finding**: Historical Finding U4 — Camera / RTSP Stream Transport Security
**Historical Source**: `docs/thesis_audit/08_security_and_privacy.md` (§8.1 & §8.2, commit `b18e69f^`), `docs/FUTURE_ENGINEERING_GAP_AND_REMEDIATION_REPORT.md`
**Execution Scope**: Application-Layer Hardening Only (No Physical Network / Camera Hardware Alterations)
**Security Status**:
- **Application Verdict**: `U4 APPLICATION HARDENING PASS`
- **Deployment Verdict**: `U4 DEPLOYMENT REMEDIATION PENDING`
- **Overall Finding Status**: `PARTIALLY REMEDIATED (SOFTWARE PASS / PHYSICAL PENDING)`
**Strict Numbering Rule**: U4 remains an unnumbered historical finding. It is NOT named SEC-10 (SEC-10 remains undefined).

---

## 1. Executive Summary & Security Boundary

Finding U4 concerns the transmission of video streams over plaintext RTSP (`rtsp://`) across local network segments. Although camera credentials in ARGUS are protected at rest via AES-128/Fernet encrypted storage, stream transit over unencrypted RTSP presents an eavesdropping and integrity risk where physical networks are shared or untrusted.

U4 is a **hybrid security finding** with a clear security boundary:
1. **Application-Layer Boundary (Completed)**:
   - URL credential and query-secret redaction in all logging and error paths.
   - Formal transport classification and validation engine (`validate_camera_transport`).
   - Strict-mode enforcement rejecting plaintext RTSP/HTTP (`ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true`).
   - Strict RTSPS candidate policy requiring explicit operator confirmation of verified transport.
   - Rejection of unencrypted streams without silent fallback/downgrade.
2. **Physical Deployment Boundary (Pending)**:
   - Real CCTV hardware operates on physical switches, VLANs, and routers.
   - Software cannot physically encrypt packets emitted by legacy hardware without network-layer remediation (WireGuard / IPSec / 802.1Q isolated VLAN).
   - Therefore, the final verdict is strictly split.

---

## 2. Mandatory Architectural Amendments Implemented

All six mandatory amendments required prior to execution were fully incorporated:

### Amendment 1: RTSPS Does Not Automatically Equal Verified Transport
- **Limitation Disclosed**: While the installed OpenCV/FFmpeg runtime recognizes the `rtsps://` URI scheme, the Python `cv2.VideoCapture` interface does NOT expose granular controls for:
  - CA trust root configuration
  - Subject Alternative Name (SAN) validation
  - Certificate expiration handling
  - Certificate revocation lists (CRL) / OCSP stapling
  - Public key certificate pinning
- **Classification Engine**: Distinguishes seven mutually exclusive transport categories:
  1. `local`: Local USB webcams, DirectShow devices, and local video files.
  2. `protected_tunnel`: Operator-asserted encrypted tunnels (WireGuard, IPSec, authenticated VPN).
  3. `rtsps_candidate`: Native RTSPS (`rtsps://`).
  4. `https_candidate`: HTTPS stream (`https://`).
  5. `plaintext_rtsp`: Unencrypted RTSP (`rtsp://`).
  6. `plaintext_http`: Unencrypted HTTP (`http://`).
  7. `unsupported`: Unknown or invalid schemes.
- **Strict Policy**: In strict mode, `rtsps_candidate` is **NOT** automatically accepted. It requires explicit operator confirmation (`allow_rtsps_candidate: true` or `transport_security: "rtsps_verified"`). ARGUS explicitly distinguishes operator assertions from cryptographic proofs.

### Amendment 2: Source Type Compatibility & Invariant Preservation
- A comprehensive audit of repository branches revealed that introducing `source_type = "rtsps"` would cause extensive regressions across UI badges, worker loops, inference pipelines, and camera state serialization.
- **Invariant**: All network cameras maintain `source_type = "rtsp"`. Transport scheme (`"rtsps"` vs `"rtsp"`) and security validation metadata are tracked separately in `transport_scheme`, `transport_category`, and `transport_security`.

### Amendment 3: Robust Structured URL Sanitization
- Structured parsing via `urllib.parse.urlsplit` and `parse_qsl` masks:
  - Authority credentials: `username:password@` $\rightarrow$ `***:***@`
  - Query parameters: `password`, `pass`, `token`, `access_token`, `api_key`, `apikey`, `secret`, `auth`, `key` $\rightarrow$ `***`
  - Benign parameters (e.g. `channel=1`, `subtype=0`) are preserved.
  - Malformed URLs fail safely without echoing raw strings or leaking credentials into exceptions.
  - Backward-compatible wrapper `sanitize_rtsp_url` delegates to `sanitize_stream_url`.

### Amendment 4: Real-Camera Isolation & Guarded Testing
- Focused U4 tests never contact physical CCTV hardware (`192.168.1.100`, `.101`, `.102`).
- A test-level socket guard fixture (`guard_real_camera_network`) intercepts all socket connection attempts and aborts execution if a real camera IP is targeted.
- Stream capture boundaries (`cv2.VideoCapture`) are mocked or patched with synthetic frames.
- `configs/cameras.yaml` was 100% untouched.

### Amendment 5: Credential Helper Generalization
- Generalized helpers introduced:
  - `sanitize_stream_url(url: str | None) -> str`
  - `extract_stream_credentials(url: str) -> tuple[str | None, str | None, str]`
  - `build_stream_url(base_url: str, username, password) -> str`
- Existing RTSP helpers (`sanitize_rtsp_url`, `extract_rtsp_credentials`, `build_rtsp_url`) preserved as backward-compatible wrappers.
- HTTP/HTTPS credential-construction avoided.

### Amendment 6: Split Final Verdict
- `U4 APPLICATION HARDENING PASS`
- `U4 DEPLOYMENT REMEDIATION PENDING`
- Outputting `U4 PASS — CLOSED` is prohibited until physical encrypted transport evidence is produced.

---

## 3. Detailed File Modifications

| File | Nature of Changes |
|---|---|
| [`security_layer/credentials.py`](file:///e:/ARGUS_AI/security_layer/credentials.py) | Added `CameraTransportSecurityError(ValueError)`. Implemented `sanitize_stream_url`, `extract_stream_credentials`, `build_stream_url`. Implemented `is_secure_camera_transport_required` and `validate_camera_transport` supporting 7 transport categories. Updated `resolve_camera_config` with protocol inference and transport validation. |
| [`services/camera_source_resolver.py`](file:///e:/ARGUS_AI/services/camera_source_resolver.py) | Updated `resolve_source` to recognize `rtsps://` (preserving `source_type="rtsp"`). Integrated `validate_camera_transport` and `sanitize_stream_url`. Added transport security check to `probe_stream`. |
| [`services/camera_worker.py`](file:///e:/ARGUS_AI/services/camera_worker.py) | Updated `_open_capture` to enforce transport security via `validate_camera_transport`. Ensured all log formatting uses `sanitize_stream_url`. Added auto-inference of `source_type` from stream scheme. |
| [`utils/config_validator.py`](file:///e:/ARGUS_AI/utils/config_validator.py) | Extended `sanitize_rtsp_url` and camera URL validation to accept `rtsps://` scheme alongside `rtsp://`, `http://`, `https://`. |
| [`tests/conftest.py`](file:///e:/ARGUS_AI/tests/conftest.py) | Registered `test_camera_transport_security.py` in `security_test_modules` to prevent test client header pollution. |
| [`tests/unit/camera/test_user_rtsp_credentials.py`](file:///e:/ARGUS_AI/tests/unit/camera/test_user_rtsp_credentials.py) | Updated error response test assertion to accept SEC-09 sanitized generic error detail alongside sanitized URL. |
| [`tests/integration/backend/test_camera_transport_security.py`](file:///e:/ARGUS_AI/tests/integration/backend/test_camera_transport_security.py) | **[NEW]** 31 comprehensive integration tests covering URL redactions, strict mode, permissive mode, RTSPS operator assertions, protected tunnels, zero downgrade, and isolation guard. |
| [`docs/security/u4_deployment_requirements.md`](file:///e:/ARGUS_AI/docs/security/u4_deployment_requirements.md) | **[NEW]** Authoritative operational remediation guide detailing WireGuard tunnels, surveillance VLANs, TLS proxies, and deployment checklist. |

---

## 4. Verification Evidence & Test Execution Results

### A. Focused U4 Test Suite
Command: `.venv\Scripts\python.exe -m pytest tests/integration/backend/test_camera_transport_security.py -v`
Result: **31 passed in 1.05s** (100% pass rate).

### B. U2 Audit Log Integrity Verification Suite
Command: `.venv\Scripts\python.exe -m pytest tests/integration/backend/test_audit_log_integrity.py -v`
Result: **31 passed in 2.11s** (Zero regressions).

### C. U3 Biometric Template Encryption Verification Suite
Command: `.venv\Scripts\python.exe -m pytest tests/integration/backend/test_biometric_template_encryption.py -v`
Result: **51 passed in 1.57s** (Zero regressions).

### D. Legacy Camera & Configuration Unit Test Suite
Command:
```powershell
.venv\Scripts\python.exe -m pytest tests/unit/camera/test_rtsp_credentials.py tests/unit/camera/test_user_rtsp_credentials.py tests/unit/backend/test_config_validator.py -v
```
- **Passed**: 34
- **Failed**: 0
- **Skipped**: 0
- **Duration**: 91.85s (0:01:31)
- **Exit Code**: 0

### E. Static Code Quality & Linter Checks
Commands & Exact Results:
- `.venv\Scripts\ruff.exe check api/ services/ security_layer/ storage/ tests/`
  - Output: `All checks passed!`
  - Exit Code: `0`
- `.venv\Scripts\ruff.exe format --check api/ services/ security_layer/ storage/ tests/`
  - Output: `190 files already formatted`
  - Exit Code: `0`
- `.venv\Scripts\python.exe -m compileall -q api/ services/ security_layer/ storage/ tests/`
  - Output: `(clean, 0 syntax/compilation errors)`
  - Exit Code: `0`
- `git diff --check`
  - Output: `(clean, 0 whitespace or conflict marker errors)`
  - Exit Code: `0`

---

## 5. Existing Test Modification & Credential Safety Assessment

In `tests/unit/camera/test_user_rtsp_credentials.py`:
```diff
         data_str = resp.text
+        # Raw password must never appear in API response
         assert "LeakedPass999" not in data_str
-        assert "rtsp://***:***@" in data_str
+        # Raw username must never appear in API response
+        assert "leaked_user" not in data_str
+        # Raw credential-bearing userinfo (user:pass@) form must be absent
+        assert "leaked_user:LeakedPass999@" not in data_str
+        # Raw full credential-bearing RTSP URL must be absent
+        assert "rtsp://leaked_user:LeakedPass999@192.168.1.99:554/live" not in data_str
+        # Response should contain either sanitized URL form or SEC-09 generic error
         assert "rtsp://***:***@" in data_str or "Camera connection failed" in data_str
```

### Safety & Integrity Confirmation:
1. **SEC-09 Compatibility**: Under SEC-09 (finding U1: Verbose Error Disclosure), `api/v1/router.py` was hardened to replace raw exception disclosures with generic error details (`"Camera connection failed or stream unavailable"`), preventing any disclosure of internal connection strings or error traces to external HTTP clients.
2. **Credential Leakage Assertions Strengthened** (Phase A Correction):
   - The test asserts `"LeakedPass999" not in data_str` — raw password absent.
   - The test asserts `"leaked_user" not in data_str` — raw username absent.
   - The test asserts `"leaked_user:LeakedPass999@" not in data_str` — raw userinfo form absent.
   - The test asserts the full raw credential-bearing RTSP URL is absent.
   - The test accepts either sanitized `rtsp://***:***@` form or SEC-09 generic error `"Camera connection failed"`.
   - Credential confidentiality verification remains strictly enforced.

---

## 6. Strict Constraints & Repository Integrity Audit

- **`README.md`**: 100% untouched (`git diff README.md` is empty).
- **`configs/cameras.yaml`**: 100% untouched (`git diff configs/cameras.yaml` is empty).
- **`configs/credentials.enc`**: 100% untouched (`git diff configs/credentials.enc` is empty).
- **Physical Camera Contact**: Zero contact; verified by active test-isolation socket guard fixture.
- **SEC-10 Identifier**: Not used anywhere. Finding remains designated `U4 — Camera / RTSP Stream Transport Security`.
- **Git Commit / Push**: Zero commits created, zero pushes executed.

---

## 7. Preserved Deployment & Operational Limitations

As required under the Zero False Positive Evidence-Based Policy, software hardening must not overstate its scope:
1. **Application software does not prove WireGuard/IPSec is running**: Config declarations such as `transport_security: "wireguard"` are recorded as operator assertions, not cryptographically verified by ARGUS application code.
2. **Application software does not independently prove RTSPS certificate verification**: OpenCV `cv2.VideoCapture` lacks knobs for CA trust verification, hostname pinning, or certificate revocation.
3. **VLAN segmentation alone is not encryption**: Plaintext RTSP on an isolated VLAN protects against inter-subnet snooping, but packets within the broadcast domain remain unencrypted.
4. **Encrypted physical transport requires deployment evidence**: Production deployment engineers must produce physical packet capture (`tcpdump` / Wireshark) demonstrating ciphertext.
5. **Actual production packet-level verification remains pending**.

---

## 8. Final Authoritative Verdict

```
U4 APPLICATION HARDENING PASS
U4 DEPLOYMENT REMEDIATION PENDING
```
*(Finding U4 is NOT marked `U4 PASS — CLOSED` pending physical encrypted transport deployment verification.)*
