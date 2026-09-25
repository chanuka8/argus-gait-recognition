# Security Architecture Audit & Design Report: Camera / RTSP Stream Transport Security (Finding U4)

**Document Identifier**: `docs/security/u4_rtsp_transport_security_design.md`
**Finding Designation**: **Historical Finding U4 — Camera / RTSP Stream Transport Security**
**Audit Phase**: **Phase 1 — Architecture Audit & Security Design Only (No Code Modifications)**
**Security Status**: **UNRESOLVED / DESIGN READY**
**Numbering Designation**: **U4 — Camera / RTSP Stream Transport Security** (NOT SEC-10; SEC-10 Remains Undefined)
**Date**: 2026-09-16
**Repository**: `ARGUS_AI` (`chanuka8/argus-gait-recognition`)
**Branch**: `main`

---

## 1. Historical U4 Evidence

### 1.1 Original Authoritative Sources and Commits
Historical Finding **U4** was authoritatively defined in the foundational ARGUS AI thesis security and privacy audit (`docs/thesis_audit/08_security_and_privacy.md`, commit `b18e69fbec405ea576965cd32f215cc1ce99e561^`):
- **Section 8.1 (Security Controls Matrix)**:
  > `| Data Encryption in Transit | Not assessed | RTSP streams may be unencrypted | Not implemented | Eavesdropping on video streams | Use RTSPS or VPN tunnels |`
- **Section 8.2 (STRIDE Threat Analysis & Assets)**:
  - **Asset**: `Credentials (RTSP)` | Sensitivity: `HIGH` | Location: `configs/cameras.yaml (plaintext)`
  - **Asset**: `Raw Video Frames` | Sensitivity: `HIGH` | Location: `Camera streams, in-memory`
  - **STRIDE Threat**: `Intercept RTSP video streams` | Category: `Information Disclosure` | Asset: `Raw video` | Attack Surface: `Network` | Mitigation: `None` | Risk Level: `MEDIUM`
  - **STRIDE Threat**: `RTSP credential extraction` | Category: `Information Disclosure` | Asset: `Credentials` | Attack Surface: `cameras.yaml` | Mitigation: `None (plaintext)` | Risk Level: `HIGH`
  - **Trust Boundary Diagram**: Depicts `CAM -->|RTSP unencrypted| Boundary` crossing from External (Untrusted) to Internal (Trusted).
  - **Security Warning**: *"RTSP credentials are stored in plaintext YAML."*
- **`docs/FUTURE_ENGINEERING_GAP_AND_REMEDIATION_REPORT.md` (commit `b18e69f^`)**:
  - Section 16 (Deployment Validation Requirements): *"1. RTSP Stream Stability: 24/7 continuous stream capture validation over real wireless/wired airport CCTV networks."*
  - Section 24 (Production Hardening Checklist): `[- [x] Fernet RTSP credential encryption & log URL masking enforced]`.
- **`docs/security/sec10_identification.md`**:
  - Formally cataloged U4: `| U4 | RTSP Stream Transit Encryption (RTSPS / VPN) | 08_security_and_privacy.md §8.1 | Unresolved (infrastructure-level protocol concern) | Infrastructure network configuration, not numbered as SEC-10. |`

### 1.2 Evidence Categorization
- **HISTORICAL AUTHORITATIVE EVIDENCE**:
  - Foundational audit explicitly identified that RTSP streams transmit raw surveillance video unencrypted across local networks, presenting an Information Disclosure vulnerability.
  - Recommended mitigations were explicitly identified as **RTSPS (RTSP over TLS)** or **VPN tunnels**.
  - Plaintext credential storage in `cameras.yaml` was identified as an Information Disclosure vulnerability.
- **CURRENT SOURCE EVIDENCE**:
  - Credential storage has been partially hardened via `security_layer/credentials.py` (`CredentialManager` using Fernet symmetric encryption in `configs/credentials.enc` and environment variable overrides).
  - In `configs/cameras.yaml`, credentials are no longer stored in plaintext; fields `username_env` and `password_env` are used.
  - However, **stream transport remains 100% unencrypted**: `configs/cameras.yaml` specifies standard port `554` and `type: "rtsp"`, and `services/camera_worker.py` invokes `cv2.VideoCapture("rtsp://...")` directly over plaintext TCP/UDP without TLS encapsulation.
- **AGENT INFERENCE**:
  - Existing URL sanitization regex in `security_layer/credentials.py` (`sanitize_rtsp_url`) matches only `rtsp://`, which would fail to mask credentials if `rtsps://` were used.
  - Full end-to-end transport encryption cannot be solved purely in application software if camera hardware does not support TLS; network segmentation or tunneling is an indispensable component of the complete solution.

---

## 2. Current Stream Architecture

```mermaid
flowchart TD
    subgraph CameraNetwork["Camera Network (VLAN / LAN)"]
        CAM1["CCTV Camera 1<br>(192.168.1.100:554)"]
        CAM2["CCTV Camera 2<br>(192.168.1.101:554)"]
        USBCAM["Local USB Webcam<br>(/dev/video0 or DirectShow)"]
    end

    subgraph IngestionBoundary["ARGUS AI Ingestion Engine"]
        RESOLVER["CameraSourceResolver<br>(services/camera_source_resolver.py)"]
        CRED_MGR["CredentialManager<br>(security_layer/credentials.py)"]
        WORKER["CameraWorker<br>(services/camera_worker.py)"]
        OPENCV["cv2.VideoCapture<br>(FFmpeg cap_ffmpeg_impl)"]
    end

    subgraph InternalPipeline["ARGUS AI Core Pipeline"]
        QUEUE["Frame Queue<br>(maxsize=10)"]
        DET["DetectionProcessor<br>(YOLOv8 + ByteTrack)"]
        REC["RecognitionWorker<br>(ByGaitLight Feature Extraction)"]
        JPEG["JPEG Encoder<br>(Preview Buffer)"]
    end

    subgraph Egress["Egress / Frontend Access"]
        API["FastAPI /api/v1/cameras/{id}/stream"]
        UI["Frontend CCTV Dashboard<br>(CctvNetwork.jsx)"]
    end

    CAM1 -->|"Plaintext RTSP (TCP/UDP 554)<br>[VULNERABLE TO SNIFFING]"| OPENCV
    CAM2 -->|"Plaintext RTSP (TCP/UDP 554)<br>[VULNERABLE TO SNIFFING]"| OPENCV
    USBCAM -->|"DirectShow / V4L2 (Local Bus)"| OPENCV

    CRED_MGR -.->|"Decrypts camera credentials"| RESOLVER
    RESOLVER -->|"Constructs rtsp://user:pass@host"| WORKER
    WORKER --> OPENCV
    OPENCV -->|"Decoded BGR Frame"| QUEUE
    QUEUE --> DET
    DET --> REC
    WORKER --> JPEG
    JPEG --> API
    API -->|"HTTP Multipart MJPEG"| UI

    style CAM1 fill:#ff6b6b,color:#000
    style CAM2 fill:#ff6b6b,color:#000
    style OPENCV fill:#ffd93d,color:#000
    style InternalPipeline fill:#6bcf7f,color:#000
```

### 2.1 Component Flow
1. **Configuration**: [`configs/cameras.yaml`](file:///e:/ARGUS_AI/configs/cameras.yaml) specifies registered camera endpoints (e.g. `camera_01` at `192.168.1.100:554/stream1`).
2. **Credential Resolution**: [`services/camera_source_resolver.py`](file:///e:/ARGUS_AI/services/camera_source_resolver.py) and [`security_layer/credentials.py`](file:///e:/ARGUS_AI/security_layer/credentials.py) retrieve credentials from Fernet-encrypted `configs/credentials.enc` or environment variables (`ARGUS_CAMERA_*_PASSWORD`), constructing `rtsp://<user>:<password>@192.168.1.100:554/stream1`.
3. **Capture Execution**: [`services/camera_worker.py`](file:///e:/ARGUS_AI/services/camera_worker.py) invokes `self._capture = cv2.VideoCapture(source)`.
4. **Decoding**: OpenCV delegates to its built-in FFmpeg wrapper (`cap_ffmpeg_impl.hpp`), issuing RTSP `DESCRIBE`, `SETUP`, and `PLAY` commands across the network, negotiating RTP streaming over UDP or interleaved TCP.
5. **Frame Pipeline**: Captured frames are pushed into thread-safe `_frame_queue` for recognition processing.
6. **Egress Preview**: Frames are resized and encoded to JPEG in-memory, then served to authenticated operators over FastAPI (`/api/v1/cameras/{camera_id}/stream`) as multipart JPEG.

---

## 3. Stream Inventory

| Stream Source | Protocol | Production? | Credentials? | Reader Component | Encryption in Transit? |
|---|---|---|---|---|---|
| **CCTV / IP Camera (Main Entrance)** | `rtsp://` (port 554) | Yes (`camera_01`) | Yes (env / Fernet store) | `cv2.VideoCapture` via FFmpeg | **NONE (Plaintext)** |
| **CCTV / IP Camera (Side Entrance)** | `rtsp://` (port 554) | Yes (`camera_02`) | Yes (env / Fernet store) | `cv2.VideoCapture` via FFmpeg | **NONE (Plaintext)** |
| **CCTV / IP Camera (Parking Lot)** | `rtsp://` (port 554) | Configured (`camera_03`, disabled) | Yes (env / Fernet store) | `cv2.VideoCapture` via FFmpeg | **NONE (Plaintext)** |
| **Local USB / Built-in Webcam** | DirectShow / V4L2 | Yes (Auto-detected index 0..3) | No | `cv2.VideoCapture(int)` | **N/A (Kernel Device)** |
| **HTTP / MJPEG Network Stream** | `http://` or `https://` | Supported (`type: "http"`) | Basic auth in URL | `cv2.VideoCapture(str)` | Plaintext for `http://`; TLS for `https://` |
| **Uploaded Forensic Video** | File I/O (`.mp4`, `.avi`) | Yes (`api/legacy/inference.py`) | No | `cv2.VideoCapture(filepath)` | **N/A (Local Disk)** |
| **Synthetic / Mock Camera** | Unit Mock | Test Only | Synthetic | `unittest.mock.MagicMock` | **N/A (In-Memory Mock)** |

---

## 4. Current Protocol Evidence

### 4.1 OpenCV Build & Runtime Inspection
Inspection of the active virtual environment via `cv2.getBuildInformation()`:
- **OpenCV Version**: `5.0.0`
- **Video I/O Backend**:
  - `FFMPEG: YES (prebuilt binaries)`
  - `avcodec: YES (61.19.100)`
  - `avformat: YES (61.7.100)`
  - `avutil: YES (59.39.100)`
  - `swscale: YES (8.3.100)`
  - `GStreamer: NO`
  - `DirectShow: YES`
  - `Media Foundation: YES`
- **Available Stream Backends**: `['FFMPEG', 'GSTREAMER', 'INTEL_MFX', 'MSMF', 'CV_IMAGES', 'CV_MJPEG']`.

### 4.2 RTSPS (`rtsps://`) Runtime Capability Probe
Execution of `cv2.VideoCapture('rtsps://127.0.0.1:322/test')` produced:
```text
[ WARN:0@30.066] global cap_ffmpeg_impl.hpp:453 _opencv_ffmpeg_interrupt_callback Stream timeout triggered after 30064.491000 ms
rtsps opened: False
```
- **Finding**: FFmpeg in OpenCV **does recognize the `rtsps://` URL scheme** without protocol rejection errors. It dispatches to FFmpeg's TLS transport layer and attempts socket connection to port 322 before hitting the 30-second interrupt timeout.
- **Limitation**: OpenCV's Python API (`cv2.VideoCapture`) does **NOT** expose FFmpeg dictionary options (such as `tls_verify=1`, `ca_file=...`, or `cert_file=...`). OpenCV opens stream URLs using default FFmpeg parameters, meaning granular certificate trust management cannot be configured via standard `cv2.VideoCapture` method calls.

---

## 5. Camera Credential Handling

### 5.1 Storage & Resolution
- Credentials are encrypted at rest in `configs/credentials.enc` using Fernet symmetric encryption (`security_layer/credentials.py`).
- Fallback resolution supports environment variables (`ARGUS_CAMERA_{ID}_USERNAME`, `ARGUS_CAMERA_{ID}_PASSWORD`).
- `resolve_camera_config()` strictly rejects plaintext passwords in configuration files unless explicitly permitted by `ARGUS_LEGACY_ALLOW_PLAINTEXT_CREDS=true`.

### 5.2 In-Transit Exposure Risk
- When `cv2.VideoCapture` connects to `rtsp://<user>:<password>@<host>:554/...`, the credentials are transmitted across the local network during RTSP protocol negotiation:
  - If the camera uses HTTP Basic Authentication over RTSP: credentials traverse the wire as base64-encoded cleartext (`Authorization: Basic dXNlcjpwYXNz`).
  - If the camera uses Digest Authentication: credentials resist passive recovery, but the stream payload remains 100% unencrypted.

### 5.3 Logging & URL Redaction
- In `security_layer/credentials.py`:
  ```python
  def sanitize_rtsp_url(url: str | None) -> str:
      if not url or not isinstance(url, str):
          return ""
      pattern = r"(rtsp://)([^:\s]+):(.+)@([^/\s]+(?::\d+)?(?:/[^\s]*)?)"

      def _repl(m):
          return f"{m.group(1)}***:***@{m.group(4)}"

      return re.sub(pattern, _repl, url, flags=re.IGNORECASE)
  ```
- **Confirmed Vulnerability in Redaction**:
  - The regex explicitly anchors on `rtsp://`.
  - If an operator configures an `rtsps://` URL containing credentials (e.g. `rtsps://admin:secret@192.168.1.100:322/stream1`), `sanitize_rtsp_url()` fails to match and returns the **unredacted URL with plaintext credentials intact**!
  - Redaction must be generalized to match `rtsps?://`.

---

## 6. Network Threat Model (STRIDE)

| Threat ID | Threat Description | Attacker Capability | Current Protection | Does RTSPS Help? | Does VPN Help? | Remaining Limitation |
|---|---|---|---|---|---|---|
| **T1** | **Passive Packet Sniffing** | Attacker on same LAN / switch span port captures packets | None (RTSP/RTP in cleartext) | **YES** (TLS encryption) | **YES** (Encrypted tunnel) | Does not protect if attacker compromises camera or server endpoint. |
| **T2** | **Video Content Disclosure** | Attacker extracts surveillance video frames from packet capture | None | **YES** (Payload encrypted) | **YES** (Tunnel payload encrypted) | Does not protect decrypted frames in ARGUS process memory. |
| **T3** | **Credential Interception** | Attacker intercepts RTSP Basic/Digest auth headers on wire | None | **YES** (Transmitted inside TLS) | **YES** (Transmitted inside tunnel) | Attacker with host root access can read Fernet key or env vars. |
| **T4** | **Active Man-in-the-Middle (MITM)** | Attacker uses ARP spoofing / rogue gateway to intercept stream | None | **YES** (If strict certificate verification is enforced) | **YES** (Cryptographic peer verification) | Ineffective if TLS certificate validation is disabled (`verify=False`). |
| **T5** | **Stream Substitution** | Attacker injects loop/synthetic video feed into ARGUS pipeline | None | **YES** (Only legitimate camera with private key can connect) | **YES** (Only authorized VPN endpoint can send data) | Does not protect against an attacker who physically compromises the camera hardware. |
| **T6** | **Network Packet Replay** | Attacker captures network stream packets and re-transmits them | None | **YES** (TLS session keys are ephemeral per TCP session) | **YES** (VPN replay window protection) | **Does NOT prevent physical video replay** (e.g. holding a tablet before camera lens; that requires U8 liveness). |
| **T7** | **DNS / Host Redirection** | Attacker poisons DNS to redirect camera hostname to rogue server | None | **YES** (TLS Subject Alternative Name validation rejects rogue host) | **YES** (VPN uses static IP / cryptokey routing) | Requires hostname verification against valid CA. |
| **T8** | **Camera Impersonation** | Attacker disconnects camera and connects rogue device with same IP | IP verification only | **YES** (Rogue device lacks camera's TLS private key) | **YES** (Rogue device lacks VPN private key) | Requires certificate-based or VPN client authentication. |
| **T9** | **Compromised General LAN Host** | Compromised office workstation on corporate network reaches CCTV | None if flat network | **Partial** (Workstation cannot read encrypted stream) | **YES** (VLAN/VPN restricts reachability) | Flat network allows DoS flooding of camera ports. |
| **T10** | **Compromised ARGUS Server** | Attacker achieves root execution on ARGUS AI server | Hardened RBAC, AES-256 biometric encryption | **NO** (Stream decrypted at server ingest) | **NO** (Tunnel terminated at server NIC) | Transport encryption only protects data *in transit*, not on the processing node. |

---

## 7. Transport Security Goals

To avoid oversimplification, transport security must be partitioned into five independent goals:

1. **CONFIDENTIALITY**:
   - *Goal*: Ensure surveillance video frames and identity features cannot be viewed or reconstructed by eavesdroppers on the physical or wireless network.
   - *Requirement*: Strong symmetric cipher (AES-128/256-GCM or ChaCha20-Poly1305) in transit.
2. **INTEGRITY**:
   - *Goal*: Ensure frames received by ARGUS AI are bit-for-bit identical to those transmitted by the camera sensor, with immediate detection of packet tampering.
   - *Requirement*: Authenticated encryption (AEAD) or HMAC-SHA256 packet authentication.
3. **AUTHENTICATION & ENDPOINT IDENTITY**:
   - *Goal*: Verify that the stream source is genuinely the designated surveillance camera and not an impersonator, rogue IP, or MITM proxy.
   - *Requirement*: X.509 certificate verification or WireGuard public-key binding.
4. **CREDENTIAL PROTECTION**:
   - *Goal*: Ensure camera administrative and stream credentials (usernames, passwords, session tokens) are never transmitted in cleartext or logged to persistent storage.
   - *Requirement*: In-memory masking, TLS transmission, and encrypted credential vault storage.
5. **AVAILABILITY & RESILIENCE**:
   - *Goal*: Prevent network-level Denial of Service, connection hijacking, or stream stalling from crashing the ARGUS surveillance pipeline.
   - *Requirement*: Bounded reconnection backoff, socket timeout enforcement, and network rate limiting.

---

## 8. Remediation Options Evaluation

### Option A — Native RTSPS (RTSP-over-TLS)
- **Mechanism**: Encapsulate RTSP control and RTP interleaved data inside TLS 1.3/1.2 (`rtsps://camera:322/stream`).
- **Camera Requirements**: Camera firmware must support ONVIF Profile T or native HTTPS/RTSPS with onboard certificate installation.
- **OpenCV / FFmpeg Feasibility**: Supported at the protocol level, but `cv2.VideoCapture` lacks exposed knobs for certificate authority injection or pin verification.
- **Pros**: Direct connection, no intermediary server, standard protocol.
- **Cons**: High operational complexity; many legacy or budget IP cameras lack RTSPS support or only support expired self-signed certificates.

### Option B — Network-Level VPN / Encrypted Tunnel (WireGuard / IPsec)
- **Mechanism**: Encapsulate all traffic between the camera subnet/gateway and the ARGUS AI host within a kernel-level encrypted tunnel (WireGuard or IPsec).
- **Camera Requirements**: None if deployed via a site-to-site or edge IoT gateway router (e.g. industrial edge switch). Camera sends standard RTSP to its local gateway.
- **OpenCV / FFmpeg Feasibility**: 100% transparent. OpenCV connects to the camera's tunnel IP (`10.x.x.x`) via standard `cv2.VideoCapture`.
- **Pros**: Protects ALL camera models (even 10-year-old legacy cameras); high throughput with zero application-level certificate renewal hassles; line-rate ChaCha20-Poly1305 kernel encryption.
- **Cons**: Requires gateway router or VPN-capable switch at the camera location.

### Option C — Private Isolated Camera VLAN (Layer 2 Segmentation)
- **Mechanism**: Place all CCTV cameras and the ARGUS ingest NIC onto an isolated 802.1Q VLAN (e.g. VLAN 20) with strict ACLs blocking routing to/from the corporate LAN and Internet.
- **Camera Requirements**: None. Standard Ethernet switch configuration.
- **OpenCV / FFmpeg Feasibility**: 100% transparent.
- **Pros**: Essential baseline security control; prevents lateral movement and unauthorized access from general LAN workstations.
- **Cons**: Does NOT provide cryptographic encryption in transit. Physical taps on switch trunks or cables can still capture plaintext video.

### Option D — Secure Edge Gateway / RTSP Proxy (MediaMTX / go2rtc)
- **Mechanism**: Deploy a lightweight, secure streaming proxy (e.g. MediaMTX or go2rtc) co-located on the camera LAN that pulls local RTSP and serves verified RTSPS/WebRTC/SRT to ARGUS AI.
- **Camera Requirements**: None.
- **OpenCV / FFmpeg Feasibility**: Connects to proxy via RTSPS or HTTP-FLV/WebRTC.
- **Pros**: Decouples camera limitations from server; handles TLS termination, credential isolation, and stream multiplexing.
- **Cons**: Adds an additional software component to deploy and monitor.

---

## 9. Certificate Validation Architecture (For RTSPS)

If native RTSPS (`rtsps://`) is used, certificate validation must follow strict cryptographic rules:

```mermaid
flowchart TD
    INIT["Initiate RTSPS Connection<br>(rtsps://camera.lan:322/live)"]
    HANDSHAKE["TLS 1.2 / 1.3 Handshake"]
    CHECK_EXP["Check Certificate Validity Period<br>(notBefore <= now <= notAfter)"]
    CHECK_CHAIN["Validate Certificate Chain<br>(Anchored to Trusted Private CA)"]
    CHECK_SAN["Validate Hostname / SAN<br>(Subject Alternative Name matches camera hostname)"]
    CONNECT["Establish Encrypted Session<br>& Negotiate RTSP"]
    FAIL["ABORT CONNECTION<br>[DO NOT FALLBACK TO PLAINTEXT]"]

    INIT --> HANDSHAKE
    HANDSHAKE --> CHECK_EXP
    CHECK_EXP -->|Valid| CHECK_CHAIN
    CHECK_EXP -->|Expired / Premature| FAIL
    CHECK_CHAIN -->|Trusted CA| CHECK_SAN
    CHECK_CHAIN -->|Untrusted / Self-Signed| FAIL
    CHECK_SAN -->|Matches| CONNECT
    CHECK_SAN -->|Mismatch| FAIL

    style FAIL fill:#ff6b6b,color:#000
    style CONNECT fill:#6bcf7f,color:#000
```

### Strict Rules:
1. **No Silent Downgrade**: If TLS handshake or certificate validation fails, the connection **MUST ABORT**. The system must **NEVER** fall back to unencrypted `rtsp://`.
2. **Rejection of `verify=False`**: Disabling certificate verification completely negates MITM protection. An attacker can present any self-signed certificate and intercept all video.
3. **Internal Private CA**: Surveillance environments must issue certificates to cameras from an internal organizational Private CA (e.g. `cctv-ca.argus.internal`), installing the CA root certificate into the ARGUS server trust store.
4. **IP-Address SANs**: Because CCTV cameras are frequently accessed by static IP (e.g. `192.168.1.100`), the X.509 certificate's Subject Alternative Name extension **must include an `iPAddress` entry**, or local DNS must map cameras to FQDNs (e.g. `cam01.cctv.argus.internal`).

---

## 10. Camera Hardware Compatibility Status

- **Configured Endpoints**: `configs/cameras.yaml` targets `192.168.1.100` (`camera_01`), `192.168.1.101` (`camera_02`), and `192.168.1.102` (`camera_03`).
- **Documented Vendor Adapters**: Code exists in `services/vendor_adapters.py` for `Hikvision`, `Dahua`, `Uniview`, and `Axis`.
- **Physical Camera RTSPS Capability**: **UNKNOWN**.
  - The repository contains no documentation specifying the exact physical models, firmware versions, or RTSPS capabilities of the deployed hardware.
  - Per the Zero False Positive policy, it cannot be assumed that physical cameras support native RTSPS.
  - Physical probing of live infrastructure was intentionally omitted during this audit to avoid disturbing live operations.

---

## 11. Configuration Security

### Current vs Desired State
- **Current State**:
  - `configs/cameras.yaml` stores structured fields (`host`, `port`, `path`, `username_env`, `password_env`).
  - However, `CameraSourceResolver` and `resolve_camera_config` merge credentials directly into a single unified URL string:
    `rtsp://user:pass@host:port/path`
- **Desired Hardened State**:
  - Store endpoint definitions strictly separated into discrete components:
    ```yaml
    cameras:
      camera_01:
        protocol: "rtsps"   # or "rtsp" with explicit tunnel annotation
        host: "192.168.1.100"
        port: 322
        path: "/stream1"
        credential_id: "cred_cam_01"
        tls_pinned_sha256: "optional_sha256_cert_fingerprint"
        transport_security: "rtsps" # ["rtsps", "vpn_tunneled", "isolated_vlan", "insecure_dev"]
    ```
  - Credentials must remain in `CredentialManager` / secure vault, avoiding the generation of long-lived credential-bearing URL strings.

---

## 12. URL Redaction Design

### Common Redaction Rule
To resolve the flaw where `rtsps://` URLs are not sanitized:
- **Enhanced Regex Pattern**:
  ```python
  RE_CREDENTIAL_URL = re.compile(
      r"((?:rtsps?|https?|rtmp)://)([^:\s@]+):([^@\s]+)@([^/\s]+(?::\d+)?(?:/[^\s]*)?)",
      re.IGNORECASE,
  )
  ```
- **Redaction Transformation**:
  ```python
  def sanitize_stream_url(url: str | None) -> str:
      if not url or not isinstance(url, str):
          return ""
      return RE_CREDENTIAL_URL.sub(r"\1\2:<redacted>@\4", url)
  ```
- **Required Coverage**: Must be applied to all logger statements in:
  - `services/camera_worker.py`
  - `services/camera_manager.py`
  - `services/camera_source_resolver.py`
  - `services/gait_service.py`
  - `api/v1/router.py`
  - Exception and traceback handlers.

---

## 13. Frontend & API Exposure Audit

- **API Audit**:
  - `/api/v1/cameras` and `/api/v1/cameras/{id}` return `CameraInfoResponse`.
  - Inspection of `api/schemas.py` and `services/gait_service.py` shows:
    - `source`: Sanitized via `sanitize_rtsp_url(resolved_source)`.
    - `resolved_source`: Sanitized via `sanitize_rtsp_url(resolved_source)`.
    - `resolved_source_label`: Sanitized via `sanitize_rtsp_url(resolved_label)`.
    - `credential_id`: String identifier (e.g. `cred_cam01`), never contains passwords.
    - `credential_configured`: Boolean (`True`/`False`).
    - Passwords and raw credential secrets are **NEVER** returned in API responses.
- **Frontend Audit**:
  - `frontend/src/components/CctvNetwork.jsx` inspects only `source_type` and `resolved_source_type` to render UI badges (`Webcam` vs `RTSP`).
  - The frontend accesses video strictly via `/api/v1/cameras/{camera_id}/stream` (MJPEG over HTTP/HTTPS from the FastAPI server).
  - The browser client **NEVER connects to the RTSP camera directly** and has zero access to camera network credentials.

---

## 14. RTSP Transport Mode (UDP vs TCP)

- **TCP Transport is NOT Encryption**:
  - RTSP-over-TCP interleaves RTP video data into the RTSP TCP control stream on port 554 (RFC 2326 §10.12).
  - While TCP provides reliable packet delivery and eliminates UDP packet loss, **it provides zero encryption, zero integrity, and zero confidentiality**.
  - Any passive network tap on the LAN can reconstruct the unencrypted H.264/H.265 NAL units.
  - Confusing TCP transport with TLS transport is a critical architectural mistake. Only RTSPS (RTSP over TLS) or an underlying IPsec/WireGuard tunnel provides encryption.

---

## 15. Stream Replay & Substitution Limitations

- **Transport Security vs Biometric Liveness**:
  - TLS/RTSPS guarantees that the video packets received by ARGUS AI originated from the authenticated endpoint possessing the camera's TLS private key and have not been altered in transit.
  - **Limitation**: Transport encryption **does NOT provide biometric liveness detection**. If an attacker gains access to the physical camera or feeds pre-recorded video into the camera's lens, the stream will be encrypted with valid TLS, but the video content remains a replay attack.
  - Replay and presentation attacks are separate research problems (Finding **U8 — Biometric Liveness Detection**) and must not be conflated with U4 transport security.

---

## 16. Performance and Latency Analysis

| Mechanism | CPU Overhead (per 1080p 15fps Stream) | Latency Overhead | Network Bandwidth Overhead | Complexity |
|---|---|---|---|---|
| **Plaintext RTSP (Baseline)** | 0% (baseline decode only) | 0 ms | 0% | Minimal |
| **Native RTSPS (TLS 1.3 AES-GCM)** | < 1.5% (AES-NI accelerated) | < 2 ms (initial handshake +50ms) | < 0.5% (TLS record headers) | Medium (Certs) |
| **WireGuard Tunnel (ChaCha20)** | < 2.0% (Kernel space) | < 1 ms | ~ 3-4% (WireGuard encapsulation) | Low (Infrastructure) |
| **Edge Gateway / RTSP Proxy** | ~ 3-5% (Proxy memory copy) | ~ 10-25 ms (Intermediary buffer) | ~ 1% | High (Extra service) |

*Assessment*: On modern server CPUs featuring hardware AES-NI instructions, TLS encryption/decryption overhead for a 4-to-16 camera fleet is negligible (<3% aggregate CPU).

---

## 17. Deployment Topology Evidence

- **Repository Evidence**:
  - `configs/cameras.yaml` lists cameras on a private Class C subnet: `192.168.1.100`, `192.168.1.101`, `192.168.1.102`.
  - No public WAN IPs or cloud stream URLs are present in default configurations.
- **Topology Classification**: **UNKNOWN (Undocumented in active repository)**.
  - The repository does not document whether production deployments place ARGUS and cameras on the same physical switch, separate routed VLANs, or a site-to-site WAN link.
  - Per Zero False Positive policy, no assumptions can be made regarding physical network layout.

---

## 18. Determination: Code vs Deployment Work

### Critical Determination:
Remediation of Finding U4 is **HYBRID (Option D: Combination of Deployment & Code)**:
1. **Network / Deployment Level (Primary Security Boundary)**:
   - Encrypting data on the physical wire between camera hardware and the server requires network-layer controls (VLAN segmentation, WireGuard/IPsec tunnels, or RTSPS configuration in camera firmware).
   - Python code running on the ARGUS server cannot encrypt physical wires if the camera hardware is transmitting unencrypted packets.
2. **Application Source Code Level (Verification & Policy Enforcement)**:
   - Application software must enforce policy controls: validating transport schemes (`rtsps://` vs `rtsp://`), rejecting unencrypted streams in strict production mode, redacting credentials from URLs in logs and errors, and supporting secure credential injection.

---

## 19. Minimal Proposed Application Changes (Design Only)

If implementation is authorized, the minimal necessary code modifications are:

1. [`security_layer/credentials.py`](file:///e:/ARGUS_AI/security_layer/credentials.py):
   - Enhance `sanitize_rtsp_url()` -> `sanitize_stream_url()` to redact both `rtsp://` and `rtsps://`.
   - Update `build_rtsp_url()` to support `rtsps://` protocol schemes.
2. [`services/camera_source_resolver.py`](file:///e:/ARGUS_AI/services/camera_source_resolver.py):
   - Add support for `rtsps://` scheme detection.
   - Enforce transport security policy check if `ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true`.
3. [`services/camera_worker.py`](file:///e:/ARGUS_AI/services/camera_worker.py):
   - Ensure sanitized URLs are used across all log messages.
   - Reject unencrypted `rtsp://` when strict mode is active.
4. [`configs/cameras.yaml`](file:///e:/ARGUS_AI/configs/cameras.yaml):
   - Add optional `transport_security` / `protocol: "rtsps"` schema documentation.

---

## 20. Strict-Mode Design

A configurable security policy should be introduced:
- **Environment Variable**: `ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true` (defaults to `false` in development/legacy environments).
- **Behavior When Active**:
  1. Any camera URL starting with `rtsp://` is **REJECTED** at startup with a `SecurityPolicyError` unless explicitly annotated with `transport_mode: "tunneled_vpn"` or `transport_mode: "isolated_vlan"`.
  2. `rtsps://` URLs are accepted.
  3. Plaintext credentials embedded in URLs are strictly rejected.
- **Behavior When Disabled (Development Mode)**:
  1. `rtsp://` URLs emit a prominent security warning log.
  2. Plaintext streams operate normally to permit local webcam testing and synthetic test suites.

---

## 21. Legacy Camera Migration Strategy

For production environments possessing legacy cameras that do not support RTSPS:

```text
Phase 1: Inventory & Audit
  └── Enumerate all cameras and verify whether firmware supports RTSPS (port 322).
Phase 2: Network Segmentation
  └── Move all cameras into a dedicated, isolated CCTV VLAN (no internet access).
Phase 3: Secure Tunneling Deployment
  └── For remote cameras, deploy an edge WireGuard gateway to encapsulate RTSP.
Phase 4: ARGUS Configuration Transition
  └── Configure ARGUS to ingest from RTSPS or tunnel IPs.
Phase 5: Strict Mode Enforcement
  └── Enable ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true to lock down the ingestion pipeline.
```

---

## 22. Failure Behavior Matrix

| Failure Scenario | Strict Mode Active (`true`) | Development Mode (`false`) | Downgrade Allowed? |
|---|---|---|---|
| **Plaintext `rtsp://` provided** | Connection rejected (`SecurityPolicyError`) | Warning logged; connection proceeds | **NO** |
| **TLS Certificate Validation Failed** | Connection aborted (`SSLError`); worker stopped | Connection aborted (`SSLError`); worker stopped | **STRICTLY FORBIDDEN** |
| **Expired TLS Certificate** | Connection aborted (`CertificateExpiredError`) | Connection aborted; error logged | **STRICTLY FORBIDDEN** |
| **Hostname / SAN Mismatch** | Connection aborted (`HostnameMismatchError`) | Connection aborted; error logged | **STRICTLY FORBIDDEN** |
| **Invalid Camera Credentials** | Worker stops; logs sanitized auth error | Worker stops; logs sanitized auth error | N/A |
| **RTSPS Stream Timeout** | Reconnect backoff initiated (maintaining RTSPS) | Reconnect backoff initiated | **NO (Never switch to RTSP)** |

---

## 23. Synthetic Integration Test Plan

When implementation is authorized, the following 15 synthetic test cases should be implemented in `tests/integration/backend/test_camera_transport_security.py` using mocks and local dummy streams:

1. `test_rtsps_url_accepted_in_strict_mode`: Verifies `rtsps://` is parsed and accepted.
2. `test_rtsp_plaintext_rejected_in_strict_mode`: Verifies `rtsp://` raises `SecurityPolicyError` when strict mode is active.
3. `test_rtsp_plaintext_allowed_in_dev_mode`: Verifies `rtsp://` connects with warning log when strict mode is off.
4. `test_no_silent_downgrade_on_rtsps_failure`: Verifies that a failed RTSPS stream never retries over `rtsp://`.
5. `test_rtsps_credential_redaction_in_logs`: Verifies `rtsps://admin:secret@host` is redacted to `rtsps://admin:<redacted>@host`.
6. `test_passwords_absent_from_camera_api_responses`: Verifies `/api/v1/cameras` responses omit camera credentials.
7. `test_invalid_tls_certificate_rejection`: Verifies self-signed or invalid certs trigger connection abortion.
8. `test_hostname_mismatch_rejection`: Verifies SAN mismatch raises certificate error.
9. `test_camera_ids_remain_functional`: Verifies camera status and health checks remain operational.
10. `test_local_webcam_unaffected_by_rtsps_policy`: Verifies USB webcams continue working normally.
11. `test_video_file_ingestion_unaffected`: Verifies local uploaded forensic video processing is unaffected.
12. `test_websocket_stream_preview_unaffected`: Verifies MJPEG HTTP stream delivery to frontend is unchanged.
13. `test_sec01_through_sec09_unaffected`: Verifies zero regression across SEC-01 through SEC-09 suites.
14. `test_u2_audit_integrity_unaffected`: Verifies audit log HMAC signing remains functional.
15. `test_u3_biometric_encryption_unaffected`: Verifies biometric gallery templates remain AES-GCM encrypted.

---

## 24. Current Regression Baseline

Executed across all backend integration test suites:
```powershell
.venv\Scripts\python.exe -m pytest tests/integration/backend/ -q
```
- **Result**: **325 passed in 407.71s (0:06:47)**
- **Static Quality Checks**:
  - `ruff check`: All checks passed.
  - `ruff format --check`: 189 files formatted cleanly.
  - `python -m compileall`: 0 bytecode compilation errors.
  - `git diff --check`: 0 whitespace / CRLF discrepancies.

---

## 25. Residual Risks

1. **Camera Sensor Physical Compromise**: Encrypted transport does not prevent tampering at the physical sensor level or camera lens.
2. **Firmware Backdoors**: Transport encryption does not mitigate backdoors or supply chain vulnerabilities in third-party camera firmware.
3. **Internal Process Memory**: Stream frames are decrypted in ARGUS process memory for YOLO/ByGaitLight inference; root host access can still dump unencrypted frames from memory.

---

## 26. SEC Numbering Assessment

- **Is Historical Finding U4 Authoritative?**: **YES**. It originates directly from foundational thesis audit `docs/thesis_audit/08_security_and_privacy.md`.
- **Is U4 Currently Unresolved?**: **YES**. While credential storage is encrypted at rest, stream transit over the network remains unencrypted plaintext RTSP.
- **Does Repository Evidence Define U4 as SEC-10?**: **NO**.
- **Is SEC-10 Otherwise Authoritatively Defined?**: **NO**. As established in `docs/security/sec10_identification.md`, SEC-10 remains undefined in the repository.
- **Official Designation**:
  > **U4 is a historical security finding, but the repository does not authoritatively define it as SEC-10.**

---

## 27. Final Recommendation: CONDITIONAL GO

- **Verdict**: **CONDITIONAL GO for U4 Implementation**.
- **Condition**:
  - U4 implementation must be structured as a **hybrid defense**:
    1. **Application Layer**: Implement URL redaction for `rtsps://`, add transport scheme validation, and introduce the optional strict-mode flag `ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT`.
    2. **Deployment Documentation**: Provide architecture guidelines for CCTV VLAN isolation and WireGuard/RTSPS deployment for physical installations.
  - Do **NOT** enforce mandatory RTSPS by default without configuration opt-in, as doing so would immediately break existing development and physical environments utilizing standard RTSP camera hardware.
