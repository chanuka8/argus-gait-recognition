# ARGUS AI Security Hardening — U4 Camera / RTSP Stream Transport Security
## Deployment Requirements & Operational Remediation Guide

**Target Security Finding**: U4 — Camera / RTSP Stream Transport Security
**Scope**: Physical CCTV Infrastructure, Network Segmentation, and Encrypted Transport Deployment
**Document Status**: Authoritative Operational Requirement
**Security Classification**: Operational Security Specification

---

## 1. Executive Summary & Security Threat Model

During the authoritative security audit of ARGUS AI (recovered from historical audit finding U4, `docs/thesis_audit/08_security_and_privacy.md`), camera streams were identified as utilizing unencrypted RTSP (`rtsp://`) transport:

> *"Camera / RTSP Stream Transport Security: Video streams from CCTV cameras are transmitted via plaintext RTSP over local network segments. Although credentials are encrypted at rest, stream transit over plaintext RTSP exposes biometric feeds and raw surveillance imagery to network sniffing, unauthorized eavesdropping, and MITM tampering."*

ARGUS AI has implemented comprehensive **Application-Layer Hardening** to enforce transport security policies, sanitize credential leaks, and reject unencrypted streams in strict mode (`ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true`).

However, **application-layer software cannot physically encrypt ethernet or Wi-Fi packets transmitted by legacy camera hardware across unencrypted network segments**. To achieve full closure of finding U4, production deployments must implement verified network-layer or transport-layer encryption.

---

## 2. The OpenCV / FFmpeg Transport Verification Boundary

The ARGUS video capture pipeline utilizes `cv2.VideoCapture` (backed by FFmpeg). Testing has established that the runtime recognizes the `rtsps://` URI scheme:

```
cv2.VideoCapture("rtsps://camera.domain:322/stream1")
```

### Critical Verification Limitations in Runtime:
1. **Lack of CA Trust Controls**: The Python OpenCV interface does not expose fine-grained controls to specify custom CA bundles or verify system certificate trust chains.
2. **No Hostname / SAN Verification**: OpenCV/FFmpeg does not expose an API hook to assert Subject Alternative Name (SAN) pinning or hostname matching.
3. **No Certificate Revocation (CRL/OCSP)**: Neither CRL distribution points nor OCSP stapling can be verified from the Python application layer.
4. **No Certificate Pinning**: Public key pinning cannot be programmatically enforced through standard `cv2.VideoCapture`.

### Operational Rule:
The ARGUS application classifies `rtsps://` streams as `rtsps_candidate`. In strict security mode, an `rtsps_candidate` stream is accepted **only with explicit deployment confirmation** (`allow_rtsps_candidate: true` or `transport_security: "rtsps_verified"`). Deployment engineers must verify that the underlying operating system and camera firmware maintain valid TLS configurations.

---

## 3. Approved Deployment Remediation Architectures

Production deployments must implement one or more of the following approved transport architectures:

```
+-----------------------------------------------------------------------------------------+
|                                APPROVED REMEDIATION OPTIONS                             |
+-----------------------------------------------------------------------------------------+

  Option A: Encrypted Point-to-Point Tunnel (Recommended)
  [Physical CCTV] ---> (Plaintext RTSP) ---> [Edge Gateway] ===(WireGuard/IPSec)====> [ARGUS Host]
                                                              (Encrypted Tunnel)

  Option B: Isolated CCTV Surveillance VLAN
  [Physical CCTV] ===(Isolated VLAN 802.1Q - Strict ACLs - Zero Internet)==========> [ARGUS Host]

  Option C: Authenticated TLS Media Proxy
  [Physical CCTV] ---> (Plaintext RTSP) ---> [Proxy: MediaMTX] ===(TLS / HTTPS)=====> [ARGUS Host]

  Option D: Native Verified RTSPS
  [Enterprise CCTV (Signed CA Cert)] =======(Native TLS / RTSPS)===================> [ARGUS Host]
```

### Option A: Encrypted Point-to-Point Tunnel (WireGuard / IPSec) — Recommended
Encapsulate all camera stream traffic within an authenticated, encrypted VPN tunnel:
- **WireGuard**: Deploy a lightweight WireGuard peer on the camera network switch/gateway and on the ARGUS server.
- **IPSec / StrongSwan**: Deploy site-to-site IPSec with AES-GCM-256 encryption.
- **ARGUS Configuration**:
  ```yaml
  cameras:
    camera_01:
      id: "camera_01"
      url: "rtsp://10.100.0.10:554/stream1"
      transport_security: "wireguard"
      tunnel_type: "wireguard_site_to_site"
      is_tunnel: true
  ```

### Option B: Isolated Surveillance VLAN with Microsegmentation
Where camera firmware cannot run VPN agents and edge gateways are unavailable:
- Place all cameras on an isolated physical or 802.1Q virtual LAN (e.g., VLAN 200).
- Prohibit direct routing between the camera VLAN and general corporate/public subnets.
- Enforce stateful firewall rules permitting only unicast traffic from camera IPs to the ARGUS server ingress IP on port 554/RTSP.
- Prohibit Internet access (WAN egress) from all camera devices.
- **ARGUS Configuration**:
  ```yaml
  cameras:
    camera_01:
      id: "camera_01"
      url: "rtsp://192.168.200.10:554/stream1"
      transport_security: "protected_tunnel"
      tunnel_type: "isolated_vlan_microsegmentation"
      is_tunnel: true
  ```

### Option D: Native Verified RTSPS
For enterprise CCTV devices supporting native RTSPS over TLS (typically port 322 or 443):
- Install a certificate issued by an enterprise internal CA onto each camera.
- Ensure the ARGUS host trust store contains the enterprise root CA.
- Configure DNS hostnames matching the certificate Common Name or Subject Alternative Name.
- **ARGUS Configuration**:
  ```yaml
  cameras:
    camera_01:
      id: "camera_01"
      protocol: "rtsps"
      host: "cam01.surveillance.internal"
      port: 322
      path: "/live"
      allow_rtsps_candidate: true
      transport_security: "rtsps_verified"
  ```

---

## 4. Application Strict-Mode Policy Configuration

To activate strict-mode transport enforcement in ARGUS AI, configure the following environment variable:

```bash
# In production environment (.env or systemd service)
ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true
```

When strict mode is enabled:
1. **Plaintext RTSP (`rtsp://`)** is immediately **REJECTED** unless `is_tunnel: true` or `transport_security: "protected_tunnel"` is declared.
2. **Plaintext HTTP (`http://`)** is immediately **REJECTED**.
3. **RTSPS (`rtsps://`)** requires explicit deployment confirmation (`allow_rtsps_candidate: true` or `transport_security: "rtsps_verified"`).
4. **Local devices (USB, DirectShow, local files)** are **PERMITTED** without network transport restrictions.
5. **No silent downgrade**: Under no circumstances will a failed encrypted stream fall back to unencrypted RTSP.

---

## 5. Deployment Verification Checklist

Before marking finding U4 as fully remediated in production, the deployment engineer must complete and sign the following checklist:

| Item | Requirement | Verification Command / Method | Status |
|---|---|---|---|
| 1 | Camera network segment isolated from general LAN | Check routing table & firewall rules | PENDING |
| 2 | Tunnel or VLAN encryption active | `tcpdump -i wg0` or packet inspection confirms ciphertext | PENDING |
| 3 | Camera WAN internet access disabled | Verify cameras cannot reach external DNS/NTP | PENDING |
| 4 | ARGUS strict mode enabled | `ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true` | READY |
| 5 | Camera passwords stored in encrypted store | `security_layer/credentials.py` credential vault | VERIFIED |
| 6 | Synthetic URL and log sanitization active | Automated test suite passes 100% | VERIFIED |

---

## 6. Verdict Classification Policy

Under the Zero False Positive Evidence-Based Reporting Policy:
- **Application Hardening**: Fully implemented and verified (`U4 APPLICATION HARDENING PASS`).
- **Physical Transport Remediation**: Remains pending until physical camera network is configured (`U4 DEPLOYMENT REMEDIATION PENDING`).
- **Final Verdict**: Finding U4 cannot be marked `U4 PASS — CLOSED` until physical deployment evidence is recorded.
