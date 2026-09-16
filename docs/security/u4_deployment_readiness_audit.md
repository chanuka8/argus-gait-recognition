# ARGUS AI Security Hardening — U4 Camera / RTSP Stream Transport Security
## Deployment Readiness Audit

**Target Finding**: U4 — Camera / RTSP Stream Transport Security
**Audit Scope**: Deployment Readiness Assessment — Design & Evidence Review Only
**Document Purpose**: Define the exact evidence and infrastructure required to move from `U4 DEPLOYMENT REMEDIATION PENDING` to `U4 PASS — CLOSED`
**Strict Numbering Rule**: U4 remains an unnumbered historical finding. It is NOT named SEC-10 (SEC-10 remains undefined).

**Infrastructure Modification Status**: NONE. This document is an audit and design artifact only. No network infrastructure, camera configuration, VPN/tunnel, or VLAN changes were performed.

---

## 1. Current U4 State

```
U4 APPLICATION HARDENING PASS
U4 DEPLOYMENT REMEDIATION PENDING
```

**Application-layer hardening** is fully implemented and verified:
- Transport classification engine (`validate_camera_transport`) with 7 mutually exclusive categories
- Strict-mode enforcement (`ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true`) rejecting plaintext RTSP/HTTP
- URL credential and query-secret redaction in all logging and error paths
- RTSPS candidate policy requiring explicit operator confirmation
- Fail-closed architecture preventing silent downgrade to plaintext
- 31 focused U4 integration tests — all passing
- 34 legacy camera/configuration unit tests — all passing
- Zero regressions against U2 (31 tests) and U3 (51 tests)

**Deployment-layer remediation** remains pending:
- No physical encrypted transport evidence exists in the repository
- No packet-capture verification has been performed
- No tunnel, VPN, or RTSPS deployment evidence exists
- Camera RTSPS capability is undocumented

---

## 2. Known Deployment Facts

The following facts are confirmed from repository artifacts (`configs/cameras.yaml`, `services/vendor_adapters.py`, `services/onvif_client.py`, test suites):

| Fact | Source | Classification |
|---|---|---|
| 3 cameras configured: `camera_01` (Main Entrance), `camera_02` (Side Entrance), `camera_03` (Parking Lot) | `configs/cameras.yaml` | Repository-confirmed |
| Camera IPs: `192.168.1.100`, `192.168.1.101`, `192.168.1.102` | `configs/cameras.yaml` | Repository-confirmed |
| All cameras use RTSP port `554` with path `/stream1` | `configs/cameras.yaml` | Repository-confirmed |
| All cameras configured as `type: "rtsp"` (plaintext) | `configs/cameras.yaml` | Repository-confirmed |
| Camera `camera_03` (Parking Lot) is `enabled: false` | `configs/cameras.yaml` | Repository-confirmed |
| Credentials resolved via environment variables (`username_env`, `password_env`) | `configs/cameras.yaml` | Repository-confirmed |
| No `transport_security`, `is_tunnel`, or `tunnel_type` fields present in camera config | `configs/cameras.yaml` | Repository-confirmed |
| No `rtsps://` URLs exist in any configuration file | Grep search across `configs/` | Repository-confirmed |
| Vendor adapters exist for Hikvision, Dahua, Axis, Uniview, and Generic RTSP | `services/vendor_adapters.py` | Repository-confirmed |
| All vendor adapters construct `rtsp://` (plaintext) URLs only | `services/vendor_adapters.py` | Repository-confirmed |
| ONVIF client exists for camera discovery and capability queries | `services/onvif_client.py` | Repository-confirmed |
| Target FPS: 15, Resolution: 640×480, Reconnect interval: 5s | `configs/cameras.yaml` | Repository-confirmed |

---

## 3. Unknown Deployment Facts

The following facts are **not documented** in the repository and cannot be determined without physical site access or operator input:

| Unknown | Impact on U4 Closure |
|---|---|
| Physical network topology (flat LAN, segmented VLAN, routed subnets) | Determines whether traffic crosses untrusted boundaries |
| Whether camera IPs `192.168.1.100–102` share a broadcast domain with non-surveillance devices | Determines eavesdropping exposure on the LAN segment |
| Whether any VPN/tunnel infrastructure exists between camera network and ARGUS host | Determines if transport encryption is already in place |
| Whether the ARGUS host is co-located on the same physical switch as cameras | Affects required encryption boundary |
| Physical camera manufacturer, model, and firmware version for each deployed device | Determines RTSPS capability per device |
| Operating system and network stack of the ARGUS production host | Affects VPN/tunnel deployment options |
| Whether the 192.168.1.0/24 subnet has Internet/WAN exposure | Affects remote eavesdropping threat model |
| Whether any wireless segments exist between cameras and ARGUS host | Wireless segments increase eavesdropping surface |
| Site-specific physical security controls (locked cabinets, restricted physical access) | Mitigating control for local eavesdropping |

**NOTE**: Private IP addresses (`192.168.1.x`) confirm a private network range but do NOT prove physical isolation, VLAN segmentation, or encrypted transport. No deployment topology should be inferred from IP addresses alone.

---

## 4. Camera Capability Status

```
CAMERA RTSPS CAPABILITY: UNKNOWN
```

**Repository evidence**:
- `configs/cameras.yaml` documents camera IPs, ports, and stream paths but does **not** record camera manufacturer, model, firmware version, or TLS/RTSPS capability.
- `services/vendor_adapters.py` includes adapters for Hikvision, Dahua, Axis, Uniview, and Generic RTSP. All adapters construct `rtsp://` (plaintext) URLs. None construct `rtsps://` URLs.
- `services/onvif_client.py` includes ONVIF discovery and device information parsing (manufacturer, model, firmware, serial), but no production ONVIF discovery results are stored in the repository.
- Test fixtures (`tests/unit/camera/test_phase5_cctv.py`) reference Hikvision as a mock manufacturer in ONVIF XML responses, but this is test data, not production evidence.

**To determine RTSPS capability**, the deployment operator must:
1. Identify the actual manufacturer, model, and firmware of each physical camera.
2. Consult vendor documentation for TLS/RTSPS support.
3. Verify whether the camera firmware supports certificate management (upload, generation, or SCEP enrollment).
4. Record results per camera in a deployment inventory.

---

## 5. Native RTSPS Closure Path (PATH A)

### Architecture
```
[Camera with RTSPS Firmware] =====(TLS/RTSPS port 322 or 443)=====> [ARGUS Host]
```

### Requirements for Closure

| # | Requirement | Evidence Type |
|---|---|---|
| A1 | Camera firmware supports RTSPS (TLS-wrapped RTSP) | Vendor documentation + firmware version |
| A2 | X.509 certificate installed on camera | Certificate inspection (`openssl s_client`) |
| A3 | Certificate chain trusted by ARGUS host OS trust store | Trust store inspection |
| A4 | Subject Alternative Name (SAN) or Common Name (CN) matches camera hostname/IP | Certificate field inspection |
| A5 | Certificate expiration date is future and monitored | Certificate inspection + monitoring alert |
| A6 | TLS version ≥ 1.2 with strong cipher suite | `openssl s_client -connect` output |
| A7 | No TLS-to-plaintext RTSP downgrade on connection failure | Application log + test: connection refused when TLS fails |
| A8 | Packet capture at observation point confirms encrypted transport | `tcpdump` / Wireshark capture |
| A9 | ARGUS config updated with `allow_rtsps_candidate: true` and `transport_security: "rtsps_verified"` | Config file inspection |
| A10 | OpenCV/FFmpeg runtime TLS behavior documented | Runtime test with certificate mismatch |

### RTSPS Verification Limitation
The ARGUS application uses `cv2.VideoCapture` (FFmpeg backend). The Python OpenCV interface does NOT expose controls for:
- CA trust root specification
- SAN/hostname verification
- Certificate revocation (CRL/OCSP)
- Certificate pinning

RTSPS verification is therefore **operator-attested** at the deployment infrastructure level, not cryptographically proven by the application. The application classifies RTSPS as `rtsps_candidate`, not `verified_secure`.

---

## 6. VPN Closure Path (PATH B)

### Architecture
```
[Camera] ---(Plaintext RTSP)---> [Edge Gateway/Switch] ===(WireGuard/IPSec)====> [ARGUS Host]
                                  (Isolated camera segment)    (Encrypted tunnel)
```

### Requirements for Closure

| # | Requirement | Evidence Type |
|---|---|---|
| B1 | Camera RTSP remains plaintext but confined to isolated camera segment | Network topology diagram + ACL rules |
| B2 | Traffic crossing the untrusted network boundary traverses an authenticated encrypted tunnel | Tunnel interface state (`wg show` / `ipsec status`) |
| B3 | WireGuard or IPSec peer authenticated with public keys or certificates | Tunnel configuration inspection |
| B4 | Encryption active (ChaCha20-Poly1305 for WireGuard; AES-GCM-256 for IPSec) | Tunnel configuration inspection |
| B5 | Routing table directs camera subnet traffic through tunnel interface | `ip route show` / `route print` |
| B6 | No alternate plaintext route exists for camera traffic | Routing table + firewall ACL audit |
| B7 | Firewall/ACL enforces tunnel-only path | Firewall rule inspection |
| B8 | Packet capture at observation point confirms encrypted transport | `tcpdump` at untrusted boundary |
| B9 | Tunnel state monitored with alerting on disconnection | Monitoring/alerting configuration |
| B10 | ARGUS config updated with `transport_security: "protected_tunnel"` and `is_tunnel: true` | Config file inspection |

---

## 7. Secure Gateway Closure Path (PATH C)

### Architecture
```
[Camera] ---(Plaintext RTSP)---> [Media Gateway (e.g. MediaMTX)] ===(TLS/HTTPS)====> [ARGUS Host]
                                  (Isolated camera network)        (Encrypted upstream)
```

### Requirements for Closure

| # | Requirement | Evidence Type |
|---|---|---|
| C1 | Gateway identity documented (software, version, hardening status) | Deployment documentation |
| C2 | Camera-side exposure constrained to isolated network segment | Network topology + ACL rules |
| C3 | Gateway-to-ARGUS path uses authenticated, encrypted protocol (TLS ≥ 1.2) | Certificate inspection + protocol test |
| C4 | Gateway credential storage isolated and protected | Gateway configuration audit |
| C5 | Gateway TLS certificate valid and trusted | Certificate chain inspection |
| C6 | No plaintext fallback from gateway to ARGUS | Gateway configuration + test |
| C7 | Gateway service hardened (minimal privileges, no default credentials, updated) | Service hardening checklist |
| C8 | Gateway logging redacts credentials | Log inspection |
| C9 | Gateway failure behavior documented (no silent revert to plaintext) | Architecture documentation |
| C10 | Packet capture confirms encrypted gateway-to-ARGUS traffic | `tcpdump` / Wireshark capture |

---

## 8. Trust Boundaries

### Deployment Data-Flow Model

```
┌──────────┐     ┌───────────────────────┐     ┌─────────────────────┐     ┌────────────────┐     ┌──────────────────┐
│ Physical │     │ Camera LAN/VLAN       │     │ Gateway / Switch /  │     │ ARGUS Host     │     │ Application      │
│ Camera   │────>│ (Local segment)       │────>│ Network Boundary    │────>│ (OS/Network    │────>│ Memory           │
│          │     │                       │     │                     │     │  Stack)        │     │ (cv2.VideoCapture)│
└──────────┘     └───────────────────────┘     └─────────────────────┘     └────────────────┘     └──────────────────┘
```

| Link | Plaintext Possible? | Encryption Mechanism | Authentication | Trust Boundary | Evidence Required |
|---|---|---|---|---|---|
| Camera → Camera LAN | YES (default RTSP) | None (unless native RTSPS) | RTSP Digest/Basic auth | Untrusted if shared segment | Packet capture on segment |
| Camera LAN → Network Boundary | YES (default) | WireGuard/IPSec/TLS if deployed | Tunnel peer auth / TLS cert | **PRIMARY TRUST BOUNDARY** | Tunnel state + packet capture |
| Network Boundary → ARGUS Host | Depends on path | Tunnel encapsulation or TLS | Tunnel key / TLS cert | Trusted if encrypted | Packet capture at host NIC |
| ARGUS Host NIC → Application Memory | NO (local kernel) | OS kernel/loopback | OS process isolation | Trusted | OS security posture |

### Critical Trust Boundary Identification
The **primary trust boundary** is between the camera LAN segment and the ARGUS host network ingress. If cameras and the ARGUS host share an unencrypted, unsegmented broadcast domain, any device on that domain can passively capture RTSP video streams.

---

## 9. VLAN Role

**VLAN provides segmentation. VLAN alone does NOT provide encryption.**

A dedicated CCTV VLAN (802.1Q) isolates camera traffic from general corporate/office network segments, providing:
- **Access control**: Only authorized ports/hosts can reach camera streams.
- **Broadcast domain isolation**: Camera traffic is not visible to hosts on other VLANs.
- **Reduced attack surface**: Non-surveillance devices cannot directly access camera IP space.

However:
- **Within the VLAN broadcast domain**, traffic remains unencrypted. Any device on the camera VLAN can passively capture RTSP streams.
- **VLAN tagging** is a Layer 2 construct and provides no cryptographic confidentiality or integrity guarantees.
- **VLAN hopping attacks** (double-tagging, switch spoofing) can breach VLAN isolation if switches are not hardened.

**Conclusion**: VLAN-only deployment is **insufficient** to close the original U4 confidentiality finding if plaintext RTSP traffic crosses a threat-relevant network boundary. VLANs are a necessary segmentation control but must be combined with encryption (PATH A, B, or C) for full remediation.

---

## 10. Packet-Capture Verification Design

> **NOTE**: This section defines a FUTURE verification procedure. Packet capture was NOT executed during this audit.

### Purpose
Prove whether surveillance video transport between cameras and the ARGUS host is encrypted.

### Procedure

#### Observation Point
Capture at the network interface on the ARGUS host that receives camera traffic, or at a network tap / mirror port between the camera segment and the ARGUS host.

#### Capture Command (Example)
```bash
# On ARGUS host — capture traffic from camera IP on RTSP port
tcpdump -i <interface> host 192.168.1.100 and port 554 -w /tmp/u4_capture.pcap -c 10000

# For WireGuard tunnel — capture on tunnel interface
tcpdump -i wg0 -w /tmp/u4_tunnel_capture.pcap -c 10000

# For RTSPS — capture on port 322 (or camera-specific TLS port)
tcpdump -i <interface> host 192.168.1.100 and port 322 -w /tmp/u4_rtsps_capture.pcap -c 10000
```

#### Capture Duration
Minimum 60 seconds of active video streaming to ensure sufficient payload data for analysis.

#### Expected Protocol Analysis

| Indicator | Plaintext (FAIL) | Encrypted (PASS) |
|---|---|---|
| RTSP `DESCRIBE` / `SETUP` / `PLAY` commands visible in payload | YES — plaintext exposure confirmed | NO — should not be readable |
| RTP video payload (NAL units, H.264/H.265 headers) visible | YES — video data exposed | NO — encrypted payload |
| Camera credentials (`Authorization: Basic/Digest`) visible | YES — critical credential leak | NO — encrypted or absent |
| TLS handshake (`ClientHello`, `ServerHello`, `Certificate`) visible | N/A | YES — confirms TLS negotiation |
| WireGuard/IPSec encapsulation headers visible | N/A | YES — confirms tunnel encapsulation |
| Random/high-entropy payload bytes | NO — structured video data | YES — encrypted content |

#### Credential/Video Exposure Assessment
1. Open the capture in Wireshark.
2. Apply display filter: `rtsp || rtp || http`.
3. If RTSP commands or RTP video frames are readable, **transport is plaintext — FAIL**.
4. If payload is high-entropy / encrypted and RTSP/RTP dissectors cannot decode content, **transport is encrypted — PASS**.
5. Search for strings: `DESCRIBE`, `SETUP`, `PLAY`, `Authorization`, `Basic`, `Digest` in packet payloads.

#### Capture Protection
Packet captures may contain sensitive surveillance video data and/or camera credentials. Captures must be:
- Stored in a temporary location with restricted file permissions (e.g., `chmod 600`)
- Deleted after verification analysis is complete
- Never committed to version control
- Never transmitted over unencrypted channels
- Access restricted to authorized security/deployment personnel only

---

## 11. RTSPS Verification Checklist

For deployment operators implementing PATH A (Native RTSPS):

| # | Check | Command / Method | Expected Result | Status |
|---|---|---|---|---|
| 1 | Trusted issuing CA | `openssl s_client -connect <camera_ip>:322 -showcerts` | Certificate chain terminates at trusted root CA | PENDING |
| 2 | Certificate chain valid | `openssl verify -CAfile <ca_bundle> <camera_cert>` | `OK` | PENDING |
| 3 | SAN/CN hostname/IP match | `openssl x509 -in <cert> -text -noout \| grep -A1 "Subject Alternative Name"` | Camera IP or hostname present in SAN | PENDING |
| 4 | Certificate not expired | `openssl x509 -in <cert> -checkend 0` | `Certificate will not expire` | PENDING |
| 5 | Algorithm/key strength | `openssl x509 -in <cert> -text -noout \| grep "Public-Key\|Signature Algorithm"` | RSA ≥ 2048-bit or ECDSA ≥ 256-bit; SHA-256+ signature | PENDING |
| 6 | TLS version ≥ 1.2 | `openssl s_client -connect <camera_ip>:322 -tls1_2` | Successful handshake | PENDING |
| 7 | Reconnect behavior | Restart camera, observe ARGUS reconnect attempt | Reconnects to RTSPS, no plaintext fallback | PENDING |
| 8 | Invalid-certificate behavior | Configure invalid/expired cert on camera | ARGUS refuses connection (fail-closed) | PENDING |
| 9 | No plaintext fallback | Block RTSPS port, observe behavior | ARGUS does NOT silently switch to `rtsp://` | PENDING |
| 10 | Camera firmware RTSPS support | Vendor documentation / firmware changelog | RTSPS explicitly listed as supported feature | PENDING |

**Important**: Checks 7–9 depend on the OpenCV/FFmpeg runtime behavior, which the application does NOT fully control. The ARGUS application classifies RTSPS as `rtsps_candidate` and requires operator assertion. These checks verify the operator's assertion is correct.

---

## 12. VPN Verification Checklist

For deployment operators implementing PATH B (WireGuard/IPSec):

| # | Check | Command / Method | Expected Result | Status |
|---|---|---|---|---|
| 1 | Tunnel interface exists and is UP | `wg show` / `ip link show wg0` / `ipsec status` | Interface active with peer | PENDING |
| 2 | Authenticated peer configured | `wg show wg0 \| grep peer` / IPSec cert | Public key or certificate listed | PENDING |
| 3 | Encryption active | `wg show wg0` (shows ChaCha20-Poly1305) / IPSec SA | Cipher suite confirmed | PENDING |
| 4 | Route for camera network through tunnel | `ip route show 192.168.1.0/24` / `route print` | Camera subnet routed via `wg0` or IPSec SA | PENDING |
| 5 | No alternate plaintext route | Routing table audit | No secondary route bypassing tunnel | PENDING |
| 6 | Firewall/ACL enforcement | `iptables -L` / `nft list ruleset` / Windows Firewall rules | Camera traffic blocked except through tunnel | PENDING |
| 7 | Packet capture confirms encryption | `tcpdump -i eth0 host 192.168.1.100` | Encrypted (WireGuard/ESP) payload, no readable RTSP | PENDING |
| 8 | Tunnel state monitoring | Monitoring system configuration | Alert fires on tunnel disconnection | PENDING |
| 9 | Tunnel auto-reconnect | Kill tunnel, observe recovery | Tunnel re-establishes within configured interval | PENDING |
| 10 | ARGUS config declares tunnel | `configs/cameras.yaml` inspection | `transport_security: "protected_tunnel"` and `is_tunnel: true` | PENDING |

---

## 13. Gateway Verification Checklist

For deployment operators implementing PATH C (Secure Media Gateway):

| # | Check | Command / Method | Expected Result | Status |
|---|---|---|---|---|
| 1 | Gateway identity documented | Deployment documentation | Software name, version, hardening baseline | PENDING |
| 2 | Source-side isolation | Network topology audit | Cameras and gateway on isolated segment; no general LAN access | PENDING |
| 3 | Encrypted upstream protocol | `openssl s_client -connect <argus_host>:<port>` | TLS ≥ 1.2 with valid certificate | PENDING |
| 4 | Credential storage protected | Gateway config audit | Credentials encrypted or secured; not plaintext in config files | PENDING |
| 5 | TLS certificate valid | Certificate inspection | Trusted CA, valid dates, correct SAN | PENDING |
| 6 | No plaintext fallback | Block TLS port, observe behavior | Gateway does NOT revert to plaintext upstream | PENDING |
| 7 | Service hardened | Service audit | Minimal privileges, no default credentials, latest patches | PENDING |
| 8 | Logging redaction | Gateway log inspection | No credentials or sensitive URLs in gateway logs | PENDING |
| 9 | Failure behavior documented | Architecture documentation | Documented failure mode; no silent plaintext fallback | PENDING |
| 10 | Packet capture confirms encrypted upstream | `tcpdump` between gateway and ARGUS | TLS-encrypted payload | PENDING |

---

## 14. Operator Attestation vs Technical Proof

This section critically distinguishes between two categories of evidence:

### Operator Attestation (Declarative)
An operator attestation is a configuration declaration stating that a security control is in place. Example:

```yaml
# In configs/cameras.yaml
cameras:
  camera_01:
    transport_security: "protected_tunnel"
    tunnel_type: "wireguard_site_to_site"
    is_tunnel: true
```

**Properties**:
- Recorded in ARGUS configuration files
- Accepted by the ARGUS application as input for transport classification
- Enables strict-mode to permit `rtsp://` URLs that would otherwise be rejected
- **Does NOT prove the tunnel actually exists**
- **Does NOT prove traffic actually traverses the tunnel**
- **Can be set incorrectly (intentionally or accidentally)**

### Technical Proof (Verification)
Technical proof is independently verifiable evidence that the security control is operational. Examples:

```bash
# Prove WireGuard tunnel exists and is active
$ wg show wg0
interface: wg0
  public key: <key>
  listening port: 51820

peer: <peer_key>
  endpoint: <gateway_ip>:51820
  latest handshake: 12 seconds ago
  transfer: 1.24 GiB received, 45.6 MiB sent

# Prove traffic is encrypted at wire level
$ tcpdump -i eth0 host 192.168.1.100 -c 100
# Output shows only WireGuard-encapsulated packets (UDP port 51820)
# No readable RTSP commands or RTP video data visible
```

**Properties**:
- Produced by executing verification commands on production infrastructure
- Independently verifiable by different personnel
- Demonstrates actual operational state, not just configuration intent
- Provides time-stamped evidence

### Critical Rule
**ARGUS configuration alone must NEVER be considered sufficient closure evidence for U4.** An operator attestation is a necessary input for application-layer policy enforcement, but it is not sufficient to prove physical transport encryption. Both operator attestation AND technical proof are required for U4 closure.

---

## 15. Strict-Mode Activation Plan

### Environment Variable
```bash
ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true
```

### Preconditions (All Must Be Met Before Activation)

| # | Precondition | Verification |
|---|---|---|
| 1 | All production camera paths classified by transport category | Config audit: every camera has `transport_security` field |
| 2 | Approved protection method documented for each camera | Deployment documentation per camera |
| 3 | Technical proof (packet capture or equivalent) exists for each camera | Evidence archive |
| 4 | ARGUS application passes all U4 integration tests | `pytest tests/integration/backend/test_camera_transport_security.py` — 31/31 pass |
| 5 | Connectivity test: all cameras accessible via declared secure transport | Operator verification per camera |
| 6 | Rollback plan documented and tested | Operations runbook |
| 7 | Monitoring/alerting configured for transport failures | Monitoring system check |
| 8 | No silent downgrade behavior confirmed | Test: disable encrypted transport, verify ARGUS rejects connection |

### Activation Procedure
1. Deploy encrypted transport for all production cameras (PATH A, B, or C).
2. Update `configs/cameras.yaml` with `transport_security` and related fields per camera.
3. Set `ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true` in production environment.
4. Restart ARGUS application.
5. Verify all cameras connect successfully.
6. Run focused U4 test suite to confirm no regressions.
7. Monitor camera health dashboard for 24 hours.
8. Archive activation evidence.

### Rollback Procedure
If strict-mode activation causes camera connectivity failures:
1. Set `ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=false` (or remove the variable).
2. Restart ARGUS application.
3. Verify cameras reconnect in permissive mode.
4. Investigate and resolve the transport issue.
5. **Do NOT leave cameras in permissive mode permanently** — document a remediation timeline.

---

## 16. Failure / Rollback Plan

### Failure Scenarios and Operator Response

| Scenario | Operator Response | Security Constraint |
|---|---|---|
| Secure camera becomes unreachable | Investigate tunnel/TLS state; check camera power/network | Do NOT silently revert to plaintext RTSP |
| VPN/tunnel fails | Check tunnel interface state; restart tunnel service; verify peer authentication | Camera traffic must stop until tunnel recovers |
| Certificate expires | Renew certificate; restart camera TLS service | Do NOT bypass certificate validation |
| Certificate mismatch (SAN/CN) | Correct certificate or DNS configuration | Do NOT accept mismatched certificates |
| Gateway fails | Restart gateway service; verify upstream TLS | Camera traffic must stop until gateway recovers |
| Strict mode rejects a camera source | Verify camera transport configuration; fix config | Do NOT disable strict mode as a workaround |

### Emergency Plaintext Exception Policy

**Security Rule**: Do NOT solve availability failures by silently reverting to plaintext RTSP.

If an emergency requires temporary plaintext operation:
1. The exception must be **explicitly authorized** by a security-responsible party.
2. The exception must be **time-bounded** (maximum 24-hour window with documented renewal).
3. The exception must be **risk-accepted** with documented acknowledgment of eavesdropping exposure.
4. The exception must be **independently documented** (incident report with justification, duration, and remediation plan).
5. The exception must be **reverted** to encrypted transport within the authorized window.

Emergency plaintext exceptions are NOT implemented in the current codebase. Any future implementation must follow this policy.

---

## 17. Closure Evidence Matrix

| Requirement | Application Evidence | Deployment Evidence | Current Status | Needed for Closure |
|---|---|---|---|---|
| Credential redaction in logs/errors | ✅ `sanitize_stream_url` + SEC-09 error handler — 34 unit tests pass | N/A (application concern) | **PASS** | ✅ Complete |
| Plaintext strict-mode rejection | ✅ `validate_camera_transport` rejects `plaintext_rtsp`/`plaintext_http` in strict mode — 31 integration tests pass | Strict mode not yet activated in production | **APPLICATION PASS** | Strict mode activated in production |
| No silent downgrade | ✅ `CameraTransportSecurityError` raised, no fallback path — tested | N/A (application concern) | **PASS** | ✅ Complete |
| Physical transport encryption | N/A (deployment concern) | ❌ No evidence exists | **PENDING** | Packet capture showing encrypted transport |
| Endpoint authentication | ✅ RTSP Digest/Basic auth via credential manager | ❌ No TLS/tunnel peer auth evidence | **PARTIAL** | TLS certificate or tunnel key verification |
| Certificate/tunnel validation | ✅ Application classifies and enforces policy | ❌ No certificate inspection or tunnel state evidence | **PENDING** | `openssl s_client` output or `wg show` output |
| Packet-capture verification | N/A (deployment concern) | ❌ Not performed | **PENDING** | Verified pcap showing encrypted video transport |
| Camera configuration updated | ✅ Application accepts secure transport declarations | ❌ No `transport_security` fields in production config | **PENDING** | Updated `configs/cameras.yaml` with security fields |
| Network segmentation | N/A (deployment concern) | ❌ No VLAN/segmentation evidence | **PENDING** | Network topology documentation |
| Operator documentation | ✅ `u4_deployment_requirements.md` created | ❌ No deployment completion record | **PARTIAL** | Signed deployment completion checklist |

---

## 18. Final U4 Closure Criteria

The following conditions must ALL be satisfied to mark:

```
U4 PASS — CLOSED
```

| # | Criterion | Current State |
|---|---|---|
| 1 | U4 application hardening remains passing (31/31 integration tests, 34/34 unit tests) | ✅ MET |
| 2 | Production camera transport protection method is documented per camera | ❌ NOT MET |
| 3 | Every production camera path has an approved secure transport (PATH A, B, or C) | ❌ NOT MET |
| 4 | Actual encrypted transport is technically verified (not just operator attestation) | ❌ NOT MET |
| 5 | No alternate plaintext route exists across the relevant threat boundary | ❌ NOT MET |
| 6 | No silent downgrade exists in application or infrastructure | ✅ MET (application); ❌ UNKNOWN (infrastructure) |
| 7 | Credentials are not exposed in API responses, logs, or error messages | ✅ MET |
| 8 | Strict transport policy is enabled where applicable (`ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true`) | ❌ NOT MET (not activated in production) |
| 9 | Packet-level or equivalent technical evidence exists proving encrypted camera transport | ❌ NOT MET |
| 10 | Residual risks are documented (e.g., OpenCV TLS verification limitations) | ✅ MET |

**Closure Count**: 4 of 10 criteria met.
**Remaining**: 6 criteria require deployment-layer action by the infrastructure/operations team.

---

## 19. Current Verdict

```
U4 APPLICATION HARDENING PASS
U4 DEPLOYMENT REMEDIATION PENDING
```

U4 is **NOT** marked `U4 PASS — CLOSED`.

**Rationale**: Application-layer software hardening is fully implemented, tested, and verified. However, no physical deployment evidence exists proving that camera video streams traverse encrypted transport in production. The 6 outstanding closure criteria (§18, items 2–6 and 8–9) all require physical infrastructure deployment, configuration, and verification by the operations/deployment team.

**Next Action**: The deployment/operations team must:
1. Complete camera capability inventory (§4).
2. Select and implement an approved closure path (§5, §6, or §7) per camera.
3. Execute the relevant verification checklist (§11, §12, or §13).
4. Produce technical proof distinguishing from operator attestation (§14).
5. Follow the strict-mode activation plan (§15).
6. Produce packet-capture evidence (§10).
7. Present completed closure evidence matrix (§17) for final review.
