"""Integration test suite for U4 — Camera / RTSP Stream Transport Security.

Tests:
- URL credential & query parameter sanitization (userinfo, token, password, api_key, etc.)
- Fail-safe handling of malformed URLs
- Transport security classification (local, protected_tunnel, rtsps_candidate, https_candidate,
  plaintext_rtsp, plaintext_http, unsupported)
- Strict mode transport policy enforcement (rejection of unencrypted transport)
- Strict mode RTSPS candidate policy: requires explicit operator verification assertion
- Operator-asserted protected tunnels (WireGuard/IPSec/VPN) accepted in strict mode
- Local capture (USB webcams, files) unaffected by strict mode
- Zero silent downgrade: failed secure streams never fall back to plaintext RTSP
- Backward compatibility of sanitize_rtsp_url, extract_rtsp_credentials, build_rtsp_url
- Source type stability: network streams retain source_type = 'rtsp'
- Test isolation guard: assert zero connection attempts to real physical camera endpoints
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from security_layer.credentials import (
    CameraTransportSecurityError,
    build_rtsp_url,
    build_stream_url,
    extract_rtsp_credentials,
    extract_stream_credentials,
    resolve_camera_config,
    sanitize_rtsp_url,
    sanitize_stream_url,
    validate_camera_transport,
)
from services.camera_source_resolver import CameraSourceResolver
from services.camera_worker import CameraWorker

# RFC 5737 TEST-NET reserved synthetic documentation addresses (guaranteed non-routable)
SYNTHETIC_RTSP_PLAINTEXT = "rtsp://192.0.2.10:554/live/ch0"
SYNTHETIC_RTSPS_SECURE = "rtsps://192.0.2.10:322/live/ch0"
SYNTHETIC_HTTP_PLAINTEXT = "http://192.0.2.10:8080/mjpeg"
SYNTHETIC_HTTPS_SECURE = "https://192.0.2.10:8443/mjpeg"

# Real physical camera addresses from configs/cameras.yaml that MUST NEVER be contacted during tests
REAL_CAMERA_IPS = {"192.168.1.100", "192.168.1.101", "192.168.1.102"}


@pytest.fixture(autouse=True)
def guard_real_camera_network(monkeypatch):
    """Safety guard fixture: abort immediately if any test attempts to open a socket to real camera IPs."""
    orig_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, (tuple, list)) else str(address)
        if any(real_ip in str(host) for real_ip in REAL_CAMERA_IPS):
            raise AssertionError(
                f"TEST ISOLATION VIOLATION: Attempted socket connection to real camera IP '{host}'! "
                f"All U4 tests must use synthetic addresses or mocks."
            )
        return orig_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def reset_transport_env(monkeypatch):
    """Ensure clean environment state for each test."""
    monkeypatch.delenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", raising=False)
    monkeypatch.delenv("ARGUS_ALLOW_RTSPS_CANDIDATE", raising=False)
    monkeypatch.delenv("ARGUS_ALLOW_HTTPS_CANDIDATE", raising=False)


# ==============================================================================
# SECTION 1: URL & QUERY SANITIZATION TESTS (Amendment 3 & 5)
# ==============================================================================


class TestUrlSanitization:
    """Validate robust structured URL parsing, credentials redaction, and query parameter masking."""

    def test_sanitize_rtsp_basic_credentials(self):
        raw = "rtsp://admin:SecretPass123@192.0.2.50:554/live"
        sanitized = sanitize_stream_url(raw)
        assert "SecretPass123" not in sanitized
        assert "admin" not in sanitized
        assert sanitized == "rtsp://***:***@192.0.2.50:554/live"

    def test_sanitize_rtsps_credentials(self):
        raw = "rtsps://camera_user:VaultPass999@camera.secure.lan:322/stream1"
        sanitized = sanitize_stream_url(raw)
        assert "VaultPass999" not in sanitized
        assert "camera_user" not in sanitized
        assert sanitized == "rtsps://***:***@camera.secure.lan:322/stream1"

    def test_sanitize_query_parameters_redaction(self):
        """Sensitive query parameters must be redacted while benign parameters are preserved."""
        raw = (
            "rtsp://192.0.2.50:554/live"
            "?channel=1&token=super_secret_token_abc&subtype=0&password=hidden_pwd&api_key=key_xyz"
        )
        sanitized = sanitize_stream_url(raw)
        assert "super_secret_token_abc" not in sanitized
        assert "hidden_pwd" not in sanitized
        assert "key_xyz" not in sanitized
        # Non-sensitive query parameters preserved
        assert "channel=1" in sanitized
        assert "subtype=0" in sanitized
        assert "token=***" in sanitized
        assert "password=***" in sanitized
        assert "api_key=***" in sanitized

    def test_sanitize_url_in_log_or_exception_string(self):
        """Sanitizer must handle error messages and log text embedding stream URLs."""
        log_message = (
            "ConnectionError: Failed to connect to rtsp://root:root1234@192.0.2.1:554/feed?token=xyz after 3 attempts"
        )
        sanitized = sanitize_stream_url(log_message)
        assert "root1234" not in sanitized
        assert "root" not in sanitized
        assert "xyz" not in sanitized
        assert "ConnectionError" in sanitized
        assert "after 3 attempts" in sanitized
        assert "rtsp://***:***@192.0.2.1:554/feed?token=***" in sanitized

    def test_sanitize_malformed_url_fails_safely(self):
        """Malformed or incomplete URLs must fail safely without leaking sensitive substrings."""
        malformed = "rtsp://admin:leaked_pass@/invalid_path?token=leaked_tok"
        sanitized = sanitize_stream_url(malformed)
        assert "leaked_pass" not in sanitized
        assert "leaked_tok" not in sanitized

    def test_sanitize_empty_and_non_string_inputs(self):
        assert sanitize_stream_url("") == ""
        assert sanitize_stream_url(None) == ""
        assert sanitize_stream_url(123) == ""

    def test_backward_compatible_sanitize_rtsp_url_wrapper(self):
        raw = "rtsp://user:pwd@192.0.2.20:554/stream"
        assert sanitize_rtsp_url(raw) == sanitize_stream_url(raw)
        assert "pwd" not in sanitize_rtsp_url(raw)


# ==============================================================================
# SECTION 2: CREDENTIAL HELPER COMPATIBILITY (Amendment 5)
# ==============================================================================


class TestCredentialHelpers:
    """Validate extraction and URL building across RTSP and RTSPS streams."""

    def test_extract_stream_credentials_rtsp(self):
        url = "rtsp://operator:op_pass@192.0.2.30:554/h264"
        user, pwd, clean = extract_stream_credentials(url)
        assert user == "operator"
        assert pwd == "op_pass"
        assert clean == "rtsp://192.0.2.30:554/h264"

    def test_extract_stream_credentials_rtsps(self):
        url = "rtsps://sec_admin:vault_pass@192.0.2.30:322/h264"
        user, pwd, clean = extract_stream_credentials(url)
        assert user == "sec_admin"
        assert pwd == "vault_pass"
        assert clean == "rtsps://192.0.2.30:322/h264"

    def test_extract_stream_credentials_no_auth(self):
        url = "rtsp://192.0.2.30:554/h264"
        user, pwd, clean = extract_stream_credentials(url)
        assert user is None
        assert pwd is None
        assert clean == url

    def test_build_stream_url_rtsp_and_rtsps(self):
        rtsp_built = build_stream_url("rtsp://192.0.2.30:554/ch1", "admin", "p@ss:word")
        assert rtsp_built == "rtsp://admin:p%40ss%3Aword@192.0.2.30:554/ch1"

        rtsps_built = build_stream_url("rtsps://192.0.2.30:322/ch1", "admin", "p@ss:word")
        assert rtsps_built == "rtsps://admin:p%40ss%3Aword@192.0.2.30:322/ch1"

    def test_build_stream_url_does_not_alter_http(self):
        """HTTP streams should not have userinfo injected unless explicitly supported."""
        clean = "http://192.0.2.30:8080/mjpeg"
        built = build_stream_url(clean, "admin", "pass")
        assert built == clean

    def test_backward_compatible_aliases(self):
        url = "rtsp://admin:secret@192.0.2.30:554/live"
        assert extract_rtsp_credentials(url) == extract_stream_credentials(url)
        assert build_rtsp_url("rtsp://192.0.2.30:554/live", "a", "b") == build_stream_url(
            "rtsp://192.0.2.30:554/live", "a", "b"
        )


# ==============================================================================
# SECTION 3: TRANSPORT CLASSIFICATION & VALIDATION (Amendment 1)
# ==============================================================================


class TestTransportValidation:
    """Validate the 7 distinct transport categories and strict/permissive policy decisions."""

    def test_classification_local_device(self):
        res_int = validate_camera_transport(0)
        assert res_int["allowed"] is True
        assert res_int["category"] == "local"
        assert res_int["verified_by_argus"] is True

        res_str = validate_camera_transport("usb:0")
        assert res_str["allowed"] is True
        assert res_str["category"] == "local"

    def test_classification_local_file(self, tmp_path):
        dummy_video = tmp_path / "test_stream.mp4"
        dummy_video.write_bytes(b"dummy_video_bytes")
        res = validate_camera_transport(str(dummy_video))
        assert res["allowed"] is True
        assert res["category"] == "local"
        assert res["transport_scheme"] == "file"

    def test_classification_plaintext_rtsp_in_permissive_mode(self):
        res = validate_camera_transport(SYNTHETIC_RTSP_PLAINTEXT, strict_mode=False)
        assert res["allowed"] is True
        assert res["category"] == "plaintext_rtsp"
        assert res["is_encrypted"] is False

    def test_classification_plaintext_rtsp_rejected_in_strict_mode(self):
        with pytest.raises(CameraTransportSecurityError) as exc_info:
            validate_camera_transport(SYNTHETIC_RTSP_PLAINTEXT, strict_mode=True, enforce=True)
        assert "Plaintext RTSP transport" in str(exc_info.value)

        # Without enforce=True, returns allowed=False dictionary
        res = validate_camera_transport(SYNTHETIC_RTSP_PLAINTEXT, strict_mode=True, enforce=False)
        assert res["allowed"] is False
        assert res["category"] == "plaintext_rtsp"

    def test_rtsps_requires_explicit_verified_transport_policy_in_strict_mode(self):
        """Amendment 1: RTSPS must NOT automatically equal verified secure transport in strict mode."""
        # In strict mode without operator confirmation, rtsps_candidate must be REJECTED
        with pytest.raises(CameraTransportSecurityError) as exc_info:
            validate_camera_transport(SYNTHETIC_RTSPS_SECURE, strict_mode=True, enforce=True)
        assert "RTSPS scheme recognized" in str(exc_info.value)
        assert "operator confirmation" in str(exc_info.value)

        res = validate_camera_transport(SYNTHETIC_RTSPS_SECURE, strict_mode=True, enforce=False)
        assert res["allowed"] is False
        assert res["category"] == "rtsps_candidate"
        assert res["verified_by_argus"] is False  # Cannot cryptographically verify in runtime

    def test_rtsps_accepted_with_operator_confirmation_in_strict_mode(self):
        """When operator explicitly confirms certificate configuration, RTSPS is allowed in strict mode."""
        cfg = {"allow_rtsps_candidate": True}
        res = validate_camera_transport(SYNTHETIC_RTSPS_SECURE, config=cfg, strict_mode=True)
        assert res["allowed"] is True
        assert res["category"] == "rtsps_candidate"
        assert res["verified_by_argus"] is False  # Explicitly distinguished from ARGUS cryptographic proof

        # Also accepts transport_security: "rtsps_verified"
        cfg2 = {"transport_security": "rtsps_verified"}
        res2 = validate_camera_transport(SYNTHETIC_RTSPS_SECURE, config=cfg2, strict_mode=True)
        assert res2["allowed"] is True

    def test_operator_asserted_protected_tunnel_accepted_in_strict_mode(self):
        """RTSP over WireGuard/IPSec/VPN tunnel is accepted when explicitly asserted by deployment."""
        cfg = {
            "transport_security": "wireguard",
            "tunnel_type": "wireguard_site_to_site",
        }
        res = validate_camera_transport(SYNTHETIC_RTSP_PLAINTEXT, config=cfg, strict_mode=True)
        assert res["allowed"] is True
        assert res["category"] == "protected_tunnel"
        assert res["verified_by_argus"] is False  # Stated as operator assertion, not ARGUS verified
        assert res["tunnel_type"] == "wireguard_site_to_site"

    def test_unsupported_scheme_rejected(self):
        with pytest.raises(CameraTransportSecurityError):
            validate_camera_transport("ftp://192.0.2.1/stream", strict_mode=False, enforce=True)


# ==============================================================================
# SECTION 4: SOURCE TYPE STABILITY AUDIT (Amendment 2)
# ==============================================================================


class TestSourceTypeCompatibility:
    """Verify that network camera source classification preserves source_type = 'rtsp'."""

    def test_source_resolver_preserves_source_type_rtsp_for_rtsps(self):
        resolver = CameraSourceResolver(config_path="configs/cameras.yaml")
        # Test synthetic RTSPS URL
        resolved = resolver.resolve_source(
            camera_id="cam_synthetic_rtsps",
            requested_source="rtsps://192.0.2.99:322/stream",
        )
        assert resolved["source_type"] == "rtsp"  # Critical invariant: must be 'rtsp'
        assert resolved["resolved_source_type"] == "rtsp"
        assert resolved["transport_scheme"] == "rtsps"
        assert resolved["transport_category"] == "rtsps_candidate"

    def test_source_resolver_preserves_source_type_rtsp_for_plaintext_rtsp(self):
        resolver = CameraSourceResolver(config_path="configs/cameras.yaml")
        resolved = resolver.resolve_source(
            camera_id="cam_synthetic_rtsp",
            requested_source="rtsp://192.0.2.99:554/stream",
        )
        assert resolved["source_type"] == "rtsp"
        assert resolved["resolved_source_type"] == "rtsp"
        assert resolved["transport_scheme"] == "rtsp"
        assert resolved["transport_category"] == "plaintext_rtsp"

    def test_source_resolver_strict_mode_rejection(self, monkeypatch):
        monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")
        resolver = CameraSourceResolver(config_path="configs/cameras.yaml")
        with pytest.raises(CameraTransportSecurityError):
            resolver.resolve_source(
                camera_id="cam_strict_reject",
                requested_source="rtsp://192.0.2.99:554/stream",
            )


# ==============================================================================
# SECTION 5: CAMERA WORKER TRANSPORT HARDENING (Amendment 4 & Fail-Closed)
# ==============================================================================


class TestCameraWorkerTransportHardening:
    """Validate CameraWorker integration with zero physical CCTV contact (Amendment 4)."""

    def test_worker_rejects_plaintext_rtsp_in_strict_mode(self, monkeypatch):
        monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")

        config = {
            "id": "cam_worker_strict_test",
            "url": SYNTHETIC_RTSP_PLAINTEXT,
            "width": 640,
            "height": 480,
        }

        worker = CameraWorker(camera_id="cam_worker_strict_test", camera_config=config)

        # Worker._open_capture() should fail closed without attempting connection
        with patch("cv2.VideoCapture") as mock_cap:
            connected = worker._open_capture()
            assert connected is False
            # cv2.VideoCapture should NOT have been called because transport validation failed first
            mock_cap.assert_not_called()

        assert worker.stats["transport_allowed"] is False
        assert worker.stats["transport_category"] == "plaintext_rtsp"

    def test_worker_allows_plaintext_rtsp_in_permissive_mode_with_warning(self, monkeypatch):
        monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "false")

        config = {
            "id": "cam_worker_permissive_test",
            "url": SYNTHETIC_RTSP_PLAINTEXT,
            "width": 640,
            "height": 480,
        }

        worker = CameraWorker(camera_id="cam_worker_permissive_test", camera_config=config)

        # Mock cv2.VideoCapture so no real network traffic occurs
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap_instance.read.return_value = (True, mock_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap_instance):
            connected = worker._open_capture()
            assert connected is True

        assert worker.stats["transport_allowed"] is True
        assert worker.stats["transport_category"] == "plaintext_rtsp"
        worker._close_capture()

    def test_worker_requires_operator_confirmation_for_rtsps_in_strict_mode(self, monkeypatch):
        monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")

        # Without confirmation: fails closed
        config_no_confirm = {
            "id": "cam_rtsps_unconfirmed",
            "url": SYNTHETIC_RTSPS_SECURE,
            "width": 640,
            "height": 480,
        }
        worker1 = CameraWorker(camera_id="cam_rtsps_unconfirmed", camera_config=config_no_confirm)
        with patch("cv2.VideoCapture") as mock_cap:
            assert worker1._open_capture() is False
            mock_cap.assert_not_called()

        # With operator confirmation: allowed
        config_confirmed = {
            "id": "cam_rtsps_confirmed",
            "url": SYNTHETIC_RTSPS_SECURE,
            "allow_rtsps_candidate": True,
            "width": 640,
            "height": 480,
        }
        worker2 = CameraWorker(camera_id="cam_rtsps_confirmed", camera_config=config_confirmed)
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap_instance.read.return_value = (True, mock_frame)

        with patch("cv2.VideoCapture", return_value=mock_cap_instance) as mock_cap:
            assert worker2._open_capture() is True
            mock_cap.assert_called_once_with(SYNTHETIC_RTSPS_SECURE)
            worker2._close_capture()

    def test_no_silent_downgrade_on_rtsps_connection_failure(self, monkeypatch):
        """Security invariant: if an RTSPS connection fails, it must NEVER retry as plaintext RTSP."""
        monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")

        config = {
            "id": "cam_no_downgrade",
            "url": SYNTHETIC_RTSPS_SECURE,
            "allow_rtsps_candidate": True,
            "width": 640,
            "height": 480,
        }
        worker = CameraWorker(camera_id="cam_no_downgrade", camera_config=config)

        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False  # Simulate connection refusal

        opened_urls = []

        def mock_video_capture(src, *args, **kwargs):
            opened_urls.append(src)
            return mock_cap_instance

        with patch("cv2.VideoCapture", side_effect=mock_video_capture):
            connected = worker._open_capture()
            assert connected is False

        # Verify: every attempted URL was the secure scheme, never plaintext
        assert len(opened_urls) == 1
        assert opened_urls[0].startswith("rtsps://")
        assert not any(url.startswith("rtsp://") for url in opened_urls)


# ==============================================================================
# SECTION 6: CONFIG RESOLVER & TEST ISOLATION GUARD (Amendment 4)
# ==============================================================================


class TestConfigResolverAndIsolationGuard:
    """Validate resolve_camera_config and prove zero real CCTV device contact."""

    def test_resolve_camera_config_handles_rtsps_scheme(self):
        raw_cfg = {
            "id": "cam_test_rtsps",
            "protocol": "rtsps",
            "host": "192.0.2.15",
            "port": 322,
            "path": "/live",
            "username": "admin",
            "password": "secret_password",
            "allow_plaintext_credentials": True,
        }
        resolved = resolve_camera_config(raw_cfg)
        assert resolved["url"].startswith("rtsps://")
        assert "322" in resolved["url"]
        assert resolved["transport_category"] == "rtsps_candidate"

    def test_resolve_camera_config_strict_mode_rejection(self, monkeypatch):
        monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")
        raw_cfg = {
            "id": "cam_test_strict_reject",
            "host": "192.0.2.15",
            "port": 554,
            "path": "/live",
            "username": "admin",
            "password": "secret_password",
            "allow_plaintext_credentials": True,
        }
        with pytest.raises(CameraTransportSecurityError):
            resolve_camera_config(raw_cfg)

    def test_camera_isolation_guard_no_real_camera_contact(self):
        """Amendment 4: Prove that test execution does not contact real camera endpoints from cameras.yaml."""
        # Verify that the safety guard fixture is active
        with pytest.raises(AssertionError) as exc_info:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("192.168.1.100", 554))
        assert "TEST ISOLATION VIOLATION" in str(exc_info.value)
