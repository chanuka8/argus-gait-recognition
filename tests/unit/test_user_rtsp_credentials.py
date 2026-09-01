import re
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import app
from security_layer.credentials import (
    CredentialManager,
    build_rtsp_url,
    extract_rtsp_credentials,
    sanitize_rtsp_url,
)
from services.camera_source_resolver import CameraSourceResolver
from services.camera_worker import CameraWorker
from services.gait_service import GaitService


def _dummy_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:150, 50:150] = [0, 255, 0]
    return frame


@pytest.fixture
def temp_credential_store(tmp_path):
    key = CredentialManager.generate_key()
    enc_file = tmp_path / "test_credentials.enc"
    cm = CredentialManager(credentials_file=str(enc_file), key=key)
    return cm


def test_rtsp_url_sanitization():
    raw = "rtsp://admin:SecretPass123@192.168.1.100:554/live"
    sanitized = sanitize_rtsp_url(raw)
    assert "SecretPass123" not in sanitized
    assert "admin" not in sanitized
    assert sanitized == "rtsp://***:***@192.168.1.100:554/live"

    no_auth = "rtsp://192.168.1.100:554/live"
    assert sanitize_rtsp_url(no_auth) == no_auth
    assert sanitize_rtsp_url("") == ""
    assert sanitize_rtsp_url(None) == ""


def test_rtsp_credential_extraction():
    raw = "rtsp://alice:p%40ssword123@10.0.0.50:554/ch1?stream=main"
    user, passwd, clean_url = extract_rtsp_credentials(raw)
    assert user == "alice"
    assert passwd == "p@ssword123"
    assert clean_url == "rtsp://10.0.0.50:554/ch1?stream=main"

    no_auth = "rtsp://10.0.0.50:554/live"
    u2, p2, clean2 = extract_rtsp_credentials(no_auth)
    assert u2 is None
    assert p2 is None
    assert clean2 == no_auth


def test_rtsp_url_reconstruction():
    base = "rtsp://192.168.1.100:554/live"
    constructed = build_rtsp_url(base, "admin_user", "SecretPass!@#")
    assert "rtsp://" in constructed
    assert "192.168.1.100:554/live" in constructed
    user, passwd, clean = extract_rtsp_credentials(constructed)
    assert user == "admin_user"
    assert passwd == "SecretPass!@#"
    assert clean == base

    assert build_rtsp_url(base, None, None) == base


def test_complex_rtsp_password(temp_credential_store):
    cm = temp_credential_store
    complex_pass = "P@$$w0rd:/?#[]@!$&'()*+,;=%"
    meta = cm.store_credential(
        owner_user_id="user_admin",
        username="sec:admin@zone",
        password=complex_pass,
        credential_id="cred_complex",
    )
    assert meta["credential_id"] == "cred_complex"

    cred = cm.get_credential("cred_complex", user_id="user_admin")
    assert cred is not None
    assert cred["username"] == "sec:admin@zone"
    assert cred["password"] == complex_pass

    url = build_rtsp_url("rtsp://10.0.0.1:554/live", cred["username"], cred["password"])
    sanitized = sanitize_rtsp_url(url)
    assert complex_pass not in sanitized
    assert "***:***" in sanitized


def test_credentials_never_appear_in_logs(caplog, temp_credential_store):
    cm = temp_credential_store
    cm.store_credential(
        owner_user_id="user_1",
        username="vault_admin",
        password="SuperSecretPassword999",
        credential_id="cred_logged",
    )

    resolver = CameraSourceResolver(credential_manager=cm)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with (
        patch("services.camera_source_resolver.cv2.VideoCapture", return_value=mock_cap),
        patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap),
    ):
        res = resolver.resolve_source(
            camera_id="CAM-LOG-TEST",
            requested_source="rtsp://10.0.0.10:554/live",
            user_id="user_1",
            credential_id="cred_logged",
        )
        assert res["credential_id"] == "cred_logged"
        resolver.release_source_by_camera_id("CAM-LOG-TEST")

    all_logs = caplog.text
    assert "SuperSecretPassword999" not in all_logs
    assert "vault_admin" not in all_logs


def test_credentials_never_appear_in_api_response():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/credentials",
            headers={"X-User-ID": "test_user_api"},
            json={
                "username": "api_camera_user",
                "password": "ApiSecretPassword888",
                "description": "Gate 1 CCTV",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "ApiSecretPassword888" not in str(data)
        assert data["password"] == "***"
        cred_id = data["credential_id"]

        list_resp = client.get(
            "/api/v1/credentials",
            headers={"X-User-ID": "test_user_api"},
        )
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert "ApiSecretPassword888" not in str(list_data)

        get_resp = client.get(
            f"/api/v1/credentials/{cred_id}",
            headers={"X-User-ID": "test_user_api"},
        )
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert "ApiSecretPassword888" not in str(get_data)
        assert get_data["password"] == "***"


def test_camera_info_does_not_expose_password(temp_credential_store):
    cm = temp_credential_store
    cm.store_credential(
        owner_user_id="user_sec",
        username="sec_user",
        password="MyUltraSecurePassword777",
        credential_id="cred_info_test",
    )

    service = GaitService()
    service.source_resolver._credential_manager = cm

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        cam_info = service.start_camera(
            camera_id="CAM-INFO-TEST",
            source="rtsp://10.0.0.2:554/live",
            user_id="user_sec",
            credential_id="cred_info_test",
        )
        assert cam_info["status"] == "ACTIVE"
        assert cam_info["credential_configured"] is True
        assert cam_info["credential_id"] == "cred_info_test"

        info_str = str(cam_info)
        assert "MyUltraSecurePassword777" not in info_str

        service.stop_camera("CAM-INFO-TEST")


def test_user_a_cannot_access_user_b_credentials(temp_credential_store):
    cm = temp_credential_store
    cm.store_credential(
        owner_user_id="user_a",
        username="user_a_cam",
        password="UserAPassword123",
        credential_id="cred_user_a",
    )

    assert cm.can_access("cred_user_a", user_id="user_a") is True
    assert cm.get_credential("cred_user_a", user_id="user_a") is not None

    assert cm.can_access("cred_user_a", user_id="user_b") is False
    assert cm.get_credential("cred_user_a", user_id="user_b") is None
    assert cm.get_credential_metadata("cred_user_a", user_id="user_b") is None

    with pytest.raises(PermissionError):
        cm.delete_credential("cred_user_a", user_id="user_b")


def test_authorized_shared_camera_access(temp_credential_store):
    cm = temp_credential_store
    cm.store_credential(
        owner_user_id="user_a",
        username="shared_cam_user",
        password="SharedSecretPassword456",
        credential_id="cred_shared_01",
    )

    assert cm.grant_access("cred_shared_01", owner_user_id="user_a", target_user_id="user_b") is True

    assert cm.can_access("cred_shared_01", user_id="user_b") is True
    cred = cm.get_credential("cred_shared_01", user_id="user_b")
    assert cred is not None
    assert cred["username"] == "shared_cam_user"

    meta = cm.get_credential_metadata("cred_shared_01", user_id="user_b")
    assert meta is not None
    assert meta["password"] == "***"
    assert meta["is_owner"] is False

    resolver = CameraSourceResolver(credential_manager=cm)
    res = resolver.resolve_source(
        camera_id="CAM-SHARED-01",
        requested_source="rtsp://10.0.0.5:554/live",
        user_id="user_b",
        credential_id="cred_shared_01",
    )
    assert res["credential_id"] == "cred_shared_01"
    assert "SharedSecretPassword456" in res["resolved_source"]
    assert "SharedSecretPassword456" not in res["resolved_source_label"]
    resolver.release_source_by_camera_id("CAM-SHARED-01")


def test_credential_delete(temp_credential_store):
    cm = temp_credential_store
    cm.store_credential(
        owner_user_id="user_del",
        username="del_user",
        password="DelPassword123",
        credential_id="cred_to_delete",
    )

    assert cm.has_credential("cred_to_delete") is True
    assert cm.delete_credential("cred_to_delete", user_id="user_del") is True
    assert cm.has_credential("cred_to_delete") is False
    assert cm.get_credential("cred_to_delete", user_id="user_del") is None


def test_missing_credential(temp_credential_store):
    resolver = CameraSourceResolver(credential_manager=temp_credential_store)

    with pytest.raises(RuntimeError) as exc_info:
        resolver.resolve_source(
            camera_id="CAM-MISSING",
            requested_source="rtsp://10.0.0.1:554/live",
            user_id="user_test",
            credential_id="cred_non_existent",
        )
    assert "not authorized to access credential" in str(exc_info.value) or "not found" in str(exc_info.value)


def test_invalid_credential_id(temp_credential_store):
    cm = temp_credential_store
    cm.store_credential(
        owner_user_id="user_isolated_owner",
        username="iso_user",
        password="IsoPassword789",
        credential_id="cred_private",
    )

    resolver = CameraSourceResolver(credential_manager=cm)
    with pytest.raises(RuntimeError) as exc_info:
        resolver.resolve_source(
            camera_id="CAM-UNAUTH",
            requested_source="rtsp://10.0.0.1:554/live",
            user_id="user_intruder",
            credential_id="cred_private",
        )
    assert "not authorized" in str(exc_info.value).lower()


def test_explicit_rtsp_without_credentials():
    resolver = CameraSourceResolver()
    res = resolver.resolve_source(
        camera_id="CAM-PUB",
        requested_source="rtsp://192.168.1.200:554/public",
    )
    assert res["resolved_source"] == "rtsp://192.168.1.200:554/public"
    assert res["credential_configured"] is False
    resolver.release_source_by_camera_id("CAM-PUB")


def test_explicit_rtsp_with_credentials():
    resolver = CameraSourceResolver()
    res = resolver.resolve_source(
        camera_id="CAM-EMBED",
        requested_source="rtsp://admin:SecretPass99@192.168.1.250:554/live",
        user_id="user_embed",
    )
    assert "SecretPass99" in res["resolved_source"]
    assert "SecretPass99" not in res["source"]
    assert "SecretPass99" not in res["resolved_source_label"]
    assert "***:***" in res["resolved_source_label"]
    assert res["credential_configured"] is True
    resolver.release_source_by_camera_id("CAM-EMBED")


def test_usb_camera_regression():
    resolver = CameraSourceResolver()
    with patch.object(resolver, "probe_usb_webcam", return_value=True):
        res = resolver.resolve_source(
            camera_id="CAM-USB",
            requested_source="0",
        )
        assert res["resolved_source_type"] == "usb"
        assert res["resolved_source"] == "0"
        assert res["credential_configured"] is False
        resolver.release_source_by_camera_id("CAM-USB")


def test_auto_source_regression():
    resolver = CameraSourceResolver()
    with patch.object(resolver, "probe_usb_webcam", return_value=True):
        res = resolver.resolve_source(
            camera_id="CAM-AUTO",
            requested_source="auto",
        )
        assert res["resolved_source_type"] == "usb"
        assert res["resolved_source"] == "0"
        resolver.release_source_by_camera_id("CAM-AUTO")


def test_rtsp_reconnect_preserves_credentials(temp_credential_store):
    cm = temp_credential_store
    cm.store_credential(
        owner_user_id="user_recon",
        username="recon_user",
        password="ReconPassword123",
        credential_id="cred_recon",
    )

    resolver = CameraSourceResolver(credential_manager=cm)
    res = resolver.resolve_source(
        camera_id="CAM-RECON-TEST",
        requested_source="rtsp://10.0.0.1:554/live",
        user_id="user_recon",
        credential_id="cred_recon",
    )
    resolved_url = res["resolved_source"]

    cfg = {
        "type": "rtsp",
        "url": resolved_url,
        "startup_timeout": 0.05,
        "startup_retry_interval": 0.01,
        "reconnect_interval": 0,
        "max_reconnect_attempts": 3,
    }
    worker = CameraWorker("CAM-RECON-TEST", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    call_count = 0

    def read_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (True, frame)
        elif call_count == 2:
            return (False, None)
        return (True, frame)

    mock_cap.read.side_effect = read_effect

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        assert worker.start() is True
        time.sleep(0.1)
        assert worker.is_running() is True
        assert worker._resolve_source() == resolved_url
        assert "ReconPassword123" in worker._resolve_source()

    worker.stop()
    resolver.release_source_by_camera_id("CAM-RECON-TEST")


def test_failed_camera_start_releases_credential_reference():
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), pytest.raises(RuntimeError):
        service.start_camera(
            camera_id="CAM-FAIL-REL",
            source="rtsp://admin:pass@10.0.0.5:554/live",
        )

    assert "CAM-FAIL-REL" not in service.active_cameras
    assert not service.source_resolver.is_source_reserved("stream:rtsp://10.0.0.5:554/live")


def test_restart_preserves_credentials(temp_credential_store):
    cm = temp_credential_store
    cm.store_credential(
        owner_user_id="user_rst",
        username="rst_user",
        password="RstPassword321",
        credential_id="cred_rst",
    )

    resolver = CameraSourceResolver(credential_manager=cm)
    res = resolver.resolve_source(
        camera_id="CAM-RST",
        requested_source="rtsp://10.0.0.1:554/live",
        user_id="user_rst",
        credential_id="cred_rst",
    )

    cfg = {
        "type": "rtsp",
        "url": res["resolved_source"],
        "startup_timeout": 0.05,
        "startup_retry_interval": 0.01,
    }
    worker = CameraWorker("CAM-RST", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, frame)

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        assert worker.start() is True
        assert worker.restart() is True
        assert worker.is_running() is True
        assert "RstPassword321" in worker._resolve_source()

    worker.stop()
    resolver.release_source_by_camera_id("CAM-RST")


def test_no_plaintext_credentials_in_persisted_camera_config():
    import yaml

    config_path = Path("configs/cameras.yaml")
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw_text = config_path.read_text(encoding="utf-8")
        assert not re.search(r"rtsp://[^:\s]+:[^@\s]+@", raw_text), (
            "Found plaintext credentials in configs/cameras.yaml"
        )
        for cam_id, cam_cfg in data.get("cameras", {}).items():
            assert "password" not in cam_cfg, f"Camera '{cam_id}' has plaintext password field in cameras.yaml"


def test_core_logger_filter_redacts_credentials(tmp_path):
    from core.logger import setup_logger

    test_logger = setup_logger("ARGUS.TestCoreLogger")
    log_file = Path("outputs/logs/system/argus.log")

    secret_str = "RTSP Stream rtsp://user:SecretLeakPassword999@192.168.1.100:554/live"
    test_logger.info(secret_str)

    if log_file.exists():
        content = log_file.read_text(encoding="utf-8")
        assert "SecretLeakPassword999" not in content
        assert "rtsp://***:***@192.168.1.100:554/live" in content


def test_monitoring_logger_filter_redacts_credentials():
    from monitoring.logging_config import get_logger

    test_logger = get_logger("camera")

    secret_str = "Connecting to rtsp://cctv_admin:MegaSecret123@10.10.10.10:554/feed"
    test_logger.info(secret_str)

    cam_log_file = Path("outputs/logs/camera/camera.log")
    if cam_log_file.exists():
        content = cam_log_file.read_text(encoding="utf-8")
        assert "MegaSecret123" not in content


def test_api_camera_start_error_response_redacts_credentials():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), TestClient(app) as client:
        resp = client.post(
            "/api/v1/cameras/start",
            json={
                "camera_id": "CAM-ERR-TEST",
                "source": "rtsp://leaked_user:LeakedPass999@192.168.1.99:554/live",
            },
        )
        data_str = resp.text
        assert "LeakedPass999" not in data_str
        assert "rtsp://***:***@" in data_str
