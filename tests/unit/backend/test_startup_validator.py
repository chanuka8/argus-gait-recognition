from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from deployment.startup_validator import DeploymentStartupValidator, StartupValidationError


def test_startup_validator_all_checks_pass():
    validator = DeploymentStartupValidator()
    summary = validator.validate_startup(raise_on_failure=False)

    assert summary["success"] is True
    assert summary["status"] in (
        DeploymentStartupValidator.STATUS_READY,
        DeploymentStartupValidator.STATUS_READY_WITH_WARNINGS,
    )
    assert isinstance(summary["blocking_issues"], list)
    assert len(summary["blocking_issues"]) == 0
    assert "Live RTSP camera streams" in summary["unable_to_verify"][0]


def test_startup_validator_missing_manifest_asset(monkeypatch, tmp_path: Path):
    validator = DeploymentStartupValidator()

    from deployment.runtime_manifest import RuntimeManifest

    mock_manifest = RuntimeManifest(runtime_assets=["non_existent_asset_path.py"])
    monkeypatch.setattr("deployment.startup_validator.get_runtime_manifest", lambda: mock_manifest)

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("missing asset" in issue for issue in summary["blocking_issues"])


def test_startup_validator_invalid_config_triggers_blocking_issue(monkeypatch):
    validator = DeploymentStartupValidator()
    monkeypatch.setattr(
        validator.config_validator,
        "validate_all",
        lambda: {"system.yaml": ["Invalid field 'bad_key'"]},
    )

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("Config error" in issue for issue in summary["blocking_issues"])


def test_startup_validator_backend_fallback_generates_warning():
    validator = DeploymentStartupValidator()

    mock_backend = MagicMock()
    mock_backend.active_backend = "pytorch"
    mock_backend.requested_backend = "onnxruntime"
    mock_backend.fallback_used = True
    mock_backend.fallback_reason = "ONNX Runtime missing"
    mock_backend.config = {}
    mock_backend.predict.return_value = np.zeros((1, 256), dtype=np.float32)
    mock_backend.metadata = {
        "requested_backend": "onnxruntime",
        "active_backend": "pytorch",
        "fallback_used": True,
    }

    summary = validator.validate_startup(raise_on_failure=False, override_backend=mock_backend)
    assert summary["success"] is True
    assert any("Backend fallback active" in w for w in summary["warnings"])


def test_startup_validator_unwritable_output_path(monkeypatch, tmp_path: Path):
    validator = DeploymentStartupValidator()

    def mock_mkdir(*args, **kwargs):
        raise PermissionError("Access Denied")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("not writable" in issue for issue in summary["blocking_issues"])


def test_startup_validator_raises_on_failure_when_enabled(monkeypatch):
    validator = DeploymentStartupValidator()
    monkeypatch.setattr(
        validator.config_validator,
        "validate_all",
        lambda: {"system.yaml": ["Fatal syntax error"]},
    )

    with pytest.raises(StartupValidationError) as exc_info:
        validator.validate_startup(raise_on_failure=True)

    assert len(exc_info.value.blocking_issues) > 0


def test_startup_validator_missing_gallery_file_produces_warning_notice(monkeypatch):
    validator = DeploymentStartupValidator()
    monkeypatch.setattr(
        "deployment.startup_validator.validate_gallery_files",
        lambda *args, **kwargs: (False, "Gallery features file missing in models/galleries/gallery", 0),
    )

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is True
    assert summary["status"] in (
        DeploymentStartupValidator.STATUS_READY_WITH_WARNINGS,
        DeploymentStartupValidator.STATUS_READY,
    )
    assert not any("Gallery defect" in issue for issue in summary["blocking_issues"])
    assert any("Gallery state notice" in w for w in summary["warnings"])


def test_startup_validator_gallery_corruption_blocks_startup(monkeypatch):
    validator = DeploymentStartupValidator()
    monkeypatch.setattr(
        "deployment.startup_validator.validate_gallery_files",
        lambda *args, **kwargs: (
            False,
            "Corrupted feature array: NaN detected in gallery_features.npy",
            0,
        ),
    )

    summary = validator.validate_startup(raise_on_failure=False)
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("Gallery defect" in issue for issue in summary["blocking_issues"])


# ==============================================================================
# U4 CAMERA TRANSPORT SECURITY TESTS
# ==============================================================================


def test_startup_validator_production_rejects_plaintext_rtsp(monkeypatch):
    """Production deployment fails closed when unencrypted RTSP cameras are enabled."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "production")
    monkeypatch.delenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", raising=False)

    validator = DeploymentStartupValidator()
    summary = validator.validate_startup(raise_on_failure=False)

    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("Insecure camera transport for 'camera_01'" in issue for issue in summary["blocking_issues"])
    assert any("Plaintext RTSP transport ('rtsp://') is rejected" in issue for issue in summary["blocking_issues"])


def test_startup_validator_strict_env_rejects_plaintext_rtsp(monkeypatch):
    """Explicit ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true triggers strict rejection."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "development")
    monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")

    validator = DeploymentStartupValidator()
    summary = validator.validate_startup(raise_on_failure=False)

    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("Insecure camera transport" in issue for issue in summary["blocking_issues"])


def test_startup_validator_development_allows_plaintext_with_warning(monkeypatch):
    """Development environment permits plaintext RTSP with non-blocking warning notice."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "development")
    monkeypatch.delenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", raising=False)

    validator = DeploymentStartupValidator()
    summary = validator.validate_startup(raise_on_failure=False)

    assert summary["success"] is True
    assert not any("Insecure camera transport" in issue for issue in summary["blocking_issues"])
    assert any("permitted in development mode" in w for w in summary["warnings"])


def test_startup_validator_disabled_plaintext_camera_does_not_block(monkeypatch, tmp_path):
    """Disabled camera configurations do not cause blocking transport defects in production."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "production")

    custom_cfg_dir = tmp_path / "configs"
    custom_cfg_dir.mkdir()
    # Copy system configs except cameras.yaml
    for cfg in Path("configs").glob("*.yaml"):
        if cfg.name != "cameras.yaml":
            (custom_cfg_dir / cfg.name).write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    disabled_cam_yaml = """
cameras:
  cam_disabled:
    id: "cam_disabled"
    type: "rtsp"
    host: "192.168.1.50"
    port: 554
    path: "/stream"
    enabled: false
"""
    (custom_cfg_dir / "cameras.yaml").write_text(disabled_cam_yaml, encoding="utf-8")

    validator = DeploymentStartupValidator(configs_dir=str(custom_cfg_dir))
    summary = validator.validate_startup(raise_on_failure=False)

    assert not any("Insecure camera transport" in issue for issue in summary["blocking_issues"])


def test_startup_validator_protected_tunnel_accepted_in_production(monkeypatch, tmp_path):
    """Operator-asserted protected tunnels are accepted in production as valid encrypted transport."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "production")

    custom_cfg_dir = tmp_path / "configs"
    custom_cfg_dir.mkdir()
    for cfg in Path("configs").glob("*.yaml"):
        if cfg.name != "cameras.yaml":
            (custom_cfg_dir / cfg.name).write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    tunnel_cam_yaml = """
cameras:
  cam_tunnel:
    id: "cam_tunnel"
    type: "rtsp"
    host: "10.8.0.2"
    port: 554
    path: "/stream1"
    is_tunnel: true
    tunnel_type: "wireguard"
    enabled: true
"""
    (custom_cfg_dir / "cameras.yaml").write_text(tunnel_cam_yaml, encoding="utf-8")

    validator = DeploymentStartupValidator(configs_dir=str(custom_cfg_dir))
    summary = validator.validate_startup(raise_on_failure=False)

    assert not any("Insecure camera transport" in issue for issue in summary["blocking_issues"])
    assert any("Operator-asserted protected tunnel (wireguard) accepted" in w for w in summary["warnings"])


def test_startup_validator_confirmed_rtsps_accepted_unconfirmed_rejected(monkeypatch, tmp_path):
    """Unconfirmed RTSPS is rejected in production; operator-confirmed RTSPS is accepted."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "production")

    custom_cfg_dir = tmp_path / "configs"
    custom_cfg_dir.mkdir()
    for cfg in Path("configs").glob("*.yaml"):
        if cfg.name != "cameras.yaml":
            (custom_cfg_dir / cfg.name).write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    # 1. Unconfirmed RTSPS
    unconfirmed_yaml = """
cameras:
  cam_rtsps:
    id: "cam_rtsps"
    type: "rtsps"
    host: "camera.secure.lan"
    port: 322
    path: "/stream1"
    enabled: true
"""
    (custom_cfg_dir / "cameras.yaml").write_text(unconfirmed_yaml, encoding="utf-8")

    validator = DeploymentStartupValidator(configs_dir=str(custom_cfg_dir))
    summary = validator.validate_startup(raise_on_failure=False)
    assert any("Insecure camera transport for 'cam_rtsps'" in issue for issue in summary["blocking_issues"])

    # 2. Confirmed RTSPS
    confirmed_yaml = """
cameras:
  cam_rtsps:
    id: "cam_rtsps"
    type: "rtsps"
    host: "camera.secure.lan"
    port: 322
    path: "/stream1"
    allow_rtsps_candidate: true
    enabled: true
"""
    (custom_cfg_dir / "cameras.yaml").write_text(confirmed_yaml, encoding="utf-8")

    validator2 = DeploymentStartupValidator(configs_dir=str(custom_cfg_dir))
    summary2 = validator2.validate_startup(raise_on_failure=False)
    assert not any("Insecure camera transport" in issue for issue in summary2["blocking_issues"])
    assert any("verified_by_argus=False" in w for w in summary2["warnings"])


def test_startup_validator_plaintext_http_and_unsupported_rejected(monkeypatch, tmp_path):
    """Plaintext HTTP and invalid/unsupported stream schemes are rejected."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "production")

    custom_cfg_dir = tmp_path / "configs"
    custom_cfg_dir.mkdir()
    for cfg in Path("configs").glob("*.yaml"):
        if cfg.name != "cameras.yaml":
            (custom_cfg_dir / cfg.name).write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    http_cam_yaml = """
cameras:
  cam_http:
    id: "cam_http"
    type: "http"
    host: "192.168.1.200"
    port: 8080
    path: "/mjpeg"
    enabled: true
"""
    (custom_cfg_dir / "cameras.yaml").write_text(http_cam_yaml, encoding="utf-8")

    validator = DeploymentStartupValidator(configs_dir=str(custom_cfg_dir))
    summary = validator.validate_startup(raise_on_failure=False)
    assert any("Plaintext HTTP" in issue for issue in summary["blocking_issues"])


def test_startup_validator_local_usb_and_file_allowed(monkeypatch, tmp_path):
    """Local hardware USB device index and local video file paths are allowed in production."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "production")

    custom_cfg_dir = tmp_path / "configs"
    custom_cfg_dir.mkdir()
    for cfg in Path("configs").glob("*.yaml"):
        if cfg.name != "cameras.yaml":
            (custom_cfg_dir / cfg.name).write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")

    dummy_video = tmp_path / "sample.mp4"
    dummy_video.write_bytes(b"dummy video content")

    local_cam_yaml = f"""
cameras:
  cam_usb:
    id: "cam_usb"
    type: "usb"
    device_index: 0
    enabled: true
  cam_file:
    id: "cam_file"
    type: "file"
    file_path: "{dummy_video.as_posix()}"
    enabled: true
"""
    (custom_cfg_dir / "cameras.yaml").write_text(local_cam_yaml, encoding="utf-8")

    validator = DeploymentStartupValidator(configs_dir=str(custom_cfg_dir))
    summary = validator.validate_startup(raise_on_failure=False)
    assert not any("Insecure camera transport" in issue for issue in summary["blocking_issues"])


def test_startup_validator_pure_static_zero_network_zero_key(monkeypatch):
    """Static camera transport validation must never instantiate CredentialManager,

    must never read, write, create, or open .credentials.key, must never call cv2.VideoCapture,
    and must never make any network calls.
    """
    import builtins
    import os
    import socket

    # 1. Guard CredentialManager.__init__ so it fails immediately if called
    def forbidden_credential_manager_init(self, *args, **kwargs):
        raise AssertionError("FORBIDDEN: CredentialManager.__init__ was called during static transport validation!")

    monkeypatch.setattr(
        "security_layer.credentials.CredentialManager.__init__",
        forbidden_credential_manager_init,
    )

    # 2. Guard against ANY access to .credentials.key (open, read, write, create)
    orig_builtin_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if ".credentials.key" in str(file):
            raise AssertionError(f"FORBIDDEN: Attempted to open .credentials.key ({file}) during static validation!")
        return orig_builtin_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    orig_os_open = os.open

    def guarded_os_open(path, *args, **kwargs):
        if ".credentials.key" in str(path):
            raise AssertionError(f"FORBIDDEN: Attempted to os.open .credentials.key ({path}) during static validation!")
        return orig_os_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_os_open)

    orig_read_bytes = Path.read_bytes
    orig_read_text = Path.read_text
    orig_write_bytes = Path.write_bytes
    orig_write_text = Path.write_text

    def guarded_read_bytes(self, *args, **kwargs):
        if ".credentials.key" in str(self):
            raise AssertionError("FORBIDDEN: read_bytes on .credentials.key")
        return orig_read_bytes(self, *args, **kwargs)

    def guarded_read_text(self, *args, **kwargs):
        if ".credentials.key" in str(self):
            raise AssertionError("FORBIDDEN: read_text on .credentials.key")
        return orig_read_text(self, *args, **kwargs)

    def guarded_write_bytes(self, *args, **kwargs):
        if ".credentials.key" in str(self):
            raise AssertionError("FORBIDDEN: write_bytes on .credentials.key")
        return orig_write_bytes(self, *args, **kwargs)

    def guarded_write_text(self, *args, **kwargs):
        if ".credentials.key" in str(self):
            raise AssertionError("FORBIDDEN: write_text on .credentials.key")
        return orig_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "write_bytes", guarded_write_bytes)
    monkeypatch.setattr(Path, "write_text", guarded_write_text)

    # 3. Guard against network calls and video capture
    def blocked_connect(*args, **kwargs):
        raise AssertionError("FORBIDDEN: Network call (socket.connect) attempted during static transport validation!")

    def blocked_create_connection(*args, **kwargs):
        raise AssertionError(
            "FORBIDDEN: Network call (socket.create_connection) attempted during static transport validation!"
        )

    def blocked_video_capture(*args, **kwargs):
        raise AssertionError("FORBIDDEN: cv2.VideoCapture called during static transport validation!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket, "create_connection", blocked_create_connection)
    monkeypatch.setattr("cv2.VideoCapture", blocked_video_capture)

    # 4. Verify in development mode: static validation succeeds from config inspection alone
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "development")
    monkeypatch.delenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", raising=False)
    validator_dev = DeploymentStartupValidator()
    blocking_dev: list[str] = []
    warnings_dev: list[str] = []
    unable_dev: list[str] = []
    validator_dev._validate_camera_transport_security(
        blocking_issues=blocking_dev,
        warnings=warnings_dev,
        unable_to_verify=unable_dev,
    )
    assert len(blocking_dev) == 0
    assert any("camera_01" in w for w in warnings_dev)

    # 5. Verify in production mode: static validation fails closed from config inspection alone
    monkeypatch.delenv("ARGUS_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", raising=False)
    validator_prod = DeploymentStartupValidator()
    blocking_prod: list[str] = []
    warnings_prod: list[str] = []
    unable_prod: list[str] = []
    validator_prod._validate_camera_transport_security(
        blocking_issues=blocking_prod,
        warnings=warnings_prod,
        unable_to_verify=unable_prod,
    )
    assert len(blocking_prod) >= 2
    assert any("Insecure camera transport for 'camera_01'" in issue for issue in blocking_prod)
    assert any("Insecure camera transport for 'camera_02'" in issue for issue in blocking_prod)


# Backward-compatibility alias
test_startup_validator_zero_credential_manager_and_zero_key_side_effects = (
    test_startup_validator_pure_static_zero_network_zero_key
)


def test_real_default_production_fail_closed_outside_fixture(monkeypatch):
    """With ARGUS_ENVIRONMENT and ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT unset,

    validator evaluates real configs/production.yaml and configs/cameras.yaml,
    identifies production environment, and fails closed with NOT_READY
    without contacting any camera endpoint.
    """
    import socket

    from security_layer.credentials import (
        get_deployment_environment,
        is_secure_camera_transport_required,
    )

    # 1. Unset environment variables to test true default resolution
    monkeypatch.delenv("ARGUS_ENVIRONMENT", raising=False)
    monkeypatch.delenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", raising=False)

    # 2. Assert environment resolution
    assert get_deployment_environment() == "production"
    assert is_secure_camera_transport_required() is True

    # 3. Guard against any network calls
    def blocked_connect(*args, **kwargs):
        raise AssertionError("TEST ISOLATION VIOLATION: Socket connect called during startup validation!")

    def blocked_create_connection(*args, **kwargs):
        raise AssertionError("TEST ISOLATION VIOLATION: Socket create_connection called during startup validation!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket, "create_connection", blocked_create_connection)

    # 4. Run DeploymentStartupValidator with repository configs and mock backend
    mock_backend = MagicMock()
    mock_backend.active_backend = "pytorch"
    mock_backend.requested_backend = "pytorch"
    mock_backend.fallback_used = False
    mock_backend.config = {}
    mock_backend.predict.return_value = np.zeros((1, 256), dtype=np.float32)
    mock_backend.metadata = {}

    validator = DeploymentStartupValidator()
    summary = validator.validate_startup(raise_on_failure=False, override_backend=mock_backend)

    # 5. Assert fail-closed behavior
    assert summary["success"] is False
    assert summary["status"] == DeploymentStartupValidator.STATUS_NOT_READY
    assert any("Insecure camera transport for 'camera_01'" in issue for issue in summary["blocking_issues"])
    assert any("Insecure camera transport for 'camera_02'" in issue for issue in summary["blocking_issues"])
