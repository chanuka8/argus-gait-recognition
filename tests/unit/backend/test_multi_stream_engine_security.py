from unittest.mock import patch

import pytest

from app.security_layer.credentials import CameraTransportSecurityError
from app.streaming.multi_stream_engine import CameraStream, MultiStreamEngine


def test_multi_stream_engine_reraises_transport_security_error(monkeypatch):
    """CameraTransportSecurityError is never swallowed by MultiStreamEngine."""
    monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")

    camera_configs = [
        {
            "id": "cam_insecure",
            "type": "rtsp",
            "url": "rtsp://192.168.1.100:554/live",
        }
    ]

    with pytest.raises(CameraTransportSecurityError) as exc_info:
        MultiStreamEngine(camera_configs=camera_configs)

    assert "cam_insecure" in str(exc_info.value)


def test_multi_stream_engine_never_falls_back_to_device_zero(monkeypatch):
    """Network camera config with missing/invalid URL never falls back to USB device 0."""
    monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")

    camera_configs = [
        {
            "id": "cam_network_missing_url",
            "type": "rtsp",
            "host": "192.168.1.50",
            # url is omitted
        }
    ]

    with pytest.raises(CameraTransportSecurityError) as exc_info:
        MultiStreamEngine(camera_configs=camera_configs)

    err_str = str(exc_info.value)
    assert "cam_network_missing_url" in err_str
    assert "cannot fall back to device 0" in err_str or "Plaintext RTSP" in err_str or "rejected" in err_str


def test_camera_stream_start_makes_zero_videocapture_on_security_rejection(monkeypatch):
    """CameraStream.start() validates transport and raises before cv2.VideoCapture is invoked."""
    monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")

    stream = CameraStream(
        camera_id="cam_strict_test",
        source="rtsp://192.168.1.50:554/live",
    )

    with patch("cv2.VideoCapture") as mock_vc:
        with pytest.raises(CameraTransportSecurityError):
            stream.start()

        # cv2.VideoCapture MUST NEVER have been called
        mock_vc.assert_not_called()


def test_multi_stream_engine_permissive_in_development(monkeypatch):
    """Development environment permits plaintext RTSP camera configurations."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "development")
    monkeypatch.delenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", raising=False)

    camera_configs = [
        {
            "id": "cam_dev",
            "type": "rtsp",
            "url": "rtsp://192.168.1.100:554/live",
        }
    ]

    engine = MultiStreamEngine(camera_configs=camera_configs)
    assert "cam_dev" in engine.streams
    assert engine.streams["cam_dev"].source == "rtsp://192.168.1.100:554/live"


def test_multi_stream_engine_tunnel_allowed_in_strict_mode(monkeypatch):
    """Operator-asserted protected tunnels are allowed in strict mode."""
    monkeypatch.setenv("ARGUS_ENVIRONMENT", "production")

    camera_configs = [
        {
            "id": "cam_tunnel",
            "type": "rtsp",
            "url": "rtsp://10.8.0.5:554/live",
            "is_tunnel": True,
            "tunnel_type": "wireguard",
        }
    ]

    engine = MultiStreamEngine(camera_configs=camera_configs)
    assert "cam_tunnel" in engine.streams
