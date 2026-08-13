"""
Comprehensive Unit & Integration Test Suite for Auto Source Discovery, Zone-Worker Binding, and Low-Latency Camera Pipeline.
Uses mocked cv2.VideoCapture to run in headless CI environments without requiring hardware webcams.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from services.camera_source import (
    AutoSourceResolver,
    ZoneSourceBindingRegistry,
    resolve_source_type,
    sanitize_url,
)
from services.camera_worker import CameraWorker, WorkerState
from services.gait_service import GaitService


def test_resolve_source_type_usb():
    stype, val = resolve_source_type(0, "auto")
    assert stype == "usb"
    assert val == 0

    stype2, val2 = resolve_source_type("1", "usb")
    assert stype2 == "usb"
    assert val2 == 1


def test_resolve_source_type_rtsp():
    stype, val = resolve_source_type("rtsp://admin:pass@192.168.1.100:554/live", "auto")
    assert stype == "rtsp"
    assert val == "rtsp://admin:pass@192.168.1.100:554/live"


def test_resolve_source_type_http():
    stype, val = resolve_source_type("http://192.168.1.50:8080/video", "auto")
    assert stype == "http"
    assert val == "http://192.168.1.50:8080/video"


def test_resolve_source_type_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="detected as 'usb'"):
        resolve_source_type(0, "rtsp")

    with pytest.raises(ValueError, match="detected as 'rtsp'"):
        resolve_source_type("rtsp://localhost/stream", "usb")


def test_sanitize_rtsp_credentials():
    raw = "rtsp://admin:secret123@192.168.1.50:554/h264"
    sanitized = sanitize_url(raw)
    assert "secret123" not in sanitized
    assert "admin" not in sanitized
    assert sanitized == "rtsp://***:***@192.168.1.50:554/h264"


@patch("cv2.VideoCapture")
def test_bounded_usb_discovery_releases_captures(mock_vc):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 30.0)
    mock_vc.return_value = mock_cap

    resolver = AutoSourceResolver(cache_ttl=0.0)
    sources = resolver.discover_usb_sources(max_index=2, force_refresh=True)

    assert len(sources) == 3
    assert sources[0].source_id == "usb_0"
    assert sources[0].available is True
    # Confirm release was called for every probed index
    assert mock_cap.release.call_count == 3


def test_zone_binding_registry_conflict_prevention():
    registry = ZoneSourceBindingRegistry()
    dummy_worker1 = MagicMock()
    dummy_worker2 = MagicMock()

    # Register USB 0 in Zone A
    registry.register_binding(camera_id="cam1", zone_id="Z01", source_type="usb", device_index=0, worker_instance=dummy_worker1)

    assert registry.is_zone_active("Z01") is True
    assert registry.is_usb_active(0) is True

    # Duplicate zone binding attempt -> ValueError
    with pytest.raises(ValueError, match="CONFLICT: Zone 'Z01' already has an active worker"):
        registry.register_binding(camera_id="cam2", zone_id="Z01", source_type="rtsp", device_index=None, worker_instance=dummy_worker2)

    # Duplicate USB device index attempt in another zone -> ValueError
    with pytest.raises(ValueError, match="CONFLICT: USB device index 0 is already in use"):
        registry.register_binding(camera_id="cam3", zone_id="Z02", source_type="usb", device_index=0, worker_instance=dummy_worker2)

    # Unregister cam1 -> frees zone Z01 and USB 0
    registry.unregister_binding("cam1")
    assert registry.is_zone_active("Z01") is False
    assert registry.is_usb_active(0) is False

    # Now Z02 + USB 0 can be registered cleanly
    registry.register_binding(camera_id="cam3", zone_id="Z02", source_type="usb", device_index=0, worker_instance=dummy_worker2)
    assert registry.is_zone_active("Z02") is True
    assert registry.is_usb_active(0) is True


@patch("cv2.VideoCapture")
def test_camera_worker_single_owner_and_low_latency(mock_vc):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    mock_cap.read.return_value = (True, dummy_frame)
    mock_vc.return_value = mock_cap

    worker = CameraWorker(
        camera_id="test_worker_1",
        source=0,
        zone_id="Z01",
        source_type="usb",
        gait_service=None,
    )

    started = worker.start(startup_timeout=2.0)
    assert started is True
    assert worker.status == WorkerState.ACTIVE.value
    assert worker.captured_frames > 0

    jpeg_bytes = worker.get_jpeg_frame()
    assert jpeg_bytes is not None

    stopped = worker.stop()
    assert stopped is True
    assert worker.status == WorkerState.STOPPED.value


def test_gait_service_start_and_stop_camera_with_zone():
    service = GaitService()

    with patch("cv2.VideoCapture") as mock_vc:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        mock_cap.read.return_value = (True, dummy_frame)
        mock_vc.return_value = mock_cap

        # Start webcam 0 in Zone Z01
        cam_info = service.start_camera(
            camera_id="cam_z01",
            source="0",
            zone_id="Z01",
            source_type="usb",
            location="Main Entrance",
        )

        assert cam_info["camera_id"] == "cam_z01"
        assert cam_info["zone_id"] == "Z01"
        assert cam_info["status"] == "ACTIVE"

        # Attempt to start webcam 0 in Zone Z02 -> conflict
        with pytest.raises(ValueError, match="CONFLICT"):
            service.start_camera(
                camera_id="cam_z02",
                source="0",
                zone_id="Z02",
                source_type="usb",
            )

        # Stop Zone Z01 worker
        stopped = service.stop_camera("cam_z01")
        assert stopped is True

        # Now start webcam 0 in Zone Z02 -> succeeds after release
        cam_info_z02 = service.start_camera(
            camera_id="cam_z02",
            source="0",
            zone_id="Z02",
            source_type="usb",
        )
        assert cam_info_z02["zone_id"] == "Z02"
        assert cam_info_z02["status"] == "ACTIVE"

        service.stop_camera("cam_z02")
