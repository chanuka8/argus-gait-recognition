import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from api.server import app
from services.camera_worker import CameraWorker, normalize_camera_source
from services.gait_service import GaitService


def test_normalize_camera_source_webcam_indices():
    """Verify numeric strings and integers are normalized to integer device indices."""
    assert normalize_camera_source("0") == 0
    assert normalize_camera_source("1") == 1
    assert normalize_camera_source(" 2 ") == 2
    assert normalize_camera_source(0) == 0
    assert normalize_camera_source(3) == 3


def test_normalize_camera_source_rtsp_and_http():
    """Verify RTSP, HTTP, HTTPS, and file paths are preserved as unchanged strings."""
    assert normalize_camera_source("rtsp://192.168.1.100:554/stream1") == "rtsp://192.168.1.100:554/stream1"
    assert normalize_camera_source("http://192.168.1.100:8080/video") == "http://192.168.1.100:8080/video"
    assert normalize_camera_source("https://example.com/live/feed.m3u8") == "https://example.com/live/feed.m3u8"
    assert normalize_camera_source("data/test_video.mp4") == "data/test_video.mp4"


def test_camera_start_and_stop_lifecycle():
    """Verify camera start, list, and stop lifecycle through GaitService."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):
        cam_info = service.start_camera(
            camera_id="CCTV-TEST-101",
            source="0",
            location="Western Transit Corridor",
        )
        assert cam_info["camera_id"] == "CCTV-TEST-101"
        assert cam_info["status"] == "ACTIVE"
        assert "CCTV-TEST-101" in service.active_cameras

        stopped = service.stop_camera("CCTV-TEST-101")
        assert stopped is True
        assert "CCTV-TEST-101" not in service.active_cameras

        assert service.stop_camera("NON_EXISTENT") is False


def test_camera_api_endpoints():
    """Verify FastAPI endpoints /api/v1/cameras/start, /stop, and /cameras using lifespan singleton."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with TestClient(app) as client, \
         patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch("services.camera_source_resolver.CameraSourceResolver.probe_usb_webcam", return_value=True):
        start_resp = client.post(
            "/api/v1/cameras/start",
            json={
                "camera_id": "CCTV-API-99",
                "source": "0",
                "location": "Test Platform",
            },
        )
        assert start_resp.status_code == 200
        data = start_resp.json()
        assert data["camera_id"] == "CCTV-API-99"
        assert data["status"] == "ACTIVE"

        list_resp = client.get("/api/v1/cameras")
        assert list_resp.status_code == 200
        cameras = list_resp.json()
        assert any(c["camera_id"] == "CCTV-API-99" for c in cameras)

        stop_resp = client.post(
            "/api/v1/cameras/stop",
            json={"camera_id": "CCTV-API-99"},
        )
        assert stop_resp.status_code == 200
        assert stop_resp.json()["success"] is True

        stop_404 = client.post(
            "/api/v1/cameras/stop",
            json={"camera_id": "CCTV-API-99"},
        )
        assert stop_404.status_code == 404


def _make_rtsp_config(**overrides):
    """Build an RTSP camera config dict with fast test timeouts."""
    cfg = {
        "type": "rtsp",
        "url": "rtsp://user:secret@10.0.0.1:554/live",
        "device_index": 0,
        "width": 640,
        "height": 480,
        "target_fps": 15,
        "jpeg_quality": 75,
        "preview_max_fps": 15,
        "startup_timeout": 2,
        "startup_retry_interval": 0.05,
        "reconnect_interval": 0,
        "max_reconnect_attempts": 3,
    }
    cfg.update(overrides)
    return cfg


def _dummy_frame():
    """Return a valid 640x480 BGR frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = [0, 255, 0]
    return frame


def test_rtsp_startup_retries_then_succeeds():
    """RTSP VideoCapture opens but first reads fail; bounded retry eventually succeeds."""
    cfg = _make_rtsp_config()
    worker = CameraWorker("CAM-RETRY", cfg)

    frame = _dummy_frame()

    call_count = 0

    def read_effect():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return (False, None)
        return (True, frame)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = read_effect

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        started = worker.start()

    assert started is True
    assert worker.is_connected() is True
    assert worker.stats["frames_captured"] >= 1
    assert worker.get_latest_jpeg() is not None

    worker.stop()
    assert worker.is_connected() is False


def test_rtsp_startup_timeout_clean_failure():
    """RTSP VideoCapture opens but read() never returns a valid frame — bounded timeout."""
    cfg = _make_rtsp_config(startup_timeout=0.5, startup_retry_interval=0.05)
    worker = CameraWorker("CAM-TIMEOUT", cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        started = worker.start()

    assert started is False
    assert worker.is_connected() is False
    assert worker._capture is None
    assert worker.is_running() is False
    mock_cap.release.assert_called()


def test_capture_open_failure():
    """VideoCapture cannot open at all — clean startup failure."""
    cfg = _make_rtsp_config()
    worker = CameraWorker("CAM-NOOPEN", cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        started = worker.start()

    assert started is False
    assert worker._capture is None
    assert worker.is_running() is False


def test_stop_during_startup_handshake():
    """Stop request during first-frame retry loop aborts startup cleanly."""
    cfg = _make_rtsp_config(startup_timeout=5, startup_retry_interval=0.05)
    worker = CameraWorker("CAM-ABORT", cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)

    def set_stop_after_delay():
        time.sleep(0.2)
        worker._stop_event.set()

    stopper = threading.Thread(target=set_stop_after_delay)
    stopper.start()

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        started = worker.start()

    stopper.join()

    assert started is False
    assert worker._capture is None
    assert worker.is_running() is False
    mock_cap.release.assert_called()


def test_duplicate_start_rejected():
    """Starting the same camera twice does not create duplicate workers."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):
        info1 = service.start_camera(camera_id="CAM-DUP", source="0", location="Test")
        assert info1["status"] == "ACTIVE"

        info2 = service.start_camera(camera_id="CAM-DUP", source="0", location="Test")
        assert info2["camera_id"] == "CAM-DUP"
        assert info2["status"] == "ACTIVE"

        assert len([k for k in service.camera_workers if k == "CAM-DUP"]) == 1

        service.stop_camera("CAM-DUP")


def test_repeated_stop_is_safe():
    """Stopping the same camera twice is idempotent and safe."""
    service = GaitService()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _dummy_frame())

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
         patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):
        service.start_camera(camera_id="CAM-RSTOP", source="0", location="Test")
        assert service.stop_camera("CAM-RSTOP") is True
        assert service.stop_camera("CAM-RSTOP") is False
        assert "CAM-RSTOP" not in service.active_cameras
        assert "CAM-RSTOP" not in service.camera_workers


def test_startup_failure_no_stale_active_worker():
    """Failed startup must not leave a stale ACTIVE entry in the registry."""
    service = GaitService()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        try:
            service.start_camera(
                camera_id="CAM-STALE",
                source="rtsp://user:pass@10.0.0.1:554/live",
                location="Test",
                zone_id="Z99",
            )
        except RuntimeError:
            pass

    assert "CAM-STALE" not in service.active_cameras
    assert "CAM-STALE" not in service.camera_workers


def test_startup_failure_releases_zone_reservation():
    """Failed camera startup must release source reservation."""
    service = GaitService()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        try:
            service.start_camera(
                camera_id="CAM-ZONE",
                source="rtsp://user:pass@10.0.0.1:554/live",
                location="Test",
                zone_id="Z50",
            )
        except RuntimeError:
            pass

    assert not service.source_resolver.is_source_reserved(
        "stream:rtsp://user:pass@10.0.0.1:554/live"
    )


def test_credential_sanitization_in_error():
    """RTSP passwords must not appear in API error responses or worker error messages."""
    service = GaitService()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        try:
            service.start_camera(
                camera_id="CAM-CRED",
                source="rtsp://admin:SuperSecret123@192.168.1.100:554/live",
                location="Test",
            )
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            error_msg = str(e)
            assert "SuperSecret123" not in error_msg
            assert "admin" not in error_msg


def test_runtime_frame_failure_triggers_reconnect():
    """Temporary runtime frame read failure triggers reconnect, not permanent death."""
    cfg = _make_rtsp_config(
        startup_timeout=1,
        startup_retry_interval=0.05,
        reconnect_interval=0,
        max_reconnect_attempts=2,
    )
    worker = CameraWorker("CAM-RECON", cfg)

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
        else:
            return (True, frame)

    mock_cap.read.side_effect = read_effect

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        started = worker.start()
        assert started is True

        time.sleep(0.5)

        assert worker.is_running() is True

    worker.stop()


def test_mjpeg_preview_available_after_startup():
    """After successful startup the JPEG preview buffer must contain valid data."""
    cfg = _make_rtsp_config()
    worker = CameraWorker("CAM-PREV", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, frame)

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        started = worker.start()

    assert started is True
    jpeg = worker.get_latest_jpeg()
    assert jpeg is not None
    assert jpeg[:2] == b"\xff\xd8"

    stats = worker.get_stats()
    assert stats["frames_captured"] >= 1
    assert stats["last_frame_at"] is not None

    worker.stop()


def test_successful_stop_releases_resources():
    """Normal stop: thread terminates, capture released, stats updated."""
    cfg = _make_rtsp_config()
    worker = CameraWorker("CAM-STOPOK", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, frame)

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        worker.start()

    assert worker.is_running() is True

    worker.stop()
    assert worker.is_running() is False
    assert worker._capture is None
    assert worker.is_connected() is False


def test_camera_worker_restart():
    """Worker restart() must cleanly stop and restart the worker."""
    cfg = _make_rtsp_config()
    worker = CameraWorker("CAM-RESTART", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, frame)

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        assert worker.start() is True
        assert worker.is_running() is True

        restarted = worker.restart()
        assert restarted is True
        assert worker.is_running() is True
        assert worker.is_connected() is True

    worker.stop()


def test_camera_worker_config_boundary_safety():
    """CameraWorker handles malformed or edge-case configs safely without crashing."""
    malformed_cfg = {
        "type": "usb",
        "device_index": 0,
        "width": -100,
        "height": 0,
        "target_fps": -5,
        "jpeg_quality": 200,
        "preview_max_fps": 0,
        "startup_timeout": -10,
        "startup_retry_interval": -0.5,
        "reconnect_interval": -3,
        "max_reconnect_attempts": -1,
    }
    worker = CameraWorker("CAM-BOUNDS", malformed_cfg)
    assert worker._width >= 16
    assert worker._height >= 16
    assert worker._jpeg_quality <= 100
    assert worker._min_jpeg_interval > 0
    assert worker._startup_timeout >= 0.1
    assert worker._startup_retry_interval >= 0.01
    assert worker._reconnect_interval >= 0


def test_source_resolution_file_and_http():
    """Test resolution of file and http sources."""
    http_cfg = {"type": "http", "url": "http://192.168.1.50:8080/video"}
    w_http = CameraWorker("CAM-HTTP", http_cfg)
    assert w_http._resolve_source() == "http://192.168.1.50:8080/video"

    w_http_empty = CameraWorker("CAM-HTTP-EMPTY", {"type": "http", "url": ""})
    try:
        w_http_empty._resolve_source()
        assert False, "Expected ValueError"
    except ValueError:
        pass

    file_cfg = {"type": "file", "file_path": "data/sample.mp4"}
    w_file = CameraWorker("CAM-FILE", file_cfg)
    assert w_file._resolve_source() == "data/sample.mp4"

    w_file_empty = CameraWorker("CAM-FILE-EMPTY", {"type": "file", "file_path": ""})
    try:
        w_file_empty._resolve_source()
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_resolver_sanitizes_credentials_in_label():
    """CameraSourceResolver must sanitize RTSP credentials in resolved_source_label."""
    from services.camera_source_resolver import CameraSourceResolver
    resolver = CameraSourceResolver()
    res = resolver.resolve_source(
        camera_id="CAM-RESOLVE-CRED",
        requested_source="rtsp://admin:SecretPass456@192.168.1.200:554/live",
    )
    assert "SecretPass456" not in res["resolved_source_label"]
    assert "***:***" in res["resolved_source_label"]
    resolver.release_source_by_camera_id("CAM-RESOLVE-CRED")


def test_reconnect_max_attempts_exceeded_exits_loop():
    """When max_reconnect_attempts is reached, capture loop exits gracefully."""
    cfg = _make_rtsp_config(
        startup_timeout=0.05,
        startup_retry_interval=0.01,
        reconnect_interval=0,
        max_reconnect_attempts=2,
    )
    worker = CameraWorker("CAM-MAXREC", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    call_count = 0

    def read_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (True, frame)
        return (False, None)

    mock_cap.read.side_effect = read_effect

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        assert worker.start() is True

        if worker._thread is not None:
            worker._thread.join(timeout=2.0)

    assert worker.is_running() is False
    assert worker.is_connected() is False
    worker.stop()


def test_system_config_propagates_to_worker():
    """Modifying startup_timeout/startup_retry_interval in system.yaml configuration reaches CameraWorker."""
    service = GaitService()

    custom_yaml = {
        "camera": {
            "startup_timeout": 14.5,
            "startup_retry_interval": 0.45,
            "width": 1280,
            "height": 720,
            "jpeg_quality": 85,
        }
    }

    import yaml
    from unittest.mock import mock_open
    yaml_content = yaml.dump(custom_yaml)

    with patch("builtins.open", mock_open(read_data=yaml_content)), \
         patch("pathlib.Path.exists", return_value=True):
        cam_defaults = service._load_camera_config()
        assert cam_defaults["startup_timeout"] == 14.5
        assert cam_defaults["startup_retry_interval"] == 0.45
        assert cam_defaults["width"] == 1280
        assert cam_defaults["height"] == 720
        assert cam_defaults["jpeg_quality"] == 85

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)

        with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap), \
             patch.object(service.source_resolver, "probe_usb_webcam", return_value=True):
            cam_info = service.start_camera("CAM-CONFIG-TEST", source="auto")
            assert cam_info["status"] == "ACTIVE"

            worker = service.get_camera_worker("CAM-CONFIG-TEST")
            assert worker is not None
            assert worker._startup_timeout == 14.5
            assert worker._startup_retry_interval == 0.45
            assert worker._width == 1280
            assert worker._height == 720
            assert worker._jpeg_quality == 85

            service.stop_camera("CAM-CONFIG-TEST")


def test_concurrent_start_attempts():
    """Multiple threads calling start() simultaneously must result in exactly one active thread."""
    cfg = _make_rtsp_config()
    worker = CameraWorker("CAM-CONC-START", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, frame)

    results = []

    def start_worker():
        with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
            res = worker.start()
            results.append(res)

    threads = [threading.Thread(target=start_worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1
    assert results.count(False) == 4
    assert worker.is_running() is True
    worker.stop()


def test_start_and_stop_race():
    """Calling stop() immediately after or during start() must not leave orphaned capture threads."""
    cfg = _make_rtsp_config(startup_timeout=2, startup_retry_interval=0.05)
    worker = CameraWorker("CAM-RACE", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, frame)

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        worker.start()
        worker.stop()

    assert worker.is_running() is False
    assert worker._capture is None
    assert worker.is_connected() is False


def test_restart_during_reconnect_is_safe():
    """Restarting a worker while it is in the middle of a reconnect cycle must recover cleanly."""
    cfg = _make_rtsp_config(
        startup_timeout=0.05,
        startup_retry_interval=0.01,
        reconnect_interval=0,
        max_reconnect_attempts=5,
    )
    worker = CameraWorker("CAM-RECON-RESTART", cfg)

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

        assert worker.restart() is True
        assert worker.is_running() is True
        assert worker.is_connected() is True

    worker.stop()


def test_failed_startup_followed_by_successful_restart():
    """A failed startup attempt followed by restart() succeeds when source becomes available."""
    cfg = _make_rtsp_config()
    worker = CameraWorker("CAM-FAIL-THEN-RESTART", cfg)

    frame = _dummy_frame()
    mock_cap_fail = MagicMock()
    mock_cap_fail.isOpened.return_value = False

    mock_cap_ok = MagicMock()
    mock_cap_ok.isOpened.return_value = True
    mock_cap_ok.read.return_value = (True, frame)

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap_fail):
        assert worker.start() is False
        assert worker.is_running() is False

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap_ok):
        assert worker.restart() is True
        assert worker.is_running() is True
        assert worker.is_connected() is True

    worker.stop()


def test_capture_release_exception_safety():
    """Even if VideoCapture.release() raises an exception, worker state is reset cleanly."""
    cfg = _make_rtsp_config()
    worker = CameraWorker("CAM-REL-EXC", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, frame)
    mock_cap.release.side_effect = RuntimeError("Driver crash during release")

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        assert worker.start() is True
        assert worker.is_connected() is True

        worker.stop()

    assert worker.is_running() is False
    assert worker._capture is None
    assert worker.is_connected() is False


def test_reconnect_attempts_semantics_zero_infinite():
    """max_reconnect_attempts=0 means infinite reconnects (does not stop on attempt 1 or 2)."""
    cfg = _make_rtsp_config(
        startup_timeout=0.02,
        startup_retry_interval=0.01,
        reconnect_interval=0,
        max_reconnect_attempts=0,
    )
    worker = CameraWorker("CAM-REC-INF", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    call_count = 0

    def read_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (True, frame)
        elif call_count <= 4:
            return (False, None)
        return (True, frame)

    mock_cap.read.side_effect = read_effect

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        assert worker.start() is True
        time.sleep(0.3)
        assert worker.is_running() is True

    worker.stop()


def test_reconnect_attempts_semantics_one():
    """max_reconnect_attempts=1 exits after 1 failed reconnect attempt."""
    cfg = _make_rtsp_config(
        startup_timeout=0.02,
        startup_retry_interval=0.01,
        reconnect_interval=0,
        max_reconnect_attempts=1,
    )
    worker = CameraWorker("CAM-REC-ONE", cfg)

    frame = _dummy_frame()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    call_count = 0

    def read_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (True, frame)
        return (False, None)

    mock_cap.read.side_effect = read_effect

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        assert worker.start() is True
        if worker._thread is not None:
            worker._thread.join(timeout=2.0)

    assert worker.is_running() is False
    worker.stop()


def test_credential_masking_complex_password():
    """RTSP credentials with special characters (e.g. p@ssword) are masked."""
    service = GaitService()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
        try:
            service.start_camera(
                camera_id="CAM-SPECIAL-CRED",
                source="rtsp://user:p@ssword@10.0.0.1:554/live",
                location="Test Zone",
            )
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            err = str(e)
            assert "p@ssword" not in err
            assert "user:p" not in err
            assert "***:***" in err
