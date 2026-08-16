import pytest
import time
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.server import app
from services.camera_worker import CameraWorker
from services.gait_service import GaitService
from services.camera_source_resolver import CameraSourceResolver


def test_camera_worker_latest_jpeg_buffer():
    """Verify CameraWorker stores and returns JPEG encoded frames."""
    cfg = {
        "type": "usb",
        "device_index": 0,
        "width": 640,
        "height": 480,
        "target_fps": 15,
        "jpeg_quality": 75,
        "preview_max_fps": 15,
    }
    worker = CameraWorker("CCTV-TEST-STREAM", cfg, None, None)

    # Initially before capture
    assert worker.get_latest_jpeg() is None
    assert worker.stats["fps"] == 0.0

    # Simulate a frame captured
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[100:200, 100:200] = [0, 255, 0]

    success, enc_buf = cv2.imencode(".jpg", dummy_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    assert success is True
    jpeg_bytes = enc_buf.tobytes()

    with worker._lock:
        worker._latest_jpeg = jpeg_bytes
        worker._last_frame_at = "2026-08-16T18:00:00Z"
        worker._frame_count = 5
        worker.stats["frames_captured"] = 5

    returned_jpeg = worker.get_latest_jpeg()
    assert returned_jpeg is not None
    assert returned_jpeg.startswith(b"\xff\xd8")  # JPEG magic header

    stats = worker.get_stats()
    assert stats["frames_captured"] == 5
    assert stats["last_frame_at"] == "2026-08-16T18:00:00Z"


def test_get_cameras_empty_by_default():
    """Verify GET /api/v1/cameras returns empty list by default."""
    with TestClient(app) as client:
        resp = client.get("/api/v1/cameras")
        assert resp.status_code == 200
        assert resp.json() == []


def test_mjpeg_stream_and_snapshot_endpoints():
    """Verify /api/v1/cameras/{camera_id}/stream and /snapshot endpoints with TestClient."""
    with TestClient(app) as client:
        # 1. Non-active camera returns 404
        resp_404 = client.get("/api/v1/cameras/NON_EXISTENT_CAM/stream")
        assert resp_404.status_code == 404

        snap_404 = client.get("/api/v1/cameras/NON_EXISTENT_CAM/snapshot")
        assert snap_404.status_code == 404

        # 2. Start Camera via API
        start_resp = client.post(
            "/api/v1/cameras/start",
            json={
                "camera_id": "CCTV-PREVIEW-1",
                "source": "0",
                "location": "Central Hub",
                "zone_id": "Z01",
            }
        )
        assert start_resp.status_code == 200
        cam_data = start_resp.json()
        assert cam_data["camera_id"] == "CCTV-PREVIEW-1"
        assert cam_data["status"] == "ACTIVE"
        assert cam_data["preview_url"] == "/api/v1/cameras/CCTV-PREVIEW-1/stream"
        assert cam_data["resolved_source_label"] == "USB Webcam 0"
        assert cam_data["processed_frames"] >= 1

        # 3. Verify in active cameras list
        list_resp = client.get("/api/v1/cameras")
        assert list_resp.status_code == 200
        active_list = list_resp.json()
        matching = [c for c in active_list if c["camera_id"] == "CCTV-PREVIEW-1"]
        assert len(matching) == 1
        assert matching[0]["preview_url"] == "/api/v1/cameras/CCTV-PREVIEW-1/stream"
        assert matching[0]["resolved_source_label"] == "USB Webcam 0"

        # 4. Stop Camera
        stop_resp = client.post(
            "/api/v1/cameras/stop",
            json={"camera_id": "CCTV-PREVIEW-1"}
        )
        assert stop_resp.status_code == 200
        assert stop_resp.json()["success"] is True

        # 5. List is empty again
        list_after = client.get("/api/v1/cameras")
        assert list_after.status_code == 200
        assert len([c for c in list_after.json() if c["camera_id"] == "CCTV-PREVIEW-1"]) == 0
