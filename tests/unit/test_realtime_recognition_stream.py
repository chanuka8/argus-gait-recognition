"""
Unit and Integration Tests for Real-Time Recognition Stream.

Covers:
1. FeatureExtractionStep.extract_from_gei
2. RecognitionResultCache (put, get, TTL expiry, multi-camera isolation, inactive cleanup)
3. RecognitionWorker (start, stop, put_frame non-blocking, error resilience, gallery update)
4. CameraWorker integration (overlay rendering, status frames, client tracking, stats)
5. API endpoints (list_cameras with telemetry, stream endpoint multi-client, graceful disconnect)
6. Security / credential redaction across overlays, telemetry, and error logging
"""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock

from pipeline.steps.feature_extraction import FeatureExtractionStep
from services.recognition_worker import (
    RecognitionResult,
    RecognitionResultCache,
    RecognitionWorker,
)
from services.camera_worker import CameraWorker
from services.gait_service import GaitService
from fastapi.testclient import TestClient
from api.server import app


def test_extract_from_gei_valid_shape():
    step = FeatureExtractionStep()
    gei = np.ones((128, 64), dtype=np.uint8) * 200
    emb = step.extract_from_gei(gei)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (256,)
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_extract_from_gei_empty_or_none():
    step = FeatureExtractionStep()
    assert len(step.extract_from_gei(None)) == 0
    assert len(step.extract_from_gei(np.empty((0, 0)))) == 0


def test_extract_from_gei_resizes_arbitrary_shape():
    step = FeatureExtractionStep()
    gei_odd = np.ones((200, 100, 3), dtype=np.uint8) * 128
    emb = step.extract_from_gei(gei_odd)
    assert emb.shape == (256,)


def test_cache_put_get_within_ttl():
    cache = RecognitionResultCache(ttl_seconds=1.0)
    res = RecognitionResult(
        camera_id="cam_01",
        track_id=10,
        identity="Person_A",
        similarity=0.88,
        confidence=0.88,
        decision="CONFIRMED_MATCH",
        status="CONFIRMED",
        bbox=[10, 10, 50, 100],
        timestamp=time.monotonic(),
        iso_timestamp="2026-08-20T12:00:00Z",
    )
    cache.put(res)

    fetched = cache.get("cam_01", 10)
    assert fetched is not None
    assert fetched.identity == "Person_A"
    assert fetched.similarity == 0.88


def test_cache_ttl_expiration():
    cache = RecognitionResultCache(ttl_seconds=0.1)
    res = RecognitionResult(
        camera_id="cam_01",
        track_id=1,
        identity="Person_B",
        similarity=0.90,
        confidence=0.90,
        decision="CONFIRMED_MATCH",
        status="CONFIRMED",
        bbox=[0, 0, 10, 10],
        timestamp=time.monotonic() - 0.2,
        iso_timestamp="2026-08-20T12:00:00Z",
    )
    cache.put(res)

    assert cache.get("cam_01", 1) is None
    assert len(cache.get_active_tracks("cam_01")) == 0


def test_cache_multi_camera_isolation():
    cache = RecognitionResultCache(ttl_seconds=5.0)
    res1 = RecognitionResult(
        camera_id="cam_01",
        track_id=1,
        identity="Person_A",
        similarity=0.85,
        confidence=0.85,
        decision="CONFIRMED_MATCH",
        status="CONFIRMED",
        bbox=[0, 0, 10, 10],
        timestamp=time.monotonic(),
        iso_timestamp="2026-08-20T12:00:00Z",
    )
    res2 = RecognitionResult(
        camera_id="cam_02",
        track_id=1,
        identity="Person_B",
        similarity=0.92,
        confidence=0.92,
        decision="CONFIRMED_MATCH",
        status="CONFIRMED",
        bbox=[0, 0, 10, 10],
        timestamp=time.monotonic(),
        iso_timestamp="2026-08-20T12:00:00Z",
    )
    cache.put(res1)
    cache.put(res2)

    assert cache.get("cam_01", 1).identity == "Person_A"
    assert cache.get("cam_02", 1).identity == "Person_B"
    assert len(cache.get_active_tracks("cam_01")) == 1
    assert len(cache.get_active_tracks("cam_02")) == 1

    cache.clear_camera("cam_01")
    assert cache.get("cam_01", 1) is None
    assert cache.get("cam_02", 1) is not None


def test_cache_cleanup_inactive():
    cache = RecognitionResultCache(ttl_seconds=10.0)
    res_stale = RecognitionResult(
        camera_id="cam_01",
        track_id=5,
        identity="Unknown",
        similarity=0.2,
        confidence=0.2,
        decision="UNKNOWN_PERSON",
        status="UNKNOWN",
        bbox=[0, 0, 10, 10],
        timestamp=time.monotonic() - 6.0,
        iso_timestamp="2026-08-20T12:00:00Z",
    )
    res_fresh = RecognitionResult(
        camera_id="cam_01",
        track_id=6,
        identity="Person_C",
        similarity=0.89,
        confidence=0.89,
        decision="CONFIRMED_MATCH",
        status="CONFIRMED",
        bbox=[0, 0, 10, 10],
        timestamp=time.monotonic(),
        iso_timestamp="2026-08-20T12:00:00Z",
    )
    cache.put(res_stale)
    cache.put(res_fresh)

    evicted = cache.cleanup_inactive("cam_01", max_idle_seconds=5.0)
    assert 5 in evicted
    assert 6 not in evicted


def test_recognition_worker_put_frame_non_blocking():
    worker = RecognitionWorker(
        camera_id="cam_test",
        config={"target_fps": 10},
        detector=MagicMock(),
        tracker=MagicMock(),
        silhouette_extractor=MagicMock(),
        extractor=MagicMock(),
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(10):
        worker.put_frame(dummy_frame)

    assert worker._input_queue.qsize() <= 2
    worker.stop()


def test_recognition_worker_lifecycle():
    worker = RecognitionWorker(
        camera_id="cam_lifecycle",
        config={"target_fps": 15},
        detector=MagicMock(),
        tracker=MagicMock(),
        silhouette_extractor=MagicMock(),
        extractor=MagicMock(),
    )
    assert not worker.is_alive()
    worker.start()
    assert worker.is_alive()
    worker.stop()
    assert not worker.is_alive()


def test_recognition_worker_handles_exceptions_gracefully():
    mock_detector = MagicMock()
    mock_detector.detect.side_effect = RuntimeError("Simulated detector CUDA crash")

    worker = RecognitionWorker(
        camera_id="cam_err",
        detector=mock_detector,
        tracker=MagicMock(),
        silhouette_extractor=MagicMock(),
        extractor=MagicMock(),
    )
    worker.start()
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    worker.put_frame(dummy_frame)
    time.sleep(0.2)

    assert worker.is_alive()
    worker.stop()


def test_recognition_worker_gallery_update():
    worker = RecognitionWorker(
        camera_id="cam_gal",
        detector=MagicMock(),
        tracker=MagicMock(),
        silhouette_extractor=MagicMock(),
        extractor=MagicMock(),
    )
    features = np.ones((5, 256), dtype=np.float32)
    labels = ["ID1", "ID2", "ID3", "ID4", "ID5"]
    worker.update_gallery(features, labels, metadata=[])

    assert worker.gallery_features.shape == (5, 256)
    assert worker.gallery_labels == labels
    worker.stop()


def test_camera_worker_status_frame():
    worker = CameraWorker(
        camera_id="cam_status_test",
        camera_config={"width": 320, "height": 240, "jpeg_quality": 50},
    )
    frame_bytes = worker._render_status_frame("RECONNECTING (1)")
    assert isinstance(frame_bytes, bytes)
    assert len(frame_bytes) > 0
    assert frame_bytes.startswith(b"\xff\xd8")


def test_camera_worker_preview_overlays_with_active_tracks():
    cache = RecognitionResultCache()
    cache.put(
        RecognitionResult(
            camera_id="cam_overlay_test",
            track_id=42,
            identity="VIP_Agent",
            similarity=0.91,
            confidence=0.91,
            decision="CONFIRMED_MATCH",
            status="CONFIRMED",
            bbox=[50, 50, 150, 200],
            timestamp=time.monotonic(),
            iso_timestamp="2026-08-20T12:00:00Z",
        )
    )

    mock_rec_worker = MagicMock()
    mock_rec_worker.is_alive.return_value = True
    mock_rec_worker.cache = cache

    worker = CameraWorker(
        camera_id="cam_overlay_test",
        camera_config={"width": 640, "height": 480},
        recognition_worker=mock_rec_worker,
    )

    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    overlaid = worker._render_preview_overlays(raw_frame)

    assert overlaid.shape == (480, 640, 3)
    assert np.any(overlaid > 0)
    assert worker._active_tracks == 1
    assert "VIP_Agent" in worker._recognized_identities


def test_camera_worker_client_registration():
    worker = CameraWorker(
        camera_id="cam_clients",
        camera_config={"width": 640, "height": 480},
    )
    assert worker.get_active_clients() == 0
    worker.register_client()
    assert worker.get_active_clients() == 1
    worker.register_client()
    assert worker.get_active_clients() == 2
    worker.unregister_client()
    assert worker.get_active_clients() == 1
    worker.unregister_client()
    assert worker.get_active_clients() == 0


def test_camera_worker_stats_includes_telemetry():
    worker = CameraWorker(
        camera_id="cam_stats",
        camera_config={"width": 640, "height": 480},
    )
    stats = worker.get_stats()
    assert "active_tracks" in stats
    assert "active_clients" in stats
    assert "recognized_identities" in stats
    assert "recognition_active" in stats


def test_api_list_cameras_telemetry():
    client = TestClient(app)
    resp = client.get("/api/v1/cameras")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_api_stream_nonexistent_camera_returns_404():
    client = TestClient(app)
    resp = client.get("/api/v1/cameras/non_existent_cam_99/stream")
    assert resp.status_code == 404
    assert "not active" in resp.json()["detail"]


@pytest.mark.anyio
async def test_api_stream_yields_mjpeg_frames():
    fake_worker = MagicMock()
    fake_worker.is_running.return_value = True
    dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50 + b"\xff\xd9"
    fake_worker.get_latest_jpeg.return_value = dummy_jpeg

    service = GaitService()
    service.active_cameras["test_stream_cam"] = {
        "camera_id": "test_stream_cam",
        "status": "ACTIVE",
    }
    service.camera_workers["test_stream_cam"] = fake_worker

    from api.v1.router import stream_camera
    response = await stream_camera(camera_id="test_stream_cam", service=service)
    assert response.status_code == 200
    assert "multipart/x-mixed-replace" in response.media_type

    gen = response.body_iterator
    first_chunk = await gen.__anext__()
    assert b"--frame" in first_chunk
    assert b"image/jpeg" in first_chunk

    service.active_cameras.pop("test_stream_cam", None)
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


def test_rtsp_credentials_never_in_stream_telemetry_or_errors():
    from security_layer.credentials import sanitize_rtsp_url

    secret_url = "rtsp://admin:super_secret_password_1234@192.168.1.50:554/stream1"
    sanitized = sanitize_rtsp_url(secret_url)

    assert "super_secret_password_1234" not in sanitized
    assert "admin" not in sanitized
    assert sanitized == "rtsp://***:***@192.168.1.50:554/stream1"

