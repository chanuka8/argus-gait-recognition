import time

import numpy as np
import pytest

from streaming.production_multicamera_engine import (
    CentralStreamScheduler,
    FramePacket,
    HardwareProfile,
    ProductionMultiCameraEngine,
    StreamIngestionQueue,
    detect_hardware_profile,
)


class DummyCollector:
    def __init__(self):
        self.observations = []

    def record_observation(self, **kwargs):
        self.observations.append(kwargs)


def test_hardware_capability_detection():
    profile = detect_hardware_profile()
    assert isinstance(profile, HardwareProfile)
    assert profile.cpu_cores >= 1
    assert profile.total_ram_mb > 0
    assert profile.recommended_batch_size in (4, 8, 16, 32)
    assert profile.stale_frame_max_age_ms > 0


def test_bounded_queue_overflow_and_backpressure():
    q = StreamIngestionQueue(camera_id="cam_test_01", maxsize=3, max_stale_age_seconds=1.0)
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)


    for i in range(5):
        p = FramePacket(camera_id="cam_test_01", frame_id=i, capture_time=time.monotonic(), frame=dummy)
        success = q.put(p)
        assert success is True

    assert q.qsize() == 3
    assert q.frames_enqueued == 5
    assert q.frames_dropped_overflow == 2


    p_first = q.get()
    assert p_first is not None
    assert p_first.frame_id == 2


def test_stale_frame_dropping():
    q = StreamIngestionQueue(camera_id="cam_stale_01", maxsize=5, max_stale_age_seconds=0.05)
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)

    old_packet = FramePacket(
        camera_id="cam_stale_01",
        frame_id=100,
        capture_time=time.monotonic() - 0.20,
        frame=dummy,
    )
    q.put(old_packet)

    time.sleep(0.01)
    retrieved = q.get()
    assert retrieved is None
    assert q.frames_dropped_stale == 1


def test_fair_stream_scheduler_starvation_prevention():
    scheduler = CentralStreamScheduler(starvation_threshold_seconds=0.1)
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)

    q1 = scheduler.register_stream("cam_high_fps", priority=10, maxsize=10)
    q2 = scheduler.register_stream("cam_low_priority", priority=1, maxsize=10)


    for i in range(5):
        q1.put(FramePacket("cam_high_fps", i, time.monotonic(), dummy))
        q2.put(FramePacket("cam_low_priority", i, time.monotonic(), dummy))

    served = []
    for _ in range(6):
        pkt = scheduler.select_next_frame()
        if pkt:
            served.append(pkt.camera_id)


    assert "cam_high_fps" in served
    assert "cam_low_priority" in served


@pytest.mark.parametrize("num_cams", [1, 4, 8, 16, 32])
def test_unbounded_camera_registration(num_cams):
    engine = ProductionMultiCameraEngine()
    dummy = np.zeros((32, 32, 3), dtype=np.uint8)

    for i in range(num_cams):
        cid = f"cam_scale_{i:03d}"
        q = engine.register_camera(cid, priority=5)
        assert q is not None
        ok = engine.put_frame(cid, dummy, frame_id=1)
        assert ok is True

    telemetry = engine.get_telemetry()
    assert telemetry["registered_cameras_count"] == num_cams
    assert len(telemetry["cameras"]) == num_cams


    for i in range(num_cams):
        engine.unregister_camera(f"cam_scale_{i:03d}")

    assert engine.get_telemetry()["registered_cameras_count"] == 0


def test_camera_stream_isolation():
    engine = ProductionMultiCameraEngine()
    engine.register_camera("cam_A")
    engine.register_camera("cam_B")
    engine.register_camera("cam_C")

    dummy = np.zeros((16, 16, 3), dtype=np.uint8)
    engine.put_frame("cam_A", dummy)
    engine.put_frame("cam_B", dummy)
    engine.put_frame("cam_C", dummy)


    engine.unregister_camera("cam_B")


    assert engine.put_frame("cam_A", dummy) is True
    assert engine.put_frame("cam_C", dummy) is True
    assert engine.scheduler.get_queue("cam_B") is None
    assert engine.scheduler.get_queue("cam_A") is not None


def test_continual_learning_observation_integration():
    collector = DummyCollector()
    engine = ProductionMultiCameraEngine(operational_collector=collector)
    engine.register_camera("cam_live_01")


    dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    pkt = FramePacket(
        camera_id="cam_live_01",
        frame_id=1,
        capture_time=time.monotonic(),
        frame=dummy_frame,
    )

    engine._process_single_frame(pkt)
    telemetry = engine.get_telemetry()
    assert telemetry["cameras"]["cam_live_01"]["processed_frames"] == 1


def test_process_single_frame_tracking_and_caching():
    class MockDetector:
        def detect(self, frame):
            return [{"bbox": [10, 10, 50, 50], "confidence": 0.95}]

    class MockTracker:
        def update(self, detections, shape):
            return [{"track_id": 42, "bbox": [10, 10, 50, 50]}]

    received_events = []
    engine = ProductionMultiCameraEngine(
        detector=MockDetector(),
        event_callback=lambda evt: received_events.append(evt),
    )
    engine.register_camera("cam_cache_test")
    engine._camera_trackers["cam_cache_test"] = MockTracker()

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pkt = FramePacket(
        camera_id="cam_cache_test",
        frame_id=10,
        capture_time=time.monotonic(),
        frame=dummy_frame,
    )


    engine._process_single_frame(pkt)


    res = engine.cache.get("cam_cache_test", 42)
    assert res is not None
    assert res.track_id == 42
    assert res.camera_id == "cam_cache_test"
    assert res.bbox == [10, 10, 50, 50]
    assert res.identity == "UNKNOWN_PERSON"

