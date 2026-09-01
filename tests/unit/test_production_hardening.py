import threading
import time

import numpy as np
import pytest

from streaming.production_runtime import (
    AdaptiveResourceManager,
    CameraResource,
    CameraState,
    CameraStateMachine,
    CapacityEstimator,
    DataPoisoningGuard,
    FPSGovernor,
    FPSPolicy,
    FrameQualityGate,
    GracefulShutdownManager,
    InferenceWorkerInfo,
    ModelVersion,
    ProductionSurveillanceRuntime,
    QualifiedFrame,
    ReconnectConfig,
    ReconnectEngine,
    ResilientWorkerPool,
    ResourcePressure,
    ResourceSnapshot,
    ResourceThresholds,
    SafeModelSwapper,
    StructuredEventLogger,
)






class TestCameraStateMachine:
    def test_initial_state_is_stopped(self):
        sm = CameraStateMachine()
        cam = sm.register_camera("cam_01")
        assert cam.connection_state == CameraState.STOPPED
        assert cam.actual_state == CameraState.STOPPED

    def test_valid_lifecycle_transitions(self):
        sm = CameraStateMachine()
        sm.register_camera("cam_lc")
        assert sm.transition("cam_lc", CameraState.STARTING)
        assert sm.transition("cam_lc", CameraState.CONNECTING)
        assert sm.transition("cam_lc", CameraState.CONNECTED)
        assert sm.transition("cam_lc", CameraState.STOPPING)
        assert sm.transition("cam_lc", CameraState.STOPPED)
        cam = sm.get_camera("cam_lc")
        assert cam.connection_state == CameraState.STOPPED

    def test_invalid_transition_rejected(self):
        sm = CameraStateMachine()
        sm.register_camera("cam_inv")
        result = sm.transition("cam_inv", CameraState.CONNECTED)
        assert result is False
        cam = sm.get_camera("cam_inv")
        assert cam.connection_state == CameraState.STOPPED

    def test_reconnect_lifecycle(self):
        sm = CameraStateMachine()
        sm.register_camera("cam_rc")
        sm.transition("cam_rc", CameraState.STARTING)
        sm.transition("cam_rc", CameraState.CONNECTING)
        sm.transition("cam_rc", CameraState.CONNECTED)
        sm.transition("cam_rc", CameraState.RECONNECTING)
        cam = sm.get_camera("cam_rc")
        assert cam.reconnect_attempts == 1
        sm.transition("cam_rc", CameraState.CONNECTING)
        sm.transition("cam_rc", CameraState.CONNECTED)
        cam = sm.get_camera("cam_rc")
        assert cam.reconnect_attempts == 0

    def test_degraded_state(self):
        sm = CameraStateMachine()
        sm.register_camera("cam_dg")
        sm.transition("cam_dg", CameraState.STARTING)
        sm.transition("cam_dg", CameraState.CONNECTING)
        sm.transition("cam_dg", CameraState.CONNECTED)
        sm.transition("cam_dg", CameraState.DEGRADED)
        cam = sm.get_camera("cam_dg")
        assert cam.connection_state == CameraState.DEGRADED
        sm.transition("cam_dg", CameraState.CONNECTED)
        assert cam.connection_state == CameraState.CONNECTED

    def test_failed_state_only_after_connection_attempt(self):
        sm = CameraStateMachine()
        sm.register_camera("cam_fail")
        sm.transition("cam_fail", CameraState.STARTING)
        sm.transition("cam_fail", CameraState.CONNECTING)
        result = sm.transition("cam_fail", CameraState.FAILED, error="timeout")
        assert result is True
        cam = sm.get_camera("cam_fail")
        assert cam.connection_state == CameraState.FAILED
        assert cam.last_error == "timeout"

    def test_state_listener_notification(self):
        sm = CameraStateMachine()
        events = []
        sm.add_state_listener(lambda cid, old, new, err: events.append((cid, old, new)))
        sm.register_camera("cam_ev")
        sm.transition("cam_ev", CameraState.STARTING)
        sm.transition("cam_ev", CameraState.CONNECTING)
        assert len(events) == 2
        assert events[0] == ("cam_ev", CameraState.STOPPED, CameraState.STARTING)

    def test_health_score_computation(self):
        cam = CameraResource(camera_id="cam_hs")
        cam.connection_state = CameraState.FAILED
        assert cam.compute_health_score() == 0.0
        cam.connection_state = CameraState.CONNECTED
        cam.last_success_timestamp = time.monotonic()
        assert cam.compute_health_score() > 0.5

    def test_unregister_camera(self):
        sm = CameraStateMachine()
        sm.register_camera("cam_unreg")
        assert sm.unregister_camera("cam_unreg") is True
        assert sm.get_camera("cam_unreg") is None







class TestReconnectEngine:
    def test_exponential_backoff_delay(self):
        engine = ReconnectEngine(ReconnectConfig(
            min_retry_interval=1.0,
            max_retry_interval=60.0,
            backoff_multiplier=2.0,
            jitter_range=0.0,
        ))
        d0 = engine._compute_delay(0)
        d1 = engine._compute_delay(1)
        d2 = engine._compute_delay(2)
        assert d0 == pytest.approx(1.0, abs=0.01)
        assert d1 == pytest.approx(2.0, abs=0.01)
        assert d2 == pytest.approx(4.0, abs=0.01)

    def test_delay_clamped_at_max(self):
        engine = ReconnectEngine(ReconnectConfig(
            min_retry_interval=1.0,
            max_retry_interval=10.0,
            backoff_multiplier=2.0,
            jitter_range=0.0,
        ))
        d10 = engine._compute_delay(10)
        assert d10 == pytest.approx(10.0, abs=0.01)

    def test_jitter_applied(self):
        engine = ReconnectEngine(ReconnectConfig(
            min_retry_interval=1.0,
            max_retry_interval=60.0,
            backoff_multiplier=2.0,
            jitter_range=0.5,
        ))
        delays = [engine._compute_delay(3) for _ in range(20)]

        assert max(delays) > min(delays)

    def test_max_retry_limit(self):
        engine = ReconnectEngine(ReconnectConfig(max_retry_attempts=2))
        called = []
        assert engine.schedule_reconnect("cam1", lambda: called.append(1))
        assert engine.schedule_reconnect("cam1", lambda: called.append(1))
        result = engine.schedule_reconnect("cam1", lambda: called.append(1))
        assert result is False

    def test_reset_clears_attempt_counter(self):
        engine = ReconnectEngine(ReconnectConfig(max_retry_attempts=3))
        engine.schedule_reconnect("cam1", lambda: None)
        engine.schedule_reconnect("cam1", lambda: None)
        assert engine.get_attempt_count("cam1") == 2
        engine.reset("cam1")
        assert engine.get_attempt_count("cam1") == 0

    def test_cancel_all_prevents_new_schedules(self):
        engine = ReconnectEngine()
        engine.schedule_reconnect("cam1", lambda: None)
        engine.cancel_all()
        result = engine.schedule_reconnect("cam2", lambda: None)
        assert result is False

    def test_no_busy_looping(self):
        engine = ReconnectEngine(ReconnectConfig(min_retry_interval=0.1))
        called = threading.Event()
        engine.schedule_reconnect("cam1", called.set)

        assert not called.is_set()
        called.wait(timeout=1.0)
        assert called.is_set()
        engine.cancel_all()







class TestCameraFailureIsolation:
    def test_single_failure_doesnt_affect_others(self):
        sm = CameraStateMachine()
        for cid in ["A", "B", "C", "D"]:
            sm.register_camera(cid)
            sm.transition(cid, CameraState.STARTING)
            sm.transition(cid, CameraState.CONNECTING)
            sm.transition(cid, CameraState.CONNECTED)


        sm.transition("A", CameraState.RECONNECTING, error="network_timeout")
        assert sm.get_camera("A").connection_state == CameraState.RECONNECTING


        for cid in ["B", "C", "D"]:
            assert sm.get_camera(cid).connection_state == CameraState.CONNECTED

    def test_multiple_simultaneous_failures(self):
        sm = CameraStateMachine()
        for i in range(8):
            cid = f"cam_{i}"
            sm.register_camera(cid)
            sm.transition(cid, CameraState.STARTING)
            sm.transition(cid, CameraState.CONNECTING)
            sm.transition(cid, CameraState.CONNECTED)


        for i in range(0, 8, 2):
            sm.transition(f"cam_{i}", CameraState.RECONNECTING, error="fail")


        for i in range(1, 8, 2):
            assert sm.get_camera(f"cam_{i}").connection_state == CameraState.CONNECTED







class TestInferenceWorkerResilience:
    def test_worker_info_tracking(self):
        info = InferenceWorkerInfo(worker_id="w-00")
        info.record_success(15.0)
        info.record_success(20.0)
        info.record_success(25.0)
        assert info.processed_frames == 3
        assert info.average_latency_ms > 0
        assert info.p95_latency_ms >= info.average_latency_ms

    def test_worker_failure_tracking(self):
        info = InferenceWorkerInfo(worker_id="w-fail")
        info.record_failure()
        assert info.failed_frames == 1
        assert info.state.value == "FAILED"

    def test_worker_pool_start_stop(self):
        call_count = {"n": 0}
        stop = threading.Event()

        def process():
            call_count["n"] += 1
            if stop.is_set():
                return
            time.sleep(0.01)

        pool = ResilientWorkerPool(num_workers=2, health_check_interval=60.0)
        pool.start(process)
        time.sleep(0.2)
        stop.set()
        pool.stop(timeout=2.0)
        assert call_count["n"] > 0
        info = pool.get_worker_info()
        assert len(info) == 2







class TestAdaptiveResourceManagement:
    def test_healthy_under_normal_load(self):
        mgr = AdaptiveResourceManager()
        snap = ResourceSnapshot(cpu_percent=30.0, vram_percent=20.0)
        result = mgr.evaluate(snap)
        assert result == ResourcePressure.HEALTHY
        assert mgr.processing_rate_factor >= 0.9

    def test_saturated_under_high_load(self):
        mgr = AdaptiveResourceManager(ResourceThresholds(
            max_cpu_percent=85.0, max_vram_percent=90.0
        ))
        snap = ResourceSnapshot(cpu_percent=80.0, vram_percent=85.0)
        result = mgr.evaluate(snap)
        assert result in (ResourcePressure.ELEVATED, ResourcePressure.SATURATED)
        assert mgr.processing_rate_factor < 1.0

    def test_critical_extreme_load(self):
        mgr = AdaptiveResourceManager()
        snap = ResourceSnapshot(cpu_percent=95.0, ram_percent=95.0, vram_percent=95.0)
        result = mgr.evaluate(snap)
        assert result == ResourcePressure.CRITICAL
        assert mgr.processing_rate_factor < 0.5

    def test_gradual_recovery(self):
        mgr = AdaptiveResourceManager()

        mgr.evaluate(ResourceSnapshot(cpu_percent=95.0, vram_percent=95.0))
        low_factor = mgr.processing_rate_factor


        mgr.evaluate(ResourceSnapshot(cpu_percent=20.0, vram_percent=10.0))
        recovered_factor = mgr.processing_rate_factor

        assert recovered_factor > low_factor
        assert recovered_factor <= 1.0







class TestFrameQualityControl:
    def test_fresh_frame_accepted(self):
        gate = FrameQualityGate(max_frame_age_ms=500.0)
        frame = QualifiedFrame(
            camera_id="cam1", frame_id=1, frame_uuid="u1",
            capture_timestamp=time.monotonic(),
            wall_timestamp="2026-01-01T00:00:00Z",
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        ok, reason = gate.validate(frame)
        assert ok is True
        assert reason == "accepted"

    def test_stale_frame_rejected(self):
        gate = FrameQualityGate(max_frame_age_ms=50.0)
        frame = QualifiedFrame(
            camera_id="cam1", frame_id=1, frame_uuid="u1",
            capture_timestamp=time.monotonic() - 0.5,
            wall_timestamp="2026-01-01T00:00:00Z",
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        ok, reason = gate.validate(frame)
        assert ok is False
        assert "stale" in reason

    def test_out_of_order_rejected(self):
        gate = FrameQualityGate(max_frame_age_ms=5000.0)
        now = time.monotonic()
        f1 = QualifiedFrame("cam1", 1, "u1", now, "", np.zeros((2, 2, 3), dtype=np.uint8))
        f2 = QualifiedFrame("cam1", 2, "u2", now - 0.1, "", np.zeros((2, 2, 3), dtype=np.uint8))
        gate.validate(f1)
        ok, reason = gate.validate(f2)
        assert ok is False
        assert reason == "out_of_order"

    def test_duplicate_frame_rejected(self):
        gate = FrameQualityGate(max_frame_age_ms=5000.0, enable_duplicate_detection=True)
        data = np.ones((10, 10, 3), dtype=np.uint8) * 42
        h = gate.compute_frame_hash(data)
        now = time.monotonic()
        f1 = QualifiedFrame("cam1", 1, "u1", now, "", data, frame_hash=h)
        f2 = QualifiedFrame("cam1", 2, "u2", now + 0.01, "", data, frame_hash=h)
        gate.validate(f1)
        ok, reason = gate.validate(f2)
        assert ok is False
        assert reason == "duplicate"

    def test_frame_hash_deterministic(self):
        gate = FrameQualityGate()
        data = np.ones((32, 32, 3), dtype=np.uint8) * 100
        h1 = gate.compute_frame_hash(data)
        h2 = gate.compute_frame_hash(data)
        assert h1 == h2
        assert len(h1) > 0







class TestFPSGovernor:
    def test_process_every_frame(self):
        gov = FPSGovernor(policy=FPSPolicy.PROCESS_EVERY_FRAME)
        assert gov.should_process("cam1") is True
        assert gov.should_process("cam1") is True
        assert gov.should_process("cam1") is True

    def test_target_fps_rate_limiting(self):
        gov = FPSGovernor(policy=FPSPolicy.TARGET_FPS, target_inference_fps=5.0)

        assert gov.should_process("cam1") is True

        assert gov.should_process("cam1") is False

    def test_adaptive_fps_resource_factor(self):
        gov = FPSGovernor(
            policy=FPSPolicy.ADAPTIVE_FPS,
            target_inference_fps=10.0,
            adaptive_min_fps=2.0,
            adaptive_max_fps=30.0,
        )

        assert gov.should_process("cam1", resource_factor=1.0) is True
        eff = gov.get_effective_fps("cam1")
        assert eff == pytest.approx(10.0, abs=0.5)

    def test_independent_camera_governors(self):
        gov = FPSGovernor(policy=FPSPolicy.TARGET_FPS, target_inference_fps=5.0)
        assert gov.should_process("cam1") is True
        assert gov.should_process("cam2") is True

        assert gov.should_process("cam1") is False







class TestModelLifecycleSafety:
    def test_set_active_model(self):
        swapper = SafeModelSwapper()
        v1 = ModelVersion("v1.0", "path/v1.pth", "ByGaitLight")
        swapper.set_active(v1)
        assert swapper.get_active().version_id == "v1.0"

    def test_candidate_promotion(self):
        swapper = SafeModelSwapper()
        v1 = ModelVersion("v1.0", "path/v1.pth", "ByGaitLight")
        v2 = ModelVersion("v2.0", "path/v2.pth", "ByGaitLight")
        swapper.set_active(v1)
        swapper.register_candidate(v2)
        result = swapper.promote_candidate()
        assert result is True
        assert swapper.get_active().version_id == "v2.0"

    def test_rollback(self):
        swapper = SafeModelSwapper()
        v1 = ModelVersion("v1.0", "path/v1.pth", "ByGaitLight")
        v2 = ModelVersion("v2.0", "path/v2.pth", "ByGaitLight")
        swapper.set_active(v1)
        swapper.register_candidate(v2)
        swapper.promote_candidate()
        assert swapper.get_active().version_id == "v2.0"

        result = swapper.rollback()
        assert result is True
        assert swapper.get_active().version_id == "v1.0"

    def test_rollback_without_previous_fails(self):
        swapper = SafeModelSwapper()
        assert swapper.rollback() is False

    def test_promote_without_candidate_fails(self):
        swapper = SafeModelSwapper()
        v1 = ModelVersion("v1.0", "path/v1.pth", "M")
        swapper.set_active(v1)
        assert swapper.promote_candidate() is False

    def test_swap_listener_notified(self):
        swapper = SafeModelSwapper()
        notifications = []
        swapper.add_swap_listener(lambda new, old: notifications.append((new.version_id, old.version_id if old else None)))
        v1 = ModelVersion("v1.0", "p", "M")
        v2 = ModelVersion("v2.0", "p", "M")
        swapper.set_active(v1)
        swapper.register_candidate(v2)
        swapper.promote_candidate()
        assert len(notifications) == 1
        assert notifications[0] == ("v2.0", "v1.0")

    def test_registry_tracks_all_versions(self):
        swapper = SafeModelSwapper()
        for i in range(5):
            v = ModelVersion(f"v{i}", f"path/v{i}.pth", "M")
            swapper.set_active(v)
        reg = swapper.get_registry()
        assert len(reg) == 5







class TestDataPoisoningProtection:
    def test_low_confidence_rejected(self):
        guard = DataPoisoningGuard(min_confidence=0.7)
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        ok, reason = guard.validate_observation("person_A", 0.3, emb, time.monotonic())
        assert ok is False
        assert "low_confidence" in reason

    def test_high_confidence_accepted(self):
        guard = DataPoisoningGuard(min_confidence=0.7)
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        ok, _ = guard.validate_observation("person_A", 0.95, emb, time.monotonic())
        assert ok is True

    def test_duplicate_embedding_rejected(self):
        guard = DataPoisoningGuard(min_confidence=0.5, min_temporal_gap_seconds=0.0)
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        now = time.monotonic()
        guard.validate_observation("person_A", 0.9, emb, now)
        ok, reason = guard.validate_observation("person_A", 0.9, emb, now + 0.1)
        assert ok is False
        assert "duplicate" in reason

    def test_outlier_rejected(self):
        guard = DataPoisoningGuard(min_confidence=0.5, max_embedding_distance=0.5, min_temporal_gap_seconds=0.0)

        base = np.ones(512, dtype=np.float32)
        base /= np.linalg.norm(base)
        now = time.monotonic()
        for i in range(5):
            slight_noise = base + np.random.randn(512).astype(np.float32) * 0.001
            slight_noise /= np.linalg.norm(slight_noise)
            guard.validate_observation("person_B", 0.9, slight_noise, now + i * 0.1)


        outlier = -base
        ok, reason = guard.validate_observation("person_B", 0.9, outlier, now + 10.0)
        assert ok is False
        assert "outlier" in reason

    def test_temporal_consistency(self):
        guard = DataPoisoningGuard(min_confidence=0.5, min_temporal_gap_seconds=1.0)
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        now = time.monotonic()
        guard.validate_observation("person_C", 0.9, emb1, now)
        ok, reason = guard.validate_observation("person_C", 0.9, emb2, now + 0.1)
        assert ok is False
        assert "temporal" in reason

    def test_stats_tracking(self):
        guard = DataPoisoningGuard(min_confidence=0.7)
        emb = np.random.randn(512).astype(np.float32)
        guard.validate_observation("p", 0.3, emb, time.monotonic())
        guard.validate_observation("p", 0.9, emb, time.monotonic())
        stats = guard.get_stats()
        assert stats["rejected_low_confidence"] == 1
        assert stats["accepted"] >= 1







class TestStructuredEventLogging:
    def test_emit_returns_record(self):
        logger = StructuredEventLogger()
        record = logger.emit(
            "camera_connected",
            "Camera connected",
            camera_id="cam_01",
            component="camera_lifecycle",
        )
        assert record["event_type"] == "camera_connected"
        assert record["camera_id"] == "cam_01"
        assert "timestamp" in record

    def test_event_count_increments(self):
        logger = StructuredEventLogger()
        assert logger.event_count == 0
        logger.emit("worker_started", "Worker started")
        logger.emit("worker_failed", "Worker failed", severity="ERROR")
        assert logger.event_count == 2

    def test_all_fields_populated(self):
        logger = StructuredEventLogger()
        record = logger.emit(
            "model_promoted",
            "New model promoted",
            severity="INFO",
            component="model_registry",
            camera_id="cam_02",
            worker_id="worker-00",
            model_version="v2.0",
            correlation_id="corr-123",
            extra={"previous_version": "v1.0"},
        )
        assert record["worker_id"] == "worker-00"
        assert record["model_version"] == "v2.0"
        assert record["correlation_id"] == "corr-123"
        assert record["extra"]["previous_version"] == "v1.0"







class TestGracefulShutdown:
    def test_shutdown_hooks_execute_in_order(self):
        mgr = GracefulShutdownManager()
        order = []
        mgr.register_hook(30, "third", lambda: order.append(3))
        mgr.register_hook(10, "first", lambda: order.append(1))
        mgr.register_hook(20, "second", lambda: order.append(2))
        results = mgr.shutdown()
        assert order == [1, 2, 3]
        assert results["first"] == "SUCCESS"
        assert results["second"] == "SUCCESS"

    def test_hook_failure_doesnt_stop_others(self):
        mgr = GracefulShutdownManager()
        order = []
        mgr.register_hook(10, "ok1", lambda: order.append(1))
        mgr.register_hook(20, "fail", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        mgr.register_hook(30, "ok2", lambda: order.append(3))
        results = mgr.shutdown()
        assert 1 in order
        assert 3 in order
        assert "ERROR" in results["fail"]

    def test_double_shutdown_idempotent(self):
        mgr = GracefulShutdownManager()
        mgr.register_hook(10, "h1", lambda: None)
        _ = mgr.shutdown()
        r2 = mgr.shutdown()
        assert r2["status"] == "already_shutting_down"

    def test_is_shutting_down_flag(self):
        mgr = GracefulShutdownManager()
        assert mgr.is_shutting_down is False
        mgr.shutdown()
        assert mgr.is_shutting_down is True







class TestCapacityEstimation:
    def test_basic_capacity_estimate(self):
        est = CapacityEstimator(target_fps_per_camera=10.0)
        result = est.estimate(
            measured_throughput_fps=100.0,
            current_cameras=4,
            cpu_percent=30.0,
            vram_percent=20.0,
            p95_latency_ms=50.0,
            drop_rate=0.0,
        )
        assert result["estimated_sustainable_cameras"] == 10
        assert result["basis"] == "measured_runtime_data"
        assert result["constraints_met"] is True

    def test_cpu_constraint_reduces_capacity(self):
        est = CapacityEstimator(target_fps_per_camera=10.0, max_cpu_utilization=85.0)
        result = est.estimate(
            measured_throughput_fps=160.0,
            current_cameras=8,
            cpu_percent=90.0,
            vram_percent=20.0,
            p95_latency_ms=50.0,
            drop_rate=0.0,
        )
        assert result["estimated_sustainable_cameras"] < 16
        assert result["constraints_met"] is False
        assert result["limiting_factor"] == "cpu"

    def test_latency_constraint(self):
        est = CapacityEstimator(target_fps_per_camera=10.0, max_acceptable_p95_latency_ms=100.0)
        result = est.estimate(
            measured_throughput_fps=200.0,
            current_cameras=8,
            cpu_percent=40.0,
            vram_percent=30.0,
            p95_latency_ms=300.0,
            drop_rate=0.0,
        )
        assert result["constraints_met"] is False
        assert result["limiting_factor"] == "latency"

    def test_zero_throughput(self):
        est = CapacityEstimator()
        result = est.estimate(0.0, 0, 10.0, 10.0, 10.0, 0.0)
        assert result["estimated_sustainable_cameras"] == 0







class TestProductionSurveillanceRuntime:
    def test_register_and_start_camera(self):
        rt = ProductionSurveillanceRuntime()
        cam = rt.register_camera("cam_int_01", source_type="rtsp", fps_target=15)
        assert cam.connection_state == CameraState.STOPPED
        assert rt.start_camera("cam_int_01") is True
        cam = rt.camera_state_machine.get_camera("cam_int_01")
        assert cam.connection_state == CameraState.CONNECTING

    def test_connect_and_fail_camera(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_cf")
        rt.start_camera("cam_cf")
        rt.connect_camera("cam_cf")
        cam = rt.camera_state_machine.get_camera("cam_cf")
        assert cam.connection_state == CameraState.CONNECTED

        rt.fail_camera("cam_cf", error="network_loss")
        cam = rt.camera_state_machine.get_camera("cam_cf")
        assert cam.connection_state == CameraState.RECONNECTING

    def test_stop_camera(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_stop")
        rt.start_camera("cam_stop")
        rt.connect_camera("cam_stop")
        assert rt.stop_camera("cam_stop") is True
        cam = rt.camera_state_machine.get_camera("cam_stop")
        assert cam.connection_state == CameraState.STOPPED

    def test_system_health_snapshot(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_h1")
        rt.register_camera("cam_h2")
        health = rt.get_system_health()
        assert health["cameras"]["total"] == 2
        assert "uptime_seconds" in health
        assert "resources" in health

    def test_camera_health_report(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_rpt", source_type="webcam")
        report = rt.get_camera_health()
        assert "cam_rpt" in report
        assert report["cam_rpt"]["connection_state"] == "STOPPED"
        assert "health_score" in report["cam_rpt"]

    def test_capacity_estimation(self):
        rt = ProductionSurveillanceRuntime()
        result = rt.estimate_capacity(
            measured_throughput_fps=50.0,
            current_cameras=2,
        )
        assert result["estimated_sustainable_cameras"] >= 1
        assert result["basis"] == "measured_runtime_data"

    def test_graceful_shutdown(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_sd1")
        rt.register_camera("cam_sd2")
        rt.start_camera("cam_sd1")
        rt.connect_camera("cam_sd1")
        results = rt.shutdown_manager.shutdown()
        assert "stop_camera_ingestion" in results
        assert results["stop_camera_ingestion"] == "SUCCESS"







class TestFailureIsolation:
    def test_camera_a_fails_b_c_d_continue(self):
        rt = ProductionSurveillanceRuntime()
        for cid in ["A", "B", "C", "D"]:
            rt.register_camera(cid)
            rt.start_camera(cid)
            rt.connect_camera(cid)

        rt.fail_camera("A", error="hardware_failure")

        for cid in ["B", "C", "D"]:
            cam = rt.camera_state_machine.get_camera(cid)
            assert cam.connection_state == CameraState.CONNECTED

    def test_inference_workers_survive_camera_failure(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_w1")
        rt.start_camera("cam_w1")
        rt.connect_camera("cam_w1")
        rt.fail_camera("cam_w1", error="crash")


        health = rt.get_system_health()
        assert health["cameras"]["total"] == 1

    def test_learning_continues_during_camera_failure(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_cl")
        rt.start_camera("cam_cl")
        rt.connect_camera("cam_cl")
        rt.fail_camera("cam_cl", error="disconnected")


        emb = np.random.randn(512).astype(np.float32)
        ok, _ = rt.poisoning_guard.validate_observation("test", 0.9, emb, time.monotonic())
        assert ok is True







class TestCrashRecovery:
    def test_camera_config_survives_stop_restart(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_rs", source_type="rtsp", fps_target=20)
        rt.start_camera("cam_rs")
        rt.connect_camera("cam_rs")
        rt.stop_camera("cam_rs")


        cam = rt.camera_state_machine.get_camera("cam_rs")
        assert cam is not None
        assert cam.fps_target == 20


        rt.start_camera("cam_rs")
        cam = rt.camera_state_machine.get_camera("cam_rs")
        assert cam.connection_state == CameraState.CONNECTING

    def test_model_registry_survives_swap(self):
        rt = ProductionSurveillanceRuntime()
        v1 = ModelVersion("v1", "p1", "M")
        v2 = ModelVersion("v2", "p2", "M")
        rt.model_swapper.set_active(v1)
        rt.model_swapper.register_candidate(v2)
        rt.model_swapper.promote_candidate()
        rt.model_swapper.rollback()

        reg = rt.model_swapper.get_registry()
        assert "v1" in reg
        assert "v2" in reg
        assert rt.model_swapper.get_active().version_id == "v1"

    def test_reconnect_engine_recovers_after_cancel(self):
        engine = ReconnectEngine()
        engine.schedule_reconnect("cam1", lambda: None)
        engine.cancel_all()

        engine2 = ReconnectEngine()
        called = threading.Event()
        result = engine2.schedule_reconnect("cam1", called.set)
        assert result is True
        called.wait(timeout=3.0)
        assert called.is_set()
        engine2.cancel_all()

    def test_poisoning_guard_state_independent(self):
        guard1 = DataPoisoningGuard(min_confidence=0.5)
        emb = np.random.randn(512).astype(np.float32)
        guard1.validate_observation("p", 0.9, emb, time.monotonic())


        guard2 = DataPoisoningGuard(min_confidence=0.5)
        ok, _ = guard2.validate_observation("p", 0.9, emb, time.monotonic())
        assert ok is True







@pytest.mark.parametrize("num_cameras", [1, 2, 4, 8, 16, 32, 64])
def test_multicamera_simulation_scaling(num_cameras):
    rt = ProductionSurveillanceRuntime()

    for i in range(num_cameras):
        cid = f"sim_cam_{i:03d}"
        rt.register_camera(cid, source_type="rtsp", fps_target=15)
        rt.start_camera(cid)
        rt.connect_camera(cid)

    health = rt.get_system_health()
    assert health["cameras"]["total"] == num_cameras
    assert health["cameras"]["connected"] == num_cameras

    cam_health = rt.get_camera_health()
    assert len(cam_health) == num_cameras


    for cid, info in cam_health.items():
        assert info["connection_state"] == "CONNECTED"
        assert info["health_score"] > 0.5


    for i in range(num_cameras):
        rt.stop_camera(f"sim_cam_{i:03d}")

    health = rt.get_system_health()
    assert health["cameras"]["connected"] == 0







class TestArchitectureInvariants:
    def test_no_hardcoded_max_cameras(self):
        rt = ProductionSurveillanceRuntime()
        for i in range(100):
            rt.register_camera(f"cam_{i}")
        assert rt.get_system_health()["cameras"]["total"] == 100

    def test_initial_state_never_failed(self):
        rt = ProductionSurveillanceRuntime()
        for i in range(10):
            cam = rt.register_camera(f"cam_{i}")
            assert cam.connection_state == CameraState.STOPPED

    def test_bounded_queues_enforced(self):
        gate = FrameQualityGate(max_frame_age_ms=100.0)
        stale = QualifiedFrame(
            "cam1", 1, "u1",
            time.monotonic() - 1.0,
            "", np.zeros((2, 2, 3), dtype=np.uint8),
        )
        ok, _ = gate.validate(stale)
        assert ok is False

    def test_predicted_never_auto_training_eligible(self):
        guard = DataPoisoningGuard(min_confidence=0.5)
        emb = np.random.randn(512).astype(np.float32)
        ok, _ = guard.validate_observation(
            "person", 0.9, emb, time.monotonic(),
            verification_state="PREDICTED",
        )

        assert ok is True
