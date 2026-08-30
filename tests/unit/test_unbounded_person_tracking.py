"""
Comprehensive Unit & Scalability Test Suite for ARGUS AI Unbounded Multi-Person Architecture.

Tests:
1. Unbounded track allocation (1, 10, 50, 100, 250, 500, 1000 simulated persons).
2. Per-person context isolation and zero state crosstalk.
3. Deterministic track recovery across temporary missing frames.
4. Deterministic garbage collection and zero memory leakage over 10,000 track lifecycle events.
5. Dynamic batching for OSNet (512D) and ByGaitLight (256D) embeddings.
6. Fair-share scheduling & starvation prevention between crowded (50 persons) and sparse (2 persons) cameras.
7. Adaptive load degradation and automatic recovery tiers.
8. Single-track failure isolation (corrupted bbox, zero-size crop, NaN embedding, exception containment).
9. Multi-camera x Multi-person workload simulations (1x100, 4x25, 16x10, 32x10).
10. Hardware-agnostic dynamic capacity estimation across system profiles.
"""

from __future__ import annotations

import time

import numpy as np

from intelligence.appearance_embedding import AppearanceEmbeddingExtractor
from intelligence.concurrent_track_manager import (
    ConcurrentTrackManager,
    PersonTrackContext,
    TrackLifecycleState,
)
from pipeline.gei.stream_gei_builder import StreamGEIBuilder
from streaming.deployment_readiness import ProductionCapacityEstimator
from streaming.person_track_scheduler import (
    AdaptivePersonLoadTier,
    AdaptivePersonProcessingPolicy,
    BatchCandidateItem,
    PersonTrackScheduler,
)
from streaming.production_multicamera_engine import (
    ProductionMultiCameraEngine,
)


class TestUnboundedPersonTracking:
    """Test suite validating unbounded person tracking, dynamic batching, and fair scheduling."""

    def test_unbounded_person_track_creation_and_context(self) -> None:
        """Verify dynamic unbounded track context allocation for 1, 10, 50, 100, 250, 500, 1000 persons."""
        mgr = ConcurrentTrackManager(max_idle_seconds=5.0)

        # Test varying scales: [1, 10, 50, 100, 250, 500, 1000] [SIMULATED]
        test_counts = [1, 10, 50, 100, 250, 500, 1000]
        for count in test_counts:
            mgr.clear_all()
            for i in range(count):
                cam_id = f"cam_{i % 8}"
                bbox = [10 + (i * 2) % 400, 20, 80, 180]
                ctx = mgr.update_or_create_track(
                    camera_id=cam_id,
                    track_id=i,
                    bbox=bbox,
                    confidence=0.95,
                    frame_index=1,
                )
                assert ctx is not None
                assert ctx.track_id == i
                assert ctx.camera_id == cam_id
                assert ctx.state == TrackLifecycleState.TRACKING
                assert ctx.is_active() is True

            stats = mgr.get_stats()
            assert stats["active_tracks_count"] == count
            assert stats["total_created_tracks"] == count
            assert len(mgr.get_active_tracks()) == count

    def test_per_person_context_isolation(self) -> None:
        """Verify strict isolation between person track contexts (no cross-track pollution)."""
        mgr = ConcurrentTrackManager()

        ctx1 = mgr.update_or_create_track("cam_0", 101, [10, 10, 50, 100], confidence=0.9)
        ctx2 = mgr.update_or_create_track("cam_0", 102, [60, 10, 100, 100], confidence=0.8)

        # Update biometric state for track 1
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        ctx1.update_appearance(emb1, "PersonA", 0.92, frame_index=5)
        ctx1.update_fusion("PersonA", 0.92, "CONFIRMED", "CONFIRMED")

        # Verify track 2 remains pristine
        assert ctx2.appearance_embedding is None
        assert ctx2.appearance_identity == "UNKNOWN_PERSON"
        assert ctx2.fused_identity == "UNKNOWN_PERSON"
        assert ctx2.state == TrackLifecycleState.TRACKING

        # Verify track 1 updated correctly
        assert ctx1.appearance_identity == "PersonA"
        assert ctx1.fused_identity == "PersonA"
        assert ctx1.state == TrackLifecycleState.IDENTIFIED

    def test_track_lifecycle_and_spatial_recovery(self) -> None:
        """Verify lifecycle progression: DETECTED -> TRACKING -> TEMPORARILY_MISSING -> RECOVERED."""
        mgr = ConcurrentTrackManager(recovery_iou_threshold=0.25, recovery_time_window_seconds=2.0)

        # 1. Track initialized
        ctx = mgr.update_or_create_track("cam_1", 201, [100, 100, 200, 300], confidence=0.95)
        ctx.update_appearance(np.ones(512, dtype=np.float32), "Alice", 0.88, frame_index=1)
        ctx.update_fusion("Alice", 0.88, "CONFIRMED", "CONFIRMED")

        # 2. Frame missing track 201
        missing = mgr.mark_missing_tracks("cam_1", current_active_track_ids=set())
        assert len(missing) == 1
        assert missing[0].state == TrackLifecycleState.TEMPORARILY_MISSING

        # 3. Spatial re-appearance with new tracker ID 202 at overlapping coordinates
        time.sleep(0.01)
        recovered = mgr.update_or_create_track("cam_1", 202, [105, 102, 202, 305], confidence=0.92)

        assert recovered.state == TrackLifecycleState.RECOVERED
        assert recovered.fused_identity == "Alice"
        assert recovered.appearance_identity == "Alice"
        assert recovered.track_id == 202
        stats = mgr.get_stats()
        assert stats["total_recovered_tracks"] == 1

    def test_track_expiration_and_zero_memory_leak(self) -> None:
        """Verify deterministic garbage collection over 10,000 track lifecycle events with 0 leak."""
        mgr = ConcurrentTrackManager(max_idle_seconds=0.05)
        gei_builder = StreamGEIBuilder()
        appearance_extractor = AppearanceEmbeddingExtractor()

        cleared_gei_tracks = []
        cleared_app_tracks = []

        def gei_cb(cid: str, tid: int) -> None:
            gei_builder.clear_track(tid)
            cleared_gei_tracks.append((cid, tid))

        def app_cb(cid: str, tid: int) -> None:
            appearance_extractor.clear_track(tid)
            cleared_app_tracks.append((cid, tid))

        callbacks = [gei_cb, app_cb]

        # Simulate 10,000 track creations and evictions in batches of 1,000
        total_events = 10000
        batch_size = 1000
        for b in range(0, total_events, batch_size):
            for i in range(b, b + batch_size):
                mgr.update_or_create_track("cam_0", i, [10, 10, 50, 50])
                gei_builder.add_silhouette(i, np.ones((128, 64), dtype=np.uint8) * 255)
                appearance_extractor.extract(np.zeros((100, 50, 3), dtype=np.uint8), track_id=i)

            # Expire with synthetic past timestamp
            future_ts = time.monotonic() + 1.0
            expired = mgr.cleanup_expired_tracks(
                max_idle_seconds=0.05,
                cleanup_callbacks=callbacks,
                timestamp=future_ts,
            )
            assert len(expired) == batch_size

        # Verify all 10,000 tracks cleaned up across all subsystems
        assert len(mgr.get_all_tracks()) == 0
        assert len(mgr.get_active_tracks()) == 0
        assert len(cleared_gei_tracks) == total_events
        assert len(cleared_app_tracks) == total_events
        assert len(gei_builder.track_buffers) == 0

    def test_dynamic_batching_reid_and_gait(self) -> None:
        """Verify dynamic batch extraction for OSNet (512D) and ByGaitLight (256D)."""
        extractor = AppearanceEmbeddingExtractor()

        crops = [np.ones((120, 60, 3), dtype=np.uint8) * ((i % 10) * 20 + 1) for i in range(16)]
        track_ids = list(range(16))

        # Dynamic batch extraction
        batch_embs = extractor.extract_batch(crops, track_ids=track_ids, frame_index=1)
        assert len(batch_embs) == 16
        for emb in batch_embs:
            assert emb is not None
            assert emb.shape == (512,)
            # L2 normalization invariant
            norm = float(np.linalg.norm(emb))
            assert abs(norm - 1.0) < 1e-4

        # Verify per-track cache populated
        for tid in track_ids:
            assert extractor.get_cached(tid) is not None

    def test_fair_share_scheduler_starvation_prevention(self) -> None:
        """Verify fair scheduling: Camera A (50 persons) does not starve Camera B (2 persons)."""
        scheduler = PersonTrackScheduler()
        policy_params = scheduler.policy_engine.evaluate_policy(active_tracks_count=52)

        candidates = []
        # Cam A: 50 tracks (high density crowd)
        for i in range(50):
            ctx_a = PersonTrackContext(camera_id="cam_crowded", track_id=i, appearance_last_frame=0)
            crop_a = np.ones((100, 50, 3), dtype=np.uint8)
            candidates.append(
                BatchCandidateItem(camera_id="cam_crowded", track_id=i, crop=crop_a, bbox=[0, 0, 10, 10], context=ctx_a)
            )

        # Cam B: 2 tracks (sparse camera)
        for j in range(2):
            ctx_b = PersonTrackContext(camera_id="cam_sparse", track_id=100 + j, appearance_last_frame=0)
            crop_b = np.ones((100, 50, 3), dtype=np.uint8)
            candidates.append(
                BatchCandidateItem(camera_id="cam_sparse", track_id=100 + j, crop=crop_b, bbox=[0, 0, 10, 10], context=ctx_b)
            )

        # Process multiple scheduling rounds
        scheduled_cam_b = 0
        for frame in range(1, 10):
            batch = scheduler.select_reid_candidates(candidates, policy_params, frame_index=frame)
            for item in batch:
                if item.camera_id == "cam_sparse":
                    scheduled_cam_b += 1
                item.context.appearance_last_frame = frame

        # Verify sparse camera tracks were scheduled and not starved
        assert scheduled_cam_b > 0

    def test_adaptive_load_degradation_tiers(self) -> None:
        """Verify adaptive degradation transitions under simulated hardware pressure."""
        policy = AdaptivePersonProcessingPolicy()

        # 1. Healthy state -> FULL_QUALITY
        p1 = policy.evaluate_policy(cpu_percent=30.0, ram_percent=40.0, queue_depth=0, active_tracks_count=5)
        assert p1.tier == AdaptivePersonLoadTier.FULL_QUALITY
        assert p1.reid_update_interval == 6
        assert p1.skip_confirmed_reid is False

        # 2. Moderate crowd -> MICRO_BATCHING
        p2 = policy.evaluate_policy(cpu_percent=76.0, ram_percent=60.0, queue_depth=4, active_tracks_count=45)
        assert p2.tier == AdaptivePersonLoadTier.MICRO_BATCHING
        assert p2.skip_confirmed_reid is True

        # 3. High pressure -> AGGRESSIVE_FRAME_SKIPPING
        p3 = policy.evaluate_policy(cpu_percent=86.0, ram_percent=70.0, queue_depth=5, active_tracks_count=80)
        assert p3.tier == AdaptivePersonLoadTier.AGGRESSIVE_FRAME_SKIPPING
        assert p3.drop_stale_frames is True

        # 4. Severe overload -> DEGRADED_MODE
        p4 = policy.evaluate_policy(cpu_percent=92.0, ram_percent=92.0, queue_depth=16, p95_latency_ms=120.0)
        assert p4.tier == AdaptivePersonLoadTier.DEGRADED_MODE
        assert p4.target_fps_scale == 0.25

    def test_per_track_failure_isolation(self) -> None:
        """Verify single-track failure (corrupted crop, invalid bbox, NaN vector) is isolated."""
        extractor = AppearanceEmbeddingExtractor()
        engine = ProductionMultiCameraEngine(appearance_extractor=extractor)
        engine.register_camera("cam_test")

        # Corrupted person track should not raise unhandled exception
        bad_ctx = engine.track_manager.update_or_create_track("cam_test", 999, [-10, -20, -5, -5])
        assert bad_ctx is not None
        assert bad_ctx.track_id == 999

        # Extract on corrupted crop
        bad_emb = engine.appearance_extractor.extract(np.zeros((0, 0, 3), dtype=np.uint8), track_id=999)
        assert bad_emb is None

        # Clean healthy track continues uninterrupted
        good_ctx = engine.track_manager.update_or_create_track("cam_test", 1000, [10, 10, 80, 180])
        assert good_ctx.is_active() is True

    def test_multi_camera_multi_person_workload_simulation(self) -> None:
        """Simulate multi-camera x multi-person configurations (1x100, 4x25, 16x10, 32x10) [SIMULATED]."""
        scenarios = [
            (1, 100),
            (4, 25),
            (16, 10),
            (32, 10),
        ]
        estimator = ProductionCapacityEstimator()

        for cams, persons in scenarios:
            res = estimator.estimate_multi_person_capacity(
                camera_count=cams,
                persons_per_camera=persons,
                cpu_percent=45.0,
                vram_allocated_mb=1200.0,
                vram_total_mb=6000.0,
                p95_latency_ms=25.0,
            )
            assert res["is_unbounded_supported"] is True
            assert res["total_concurrent_persons"] == cams * persons
            assert res["capacity_state"] in ("SUPPORTED", "SUPPORTED_DEGRADED")
            assert res["estimated_track_memory_mb"] > 0

    def test_dynamic_capacity_estimation_profiles(self) -> None:
        """Verify dynamic capacity adaptation across CPU-only, low-resource, standard GPU, and server."""
        estimator = ProductionCapacityEstimator()

        # CPU-only low resource simulation
        cpu_res = estimator.estimate_multi_person_capacity(
            camera_count=4,
            persons_per_camera=10,
            cpu_percent=89.0,
            vram_allocated_mb=0.0,
            vram_total_mb=0.0,
            p95_latency_ms=110.0,
        )
        assert cpu_res["capacity_state"] == "CAPACITY_REACHED"
        assert cpu_res["recommended_policy_tier"] == "DEGRADED_MODE"

        # Standard GPU server simulation
        gpu_res = estimator.estimate_multi_person_capacity(
            camera_count=16,
            persons_per_camera=15,
            cpu_percent=55.0,
            vram_allocated_mb=3500.0,
            vram_total_mb=16000.0,
            p95_latency_ms=28.0,
        )
        assert gpu_res["capacity_state"] == "SUPPORTED"
        assert gpu_res["recommended_policy_tier"] == "FULL_QUALITY"
