"""
Unit tests for Stage 5 Track Recovery Manager.
"""

from intelligence.track_recovery_manager import TrackRecoveryManager


def test_lost_track_registration_and_recovery():
    mgr = TrackRecoveryManager(max_lost_seconds=3.0, min_recovery_iou=0.30)

    # Register lost track on cam_01 at t=10.0
    mgr.register_lost_track(
        camera_id="cam_01",
        track_id=42,
        last_bbox=[100, 100, 200, 300],
        identity="Subject_X",
        timestamp=10.0,
    )

    # Attempt recovery with new track 99 at t=11.5 with high spatial IoU overlap
    recovered = mgr.attempt_recovery(
        camera_id="cam_01",
        new_track_id=99,
        new_bbox=[105, 100, 205, 300],  # Heavy overlap
        timestamp=11.5,
    )

    assert recovered is not None
    assert recovered.track_id == 42
    assert recovered.identity == "Subject_X"

    # Verify track was removed from lost buffer after recovery
    assert ("cam_01", 42) not in mgr.lost_tracks


def test_expired_lost_track_not_recovered():
    mgr = TrackRecoveryManager(max_lost_seconds=2.0)
    mgr.register_lost_track("cam_01", 10, [100, 100, 200, 300], "Subject_Y", timestamp=10.0)

    # Attempt recovery at t=15.0 (> 2s max_lost)
    recovered = mgr.attempt_recovery("cam_01", 88, [100, 100, 200, 300], timestamp=15.0)
    assert recovered is None


def test_simultaneous_people_non_merging():
    mgr = TrackRecoveryManager(max_lost_seconds=3.0, min_recovery_iou=0.30)
    mgr.register_lost_track("cam_01", 1, [10, 10, 50, 100], "Person_1", timestamp=10.0)

    # New track 2 in completely different location [500, 500, 550, 600]
    recovered = mgr.attempt_recovery("cam_01", 2, [500, 500, 550, 600], timestamp=11.0)
    assert recovered is None


def test_max_buffer_size_eviction():
    mgr = TrackRecoveryManager(max_buffered_tracks=3)
    for i in range(5):
        mgr.register_lost_track("cam_01", i, [10 * i, 10, 50 * i + 10, 100], timestamp=float(i))

    assert len(mgr.lost_tracks) <= 3
