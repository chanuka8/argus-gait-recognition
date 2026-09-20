import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intelligence.operational_embedding_collector import (
    OperationalEmbeddingCollector,
)
from services.recognition_worker import RecognitionResultCache, RecognitionWorker


@pytest.fixture
def temp_obs_dir():
    temp_dir = tempfile.mkdtemp(prefix="argus_collector_durability_")
    obs_path = Path(temp_dir) / "operational_observations"
    obs_path.mkdir(parents=True, exist_ok=True)
    yield obs_path
    shutil.rmtree(temp_dir, ignore_errors=True)


def _make_unit_vector(dim: int = 256) -> list[float]:
    vec = np.random.randn(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


def test_a_six_observation_tail(temp_obs_dir):
    """A. SIX-OBSERVATION TAIL: Record 6 observations, flush, reload, confirm 6 persisted."""
    collector = OperationalEmbeddingCollector(
        output_dir=str(temp_obs_dir),
        dedup_window_seconds=0.01,
    )

    for i in range(6):
        collector.record_observation(
            camera_id="cam_test",
            track_id=i + 1,
            vector=_make_unit_vector(256),
            predicted_identity=f"Person_{i}",
            confidence=0.90,
            modality="gait",
        )

    assert len(collector.get_recent_observations()) == 6, "In-memory collector must have 6 observations"

    # Prior to explicit flush, because 6 > 5 and 6 % 10 != 0, disk only had 5 observations
    disk_file = temp_obs_dir / "recent_observations.json"
    assert disk_file.exists()
    with open(disk_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert len(disk_data) == 5, "Unflushed 6th observation was not yet persisted before flush"

    # Explicit flush
    success = collector.flush()
    assert success is True, "flush() must return True on success"

    # New collector reloaded from same directory
    new_collector = OperationalEmbeddingCollector(output_dir=str(temp_obs_dir))
    reloaded = new_collector.get_recent_observations()
    assert len(reloaded) == 6, f"Expected 6 reloaded observations, got {len(reloaded)}"

    # Confirm all IDs match
    orig_ids = [o.observation_id for o in collector.get_recent_observations()]
    reloaded_ids = [o.observation_id for o in reloaded]
    assert reloaded_ids == orig_ids, "Observation IDs must match exactly"


@pytest.mark.parametrize("tail_count", [7, 8, 9])
def test_b_seven_eight_nine_tails(temp_obs_dir, tail_count):
    """B. SEVEN / EIGHT / NINE TAILS: Non-batch-aligned tails survive explicit flush."""
    sub_dir = temp_obs_dir / f"tail_{tail_count}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    collector = OperationalEmbeddingCollector(
        output_dir=str(sub_dir),
        dedup_window_seconds=0.01,
    )

    for i in range(tail_count):
        collector.record_observation(
            camera_id="cam_tail",
            track_id=i + 1,
            vector=_make_unit_vector(256),
            predicted_identity=f"Person_{i}",
            confidence=0.88,
            modality="gait",
        )

    assert len(collector.get_recent_observations()) == tail_count

    # Before flush, on-disk file was flushed at 5
    disk_file = sub_dir / "recent_observations.json"
    with open(disk_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert len(disk_data) == 5

    # Flush tail
    assert collector.flush() is True

    # Reload from disk
    new_collector = OperationalEmbeddingCollector(output_dir=str(sub_dir))
    reloaded = new_collector.get_recent_observations()
    assert len(reloaded) == tail_count, f"All {tail_count} observations must survive reload"


@pytest.mark.parametrize("batch_count", [5, 10])
def test_c_batch_boundaries(temp_obs_dir, batch_count):
    """C. BATCH BOUNDARIES: 5 and 10 observations persist correctly."""
    sub_dir = temp_obs_dir / f"batch_{batch_count}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    collector = OperationalEmbeddingCollector(
        output_dir=str(sub_dir),
        dedup_window_seconds=0.01,
    )

    for i in range(batch_count):
        collector.record_observation(
            camera_id="cam_batch",
            track_id=i + 1,
            vector=_make_unit_vector(256),
            predicted_identity=f"Person_{i}",
            confidence=0.91,
            modality="gait",
        )

    assert len(collector.get_recent_observations()) == batch_count

    # Already persisted at boundary
    disk_file = sub_dir / "recent_observations.json"
    with open(disk_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert len(disk_data) == batch_count

    # Calling flush() explicitly maintains consistency
    assert collector.flush() is True

    new_collector = OperationalEmbeddingCollector(output_dir=str(sub_dir))
    reloaded = new_collector.get_recent_observations()
    assert len(reloaded) == batch_count


def test_d_idempotence(temp_obs_dir):
    """D. IDEMPOTENCE: Calling flush() repeatedly produces no duplicates or corruption."""
    collector = OperationalEmbeddingCollector(
        output_dir=str(temp_obs_dir),
        dedup_window_seconds=0.01,
    )

    for i in range(6):
        collector.record_observation(
            camera_id="cam_idemp",
            track_id=i + 1,
            vector=_make_unit_vector(256),
            predicted_identity=f"Person_{i}",
            confidence=0.89,
            modality="gait",
        )

    # Call flush multiple times consecutively
    for _ in range(5):
        assert collector.flush() is True

    # Validate JSON file
    disk_file = temp_obs_dir / "recent_observations.json"
    with open(disk_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 6

    ids = [d["observation_id"] for d in data]
    assert len(set(ids)) == 6, "No duplicate observation IDs allowed"

    new_collector = OperationalEmbeddingCollector(output_dir=str(temp_obs_dir))
    assert len(new_collector.get_recent_observations()) == 6


def test_e_empty_buffer(temp_obs_dir):
    """E. EMPTY BUFFER: flush() succeeds safely when buffer is empty."""
    collector = OperationalEmbeddingCollector(output_dir=str(temp_obs_dir))
    assert len(collector.get_recent_observations()) == 0

    assert collector.flush() is True

    disk_file = temp_obs_dir / "recent_observations.json"
    assert disk_file.exists()
    with open(disk_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == []

    new_collector = OperationalEmbeddingCollector(output_dir=str(temp_obs_dir))
    assert len(new_collector.get_recent_observations()) == 0


def test_f_worker_stop(temp_obs_dir):
    """F. WORKER STOP: Non-batch-aligned observations survive worker.stop() gracefully."""
    collector = OperationalEmbeddingCollector(
        output_dir=str(temp_obs_dir),
        dedup_window_seconds=0.001,
    )
    cache = RecognitionResultCache(ttl_seconds=2.0)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [{"bbox": [10, 10, 80, 160], "confidence": 0.95, "class_id": 0}]

    mock_sil = MagicMock()
    mock_sil.extract_from_frame.return_value = np.ones((128, 64), dtype=np.uint8) * 255

    mock_app_extractor = MagicMock()
    mock_app_extractor.extract.side_effect = lambda **kw: np.array(_make_unit_vector(512), dtype=np.float32)

    mock_app_matcher = MagicMock()
    mock_app_matcher.threshold = 0.60
    mock_app_matcher.match.return_value = ("Alice", 0.88)

    mock_gait_extractor = MagicMock()
    mock_gait_extractor.extract_from_gei.return_value = np.array(_make_unit_vector(256), dtype=np.float32)

    mock_gait_matcher = MagicMock()
    mock_gait_matcher.threshold = 0.85

    def make_tracks(dets, shape):
        count = len(collector.get_recent_observations())
        if count < 7:
            return [{"track_id": count + 1, "bbox": [10, 10, 80, 160], "confidence": 0.95}]
        return []

    mock_tracker = MagicMock()
    mock_tracker.update.side_effect = make_tracks

    from pipeline.gei.stream_gei_builder import StreamGEIBuilder

    gei_builder = StreamGEIBuilder()
    gei_builder.min_frames = 3

    worker = RecognitionWorker(
        camera_id="cam_worker_stop",
        config={"target_fps": 30.0, "cooldown_seconds": 0.001, "threshold": 0.80},
        cache=cache,
        detector=mock_detector,
        tracker=mock_tracker,
        silhouette_extractor=mock_sil,
        gei_builder=gei_builder,
        extractor=mock_gait_extractor,
        matcher=mock_gait_matcher,
        appearance_extractor=mock_app_extractor,
        appearance_matcher=mock_app_matcher,
        appearance_gallery_features=np.random.randn(1, 512).astype(np.float32),
        appearance_gallery_labels=["Alice"],
        gallery_features=np.random.randn(1, 256).astype(np.float32),
        gallery_labels=["Alice"],
        operational_collector=collector,
    )

    worker.start()

    # Feed frames until exactly 7 observations are generated in memory
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(100):
        worker.put_frame(frame)
        if len(collector.get_recent_observations()) >= 7:
            break
        time.sleep(0.03)

    assert len(collector.get_recent_observations()) == 7, "Must generate exactly 7 observations before stop"

    # Before stop: disk should have only 5 observations (due to batching <= 5 or % 10 == 0)
    disk_file = temp_obs_dir / "recent_observations.json"
    with open(disk_file, "r", encoding="utf-8") as f:
        data_before_stop = json.load(f)
    assert len(data_before_stop) == 5, "Pre-stop disk state must only contain 5 batch-persisted observations"

    # Stop worker gracefully
    stopped = worker.stop(timeout=2.0)
    assert stopped is True, "worker.stop() must return True on clean graceful shutdown"

    # Reload new collector from disk
    new_collector = OperationalEmbeddingCollector(output_dir=str(temp_obs_dir))
    reloaded = new_collector.get_recent_observations()
    assert len(reloaded) == 7, (
        f"All 7 accepted observations must survive worker.stop() without loss (got {len(reloaded)})"
    )


def test_g_thread_shutdown_ordering(temp_obs_dir):
    """G. THREAD SHUTDOWN ORDERING: Prove flush occurs only after the writer thread has terminated."""
    collector = OperationalEmbeddingCollector(output_dir=str(temp_obs_dir))

    worker = RecognitionWorker(
        camera_id="cam_order",
        operational_collector=collector,
    )

    flush_events = []
    original_flush = collector.flush

    def instrumented_flush():
        # Record whether worker thread is alive at the moment flush is called
        flush_events.append(
            {
                "is_alive_during_flush": worker.is_alive(),
                "worker_thread_ref": worker._thread,
            }
        )
        return original_flush()

    collector.flush = instrumented_flush

    worker.start()
    time.sleep(0.05)
    assert worker.is_alive() is True

    # Stop worker
    stopped = worker.stop(timeout=2.0)
    assert stopped is True
    assert worker.is_alive() is False

    assert len(flush_events) == 1, "flush must be called exactly once during graceful stop"
    assert flush_events[0]["is_alive_during_flush"] is False, (
        "flush() must be executed strictly after the worker thread has terminated"
    )
    assert flush_events[0]["worker_thread_ref"] is None, (
        "worker._thread must be verified terminated and cleared before flush"
    )


def test_h_join_timeout(temp_obs_dir):
    """H. JOIN TIMEOUT: Simulate a worker thread that does not terminate before timeout."""
    collector = OperationalEmbeddingCollector(output_dir=str(temp_obs_dir))

    worker = RecognitionWorker(
        camera_id="cam_timeout",
        operational_collector=collector,
    )

    hang_event = threading.Event()

    def stubborn_loop():
        # Ignores _stop_event until hang_event is explicitly set by the test cleanup
        hang_event.wait(timeout=5.0)

    # Start worker and replace thread target with stubborn_loop
    worker._thread = threading.Thread(target=stubborn_loop, daemon=True)
    worker._thread.start()

    flush_spy = MagicMock(wraps=collector.flush)
    collector.flush = flush_spy

    try:
        # Attempt to stop with very short timeout
        stopped = worker.stop(timeout=0.05)
        assert stopped is False, "worker.stop() must return False when worker thread join times out"
        assert worker.is_alive() is True, "Worker must remain alive if thread did not terminate"
        assert flush_spy.call_count == 0, "Collector flush must NOT be called while thread is still running"
    finally:
        hang_event.set()
        if worker._thread and worker._thread.is_alive():
            worker._thread.join(timeout=1.0)


def test_i_persistence_failure(temp_obs_dir):
    """I. PERSISTENCE FAILURE: Failure during write/replace leaves valid JSON intact and buffer preserved."""
    collector = OperationalEmbeddingCollector(
        output_dir=str(temp_obs_dir),
        dedup_window_seconds=0.01,
    )

    # Record 5 observations (flushed to disk)
    for i in range(5):
        collector.record_observation(
            camera_id="cam_fail",
            track_id=i + 1,
            vector=_make_unit_vector(256),
            predicted_identity=f"Person_{i}",
            confidence=0.95,
            modality="gait",
        )

    disk_file = temp_obs_dir / "recent_observations.json"
    assert disk_file.exists()
    with open(disk_file, "r", encoding="utf-8") as f:
        data_before = json.load(f)
    assert len(data_before) == 5

    # Record 6th observation in memory
    collector.record_observation(
        camera_id="cam_fail",
        track_id=6,
        vector=_make_unit_vector(256),
        predicted_identity="Person_5",
        confidence=0.95,
        modality="gait",
    )
    assert len(collector.get_recent_observations()) == 6

    # Simulate filesystem replace failure (e.g. disk full, permission denied, locking)
    with patch.object(Path, "replace", side_effect=OSError("Simulated filesystem I/O error")):
        success = collector.flush()
        assert success is False, "flush() must return False when disk write/replace fails"

    # Verify:
    # 1. In-memory observations remain intact
    assert len(collector.get_recent_observations()) == 6, "In-memory buffer must not be discarded on flush failure"

    # 2. Previous valid JSON on disk remains readable and uncorrupted
    assert disk_file.exists(), "Previous valid JSON file must still exist"
    with open(disk_file, "r", encoding="utf-8") as f:
        data_after = json.load(f)
    assert len(data_after) == 5, "Previous valid JSON file must remain uncorrupted"

    # 3. RecognitionWorker.stop() reflects persistence failure
    worker = RecognitionWorker(
        camera_id="cam_fail_worker",
        operational_collector=collector,
    )
    with patch.object(collector, "flush", return_value=False):
        worker_stopped = worker.stop()
        assert worker_stopped is False, "worker.stop() must return False if final persistence failed"


def test_j_concurrent_flush(temp_obs_dir):
    """J. CONCURRENT FLUSH: Concurrent record/flush access must not corrupt JSON or raise race errors."""
    collector = OperationalEmbeddingCollector(
        output_dir=str(temp_obs_dir),
        max_buffer_size=200,
        dedup_window_seconds=0.001,
    )

    errors = []

    def writer_worker(thread_id: int, count: int):
        try:
            for i in range(count):
                collector.record_observation(
                    camera_id=f"cam_{thread_id}",
                    track_id=i,
                    vector=_make_unit_vector(256),
                    predicted_identity=f"P_{thread_id}_{i}",
                    confidence=0.85,
                    modality="gait",
                )
                time.sleep(0.001)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            errors.append(exc)

    def flusher_worker(count: int):
        try:
            for _ in range(count):
                success = collector.flush()
                if not success:
                    errors.append(RuntimeError("Flush returned False during concurrency test"))
                time.sleep(0.002)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            errors.append(exc)

    threads = []
    # 2 writer threads
    for t_id in range(2):
        t = threading.Thread(target=writer_worker, args=(t_id, 20))
        threads.append(t)

    # 2 flusher threads
    for _ in range(2):
        t = threading.Thread(target=flusher_worker, args=(15,))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Concurrent operations encountered errors: {errors}"

    # Final explicit flush
    assert collector.flush() is True

    # Reload from disk and verify file validity
    disk_file = temp_obs_dir / "recent_observations.json"
    with open(disk_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == len(collector.get_recent_observations())
    assert len(data) == 40, f"Expected 40 observations, got {len(data)}"
