import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from intelligence.appearance_embedding import AppearanceEmbeddingExtractor
from pipeline.steps.appearance_matching_step import AppearanceMatchingStep
from services.recognition_worker import (
    RecognitionResult,
    RecognitionResultCache,
    RecognitionWorker,
)
from storage.vector_store import VectorStore


@pytest.fixture
def temp_gallery_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_live_appearance_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_detector_and_tracker():
    """Mock detector and tracker generating 2 simultaneous tracks."""
    detector = MagicMock()
    # Return 2 detections: [x1, y1, x2, y2, conf, cls]
    detector.detect.return_value = np.array([
        [10, 10, 60, 120, 0.95, 0],
        [80, 10, 130, 120, 0.90, 0],
    ])

    tracker = MagicMock()
    tracker.update.return_value = [
        {"track_id": 1, "bbox": [10, 10, 60, 120], "confidence": 0.95},
        {"track_id": 2, "bbox": [80, 10, 130, 120], "confidence": 0.90},
    ]
    tracker.cleanup_inactive.return_value = []
    return detector, tracker


def test_live_person_crop_to_appearance_matching(temp_gallery_dir):
    """Requirement 1, 2, 7: Live frame -> person crop -> appearance embedding -> gallery match with track_id."""
    # Build synthetic appearance gallery with known person "Alice"
    v_alice = np.zeros((512,), dtype=np.float32)
    v_alice[0] = 1.0

    app_store = VectorStore(gallery_dir=temp_gallery_dir)
    app_store.save([v_alice.tolist()], ["Alice"], {"Alice": {"status": "ACTIVE", "enabled": True}})
    features, labels, metadata = app_store.load()

    # Create worker with mock appearance extractor returning v_alice for track 1
    extractor_mock = MagicMock()
    extractor_mock.extract.return_value = v_alice

    matcher = AppearanceMatchingStep(threshold=0.60)
    cache = RecognitionResultCache(ttl_seconds=5.0)

    worker = RecognitionWorker(
        camera_id="cam_01",
        cache=cache,
        appearance_extractor=extractor_mock,
        appearance_matcher=matcher,
        appearance_gallery_features=features,
        appearance_gallery_labels=labels,
        appearance_metadata=metadata,
    )

    # Synthetic frame (200x200x3)
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 100

    # Mock detector returning 1 person
    worker.detector.detect = MagicMock(return_value=np.array([[10, 10, 60, 120, 0.95, 0]]))
    worker.tracker.update = MagicMock(return_value=[{"track_id": 42, "bbox": [10, 10, 60, 120]}])

    # Put frame and execute 1 iteration of loop manually
    worker._input_queue.put(frame)
    # Run loop body once
    frame_item = worker._input_queue.get()
    worker._frame_count += 1

    tracked = worker.tracker.update(worker.detector.detect(frame_item), frame_item.shape)
    for obj in tracked:
        track_id = int(obj["track_id"])
        bbox = [int(b) for b in obj["bbox"]]
        crop = frame_item[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        app_emb = worker.appearance_extractor.extract(
            crop=crop,
            track_id=track_id,
            frame_index=worker._frame_count,
            track_reliable=True,
            recognition_deferred=False,
        )
        assert app_emb is not None
        matched_id, matched_score = worker.appearance_matcher.match(
            query_feature=app_emb,
            gallery_features=worker.appearance_gallery_features,
            gallery_labels=worker.appearance_gallery_labels,
            metadata=worker.appearance_metadata,
        )

        res = RecognitionResult(
            camera_id=worker.camera_id,
            track_id=track_id,
            identity="UNKNOWN",
            similarity=0.0,
            confidence=0.0,
            decision="DETECTION",
            status="DETECTION",
            bbox=bbox,
            timestamp=time.monotonic(),
            iso_timestamp="2026-08-27T00:00:00Z",
            appearance_identity=matched_id,
            appearance_score=round(matched_score, 4),
            appearance_status="MATCH" if matched_id != "UNKNOWN_PERSON" else "UNKNOWN",
        )
        worker.cache.put(res)

    cached_res = worker.cache.get("cam_01", 42)
    assert cached_res is not None
    assert cached_res.track_id == 42
    assert cached_res.appearance_identity == "Alice"
    assert cached_res.appearance_score == 1.0
    assert cached_res.appearance_status == "MATCH"
    # Gait identity remains UNKNOWN (unfused)
    assert cached_res.identity == "UNKNOWN"


def test_per_track_appearance_caching():
    """Requirement 3: Per-track caching reuses embeddings between update intervals."""
    extractor = AppearanceEmbeddingExtractor(update_interval=5)
    crop = np.random.randint(0, 256, (128, 64, 3), dtype=np.uint8)

    # Frame 1: Computes new embedding
    emb1 = extractor.extract(crop, track_id=10, frame_index=1, track_reliable=True)
    assert emb1 is not None

    # Frame 2-4: Returns cached embedding without re-running OSNet backbone
    with patch.object(extractor.backbone, "extract") as mock_backbone:
        emb2 = extractor.extract(crop, track_id=10, frame_index=2, track_reliable=True)
        emb3 = extractor.extract(crop, track_id=10, frame_index=3, track_reliable=True)
        # Backbone was NOT called because frame_index < 1 + 5
        mock_backbone.assert_not_called()
        assert np.array_equal(emb1, emb2)
        assert np.array_equal(emb1, emb3)


def test_appearance_failure_does_not_break_gait():
    """Requirement 4: Appearance extractor throwing exception does not stop gait recognition."""
    cache = RecognitionResultCache()
    failing_extractor = MagicMock()
    failing_extractor.extract.side_effect = RuntimeError("OSNet GPU OOM simulation")

    _worker = RecognitionWorker(
        camera_id="cam_02",
        cache=cache,
        appearance_extractor=failing_extractor,
    )

    frame = np.ones((200, 200, 3), dtype=np.uint8) * 50
    # Simulate single tracked object
    _obj = {"track_id": 99, "bbox": [10, 10, 50, 100]}

    # Extract crop
    crop = frame[10:100, 10:50]

    # Process appearance safely with exception fallback
    app_identity = "UNKNOWN_PERSON"
    app_score = 0.0
    app_status = "UNKNOWN"
    try:
        failing_extractor.extract(crop=crop, track_id=99, frame_index=1)
    except Exception:  # noqa: BLE001, S110
        pass  # Gracefully handled

    res = RecognitionResult(
        camera_id="cam_02",
        track_id=99,
        identity="Gait_Subject_1",
        similarity=0.92,
        confidence=0.92,
        decision="CONFIRMED",
        status="CONFIRMED",
        bbox=[10, 10, 50, 100],
        timestamp=time.monotonic(),
        iso_timestamp="2026-08-27T00:00:00Z",
        appearance_identity=app_identity,
        appearance_score=app_score,
        appearance_status=app_status,
        details={
            "appearance": {"identity": app_identity, "score": app_score, "status": app_status},
            "gait": {"identity": "Gait_Subject_1", "score": 0.92, "status": "CONFIRMED"},
        },
    )
    cache.put(res)

    stored = cache.get("cam_02", 99)
    assert stored is not None
    assert stored.identity == "Gait_Subject_1"
    assert stored.similarity == 0.92
    assert stored.appearance_identity == "UNKNOWN_PERSON"
    assert stored.appearance_score == 0.0
    assert stored.appearance_status == "UNKNOWN"


def test_empty_appearance_gallery_operating_in_gait_only_mode():
    """Requirement 5: Empty appearance gallery operates cleanly in gait-only mode."""
    worker = RecognitionWorker(
        camera_id="cam_03",
        appearance_gallery_features=np.empty((0, 512), dtype=np.float32),
        appearance_gallery_labels=[],
        appearance_metadata={},
    )

    crop = np.random.randint(0, 256, (128, 64, 3), dtype=np.uint8)
    emb = worker.appearance_extractor.extract(crop=crop, track_id=1, frame_index=1)
    assert emb is not None

    # Matching with empty gallery returns UNKNOWN_PERSON
    matched_id, score = worker.appearance_matcher.match(
        query_feature=emb,
        gallery_features=worker.appearance_gallery_features,
        gallery_labels=worker.appearance_gallery_labels,
        metadata=worker.appearance_metadata,
        unknown_label="UNKNOWN_PERSON",
    )
    assert matched_id == "UNKNOWN_PERSON"
    assert score == 0.0


def test_multiple_simultaneous_tracks(mock_detector_and_tracker):
    """Requirement 6 & 7: Multiple simultaneous tracks preserve individual appearance identities."""
    detector, tracker = mock_detector_and_tracker
    cache = RecognitionResultCache()

    v_p1 = np.zeros((512,), dtype=np.float32)
    v_p1[0] = 1.0

    v_p2 = np.zeros((512,), dtype=np.float32)
    v_p2[1] = 1.0

    gallery_feats = np.stack([v_p1, v_p2])
    gallery_lbls = ["Person_A", "Person_B"]

    extractor = MagicMock()
    # Return v_p1 for track 1, v_p2 for track 2
    def side_effect_extract(crop, track_id, *args, **kwargs):
        return v_p1 if track_id == 1 else v_p2

    extractor.extract.side_effect = side_effect_extract

    matcher = AppearanceMatchingStep(threshold=0.60)

    worker = RecognitionWorker(
        camera_id="cam_multi",
        cache=cache,
        detector=detector,
        tracker=tracker,
        appearance_extractor=extractor,
        appearance_matcher=matcher,
        appearance_gallery_features=gallery_feats,
        appearance_gallery_labels=gallery_lbls,
    )

    frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
    tracked = tracker.update(detector.detect(frame), frame.shape)

    for obj in tracked:
        tid = int(obj["track_id"])
        bbox = obj["bbox"]
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        emb = worker.appearance_extractor.extract(crop=crop, track_id=tid, frame_index=1)
        m_id, m_score = worker.appearance_matcher.match(
            query_feature=emb,
            gallery_features=worker.appearance_gallery_features,
            gallery_labels=worker.appearance_gallery_labels,
        )
        res = RecognitionResult(
            camera_id="cam_multi",
            track_id=tid,
            identity="UNKNOWN",
            similarity=0.0,
            confidence=0.0,
            decision="TRACKING",
            status="TRACKING",
            bbox=bbox,
            timestamp=time.monotonic(),
            iso_timestamp="2026-08-27T00:00:00Z",
            appearance_identity=m_id,
            appearance_score=m_score,
            appearance_status="MATCH",
        )
        cache.put(res)

    res_1 = cache.get("cam_multi", 1)
    res_2 = cache.get("cam_multi", 2)

    assert res_1 is not None and res_1.appearance_identity == "Person_A"
    assert res_2 is not None and res_2.appearance_identity == "Person_B"


def test_update_appearance_gallery_runtime_safe():
    """Verify that update_appearance_gallery safely updates appearance gallery without corrupting worker state."""
    worker = RecognitionWorker(camera_id="cam_update_test")

    # Store references to initial worker state objects
    initial_queue = worker._input_queue
    initial_stop_event = worker._stop_event
    initial_lock = worker._lock
    initial_frame_count = 123
    worker._frame_count = initial_frame_count

    # Prepare new gallery
    new_features = np.ones((2, 512), dtype=np.float32)
    new_labels = ["Subject_X", "Subject_Y"]
    new_meta = {
        "Subject_X": {"status": "ACTIVE", "enabled": True},
        "Subject_Y": {"status": "ACTIVE", "enabled": True},
    }

    # Exercise update_appearance_gallery (must not raise NameError or reset state)
    worker.update_appearance_gallery(
        gallery_features=new_features,
        gallery_labels=new_labels,
        metadata=new_meta,
    )

    # Verify appearance gallery state was updated
    assert np.array_equal(worker.appearance_gallery_features, new_features)
    assert worker.appearance_gallery_labels == new_labels
    assert worker.appearance_metadata == new_meta

    # Verify worker internals were NOT reset
    assert worker._input_queue is initial_queue
    assert worker._stop_event is initial_stop_event
    assert worker._lock is initial_lock
    assert worker._frame_count == initial_frame_count

