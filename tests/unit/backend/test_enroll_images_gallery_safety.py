"""Regression tests for GaitService.enroll_images' persistence safety.

Audit finding (Phase 2 of the enrollment-hardening pass): enroll_images had
exactly one call site (the API layer, already gated to one concurrent
inference call by app.core.inference_gate), so it can't currently race
against a second concurrent enroll_images call. But it computed its new
gallery state from self.gallery_features/self.gallery_labels - an in-memory
cache that can be stale relative to disk - and then called
self.store.save(...) directly, bypassing VectorStore's locked
read-modify-write cycle. A concurrent writer using the same on-disk gallery
through VectorStore.update() (e.g. a background reference-job enrollment)
could have its write silently discarded, because enroll_images would
recompute its "new" state from a snapshot that predates that write and then
overwrite the file with it.

A second, independent bug found in the same audit: enroll_images never
updated self.metadata/self.appearance_metadata, so every enrollment through
this path left the metadata file's per-person embedding counts permanently
out of sync with the real labels array.

Fix: enroll_images now persists via VectorStore.update() (the same locked
mutator pattern used elsewhere in the codebase), which loads fresh from
disk, applies the mutation, and saves - all under one held inter-process
lock - then refreshes the in-memory cache via the existing, tested
reload_gallery(). This test proves the fix directly: a "concurrent" writer
using store.update() between enroll_images' embedding extraction and its
own store.update() call must not be lost, and metadata must be updated.

Uses a lightweight fake GaitService (real VectorStore/tempdir, mocked
detector/extractor - no real model loading) so it stays fast and safe to
run under low memory headroom.
"""

import shutil
import tempfile
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from app.services.gait_service import GaitService
from app.storage.vector_store import VectorStore


def _make_fake_service(gait_dir: Path, appearance_dir: Path):
    fake = SimpleNamespace()
    fake._lock = threading.RLock()
    fake.camera_workers = {}
    fake._readiness = {}
    fake.logger = MagicMock()
    fake.embedding_db = None
    fake.reload_gallery = MethodType(GaitService.reload_gallery, fake)

    fake.store = VectorStore(gallery_dir=str(gait_dir), gallery_type="baseline_gait")
    fake.appearance_store = VectorStore(gallery_dir=str(appearance_dir), gallery_type="appearance")

    fake.gallery_features = np.empty((0, 256), dtype=np.float32)
    fake.gallery_labels = []
    fake.metadata = {}
    fake.appearance_gallery_features = np.empty((0, 512), dtype=np.float32)
    fake.appearance_gallery_labels = []
    fake.appearance_metadata = {}

    fake.silhouette_extractor = MagicMock()
    fake.silhouette_extractor.extract_from_crop.return_value = np.zeros((128, 64), dtype=np.uint8)

    fake.extractor = MagicMock()
    fake.extractor.backend.predict.side_effect = lambda sil: (
        np.random.RandomState(abs(hash(sil.tobytes())) % (2**31)).randn(256).astype(np.float32)
    )

    fake.appearance_extractor = None  # keep this test focused on the gait path

    return fake


def _dummy_image_bytes():
    import cv2

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_enroll_images_updates_metadata():
    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_enroll_metadata_"))
    try:
        fake = _make_fake_service(tmp_dir / "gait", tmp_dir / "appearance")
        result = GaitService.enroll_images(fake, "TEST_PERSON_A", [_dummy_image_bytes()])

        assert result["success"] is True
        assert "TEST_PERSON_A" in fake.metadata
        assert fake.metadata["TEST_PERSON_A"]["embeddings"] == 1
        assert fake.metadata["TEST_PERSON_A"]["status"] == "ACTIVE"

        # A second enrollment for the same person must accumulate, not reset.
        GaitService.enroll_images(fake, "TEST_PERSON_A", [_dummy_image_bytes()])
        assert fake.metadata["TEST_PERSON_A"]["embeddings"] == 2
        assert len(fake.gallery_labels) == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_enroll_images_survives_concurrent_external_writer():
    """The core lost-update proof: a write made via VectorStore.update()
    strictly between enroll_images' feature extraction and its own
    store.update() call must survive enroll_images' persistence."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_enroll_race_"))
    try:
        gait_dir = tmp_dir / "gait"
        fake = _make_fake_service(gait_dir, tmp_dir / "appearance")

        # Simulate a concurrent background writer (e.g. a reference-job
        # enrollment) committing a different person to the SAME gallery
        # file, using a SEPARATE VectorStore instance pointed at the same
        # directory - exactly what a different process/thread would do.
        concurrent_store = VectorStore(gallery_dir=str(gait_dir), gallery_type="baseline_gait")

        def concurrent_mutator(current):
            if current is None:
                features, labels, metadata = [], [], {}
            else:
                cur_f, cur_l, cur_m = current
                features = cur_f.tolist() if hasattr(cur_f, "tolist") else list(cur_f)
                labels = list(cur_l)
                metadata = dict(cur_m)
            features.append(np.random.RandomState(999).randn(256).astype(np.float32).tolist())
            labels.append("CONCURRENT_PERSON")
            metadata["CONCURRENT_PERSON"] = {"embeddings": 1, "status": "ACTIVE", "enabled": True, "updated_at": 0.0}
            return features, labels, metadata

        # Patch enroll_images' own gait_mutator construction point: run the
        # concurrent write right before enroll_images calls store.update(),
        # by wrapping VectorStore.update on fake.store to inject it once.
        original_update = fake.store.update
        state = {"done": False}

        def update_with_interleaved_write(mutator):
            if not state["done"]:
                state["done"] = True
                concurrent_store.update(concurrent_mutator)
            return original_update(mutator)

        fake.store.update = update_with_interleaved_write

        result = GaitService.enroll_images(fake, "TEST_PERSON_B", [_dummy_image_bytes()])
        assert result["success"] is True

        features, labels, metadata = VectorStore(gallery_dir=str(gait_dir), gallery_type="baseline_gait").load()
        labels_list = list(labels)

        assert "CONCURRENT_PERSON" in labels_list, (
            "Lost update: the concurrent writer's entry was overwritten by enroll_images' save - "
            "this is exactly the race the VectorStore.update() refactor exists to close."
        )
        assert "TEST_PERSON_B" in labels_list
        assert len(features) == len(labels_list) == 2
        assert "CONCURRENT_PERSON" in metadata
        assert "TEST_PERSON_B" in metadata
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_enroll_images_features_labels_stay_aligned():
    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_enroll_align_"))
    try:
        fake = _make_fake_service(tmp_dir / "gait", tmp_dir / "appearance")
        GaitService.enroll_images(fake, "PERSON_X", [_dummy_image_bytes(), _dummy_image_bytes()])
        GaitService.enroll_images(fake, "PERSON_Y", [_dummy_image_bytes()])

        features, labels, metadata = VectorStore(gallery_dir=str(tmp_dir / "gait"), gallery_type="baseline_gait").load()
        assert len(features) == len(labels) == 3
        assert sum(1 for lbl in labels if lbl == "PERSON_X") == 2
        assert sum(1 for lbl in labels if lbl == "PERSON_Y") == 1
        assert metadata["PERSON_X"]["embeddings"] == 2
        assert metadata["PERSON_Y"]["embeddings"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
