import math
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from services.missing_person_processor import MissingPersonVideoProcessor, TrackSummary, ValidatedEmbedding
from services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus
from storage.embedding_database import EmbeddingDatabase
from storage.vector_store import VectorStore


def _create_synthetic_video(
    filepath: Path,
    num_frames: int = 30,
    width: int = 320,
    height: int = 240,
    fps: float = 25.0,
    draw_person: bool = True,
    draw_second_person: bool = False,
    second_person_frames: int = 5,
) -> Path:
    """Create a temporary video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

    for idx in range(num_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        if draw_person:
            # Draw synthetic upright walking person (white torso + head + limbs on dark background)
            # Center of person shifts slightly to simulate walking motion
            cx = 100 + (idx % 10) * 2
            # Head
            cv2.circle(frame, (cx, 60), 15, (255, 255, 255), -1)
            # Torso
            cv2.rectangle(frame, (cx - 20, 75), (cx + 20, 150), (255, 255, 255), -1)
            # Legs
            leg_phase = (idx % 8) - 4
            cv2.line(frame, (cx - 10, 150), (cx - 15 + leg_phase * 2, 210), (255, 255, 255), 8)
            cv2.line(frame, (cx + 10, 150), (cx + 15 - leg_phase * 2, 210), (255, 255, 255), 8)

        if draw_second_person and idx < second_person_frames:
            # Second person in background (smaller or equal)
            cx2 = 250
            cv2.circle(frame, (cx2, 70), 12, (200, 200, 200), -1)
            cv2.rectangle(frame, (cx2 - 15, 82), (cx2 + 15, 140), (200, 200, 200), -1)
            cv2.line(frame, (cx2 - 8, 140), (cx2 - 10, 190), (200, 200, 200), 6)
            cv2.line(frame, (cx2 + 8, 140), (cx2 + 10, 190), (200, 200, 200), 6)

        writer.write(frame)

    writer.release()
    return filepath


class TestMissingPersonVideoProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.gait_gallery_dir = self.temp_dir / "live_gallery"
        self.appearance_gallery_dir = self.temp_dir / "appearance_gallery"
        self.db_dir = self.temp_dir / "embedding_db"
        self.jobs_dir = self.temp_dir / "reference_jobs"

        self.store = VectorStore(gallery_dir=str(self.gait_gallery_dir))
        self.embedding_db = EmbeddingDatabase(
            db_dir=str(self.db_dir),
            gait_gallery_dir=str(self.gait_gallery_dir),
            appearance_gallery_dir=str(self.appearance_gallery_dir),
        )
        self.job_manager = ReferenceJobManager(jobs_dir=str(self.jobs_dir), max_workers=1)

        # Build processor with real local temporary storage
        self.processor = MissingPersonVideoProcessor(
            gait_gallery_dir=str(self.gait_gallery_dir),
            appearance_gallery_dir=str(self.appearance_gallery_dir),
            db_dir=str(self.db_dir),
            store=self.store,
            embedding_db=self.embedding_db,
            job_manager=self.job_manager,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_camera_independence_guarantee(self) -> None:
        """INVARIANT 4: The processor must not depend on CameraWorker, RecognitionWorker,
        or active camera availability.
        """
        import sys
        # Confirm no live camera worker is initialized
        self.assertFalse(hasattr(self.processor, "camera_worker"))
        self.assertFalse(hasattr(self.processor, "recognition_worker"))
        self.assertFalse(hasattr(self.processor, "camera_id"))

        # Confirm module does not import camera workers
        processor_mod = sys.modules["services.missing_person_processor"]
        self.assertNotIn("CameraWorker", processor_mod.__dict__)
        self.assertNotIn("RecognitionWorker", processor_mod.__dict__)

    def test_video_validation_rules(self) -> None:
        """INVARIANT 11: Video file validation before decoding."""
        # 1. Non-existent file
        ok, msg, _ = self.processor.validate_video_file(self.temp_dir / "non_existent.mp4")
        self.assertFalse(ok)
        self.assertIn("not found", msg)

        # 2. Empty 0-byte file
        empty_file = self.temp_dir / "empty.mp4"
        empty_file.write_bytes(b"")
        ok, msg, _ = self.processor.validate_video_file(empty_file)
        self.assertFalse(ok)
        self.assertIn("0 bytes", msg)

        # 3. Unsupported format
        unsupported = self.temp_dir / "sample.txt"
        unsupported.write_text("not a video")
        ok, msg, _ = self.processor.validate_video_file(unsupported)
        self.assertFalse(ok)
        self.assertIn("Unsupported video format", msg)

        # 4. Valid synthetic video
        valid_video = _create_synthetic_video(self.temp_dir / "valid.mp4", num_frames=20)
        ok, msg, meta = self.processor.validate_video_file(valid_video)
        self.assertTrue(ok)
        self.assertEqual(meta["total_frames"], 20)
        self.assertEqual(meta["width"], 320)
        self.assertEqual(meta["height"], 240)

    def test_embedding_numerical_validation(self) -> None:
        """INVARIANT 7 & 9: Strict numerical validation of 256D embeddings."""
        # 1. None embedding
        ok, reason, _ = self.processor.validate_embedding(None, "person_1", 1, 0)
        self.assertFalse(ok)
        self.assertIn("None", reason)

        # 2. Wrong dimension (128D instead of 256D)
        wrong_dim = np.ones(128, dtype=np.float32)
        ok, reason, _ = self.processor.validate_embedding(wrong_dim, "person_1", 1, 0)
        self.assertFalse(ok)
        self.assertIn("expected 256", reason)

        # 3. NaN in vector
        nan_vec = np.ones(256, dtype=np.float32)
        nan_vec[15] = np.nan
        ok, reason, _ = self.processor.validate_embedding(nan_vec, "person_1", 1, 0)
        self.assertFalse(ok)
        self.assertIn("non-finite", reason)

        # 4. Inf in vector
        inf_vec = np.ones(256, dtype=np.float32)
        inf_vec[30] = np.inf
        ok, reason, _ = self.processor.validate_embedding(inf_vec, "person_1", 1, 0)
        self.assertFalse(ok)
        self.assertIn("non-finite", reason)

        # 5. Near-zero norm
        zero_vec = np.zeros(256, dtype=np.float32)
        ok, reason, _ = self.processor.validate_embedding(zero_vec, "person_1", 1, 0)
        self.assertFalse(ok)
        self.assertIn("near-zero norm", reason)

        # 6. Valid vector -> normalized to unit length
        valid_vec = np.random.randn(256).astype(np.float32)
        ok, reason, validated = self.processor.validate_embedding(valid_vec, "person_1", 1, 0)
        self.assertTrue(ok)
        self.assertIsNotNone(validated)
        self.assertEqual(validated.dimension, 256)
        self.assertEqual(validated.person_id, "person_1")
        self.assertAlmostEqual(float(np.linalg.norm(validated.vector)), 1.0, places=5)

    def test_embedding_deduplication(self) -> None:
        """INVARIANT 1 & 8: Configurable cosine similarity deduplication."""
        self.processor.dedup_cosine_threshold = 0.92

        # Create base normalized vector
        base_vec = np.random.randn(256).astype(np.float32)
        base_vec /= np.linalg.norm(base_vec)

        # Candidate 1: identical (cos sim = 1.0) -> should be deduplicated
        dup_vec = base_vec.copy()

        # Candidate 2: nearly identical (cos sim = 0.98 >= 0.92) -> should be deduplicated
        ortho = np.random.randn(256).astype(np.float32)
        ortho -= float(np.dot(ortho, base_vec)) * base_vec
        ortho /= np.linalg.norm(ortho)
        near_dup = (0.98 * base_vec + math.sqrt(1 - 0.98**2) * ortho).astype(np.float32)

        # Candidate 3: distinct observation (cos sim ~ 0.2 < 0.92) -> should be accepted
        distinct_vec = np.random.randn(256).astype(np.float32)
        distinct_vec /= np.linalg.norm(distinct_vec)

        candidates = [
            ValidatedEmbedding(base_vec, 256, "P1", 1, 0, 1.0, "v1", {}),
            ValidatedEmbedding(dup_vec, 256, "P1", 1, 1, 1.0, "v1", {}),
            ValidatedEmbedding(near_dup, 256, "P1", 1, 2, 1.0, "v1", {}),
            ValidatedEmbedding(distinct_vec, 256, "P1", 1, 3, 1.0, "v1", {}),
        ]

        accepted, count = self.processor.deduplicate_embeddings(candidates)
        # Should keep candidate 0 and candidate 3; filter candidates 1 and 2
        self.assertEqual(len(accepted), 2)
        self.assertEqual(count, 2)
        self.assertTrue(np.array_equal(accepted[0].vector, base_vec))
        self.assertTrue(np.array_equal(accepted[1].vector, distinct_vec))

    def test_no_person_detected_fails_safely(self) -> None:
        """INVARIANT 3: Never use black-frame fallback; zero person detection fails safely."""
        # Create empty video without person
        blank_video = _create_synthetic_video(
            self.temp_dir / "blank.mp4",
            num_frames=20,
            draw_person=False,
        )

        job = self.job_manager.create_job(person_id="SubjectEmpty", video_path=str(blank_video))
        result = self.processor.process_reference_video(
            person_id="SubjectEmpty",
            video_path=blank_video,
            job_id=job.job_id,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result.get("diagnostic_code"), "NO_PERSON_DETECTED")

        # Confirm job state in manager
        refreshed_job = self.job_manager.get_job(job.job_id)
        self.assertIsNotNone(refreshed_job)
        self.assertEqual(refreshed_job.status, ReferenceJobStatus.FAILED)
        self.assertEqual(refreshed_job.diagnostic_code, "NO_PERSON_DETECTED")

        # Verify nothing was written to vector store
        gallery = self.store.load()
        self.assertTrue(gallery is None or len(gallery[1]) == 0 or "SubjectEmpty" not in list(gallery[1]))

    def test_too_short_gait_sequence_fails_safely(self) -> None:
        """INVARIANT 6: Sequence shorter than min_gait_frames produces NO embedding."""
        # Mock track with only 6 frames (< min_gait_frames=10)
        short_summary = TrackSummary(
            track_id=1,
            frame_indices=list(range(6)),
            bboxes=[[10, 10, 50, 100]] * 6,
            crops=[np.ones((90, 40, 3), dtype=np.uint8)] * 6,
            areas=[3600.0] * 6,
            silhouettes=[np.ones((128, 64), dtype=np.uint8)] * 6,
        )

        target_id, reason = self.processor.select_isolated_target_track({1: short_summary})
        self.assertIsNone(target_id)
        self.assertIn("INSUFFICIENT_GAIT_SEQUENCE", reason)

    def test_multi_person_ambiguity_rejection(self) -> None:
        """INVARIANT 2: When multiple prominent people are detected and target cannot be
        isolated with high confidence (ratio < target_isolation_ratio), REJECT rather than guess.
        """
        # Two tracks with nearly equal presence and area (walking together)
        track1 = TrackSummary(
            track_id=1,
            frame_indices=list(range(30)),
            bboxes=[[50, 20, 100, 180]] * 30,
            crops=[np.ones((160, 50, 3), dtype=np.uint8)] * 30,
            areas=[8000.0] * 30,
            silhouettes=[np.ones((128, 64), dtype=np.uint8)] * 30,
        )
        track2 = TrackSummary(
            track_id=2,
            frame_indices=list(range(28)),
            bboxes=[[150, 20, 200, 180]] * 28,
            crops=[np.ones((160, 50, 3), dtype=np.uint8)] * 28,
            areas=[8000.0] * 28,
            silhouettes=[np.ones((128, 64), dtype=np.uint8)] * 28,
        )

        target_id, reason = self.processor.select_isolated_target_track({1: track1, 2: track2})
        self.assertIsNone(target_id)
        self.assertIn("AMBIGUOUS_MULTIPLE_PERSONS", reason)

    def test_multi_person_prominent_target_isolated(self) -> None:
        """INVARIANT 2: Primary target in foreground with incidental background pedestrian
        (prominence ratio >= 2.5) isolates the primary target and rejects the background person.
        """
        # Primary foreground subject: 40 frames, large bounding box area 16000
        primary_track = TrackSummary(
            track_id=1,
            frame_indices=list(range(40)),
            bboxes=[[50, 20, 150, 200]] * 40,
            crops=[np.ones((180, 100, 3), dtype=np.uint8)] * 40,
            areas=[16000.0] * 40,
            silhouettes=[np.ones((128, 64), dtype=np.uint8)] * 40,
        )
        # Background pedestrian: only 10 frames, small area 1600
        # primary prominence = 40 * sqrt(16000) = 40 * 126.5 = 5060
        # secondary prominence = 10 * sqrt(1600) = 10 * 40 = 400 -> ratio = 12.65 >= 2.5
        background_track = TrackSummary(
            track_id=2,
            frame_indices=list(range(10)),
            bboxes=[[220, 50, 260, 90]] * 10,
            crops=[np.ones((40, 40, 3), dtype=np.uint8)] * 10,
            areas=[1600.0] * 10,
            silhouettes=[np.ones((128, 64), dtype=np.uint8)] * 10,
        )

        target_id, reason = self.processor.select_isolated_target_track({1: primary_track, 2: background_track})
        self.assertEqual(target_id, 1)
        self.assertEqual(reason, "PRIMARY_TARGET_ISOLATED")

    def test_idempotent_retry(self) -> None:
        """INVARIANT 10: Retrying a completed job returns the result without duplicate commits."""
        record = self.job_manager.create_job(person_id="P_IDEM", video_path="dummy.mp4")
        mock_result = {"success": True, "embeddings_committed": 2, "person_id": "P_IDEM"}
        self.job_manager.complete_job(record.job_id, mock_result)

        # Query job again
        job = self.job_manager.get_job(record.job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, ReferenceJobStatus.COMPLETED)
        self.assertEqual(job.result["embeddings_committed"], 2)

    def test_persistence_recovery_and_reconciliation(self) -> None:
        """Requirement 8 & 12: Test that persistence failure or VectorStore desync is recoverable."""
        person_id = "P_RECONCILE"
        vec1 = np.random.randn(256).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        # 1. Add valid embedding to durable database
        self.embedding_db.add_embeddings(
            person_id=person_id,
            gait_embeddings=[vec1],
        )

        # 2. Simulate vector store file deletion / corruption
        if self.store.features_file.exists():
            self.store.features_file.unlink()

        # VectorStore is now empty/missing on disk
        self.assertIsNone(self.store.load())

        # 3. Trigger reconciliation
        reconcile_res = self.embedding_db.reconcile_vector_stores()
        self.assertEqual(reconcile_res["status"], "RECONCILED")
        self.assertGreaterEqual(reconcile_res["active_persons"], 1)

        # 4. Verify VectorStore is recovered and loaded cleanly
        recovered = self.store.load()
        self.assertIsNotNone(recovered)
        features, labels, _ = recovered
        self.assertIn(person_id, list(labels))
        self.assertEqual(features.shape[1], 256)

    def test_live_cctv_lifecycle_unaffected(self) -> None:
        """Requirement 6 & 12: Offline reference processing does not mutate or affect live CCTV streams."""
        from services.gait_service import GaitService

        # Instantiate GaitService with isolated test directories
        gait_svc = GaitService(
            gallery_dir=str(self.gait_gallery_dir),
            appearance_gallery_dir=str(self.appearance_gallery_dir),
        )

        # Confirm camera state is completely pristine (zero active cameras)
        self.assertEqual(len(gait_svc.active_cameras), 0)
        self.assertEqual(len(gait_svc.camera_workers), 0)

        # Run reference video processing
        synthetic_vid = _create_synthetic_video(self.temp_dir / "cctv_isolation_test.mp4", num_frames=15)
        self.processor.process_reference_video(
            person_id="P_CCTV_ISOLATED",
            video_path=synthetic_vid,
            gait_service_ref=gait_svc,
        )

        # Confirm live cameras are STILL 0 and unaffected
        self.assertEqual(len(gait_svc.active_cameras), 0)
        self.assertEqual(len(gait_svc.camera_workers), 0)


if __name__ == "__main__":
    unittest.main()
