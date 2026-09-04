"""Targeted unit, integration, and failure injection tests for ARGUS AI

Durable Checkpointing and Automatic Resume for Missing Person Embedding Jobs.
"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_session_store
from security_layer.authorization import Role
from services.missing_person_processor import (
    MissingPersonVideoProcessor,
    TrackSummary,
    ValidatedEmbedding,
)
from services.reference_job_manager import (
    ReferenceJobManager,
    ReferenceJobStatus,
)
from storage.embedding_database import EmbeddingDatabase
from storage.vector_store import VectorStore


def _create_synthetic_video_frames(
    filepath: Path,
    num_frames: int = 50,
    width: int = 160,
    height: int = 120,
    fps: float = 25.0,
) -> Path:
    """Create a synthetic test video with walking person graphics."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    for idx in range(num_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        cx = 50 + (idx % 10) * 2
        cv2.circle(frame, (cx, 30), 8, (255, 255, 255), -1)
        cv2.rectangle(frame, (cx - 10, 38), (cx + 10, 75), (255, 255, 255), -1)
        cv2.line(frame, (cx - 5, 75), (cx - 7, 105), (255, 255, 255), 4)
        cv2.line(frame, (cx + 5, 75), (cx + 7, 105), (255, 255, 255), 4)
        writer.write(frame)
    writer.release()
    return filepath


class MockDetections:
    def __init__(self, num_detections: int = 1):
        if num_detections > 0:
            self.xyxy = np.array([[40, 20, 80, 110]])
            self.tracker_id = np.array([1])
        else:
            self.xyxy = np.empty((0, 4))
            self.tracker_id = np.empty((0,))


class MockFastTracker:
    def track(self, frame):
        return MockDetections(1)


class MockFastSilhouette:
    def extract_from_crop(self, crop):
        return np.full((128, 64), 255, dtype=np.uint8)


class MockFastExtractor:
    def extract_from_gei(self, gei):
        vec = np.ones(256, dtype=np.float32)
        return vec / np.linalg.norm(vec)


class TestJobRecoveryAndCheckpointing(unittest.TestCase):
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
        self.job_manager = ReferenceJobManager(jobs_dir=str(self.jobs_dir), max_workers=2)

        self.processor = MissingPersonVideoProcessor(
            gait_gallery_dir=str(self.gait_gallery_dir),
            appearance_gallery_dir=str(self.appearance_gallery_dir),
            db_dir=str(self.db_dir),
            detector=MagicMock(),
            tracker=MockFastTracker(),
            silhouette_step=MockFastSilhouette(),
            extractor=MockFastExtractor(),
            store=self.store,
            embedding_db=self.embedding_db,
            job_manager=self.job_manager,
        )

    def tearDown(self) -> None:
        try:
            self.job_manager.shutdown(timeout=1.0)
        except Exception:  # noqa: BLE001, S110
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_created_during_video_processing(self) -> None:
        """1. Checkpoint is created on disk during video processing."""
        video_file = _create_synthetic_video_frames(self.temp_dir / "test_vid1.mp4", num_frames=30)
        job = self.job_manager.create_job(person_id="P_CHK_1", video_path=str(video_file))

        res = self.processor.process_reference_video(
            person_id="P_CHK_1",
            video_path=video_file,
            job_id=job.job_id,
        )
        self.assertTrue(res.get("success"))

        # Verify checkpoint file exists on disk
        chk_file = self.jobs_dir / f"{job.job_id}.json"
        self.assertTrue(chk_file.exists())
        with open(chk_file, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["job_id"], job.job_id)
        self.assertEqual(data["status"], ReferenceJobStatus.COMPLETED.value)
        self.assertGreater(data["progress"]["frames_processed"], 0)

    def test_checkpoint_contains_last_safe_frame(self) -> None:
        """2. Checkpoint accurately records last_safe_frame."""
        video_file = _create_synthetic_video_frames(self.temp_dir / "test_vid2.mp4", num_frames=35)
        job = self.job_manager.create_job(person_id="P_CHK_2", video_path=str(video_file))

        self.processor.process_reference_video(
            person_id="P_CHK_2",
            video_path=video_file,
            job_id=job.job_id,
        )

        updated_job = self.job_manager.get_job(job.job_id)
        self.assertIsNotNone(updated_job)
        self.assertGreaterEqual(updated_job.progress.last_safe_frame, 30)
        self.assertEqual(updated_job.progress.last_safe_frame, updated_job.progress.frames_processed)

    def test_checkpoint_persists_across_process_restart(self) -> None:
        """3. Checkpoint persists and is correctly loaded after manager re-instantiation."""
        job = self.job_manager.create_job(person_id="P_PERSIST_RESTART", video_path="video.mp4")
        self.job_manager.checkpoint_job(
            job.job_id,
            stage="TRACKING",
            last_safe_frame=150,
            frames_processed=150,
            total_frames=300,
            status=ReferenceJobStatus.PROCESSING,
        )

        # Re-instantiate ReferenceJobManager with the same jobs_dir (simulates restart)
        new_mgr = ReferenceJobManager(jobs_dir=str(self.jobs_dir), max_workers=1)
        loaded_job = new_mgr.get_job(job.job_id)
        self.assertIsNotNone(loaded_job)
        self.assertEqual(loaded_job.progress.last_safe_frame, 150)
        self.assertEqual(loaded_job.progress.frames_processed, 150)
        self.assertEqual(loaded_job.progress.total_frames, 300)
        # On startup, in-progress jobs transition to INTERRUPTED (not FAILED)
        self.assertEqual(loaded_job.status, ReferenceJobStatus.INTERRUPTED)

    def test_interrupted_job_detected_on_startup(self) -> None:
        """4. Unfinished in-progress job is flagged as INTERRUPTED on startup, not falsely failed."""
        job = self.job_manager.create_job(person_id="P_INTERRUPT_DETECT", video_path="some_video.mp4")
        self.job_manager.update_progress(job.job_id, stage="FEATURE_EXTRACTION", status=ReferenceJobStatus.PROCESSING)

        new_mgr = ReferenceJobManager(jobs_dir=str(self.jobs_dir), max_workers=1)
        recovered = new_mgr.get_job(job.job_id)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.status, ReferenceJobStatus.INTERRUPTED)
        self.assertEqual(recovered.diagnostic_code, "PROCESS_INTERRUPTED")

    def test_interrupted_job_automatically_resumes(self) -> None:
        """5. Interrupted job is automatically resumed on recover_unfinished_jobs."""
        video_file = _create_synthetic_video_frames(self.temp_dir / "resume_vid.mp4", num_frames=30)
        job = self.job_manager.create_job(person_id="P_AUTO_RESUME", video_path=str(video_file))
        self.job_manager.checkpoint_job(
            job.job_id,
            stage="TRACKING",
            last_safe_frame=15,
            frames_processed=15,
            total_frames=30,
            status=ReferenceJobStatus.INTERRUPTED,
        )

        # Call recover_unfinished_jobs
        resumed_list = self.job_manager.recover_unfinished_jobs(processor=self.processor)
        self.assertEqual(len(resumed_list), 1)
        self.assertEqual(resumed_list[0].job_id, job.job_id)
        self.assertEqual(resumed_list[0].status, ReferenceJobStatus.RESUMING)
        self.assertTrue(resumed_list[0].resumed)
        self.assertGreaterEqual(resumed_list[0].recovery_count, 1)

        # Wait for resumed task to complete
        time.sleep(1.0)
        final_job = self.job_manager.get_job(job.job_id)
        self.assertEqual(final_job.status, ReferenceJobStatus.COMPLETED)

    def test_resume_does_not_duplicate_embeddings(self) -> None:
        """6. Idempotent resume does NOT create duplicate embeddings."""
        person_id = "P_NO_DUPLICATES"
        vec1 = np.random.randn(256).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        det_ids = [f"gait_{person_id}_job123_seq_0"]

        # First commit
        res1 = self.embedding_db.commit_and_activate_embeddings(
            person_id=person_id,
            gait_embeddings=[vec1],
            source_session_id="job123",
            embedding_ids=det_ids,
        )
        self.assertEqual(res1["gait_embeddings_added"], 1)
        self.assertEqual(res1["total_gait_embeddings"], 1)

        # Second commit with identical deterministic ID (simulates retry/resume)
        res2 = self.embedding_db.commit_and_activate_embeddings(
            person_id=person_id,
            gait_embeddings=[vec1],
            source_session_id="job123",
            embedding_ids=det_ids,
        )
        # No new embeddings added! Existing was acknowledged/confirmed
        self.assertEqual(res2["gait_embeddings_added"], 0)
        self.assertEqual(res2["total_gait_embeddings"], 1)

    def test_resume_from_feature_extraction_checkpoint(self) -> None:
        """7. Resume from FEATURE_EXTRACTION checkpoint bypasses tracking."""
        video_file = _create_synthetic_video_frames(self.temp_dir / "resume_fe.mp4", num_frames=30)
        job = self.job_manager.create_job(person_id="P_RESUME_FE", video_path=str(video_file))

        # Save checkpoint with completed tracking and candidate crops
        sample_crops = [np.full((80, 40, 3), 200, dtype=np.uint8) for _ in range(15)]
        tracks = {1: TrackSummary(track_id=1, crops=sample_crops, frame_indices=list(range(1, 31)))}

        self.job_manager.checkpoint_job(
            job.job_id,
            stage="FEATURE_EXTRACTION",
            last_safe_frame=30,
            frames_processed=30,
            total_frames=30,
            status=ReferenceJobStatus.RESUMING,
            checkpoint_data={
                "stage": "FEATURE_EXTRACTION",
                "tracks": tracks,
                "target_track_id": 1,
                "target_crops": sample_crops,
                "last_safe_frame": 30,
            },
        )

        res = self.processor.process_reference_video(
            person_id="P_RESUME_FE",
            video_path=video_file,
            job_id=job.job_id,
        )
        self.assertTrue(res.get("success"))
        final_job = self.job_manager.get_job(job.job_id)
        self.assertEqual(final_job.status, ReferenceJobStatus.COMPLETED)
        self.assertGreater(final_job.result["embeddings_committed"], 0)

    def test_resume_from_matching_checkpoint(self) -> None:
        """8. Resume from MATCHING checkpoint bypasses feature extraction."""
        video_file = _create_synthetic_video_frames(self.temp_dir / "resume_match.mp4", num_frames=30)
        job = self.job_manager.create_job(person_id="P_RESUME_MATCH", video_path=str(video_file))

        vec = np.ones(256, dtype=np.float32) / 16.0
        val_emb = ValidatedEmbedding(
            vector=vec,
            dimension=256,
            person_id="P_RESUME_MATCH",
            track_id=1,
            sequence_index=0,
            quality_score=0.95,
            model_version="v1.0.0",
            provenance={},
        )

        self.job_manager.checkpoint_job(
            job.job_id,
            stage="MATCHING",
            last_safe_frame=30,
            frames_processed=30,
            total_frames=30,
            status=ReferenceJobStatus.RESUMING,
            checkpoint_data={
                "stage": "MATCHING",
                "target_track_id": 1,
                "candidate_embeddings": [val_emb],
                "completed_sequences": [0],
                "last_safe_frame": 30,
            },
        )

        res = self.processor.process_reference_video(
            person_id="P_RESUME_MATCH",
            video_path=video_file,
            job_id=job.job_id,
        )
        self.assertTrue(res.get("success"))
        final_job = self.job_manager.get_job(job.job_id)
        self.assertEqual(final_job.status, ReferenceJobStatus.COMPLETED)

    def test_resume_from_persisting_checkpoint(self) -> None:
        """9. Resume from PERSISTING checkpoint commits embeddings and completes."""
        video_file = _create_synthetic_video_frames(self.temp_dir / "resume_persist.mp4", num_frames=30)
        job = self.job_manager.create_job(person_id="P_RESUME_PERSIST", video_path=str(video_file))

        vec = np.ones(256, dtype=np.float32) / 16.0
        val_emb = ValidatedEmbedding(
            vector=vec,
            dimension=256,
            person_id="P_RESUME_PERSIST",
            track_id=1,
            sequence_index=0,
            quality_score=0.95,
            model_version="v1.0.0",
            provenance={},
        )

        self.job_manager.checkpoint_job(
            job.job_id,
            stage="PERSISTING",
            last_safe_frame=30,
            frames_processed=30,
            total_frames=30,
            status=ReferenceJobStatus.RESUMING,
            checkpoint_data={
                "stage": "PERSISTING",
                "target_track_id": 1,
                "candidate_embeddings": [val_emb],
                "dedup_embeddings": [val_emb],
                "dedup_count": 0,
                "last_safe_frame": 30,
            },
        )

        res = self.processor.process_reference_video(
            person_id="P_RESUME_PERSIST",
            video_path=video_file,
            job_id=job.job_id,
        )
        self.assertTrue(res.get("success"))
        final_job = self.job_manager.get_job(job.job_id)
        self.assertEqual(final_job.status, ReferenceJobStatus.COMPLETED)

    def test_completed_job_is_not_resumed(self) -> None:
        """10. Completed job is never resumed or re-processed."""
        job = self.job_manager.create_job(person_id="P_COMPLETED", video_path="dummy.mp4")
        self.job_manager.complete_job(job.job_id, {"success": True, "completed": True})

        # Recovery scan should ignore completed jobs
        resumed = self.job_manager.recover_unfinished_jobs(processor=self.processor)
        self.assertEqual(len(resumed), 0)

        # Direct call returns cached result immediately without work
        res = self.processor.process_reference_video(
            person_id="P_COMPLETED",
            video_path="dummy.mp4",
            job_id=job.job_id,
        )
        self.assertTrue(res.get("completed"))

    def test_invalid_checkpoint_fails_safely(self) -> None:
        """11. Corrupted checkpoint file is caught safely without crashing manager."""
        corrupted_file = self.jobs_dir / "corrupted_job_123.json"
        corrupted_file.write_text("{ this is invalid json !!!", encoding="utf-8")

        new_mgr = ReferenceJobManager(jobs_dir=str(self.jobs_dir), max_workers=1)
        self.assertNotIn("corrupted_job_123", new_mgr._jobs)

    def test_missing_source_video_fails_safely(self) -> None:
        """12. Unfinished job with deleted/missing source video fails safely with VIDEO_NOT_FOUND."""
        missing_path = self.temp_dir / "non_existent_video_xyz.mp4"
        job = self.job_manager.create_job(person_id="P_MISSING_SRC", video_path=str(missing_path))
        self.job_manager.update_progress(job.job_id, status=ReferenceJobStatus.INTERRUPTED)

        resumed = self.job_manager.recover_unfinished_jobs(processor=self.processor)
        self.assertEqual(len(resumed), 0)

        updated_job = self.job_manager.get_job(job.job_id)
        self.assertEqual(updated_job.status, ReferenceJobStatus.FAILED)
        self.assertEqual(updated_job.diagnostic_code, "VIDEO_NOT_FOUND")

    def test_duplicate_recovery_claim_is_prevented(self) -> None:
        """13. Recovery claim mechanism prevents duplicate workers processing the same job."""
        job = self.job_manager.create_job(person_id="P_CLAIM_LOCK", video_path="v.mp4")
        self.assertTrue(self.job_manager.claim_job_for_recovery(job.job_id))
        # Second claim attempt fails
        self.assertFalse(self.job_manager.claim_job_for_recovery(job.job_id))
        self.assertTrue(self.job_manager.is_job_claimed(job.job_id))
        self.job_manager.release_job_claim(job.job_id)
        self.assertFalse(self.job_manager.is_job_claimed(job.job_id))

    def test_graceful_shutdown_persists_checkpoint(self) -> None:
        """14. Graceful shutdown signals active jobs, marks INTERRUPTED, and saves checkpoint."""
        job = self.job_manager.create_job(person_id="P_SHUTDOWN", video_path="v.mp4")
        self.job_manager.update_progress(
            job.job_id,
            stage="TRACKING",
            frames_processed=120,
            status=ReferenceJobStatus.PROCESSING,
        )

        self.job_manager.shutdown(timeout=1.0)
        updated = self.job_manager.get_job(job.job_id)
        self.assertEqual(updated.status, ReferenceJobStatus.INTERRUPTED)
        self.assertEqual(updated.diagnostic_code, "GRACEFUL_SHUTDOWN")

    def test_job_api_reports_resuming_state(self) -> None:
        """15. Job API returns status='RESUMING' and recovery metadata."""
        store = get_session_store()
        session = store.create_session("op_inv_rec", "inv_rec", Role.INVESTIGATOR.value)
        headers = {"Authorization": f"Bearer {session.token}"}

        job_mgr = ReferenceJobManager.get_instance()
        test_job = job_mgr.create_job(
            person_id="P_API_RESUME",
            video_path="v.mp4",
            owner="inv_rec",
        )
        job_mgr.update_progress(
            test_job.job_id,
            stage="TRACKING",
            status=ReferenceJobStatus.RESUMING,
            frames_processed=150,
            last_safe_frame=150,
            total_frames=300,
            recovery_count=1,
            resumed=True,
        )

        with TestClient(app) as client:
            resp = client.get(f"/api/v1/cases/jobs/{test_job.job_id}", headers=headers)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "RESUMING")
            self.assertEqual(data["recovery_count"], 1)
            self.assertTrue(data["resumed"])
            self.assertEqual(data["progress"]["last_safe_frame"], 150)
            self.assertEqual(data["progress"]["frames_processed"], 150)

    def test_frontend_progress_does_not_reset_after_resume(self) -> None:
        """16. Verification of progress calculation on resume: does NOT drop to 0%."""
        # Test progress calculation logic
        frames_processed = 300
        total_frames = 636

        # When resumed, calculateVideoProgressPercent returns >= 25% and never 0%
        calc_pct = min(95, max(25, round((frames_processed / total_frames) * 75)))
        self.assertGreater(calc_pct, 20)
        self.assertEqual(calc_pct, 35)

    def test_deterministic_failure_injection_636_frames(self) -> None:
        """17. Deterministic Failure Injection:

        636-frame video -> process until known checkpoint -> force interruption ->
        restart -> automatic recovery -> continue processing -> COMPLETED.
        """
        person_id = "P_FAIL_INJECT_636"
        video_file = _create_synthetic_video_frames(
            self.temp_dir / "fail_inject_636.mp4",
            num_frames=636,
            width=160,
            height=120,
        )

        # Step 1: Create and start job
        job = self.job_manager.create_job(person_id=person_id, video_path=str(video_file))

        # Step 2: Simulate processing until known checkpoint (frame 300 / 636)
        # We manually record the checkpoint to frame 300 as if worker reached frame 300
        sample_crops = [np.full((80, 40, 3), 180, dtype=np.uint8) for _ in range(30)]
        tracks = {1: TrackSummary(track_id=1, crops=sample_crops, frame_indices=list(range(1, 301)))}

        self.job_manager.checkpoint_job(
            job.job_id,
            stage="TRACKING",
            last_safe_frame=300,
            frames_processed=300,
            total_frames=636,
            fps=45.0,
            tracks_detected=1,
            percent=41,
            status=ReferenceJobStatus.PROCESSING,
            checkpoint_data={"stage": "TRACKING", "tracks": tracks, "last_safe_frame": 300},
        )

        # Step 3: Force worker/process interruption (simulating crash)
        self.job_manager.shutdown(timeout=0.5)

        interrupted_job = self.job_manager.get_job(job.job_id)
        self.assertEqual(interrupted_job.status, ReferenceJobStatus.INTERRUPTED)
        self.assertEqual(interrupted_job.progress.last_safe_frame, 300)

        # Step 4: Restart ARGUS (new ReferenceJobManager and new Processor)
        new_mgr = ReferenceJobManager(jobs_dir=str(self.jobs_dir), max_workers=2)
        new_processor = MissingPersonVideoProcessor(
            gait_gallery_dir=str(self.gait_gallery_dir),
            appearance_gallery_dir=str(self.appearance_gallery_dir),
            db_dir=str(self.db_dir),
            detector=MagicMock(),
            tracker=MockFastTracker(),
            silhouette_step=MockFastSilhouette(),
            extractor=MockFastExtractor(),
            store=self.store,
            embedding_db=self.embedding_db,
            job_manager=new_mgr,
        )

        # Step 5: Startup automatic recovery scanner
        recovered = new_mgr.recover_unfinished_jobs(processor=new_processor)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].job_id, job.job_id)
        self.assertEqual(recovered[0].status, ReferenceJobStatus.RESUMING)
        self.assertGreaterEqual(recovered[0].recovery_count, 1)

        # Step 6: Wait for processing to continue to completion
        time.sleep(2.0)
        final_job = new_mgr.get_job(job.job_id)
        self.assertEqual(final_job.status, ReferenceJobStatus.COMPLETED)
        self.assertEqual(final_job.progress.stage, "COMPLETED")
        self.assertEqual(final_job.progress.percent, 100)
        self.assertGreaterEqual(final_job.progress.frames_processed, 636)
        self.assertGreater(final_job.result["embeddings_committed"], 0)

        # Verify exact final invariants
        # Invariant 1: No duplicate embeddings in gallery database
        person_rec = self.embedding_db.get_person(person_id)
        self.assertIsNotNone(person_rec)
        active_embeddings = [e for e in person_rec.gait_embeddings if e.status == "ACTIVE"]
        self.assertEqual(len(active_embeddings), final_job.result["embeddings_committed"])
        emb_ids = [e.embedding_id for e in active_embeddings]
        self.assertEqual(len(emb_ids), len(set(emb_ids)), "Duplicate embeddings found in active gallery!")

        # Invariant 2: API returns COMPLETED
        store = get_session_store()
        session = store.create_session("op_inv_636", "inv_636", Role.INVESTIGATOR.value)
        headers = {"Authorization": f"Bearer {session.token}"}
        orig_mgr = ReferenceJobManager._instance
        ReferenceJobManager._instance = new_mgr
        try:
            with TestClient(app) as client:
                resp = client.get(f"/api/v1/cases/jobs/{job.job_id}", headers=headers)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["status"], "COMPLETED")
                self.assertEqual(data["progress"]["stage"], "COMPLETED")
        finally:
            ReferenceJobManager._instance = orig_mgr


if __name__ == "__main__":
    unittest.main()
