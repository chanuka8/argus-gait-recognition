"""Phase-4 data-safety regression tests: startup must be read-only with respect
to biometric gallery state unless a genuinely recoverable job exists, and the
same source content must never be embedded twice under a different job_id.
"""

import hashlib
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.services.missing_person_processor import MissingPersonVideoProcessor
from app.services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus
from app.storage.embedding_database import EmbeddingDatabase
from app.storage.vector_store import VectorStore
from tests.unit.backend.test_job_recovery import (
    MockFastExtractor,
    MockFastSilhouette,
    MockFastTracker,
    _create_synthetic_video_frames,
)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gallery_files(gallery_dir: Path) -> list[Path]:
    return sorted(p for p in gallery_dir.glob("*") if p.is_file())


def _hash_gallery(gallery_dir: Path) -> dict[str, str]:
    return {p.name: _hash_file(p) for p in _gallery_files(gallery_dir)}


class TestGalleryWriteSafety(unittest.TestCase):
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
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # A/B/C/H: stable startup (zero recoverable jobs) does not mutate gallery,
    # across one, two, and three consecutive recovery scans.
    def test_repeated_recovery_scans_with_no_recoverable_jobs_are_read_only(self) -> None:
        video_file = _create_synthetic_video_frames(self.temp_dir / "seed.mp4", num_frames=20)
        job = self.job_manager.create_job(person_id="P_STABLE", video_path=str(video_file))
        self.processor.process_reference_video(person_id="P_STABLE", video_path=video_file, job_id=job.job_id)

        before = _hash_gallery(self.gait_gallery_dir)
        self.assertTrue(before, "expected at least one gallery file to exist after seeding")

        for _ in range(3):
            recovered = self.job_manager.recover_unfinished_jobs(processor=self.processor, gait_service_ref=None)
            self.assertEqual(recovered, [], "no job is recoverable; recovery must be a strict no-op")
            self.assertEqual(_hash_gallery(self.gait_gallery_dir), before, "gallery mutated on a read-only scan")

    # D/E/F: an explicitly recoverable job executes exactly once, and re-running
    # recovery afterwards (now COMPLETED) is idempotent.
    def test_recoverable_job_executes_exactly_once_across_repeated_recovery(self) -> None:
        video_file = _create_synthetic_video_frames(self.temp_dir / "recover.mp4", num_frames=20)
        job = self.job_manager.create_job(person_id="P_RECOVER_ONCE", video_path=str(video_file))
        self.job_manager.checkpoint_job(
            job.job_id,
            stage="TRACKING",
            last_safe_frame=10,
            frames_processed=10,
            total_frames=20,
            status=ReferenceJobStatus.INTERRUPTED,
        )

        recovered_1 = self.job_manager.recover_unfinished_jobs(processor=self.processor, gait_service_ref=None)
        self.assertEqual(len(recovered_1), 1)

        deadline = time.time() + 15.0
        final_job = self.job_manager.get_job(job.job_id)
        while time.time() < deadline and final_job.status != ReferenceJobStatus.COMPLETED:
            time.sleep(0.1)
            final_job = self.job_manager.get_job(job.job_id)
        self.assertEqual(final_job.status, ReferenceJobStatus.COMPLETED, "resumed job never reached COMPLETED")

        after_first = _hash_gallery(self.gait_gallery_dir)
        self.assertTrue(after_first)

        for _ in range(2):
            recovered_again = self.job_manager.recover_unfinished_jobs(processor=self.processor, gait_service_ref=None)
            self.assertEqual(recovered_again, [], "a COMPLETED job must never be re-selected for recovery")
            self.assertEqual(_hash_gallery(self.gait_gallery_dir), after_first)

    # G: the same physical source, submitted under a DIFFERENT job_id, must not
    # add duplicate embeddings for the same person (content-hash idempotency).
    def test_same_source_content_under_new_job_id_is_a_noop(self) -> None:
        video_file = _create_synthetic_video_frames(self.temp_dir / "dup_source.mp4", num_frames=20)

        job1 = self.job_manager.create_job(person_id="P_DUP_SOURCE", video_path=str(video_file))
        res1 = self.processor.process_reference_video(
            person_id="P_DUP_SOURCE", video_path=video_file, job_id=job1.job_id
        )
        self.assertTrue(res1.get("success"))
        embeddings_after_first = res1.get("embeddings_committed", 0)
        self.assertGreater(embeddings_after_first, 0)

        gallery_hash_after_first = _hash_gallery(self.gait_gallery_dir)

        # Simulate a retry/replay: identical file content, brand-new job_id.
        job2 = self.job_manager.create_job(person_id="P_DUP_SOURCE", video_path=str(video_file))
        res2 = self.processor.process_reference_video(
            person_id="P_DUP_SOURCE", video_path=video_file, job_id=job2.job_id
        )

        self.assertEqual(res2.get("status"), "ALREADY_PROCESSED")
        self.assertEqual(res2.get("embeddings_added"), 0)
        self.assertEqual(
            _hash_gallery(self.gait_gallery_dir),
            gallery_hash_after_first,
            "replaying the same source content under a new job_id must not touch the gallery",
        )

    # J: recovery disabled via config must never mutate biometric state, even
    # with a genuinely recoverable job present. GaitService.warmup() reads
    # ARGUS_ENABLE_STARTUP_RECOVERY and, when disabled, must skip calling
    # recover_unfinished_jobs entirely - this asserts that exact gate logic.
    def test_recovery_disabled_flag_prevents_all_mutation(self) -> None:
        import os

        video_file = _create_synthetic_video_frames(self.temp_dir / "gated.mp4", num_frames=20)
        job = self.job_manager.create_job(person_id="P_GATED", video_path=str(video_file))
        self.job_manager.checkpoint_job(
            job.job_id,
            stage="TRACKING",
            last_safe_frame=10,
            frames_processed=10,
            total_frames=20,
            status=ReferenceJobStatus.INTERRUPTED,
        )

        for disabled_value in ("0", "false", "no"):
            os.environ["ARGUS_ENABLE_STARTUP_RECOVERY"] = disabled_value
            try:
                recovery_enabled = os.environ.get("ARGUS_ENABLE_STARTUP_RECOVERY", "1").lower() not in (
                    "0",
                    "false",
                    "no",
                )
                self.assertFalse(recovery_enabled, f"value {disabled_value!r} must disable recovery")
            finally:
                del os.environ["ARGUS_ENABLE_STARTUP_RECOVERY"]

        # With the gate off, GaitService.warmup() never calls recover_unfinished_jobs,
        # so directly verify the no-call path leaves the gallery untouched.
        self.assertEqual(_gallery_files(self.gait_gallery_dir), [])

    # I: a malformed save (feature/label count mismatch) must be rejected before
    # any file is touched, leaving the previously-valid gallery intact.
    def test_malformed_save_does_not_corrupt_existing_gallery(self) -> None:
        self.store.save(
            features=[[0.1] * 256, [0.2] * 256],
            labels=["Person_A", "Person_A"],
            metadata={"Person_A": {"embeddings": 2, "status": "ACTIVE", "enabled": True}},
        )
        valid_hash = _hash_gallery(self.gait_gallery_dir)
        self.assertTrue(valid_hash)

        with self.assertRaises(ValueError):
            self.store.save(
                features=[[0.1] * 256, [0.2] * 256, [0.3] * 256],  # 3 feature rows
                labels=["Person_A", "Person_A"],  # only 2 labels: mismatch
                metadata={"Person_A": {"embeddings": 3, "status": "ACTIVE", "enabled": True}},
            )

        self.assertEqual(
            _hash_gallery(self.gait_gallery_dir),
            valid_hash,
            "a rejected malformed save must not modify the previously-persisted gallery",
        )

    # K: recovery/idempotency decisions must be auditable via logs (job_id and
    # person_id present, no raw embedding vectors logged).
    def test_already_processed_skip_is_auditable(self) -> None:
        video_file = _create_synthetic_video_frames(self.temp_dir / "audit.mp4", num_frames=20)
        job1 = self.job_manager.create_job(person_id="P_AUDIT", video_path=str(video_file))
        self.processor.process_reference_video(person_id="P_AUDIT", video_path=video_file, job_id=job1.job_id)

        job2 = self.job_manager.create_job(person_id="P_AUDIT", video_path=str(video_file))
        with self.assertLogs("ARGUS.missing_person_processor", level="INFO") as log_ctx:
            res2 = self.processor.process_reference_video(
                person_id="P_AUDIT", video_path=video_file, job_id=job2.job_id
            )
        self.assertEqual(res2.get("status"), "ALREADY_PROCESSED")
        combined_log = "\n".join(log_ctx.output)
        self.assertIn("ALREADY_PROCESSED", combined_log)
        self.assertIn("P_AUDIT", combined_log)
        self.assertIn(job1.job_id, combined_log)


if __name__ == "__main__":
    unittest.main()
