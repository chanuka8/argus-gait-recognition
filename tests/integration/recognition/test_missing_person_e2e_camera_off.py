import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.server import app
from services.gait_service import GaitService
from services.missing_person_processor import MissingPersonVideoProcessor
from services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus
from storage.embedding_database import EmbeddingDatabase
from storage.vector_store import VectorStore


def _create_synthetic_person_video(
    filepath: Path,
    num_frames: int = 25,
    width: int = 320,
    height: int = 240,
    fps: float = 25.0,
) -> Path:
    """Creates a synthetic video containing a clearly visible walking human figure."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

    for idx in range(num_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)

        cx = 120 + (idx % 12) * 2
        # Head
        cv2.circle(frame, (cx, 55), 16, (255, 255, 255), -1)
        # Torso
        cv2.rectangle(frame, (cx - 22, 72), (cx + 22, 150), (255, 255, 255), -1)
        # Arms
        arm_offset = int((idx % 6 - 3) * 3)
        cv2.line(frame, (cx - 22, 85), (cx - 30, 130 + arm_offset), (255, 255, 255), 6)
        cv2.line(frame, (cx + 22, 85), (cx + 30, 130 - arm_offset), (255, 255, 255), 6)
        # Legs
        cv2.line(frame, (cx - 12, 150), (cx - 16 + arm_offset, 215), (255, 255, 255), 8)
        cv2.line(frame, (cx + 12, 150), (cx + 16 - arm_offset, 215), (255, 255, 255), 8)

        writer.write(frame)

    writer.release()
    return filepath


class TestMissingPersonCameraOffE2E(unittest.TestCase):
    """Mandatory Phase 2 Camera-OFF E2E and Integration Test Suite.
    Proves that missing-person video processing, embedding generation,
    validation, and persistence execute end-to-end when all live cameras are OFF.
    """

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

        self.processor = MissingPersonVideoProcessor(
            gait_gallery_dir=str(self.gait_gallery_dir),
            appearance_gallery_dir=str(self.appearance_gallery_dir),
            db_dir=str(self.db_dir),
            store=self.store,
            embedding_db=self.embedding_db,
            job_manager=self.job_manager,
        )

        self.client_cm = TestClient(app)
        self.client = self.client_cm.__enter__()

    def tearDown(self) -> None:
        self.client_cm.__exit__(None, None, None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_camera_off_end_to_end_pipeline(self) -> None:
        """Requirement 17: Camera-OFF End-to-End Test.
        1. Confirm every camera is OFF / stopped.
        2. Submit valid reference video.
        3. Offline processing runs without camera dependencies.
        4. Valid 256D embeddings are generated, validated, and persisted.
        5. Retrieval confirms embeddings are available in gallery.
        """
        # Step 1: Explicitly verify that no cameras are active
        gait_service: GaitService = app.state.gait_service if hasattr(app.state, "gait_service") else None
        if gait_service:
            # Shut down any active cameras
            self.assertEqual(len(gait_service.active_cameras), 0, "Precondition violated: Active cameras found")
            self.assertEqual(len(gait_service.camera_workers), 0, "Precondition violated: Camera workers running")

        # Step 2: Create reference video
        person_id = "Case_Target_999"
        video_path = _create_synthetic_person_video(
            self.temp_dir / "ref_target.mp4",
            num_frames=25,
        )

        # Step 3: Mock the tracker to return deterministic tracking detections for the synthetic walking person
        import supervision as sv

        mock_detections = sv.Detections(
            xyxy=np.array([[100, 40, 160, 220]], dtype=np.float32),
            confidence=np.array([0.95], dtype=np.float32),
            tracker_id=np.array([1], dtype=int),
        )
        mock_tracker = MagicMock()
        mock_tracker.track.return_value = mock_detections
        self.processor.tracker = mock_tracker

        # Run offline processing
        job = self.job_manager.create_job(person_id=person_id, video_path=str(video_path), case_id=person_id)
        result = self.processor.process_reference_video(
            person_id=person_id,
            video_path=video_path,
            job_id=job.job_id,
        )

        # Step 4: Verify processing succeeded with zero cameras
        self.assertTrue(result["success"], f"Processing failed: {result.get('error')}")
        self.assertEqual(result["person_id"], person_id)
        self.assertGreater(result["embeddings_committed"], 0)
        self.assertGreater(result["valid_sequences"], 0)
        self.assertTrue(result["persistence_verified"])

        # Step 5: Verify job status in manager
        completed_job = self.job_manager.get_job(job.job_id)
        self.assertIsNotNone(completed_job)
        self.assertEqual(completed_job.status, ReferenceJobStatus.COMPLETED)
        self.assertIsNone(completed_job.error_message)

        # Step 6: Verify embeddings are in local VectorStore
        gallery_data = self.store.load()
        self.assertIsNotNone(gallery_data, "VectorStore gallery is empty")
        features, labels, _ = gallery_data
        self.assertGreater(len(features), 0)
        self.assertIn(person_id, list(labels))
        self.assertEqual(features.shape[1], 256, "Embedding dimension must be exactly 256D")

        # Step 7: Verify embedding records in durable database
        person_rec = self.embedding_db.get_person(person_id)
        self.assertIsNotNone(person_rec)
        self.assertGreater(len(person_rec.gait_embeddings), 0)
        first_emb = person_rec.gait_embeddings[0]
        self.assertEqual(first_emb.embedding_dim, 256)
        self.assertEqual(first_emb.modality, "gait")
        self.assertEqual(len(first_emb.vector), 256)
        self.assertTrue(np.isfinite(first_emb.vector).all())

        # Step 8: Re-confirm zero live cameras were started or required
        if gait_service:
            self.assertEqual(len(gait_service.active_cameras), 0)
            self.assertEqual(len(gait_service.camera_workers), 0)

    def test_api_v1_reference_endpoints(self) -> None:
        """Integration test for POST /api/v1/cases/upload-reference and GET /api/v1/cases/jobs/{job_id}."""
        # Create a small valid test video in memory
        video_path = _create_synthetic_person_video(
            self.temp_dir / "api_test.mp4",
            num_frames=15,
        )
        video_bytes = video_path.read_bytes()

        # POST /cases/upload-reference
        response = self.client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "Case_API_001", "case_id": "Case_API_001"},
            files={"file": ("reference.mp4", io.BytesIO(video_bytes), "video/mp4")},
        )
        self.assertIn(response.status_code, (200, 202))
        data = response.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["person_id"], "Case_API_001")
        self.assertEqual(data["status"], "QUEUED")
        job_id = data["job_id"]

        # GET /cases/jobs/{job_id}
        status_res = self.client.get(f"/api/v1/cases/jobs/{job_id}")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertEqual(status_data["job_id"], job_id)
        self.assertEqual(status_data["person_id"], "Case_API_001")
        self.assertIn(status_data["status"], ["QUEUED", "PROCESSING", "COMPLETED", "FAILED"])
        self.assertIn("progress", status_data)

    def test_api_v1_rejects_empty_and_invalid_formats(self) -> None:
        """API validation: rejects invalid MIME types and empty files with proper HTTP codes."""
        # Empty file
        empty_res = self.client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "Case_Bad_001"},
            files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
        )
        self.assertEqual(empty_res.status_code, 400)

        # Invalid extension
        invalid_res = self.client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "Case_Bad_002"},
            files={"file": ("document.pdf", io.BytesIO(b"%PDF-1.4 test"), "application/pdf")},
        )
        self.assertEqual(invalid_res.status_code, 415)


if __name__ == "__main__":
    unittest.main()
