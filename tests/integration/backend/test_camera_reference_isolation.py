"""Production Integration Tests: Camera ON / Camera OFF Isolation,
Interruption Resilience, Failure Injection, and Performance Profiling
for ARGUS AI Missing Person Reference Media Processing.
"""

import io
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import app
from security_layer.auth import get_session_store
from security_layer.authorization import Role
from services.camera_worker import CameraWorker
from services.gait_service import GaitService
from services.missing_person_processor import MissingPersonVideoProcessor
from services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus
from storage.embedding_database import EmbeddingDatabase
from storage.vector_store import VectorStore


def _dummy_frame(width: int = 320, height: int = 240) -> np.ndarray:
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    # Draw simple walking figure for realistic detection
    cx = 160
    cv2.circle(frame, (cx, 60), 16, (255, 255, 255), -1)
    cv2.rectangle(frame, (cx - 20, 78), (cx + 20, 150), (255, 255, 255), -1)
    cv2.line(frame, (cx - 12, 150), (cx - 16, 215), (255, 255, 255), 8)
    cv2.line(frame, (cx + 12, 150), (cx + 16, 215), (255, 255, 255), 8)
    return frame


def _create_synthetic_person_video(
    filepath: Path,
    num_frames: int = 30,
    width: int = 320,
    height: int = 240,
    fps: float = 25.0,
) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    for idx in range(num_frames):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        cx = 120 + (idx % 12) * 3
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


def _create_synthetic_photo_bytes() -> bytes:
    img = np.full((240, 120, 3), 35, dtype=np.uint8)
    cx = 60
    cv2.circle(img, (cx, 40), 16, (255, 255, 255), -1)
    cv2.rectangle(img, (cx - 22, 58), (cx + 22, 140), (255, 255, 255), -1)
    cv2.line(img, (cx - 12, 140), (cx - 15, 220), (255, 255, 255), 8)
    cv2.line(img, (cx + 12, 140), (cx + 15, 220), (255, 255, 255), 8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


class ActiveStreamingCapture:
    """Mock OpenCV VideoCapture that continuously produces frames at a target rate."""

    def __init__(self, fps: float = 20.0, width: int = 320, height: int = 240) -> None:
        self.fps = fps
        self.interval = 1.0 / fps
        self.width = width
        self.height = height
        self._is_opened = True
        self._frame_count = 0
        self._lock = threading.Lock()
        self._last_time = time.monotonic()

    def isOpened(self) -> bool:
        with self._lock:
            return self._is_opened

    def read(self) -> tuple[bool, np.ndarray]:
        with self._lock:
            if not self._is_opened:
                return False, np.empty((0,))
            self._frame_count += 1
            now = time.monotonic()
            sleep_needed = self.interval - (now - self._last_time)
            if sleep_needed > 0:
                time.sleep(min(sleep_needed, 0.05))
            self._last_time = time.monotonic()
            return True, _dummy_frame(self.width, self.height)

    def release(self) -> None:
        with self._lock:
            self._is_opened = False

    def disconnect(self) -> None:
        with self._lock:
            self._is_opened = False

    def reconnect(self) -> None:
        with self._lock:
            self._is_opened = True
            self._last_time = time.monotonic()

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return True


class DeterministicTestTracker:
    """Wrapper around real TrackingStep that falls back to synthetic figure detection
    when running integration tests on computer-generated test videos where YOLOv8n returns 0 detections.
    """

    def __init__(self, real_tracker) -> None:
        self.real_tracker = real_tracker
        self.detector = getattr(real_tracker, "detector", None)
        self.tracker = getattr(real_tracker, "tracker", None)

    def reset(self) -> None:
        if hasattr(self.real_tracker, "reset"):
            self.real_tracker.reset()

    def track(self, frame: np.ndarray):
        det = self.real_tracker.track(frame)
        if len(det) > 0:
            return det
        if frame is not None and getattr(frame, "size", 0) > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            mask = gray > 100
            if np.any(mask):
                ys, xs = np.where(mask)
                h, w = frame.shape[:2]
                x1, x2 = max(0, int(xs.min()) - 10), min(w, int(xs.max()) + 10)
                y1, y2 = max(0, int(ys.min()) - 10), min(h, int(ys.max()) + 10)
                import supervision as sv

                return sv.Detections(
                    xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    confidence=np.array([0.95], dtype=np.float32),
                    class_id=np.array([0], dtype=int),
                    tracker_id=np.array([1], dtype=int),
                )
        return det


@pytest.fixture
def auth_headers():
    store = get_session_store()
    inv_session = store.create_session("op_test", "test_user", Role.INVESTIGATOR.value)
    return {"Authorization": f"Bearer {inv_session.token}"}


@pytest.fixture
def test_env():
    temp_dir = Path(tempfile.mkdtemp(prefix="argus_isolation_test_"))
    gallery_dir = temp_dir / "live_gallery"
    app_gallery_dir = temp_dir / "appearance_gallery"
    db_dir = temp_dir / "embedding_db"
    jobs_dir = temp_dir / "reference_jobs"

    gallery_dir.mkdir(parents=True, exist_ok=True)
    app_gallery_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_mgr = ReferenceJobManager(jobs_dir=str(jobs_dir), max_workers=2)
    ReferenceJobManager._instance = job_mgr

    service = GaitService()
    service.gallery_dir = str(gallery_dir)
    service.appearance_gallery_dir = str(app_gallery_dir)
    service.store = VectorStore(gallery_dir=str(gallery_dir))
    service.appearance_store = VectorStore(gallery_dir=str(app_gallery_dir))
    service.embedding_db = EmbeddingDatabase(
        db_dir=str(db_dir),
        gait_gallery_dir=str(gallery_dir),
        appearance_gallery_dir=str(app_gallery_dir),
    )
    service.reload_gallery()

    if service.tracker is not None:
        service.tracker = DeterministicTestTracker(service.tracker)

    app.state.gait_service = service

    yield {
        "temp_dir": temp_dir,
        "service": service,
        "job_mgr": job_mgr,
        "gallery_dir": gallery_dir,
        "db_dir": db_dir,
        "jobs_dir": jobs_dir,
    }

    # Teardown
    for cam_id in list(service.camera_workers.keys()):
        service.stop_camera(cam_id)
    job_mgr.shutdown(timeout=1.0)
    ReferenceJobManager._instance = None
    shutil.rmtree(temp_dir, ignore_errors=True)


# =========================================================================
# SECTION 14: CAMERA OFF TESTS
# =========================================================================


def test_camera_off_photo_upload_e2e(test_env, auth_headers):
    """TEST 1: Camera OFF + photo upload.
    Verify: Upload 100% -> job created -> 256D embedding generated -> persisted -> COMPLETED.
    """
    service = test_env["service"]
    assert len(service.active_cameras) == 0, "Precondition failed: active camera found"

    photo_bytes = _create_synthetic_photo_bytes()

    with TestClient(app) as client:
        t_upload_start = time.perf_counter()
        resp = client.post(
            "/api/v1/enroll",
            data={"person_id": "MP_OFF_PHOTO_01", "async_mode": "true"},
            files=[("files", ("portrait.jpg", photo_bytes, "image/jpeg"))],
            headers=auth_headers,
        )
        upload_duration = time.perf_counter() - t_upload_start
        assert resp.status_code == 200, f"Enroll returned {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["success"] is True
        job_id = data["job_id"]
        assert job_id is not None

        # Poll job until completion
        t_poll_start = time.perf_counter()
        final_job = None
        for _ in range(60):
            st_resp = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers)
            assert st_resp.status_code == 200
            st_data = st_resp.json()
            if st_data["status"] == ReferenceJobStatus.COMPLETED.value:
                final_job = st_data
                break
            if st_data["status"] == ReferenceJobStatus.FAILED.value:
                pytest.fail(f"Job failed: {st_data.get('error_message')}")
            time.sleep(0.1)

        assert final_job is not None, "Photo job timed out before COMPLETED"
        proc_duration = time.perf_counter() - t_poll_start

        # Validate embedding correctness
        person = service.embedding_db.get_person("MP_OFF_PHOTO_01")
        assert person is not None
        assert len(person.gait_embeddings) >= 1
        emb = person.gait_embeddings[0]
        assert emb.embedding_dim == 256
        assert emb.status == "ACTIVE"
        assert len(emb.vector) == 256
        assert np.isfinite(emb.vector).all()
        norm = np.linalg.norm(emb.vector)
        assert 0.95 <= norm <= 1.05, f"Expected unit norm, got {norm}"

        # Verify VectorStore synchronization
        g_data = service.store.load()
        assert g_data is not None
        _, labels, _ = g_data
        assert "MP_OFF_PHOTO_01" in list(labels)

        print(f"\n[METRIC] Camera OFF Photo: Upload={upload_duration*1000:.1f}ms, Processing={proc_duration*1000:.1f}ms")


def test_camera_off_video_upload_e2e(test_env, auth_headers):
    """TEST 2: Camera OFF + video upload.
    Verify: Upload 100% -> job created -> tracking -> GEI -> ByGaitLight 256D -> persistence -> COMPLETED.
    """
    service = test_env["service"]
    assert len(service.active_cameras) == 0

    vid_path = test_env["temp_dir"] / "reference_off.mp4"
    _create_synthetic_person_video(vid_path, num_frames=30)
    video_bytes = vid_path.read_bytes()

    with TestClient(app) as client:
        t_upload_start = time.perf_counter()
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_OFF_VID_01", "case_id": "CASE_OFF_01"},
            files={"file": ("reference.mp4", io.BytesIO(video_bytes), "video/mp4")},
            headers=auth_headers,
        )
        upload_duration = time.perf_counter() - t_upload_start
        assert resp.status_code == 202
        body = resp.json()
        job_id = body["job_id"]

        # Poll until COMPLETED
        t_proc_start = time.perf_counter()
        final_job = None
        stages_observed = []
        for _ in range(80):
            st_resp = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers)
            assert st_resp.status_code == 200
            st_data = st_resp.json()
            curr_stage = st_data["progress"].get("stage")
            if curr_stage and curr_stage not in stages_observed:
                stages_observed.append(curr_stage)
            if st_data["status"] == ReferenceJobStatus.COMPLETED.value:
                final_job = st_data
                break
            if st_data["status"] == ReferenceJobStatus.FAILED.value:
                pytest.fail(f"Video job failed: {st_data.get('error_message')}")
            time.sleep(0.1)

        assert final_job is not None, "Video job timed out before reaching COMPLETED"
        proc_duration = time.perf_counter() - t_proc_start

        # Assert correct stages were reached
        assert final_job["status"] == "COMPLETED"
        res = final_job["result"]
        assert res["frames_processed"] >= 25
        assert res["embeddings_committed"] >= 1
        assert res["valid_silhouettes"] >= 10

        # Assert 256D embedding in database
        person = service.embedding_db.get_person("MP_OFF_VID_01")
        assert person is not None
        assert len(person.gait_embeddings) >= 1
        for e in person.gait_embeddings:
            assert e.embedding_dim == 256
            assert np.isfinite(e.vector).all()
            assert 0.95 <= np.linalg.norm(e.vector) <= 1.05

        print(
            f"\n[METRIC] Camera OFF Video: Upload={upload_duration*1000:.1f}ms, "
            f"Processing={proc_duration*1000:.1f}ms, FPS={res.get('effective_fps', 0)}"
        )


# =========================================================================
# SECTION 15: CAMERA ON TESTS
# =========================================================================


def test_camera_on_photo_upload_e2e(test_env, auth_headers):
    """TEST 3: Camera ON + photo upload.
    Verify: Camera remains ACTIVE and streaming + photo upload succeeds +
    embedding generation completes + camera recognition gallery is synchronized.
    """
    service = test_env["service"]
    mock_cap = ActiveStreamingCapture(fps=25.0)

    # Start live CCTV camera
    resolution = {
        "resolved_source": "0",
        "resolved_source_type": "webcam",
        "resolved_source_label": "CCTV-West-Gate",
        "capture": mock_cap,
        "initial_frame": _dummy_frame(),
    }
    with patch.object(service.source_resolver, "resolve_source", return_value=resolution):
        cam_info = service.start_camera("cam_cctv_01", source="0", location="West Gate")
        assert cam_info["status"] == "ACTIVE"
        assert "cam_cctv_01" in service.active_cameras

    worker: CameraWorker = service.get_camera_worker("cam_cctv_01")
    assert worker is not None
    assert worker.is_running()

    photo_bytes = _create_synthetic_photo_bytes()

    with TestClient(app) as client:
        # Camera is streaming actively
        initial_frames = worker.get_stats().get("frames_captured", 0)

        t_upload_start = time.perf_counter()
        resp = client.post(
            "/api/v1/enroll",
            data={"person_id": "MP_ON_PHOTO_01", "async_mode": "true"},
            files=[("files", ("portrait.jpg", photo_bytes, "image/jpeg"))],
            headers=auth_headers,
        )
        upload_duration = time.perf_counter() - t_upload_start
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Poll job to completion
        t_proc_start = time.perf_counter()
        final_job = None
        for _ in range(60):
            st_resp = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers)
            st_data = st_resp.json()
            if st_data["status"] == ReferenceJobStatus.COMPLETED.value:
                final_job = st_data
                break
            time.sleep(0.1)

        assert final_job is not None, "Camera ON photo job timed out"
        proc_duration = time.perf_counter() - t_proc_start

        # CRITICAL ASSERTION: Camera remained operational and continued capturing frames
        assert worker.is_running()
        final_frames = worker.get_stats().get("frames_captured", 0)
        assert final_frames > initial_frames, "Camera stopped capturing frames during photo processing"

        # Verify live camera worker received the updated gallery
        rec_worker = worker.recognition_worker
        assert rec_worker is not None
        assert "MP_ON_PHOTO_01" in rec_worker.gallery_labels

        print(
            f"\n[METRIC] Camera ON Photo: Upload={upload_duration*1000:.1f}ms, "
            f"Processing={proc_duration*1000:.1f}ms, CamFramesDelta={final_frames - initial_frames}"
        )


def test_camera_on_video_upload_e2e(test_env, auth_headers):
    """TEST 4: Camera ON + video upload.
    Verify: Camera remains ACTIVE + reference video upload succeeds +
    ByteTrack and ByGaitLight complete without blocking camera workers +
    gallery is synchronized for live recognition.
    """
    service = test_env["service"]
    mock_cap = ActiveStreamingCapture(fps=25.0)

    resolution = {
        "resolved_source": "0",
        "resolved_source_type": "webcam",
        "resolved_source_label": "CCTV-Perimeter-01",
        "capture": mock_cap,
        "initial_frame": _dummy_frame(),
    }
    with patch.object(service.source_resolver, "resolve_source", return_value=resolution):
        cam_info = service.start_camera("cam_cctv_02", source="0", location="Perimeter Zone")
        assert cam_info["status"] == "ACTIVE"

    worker: CameraWorker = service.get_camera_worker("cam_cctv_02")
    assert worker.is_running()

    vid_path = test_env["temp_dir"] / "reference_on.mp4"
    _create_synthetic_person_video(vid_path, num_frames=30)
    video_bytes = vid_path.read_bytes()

    with TestClient(app) as client:
        initial_cam_frames = worker.get_stats().get("frames_captured", 0)

        t_upload_start = time.perf_counter()
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_ON_VID_01", "case_id": "CASE_ON_01"},
            files={"file": ("reference.mp4", io.BytesIO(video_bytes), "video/mp4")},
            headers=auth_headers,
        )
        upload_duration = time.perf_counter() - t_upload_start
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Poll job to completion
        t_proc_start = time.perf_counter()
        final_job = None
        for _ in range(80):
            st_resp = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers)
            st_data = st_resp.json()
            if st_data["status"] == ReferenceJobStatus.COMPLETED.value:
                final_job = st_data
                break
            time.sleep(0.1)

        assert final_job is not None, "Camera ON video job timed out"
        proc_duration = time.perf_counter() - t_proc_start

        # Camera must have continued streaming without interruption
        assert worker.is_running()
        final_cam_frames = worker.get_stats().get("frames_captured", 0)
        assert final_cam_frames > initial_cam_frames

        # Live gallery in recognition worker was refreshed
        assert "MP_ON_VID_01" in worker.recognition_worker.gallery_labels

        print(
            f"\n[METRIC] Camera ON Video: Upload={upload_duration*1000:.1f}ms, "
            f"Processing={proc_duration*1000:.1f}ms, CamFramesDelta={final_cam_frames - initial_cam_frames}"
        )


# =========================================================================
# SECTION 16: CAMERA INTERRUPTION TESTS
# =========================================================================


def test_camera_interruption_disconnect_during_video_processing(test_env, auth_headers):
    """Scenario A: Camera ON -> video processing runs -> camera disconnects.
    Expected: Reference video job continues unharmed to COMPLETED.
    """
    service = test_env["service"]
    mock_cap = ActiveStreamingCapture(fps=25.0)

    resolution = {
        "resolved_source": "0",
        "resolved_source_type": "webcam",
        "resolved_source_label": "CCTV-Disconnect-Test",
        "capture": mock_cap,
        "initial_frame": _dummy_frame(),
    }
    with patch.object(service.source_resolver, "resolve_source", return_value=resolution):
        service.start_camera("cam_disc", source="0")

    worker = service.get_camera_worker("cam_disc")
    assert worker.is_running()

    vid_path = test_env["temp_dir"] / "vid_disc.mp4"
    _create_synthetic_person_video(vid_path, num_frames=30)
    video_bytes = vid_path.read_bytes()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_DISC_01"},
            files={"file": ("reference.mp4", io.BytesIO(video_bytes), "video/mp4")},
            headers=auth_headers,
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Simulate camera disconnect mid-processing
        time.sleep(0.2)
        mock_cap.disconnect()

        # Job must still finish COMPLETED
        final_job = None
        for _ in range(80):
            st_data = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers).json()
            if st_data["status"] == ReferenceJobStatus.COMPLETED.value:
                final_job = st_data
                break
            time.sleep(0.1)

        assert final_job is not None, "Job failed or stalled following camera disconnect"
        assert final_job["status"] == "COMPLETED"


def test_camera_interruption_reconnect_during_video_processing(test_env, auth_headers):
    """Scenario B: Camera ON -> camera disconnects and reconnects while video job runs.
    Expected: Reference job continues uninterrupted to COMPLETED.
    """
    service = test_env["service"]
    mock_cap = ActiveStreamingCapture(fps=25.0)

    resolution = {
        "resolved_source": "0",
        "resolved_source_type": "webcam",
        "resolved_source_label": "CCTV-Reconnect-Test",
        "capture": mock_cap,
        "initial_frame": _dummy_frame(),
    }
    with patch.object(service.source_resolver, "resolve_source", return_value=resolution):
        service.start_camera("cam_reconn", source="0")

    vid_path = test_env["temp_dir"] / "vid_reconn.mp4"
    _create_synthetic_person_video(vid_path, num_frames=30)
    video_bytes = vid_path.read_bytes()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_RECONN_01"},
            files={"file": ("reference.mp4", io.BytesIO(video_bytes), "video/mp4")},
            headers=auth_headers,
        )
        job_id = resp.json()["job_id"]

        # Disconnect then reconnect camera
        time.sleep(0.15)
        mock_cap.disconnect()
        time.sleep(0.1)
        mock_cap.reconnect()

        final_job = None
        for _ in range(80):
            st_data = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers).json()
            if st_data["status"] == ReferenceJobStatus.COMPLETED.value:
                final_job = st_data
                break
            time.sleep(0.1)

        assert final_job is not None
        assert final_job["status"] == "COMPLETED"


def test_camera_interruption_stop_during_video_processing(test_env, auth_headers):
    """Scenario C: Camera ON -> camera is stopped via API mid-job.
    Expected: Reference job continues to COMPLETED.
    """
    service = test_env["service"]
    mock_cap = ActiveStreamingCapture(fps=25.0)

    resolution = {
        "resolved_source": "0",
        "resolved_source_type": "webcam",
        "resolved_source_label": "CCTV-Stop-Test",
        "capture": mock_cap,
        "initial_frame": _dummy_frame(),
    }
    with patch.object(service.source_resolver, "resolve_source", return_value=resolution):
        service.start_camera("cam_stop_mid", source="0")

    vid_path = test_env["temp_dir"] / "vid_stop.mp4"
    _create_synthetic_person_video(vid_path, num_frames=30)
    video_bytes = vid_path.read_bytes()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_STOP_MID_01"},
            files={"file": ("reference.mp4", io.BytesIO(video_bytes), "video/mp4")},
            headers=auth_headers,
        )
        job_id = resp.json()["job_id"]

        # Stop camera via API while job is actively running
        time.sleep(0.15)
        service.stop_camera("cam_stop_mid")
        assert "cam_stop_mid" not in service.active_cameras

        final_job = None
        for _ in range(80):
            st_data = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers).json()
            if st_data["status"] == ReferenceJobStatus.COMPLETED.value:
                final_job = st_data
                break
            time.sleep(0.1)

        assert final_job is not None
        assert final_job["status"] == "COMPLETED"


def test_camera_interruption_start_while_processing(test_env, auth_headers):
    """Scenario D: Camera OFF -> video job processing -> camera started dynamically.
    Expected: Reference job continues safely to COMPLETED and camera initializes cleanly.
    """
    service = test_env["service"]
    assert len(service.active_cameras) == 0

    vid_path = test_env["temp_dir"] / "vid_start_mid.mp4"
    _create_synthetic_person_video(vid_path, num_frames=30)
    video_bytes = vid_path.read_bytes()

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_START_MID_01"},
            files={"file": ("reference.mp4", io.BytesIO(video_bytes), "video/mp4")},
            headers=auth_headers,
        )
        job_id = resp.json()["job_id"]

        # Start a camera while job is running in the background
        time.sleep(0.1)
        mock_cap = ActiveStreamingCapture(fps=25.0)
        resolution = {
            "resolved_source": "0",
            "resolved_source_type": "webcam",
            "resolved_source_label": "CCTV-Dynamic-Start",
            "capture": mock_cap,
            "initial_frame": _dummy_frame(),
        }
        with patch.object(service.source_resolver, "resolve_source", return_value=resolution):
            service.start_camera("cam_dyn_start", source="0")
        assert "cam_dyn_start" in service.active_cameras

        final_job = None
        for _ in range(80):
            st_data = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers).json()
            if st_data["status"] == ReferenceJobStatus.COMPLETED.value:
                final_job = st_data
                break
            time.sleep(0.1)

        assert final_job is not None
        assert final_job["status"] == "COMPLETED"
        assert service.get_camera_worker("cam_dyn_start").is_running()


# =========================================================================
# SECTION 17: FAILURE INJECTION & IDEMPOTENCY TESTS
# =========================================================================


def test_failure_injection_corrupt_video(test_env, auth_headers):
    """Failure Injection 1: Corrupted video stream.
    Expected: Job safely records FAILED status with DECODER_ERROR without crashing server.
    """
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_CORRUPT_01"},
            files={"file": ("reference.mp4", io.BytesIO(b"NOT_A_VALID_MP4_HEADER_DATA_12345"), "video/mp4")},
            headers=auth_headers,
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        final_job = None
        for _ in range(30):
            st_data = client.get(f"/api/v1/cases/jobs/{job_id}", headers=auth_headers).json()
            if st_data["status"] == ReferenceJobStatus.FAILED.value:
                final_job = st_data
                break
            time.sleep(0.1)

        assert final_job is not None
        assert final_job["status"] == ReferenceJobStatus.FAILED.value
        assert (
            final_job.get("diagnostic_code") in ("INVALID_VIDEO", "DECODER_ERROR")
            or "decoder" in str(final_job.get("error_message", "")).lower()
        )


def test_failure_injection_empty_video(auth_headers):
    """Failure Injection 2: Empty 0-byte video upload.
    Expected: HTTP 400 Bad Request immediately.
    """
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/cases/upload-reference",
            data={"person_id": "MP_EMPTY_01"},
            files={"file": ("reference.mp4", io.BytesIO(b""), "video/mp4")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "empty" in resp.text.lower()


def test_idempotent_duplicate_prevention(test_env):
    """Verify that repeatedly committing the same reference job does not produce duplicate active embeddings."""
    service = test_env["service"]
    db: EmbeddingDatabase = service.embedding_db

    fake_vector = np.random.randn(256).astype(np.float32)
    fake_vector /= np.linalg.norm(fake_vector)

    deterministic_id = "gait_DEDUP_TEST_job_101_seq_0"

    # First commit
    res1 = db.commit_and_activate_embeddings(
        person_id="DEDUP_TEST",
        gait_embeddings=[fake_vector],
        source_session_id="job_101",
        embedding_ids=[deterministic_id],
    )
    assert res1["success"] is True
    assert res1["gait_embeddings_added"] == 1

    person1 = db.get_person("DEDUP_TEST")
    assert len(person1.gait_embeddings) == 1

    # Second commit with identical deterministic ID (simulating retry or crash recovery)
    res2 = db.commit_and_activate_embeddings(
        person_id="DEDUP_TEST",
        gait_embeddings=[fake_vector],
        source_session_id="job_101",
        embedding_ids=[deterministic_id],
    )
    assert res2["success"] is True
    assert res2["gait_embeddings_added"] == 0, "Duplicate embedding was erroneously added"

    person2 = db.get_person("DEDUP_TEST")
    assert len(person2.gait_embeddings) == 1, "Expected exactly 1 embedding after repeated commit"
    assert person2.gait_embeddings[0].embedding_id == deterministic_id


def test_embedding_validation_numerical_rejection(test_env):
    """Verify that validate_embedding strictly rejects non-finite vectors or wrong dimensions."""
    processor = MissingPersonVideoProcessor(
        store=test_env["service"].store,
        embedding_db=test_env["service"].embedding_db,
    )

    # 1. NaN vector
    nan_vec = np.zeros(256, dtype=np.float32)
    nan_vec[10] = np.nan
    valid, msg, _ = processor.validate_embedding(nan_vec, "TEST_P", 1, 0)
    assert valid is False
    assert "NaN" in msg or "Inf" in msg

    # 2. Infinite vector
    inf_vec = np.zeros(256, dtype=np.float32)
    inf_vec[5] = np.inf
    valid, msg, _ = processor.validate_embedding(inf_vec, "TEST_P", 1, 0)
    assert valid is False

    # 3. Wrong dimension (e.g. 512 instead of 256)
    wrong_dim = np.ones(512, dtype=np.float32)
    valid, msg, _ = processor.validate_embedding(wrong_dim, "TEST_P", 1, 0)
    assert valid is False
    assert "256" in msg

    # 4. Zero norm vector
    zero_vec = np.zeros(256, dtype=np.float32)
    valid, msg, _ = processor.validate_embedding(zero_vec, "TEST_P", 1, 0)
    assert valid is False
    assert "norm" in msg.lower()
