import tempfile
import unittest
import unittest.mock
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.server import app
from api.v1.router import get_gait_service
from services.gait_service import GaitService
from storage.embedding_database import EmbeddingDatabase


class TestApiV1Integration(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.test_gallery = temp_path / "live_gallery"
        self.test_appearance = temp_path / "appearance_gallery"
        self.test_db = temp_path / "embedding_db"
        self.test_gallery.mkdir(parents=True, exist_ok=True)
        self.test_appearance.mkdir(parents=True, exist_ok=True)
        self.test_db.mkdir(parents=True, exist_ok=True)

        self.service = GaitService(
            gallery_dir=str(self.test_gallery),
            appearance_gallery_dir=str(self.test_appearance),
        )
        self.service.embedding_db = EmbeddingDatabase(
            db_dir=str(self.test_db),
            gait_gallery_dir=str(self.test_gallery),
            appearance_gallery_dir=str(self.test_appearance),
        )

        app.state.gait_service = self.service
        app.dependency_overrides[get_gait_service] = lambda: self.service

        self.client_cm = TestClient(app)
        self.client = self.client_cm.__enter__()

    def tearDown(self) -> None:
        self.client_cm.__exit__(None, None, None)
        app.dependency_overrides.pop(get_gait_service, None)
        app.state.gait_service = None
        self.temp_dir.cleanup()


    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["pipeline_loaded"])

    def test_status_endpoint(self) -> None:
        response = self.client.get("/api/v1/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "operational")
        self.assertIn("gallery", data)

    def test_metrics_endpoint(self) -> None:
        response = self.client.get("/api/v1/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("people", data)
        self.assertIn("embeddings", data)

    def test_identify_image_endpoint(self) -> None:
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (40, 90), (255, 255, 255), -1)
        _, encoded = cv2.imencode(".jpg", img)

        response = self.client.post(
            "/api/v1/identify/image",
            files={"file": ("test.jpg", encoded.tobytes(), "image/jpeg")},
            data={"camera_id": "test-cam-01"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("event_id", data)
        self.assertEqual(data["camera_id"], "test-cam-01")
        self.assertIn(data["decision"], ["KNOWN", "UNCERTAIN", "UNKNOWN"])
        self.assertEqual(data["recognition_branch"], "2D_GEI")

    def test_camera_lifecycle_endpoints(self) -> None:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[50:150, 50:150] = [0, 200, 0]

        mock_cap = unittest.mock.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)

        with unittest.mock.patch("services.camera_worker.cv2.VideoCapture", return_value=mock_cap):
            start_res = self.client.post(
                "/api/v1/cameras/start",
                json={
                    "camera_id": "cam_gate_01",
                    "source": "rtsp://user:pass@192.168.1.100:554/live",
                    "location": "Main Gate",
                },
            )
            self.assertEqual(start_res.status_code, 200)
            cam_data = start_res.json()
            self.assertEqual(cam_data["camera_id"], "cam_gate_01")
            self.assertNotIn("pass", cam_data["source"])

            list_res = self.client.get("/api/v1/cameras")
            self.assertEqual(list_res.status_code, 200)
            cams = list_res.json()
            self.assertTrue(any(c["camera_id"] == "cam_gate_01" for c in cams))

            stop_res = self.client.post(
                "/api/v1/cameras/stop",
                json={"camera_id": "cam_gate_01"},
            )
            self.assertEqual(stop_res.status_code, 200)

    def test_enroll_endpoint(self) -> None:
        img = np.zeros((100, 50, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (40, 90), (255, 255, 255), -1)
        _, encoded = cv2.imencode(".jpg", img)

        response = self.client.post(
            "/api/v1/enroll",
            data={"person_id": "subject999"},
            files=[("files", ("enroll1.jpg", encoded.tobytes(), "image/jpeg"))],
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["person_id"], "subject999")
        self.assertGreater(data["embeddings_added"], 0)

    def test_events_endpoint(self) -> None:
        response = self.client.get("/api/v1/events")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_websocket_recognition(self) -> None:
        from security_layer.auth import get_session_store

        session = get_session_store().create_session(
            operator_id="test_ws_operator",
            username="test_ws_operator",
            role="investigator",
        )
        with self.client.websocket_connect(
            "/api/v1/ws/recognition",
            subprotocols=["argus-auth", session.token],
        ) as websocket:
            websocket.send_text("ping")


if __name__ == "__main__":
    unittest.main()
