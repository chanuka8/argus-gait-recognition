import unittest
import unittest.mock
import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.server import app


class TestApiV1Integration(unittest.TestCase):
    def setUp(self) -> None:
        self.client_cm = TestClient(app)
        self.client = self.client_cm.__enter__()

    def tearDown(self) -> None:
        self.client_cm.__exit__(None, None, None)

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
                json={"camera_id": "cam_gate_01", "source": "rtsp://user:pass@192.168.1.100:554/live", "location": "Main Gate"},
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
        with self.client.websocket_connect("/api/v1/ws/recognition") as websocket:
            websocket.send_text("ping")


if __name__ == "__main__":
    unittest.main()
