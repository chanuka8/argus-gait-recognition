import unittest
import numpy as np

from services.camera_service import CameraService


class TestCameraService(unittest.TestCase):
    def test_camera_service_initialization(self) -> None:
        config = {
            "type": "usb",
            "device_index": 0,
            "width": 320,
            "height": 240,
            "target_fps": 10,
            "reconnect_seconds": 1,
            "max_queue_size": 5,
        }
        service = CameraService(config=config)
        self.assertFalse(service.is_alive())
        self.assertFalse(service.is_connected())
        self.assertEqual(service.queue_size, 0)
        self.assertEqual(service.frame_count, 0)

    def test_camera_service_status_dict(self) -> None:
        config = {
            "type": "usb",
            "device_index": 0,
            "max_queue_size": 5,
        }
        service = CameraService(config=config)
        status = service.get_status()
        self.assertIn("alive", status)
        self.assertIn("connected", status)
        self.assertIn("fps", status)
        self.assertIn("queue_size", status)
        self.assertIn("frame_count", status)
        self.assertIn("reconnect_count", status)
        self.assertEqual(status["source_type"], "usb")

    def test_camera_service_synthetic_queue(self) -> None:
        config = {
            "type": "usb",
            "device_index": 0,
            "max_queue_size": 3,
        }
        service = CameraService(config=config)

        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8)

        service._queue.put(frame1)
        service._queue.put(frame2)

        self.assertEqual(service.queue_size, 2)

        retrieved1 = service.get_frame()
        self.assertIsNotNone(retrieved1)
        self.assertEqual(retrieved1.shape, (100, 100, 3))

        retrieved2 = service.get_frame()
        self.assertIsNotNone(retrieved2)

        retrieved_none = service.get_frame()
        self.assertIsNone(retrieved_none)


if __name__ == "__main__":
    unittest.main()
