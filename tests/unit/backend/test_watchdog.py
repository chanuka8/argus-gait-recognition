import unittest
from unittest.mock import MagicMock

from monitoring.watchdog import Watchdog


class TestWatchdog(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_service = MagicMock()
        self.mock_service.get_status.return_value = {
            "running": True,
            "uptime_seconds": 100.0,
            "restart_count": 0,
            "recognition_alive": True,
            "camera": {
                "alive": True,
                "connected": True,
                "fps": 15.0,
                "queue_size": 2,
                "frame_count": 1500,
                "reconnect_count": 0,
                "source_type": "usb",
            },
        }

    def test_watchdog_healthy_check(self) -> None:
        config = {
            "interval_seconds": 30,
            "max_queue_warning": 8,
            "min_fps_warning": 2.0,
            "auto_restart": True,
            "max_restart_attempts": 5,
        }
        watchdog = Watchdog(service=self.mock_service, config=config)
        report = watchdog.check_health()

        self.assertTrue(report["camera_alive"])
        self.assertTrue(report["camera_connected"])
        self.assertTrue(report["recognition_alive"])
        self.assertEqual(report["fps"], 15.0)
        self.assertEqual(report["queue_size"], 2)

        self.mock_service.restart_recognition.assert_not_called()
        self.mock_service.restart_camera.assert_not_called()

    def test_watchdog_detects_recognition_failure(self) -> None:
        self.mock_service.get_status.return_value["recognition_alive"] = False

        config = {
            "interval_seconds": 30,
            "auto_restart": True,
            "max_restart_attempts": 5,
        }
        watchdog = Watchdog(service=self.mock_service, config=config)
        watchdog.check_health()

        self.mock_service.restart_recognition.assert_called_once()
        self.assertEqual(watchdog._consecutive_restarts, 1)

    def test_watchdog_detects_camera_failure(self) -> None:
        self.mock_service.get_status.return_value["camera"]["alive"] = False

        config = {
            "interval_seconds": 30,
            "auto_restart": True,
            "max_restart_attempts": 5,
        }
        watchdog = Watchdog(service=self.mock_service, config=config)
        watchdog.check_health()

        self.mock_service.restart_camera.assert_called_once()
        self.assertEqual(watchdog._consecutive_restarts, 1)

    def test_watchdog_resource_metrics_collection(self) -> None:
        metrics = Watchdog._collect_resource_usage()
        self.assertIn("cpu_percent", metrics)
        self.assertIn("ram_percent", metrics)
        self.assertIn("ram_used_mb", metrics)
        self.assertIn("gpu_name", metrics)


if __name__ == "__main__":
    unittest.main()
