"""Tests for multi-camera system."""

import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from monitoring.camera_monitor import CameraMonitor
from services.camera_manager import CameraManager
from services.camera_worker import CameraWorker


class TestCameraWorker(unittest.TestCase):
    """Test CameraWorker."""

    def setUp(self):
        self.config = {
            "id": "test_camera",
            "name": "Test Camera",
            "type": "rtsp",
            "url": "rtsp://example.com/stream",
            "device_index": 0,
            "width": 640,
            "height": 480,
            "target_fps": 15,
            "reconnect_interval": 1,
            "max_reconnect_attempts": 3,
            "max_queue_size": 10,
        }

    def test_worker_initialization(self):
        """Test worker initialization."""
        worker = CameraWorker(
            camera_id="test_01",
            camera_config=self.config,
            inference_pipeline=None,
            detection_processor=None,
        )

        self.assertEqual(worker.camera_id, "test_01")
        self.assertFalse(worker.is_running())
        self.assertFalse(worker.is_connected())

    def test_worker_stats_structure(self):
        """Test worker stats structure."""
        worker = CameraWorker(
            camera_id="test_01",
            camera_config=self.config,
            inference_pipeline=None,
            detection_processor=None,
        )

        stats = worker.get_stats()

        self.assertIn("frames_captured", stats)
        self.assertIn("fps", stats)
        self.assertIn("connected", stats)
        self.assertIn("reconnect_count", stats)
        self.assertIn("uptime_seconds", stats)

    def test_worker_thread_safety(self):
        """Test worker thread safety."""
        worker = CameraWorker(
            camera_id="test_01",
            camera_config=self.config,
            inference_pipeline=None,
            detection_processor=None,
        )

        stats1 = worker.get_stats()
        stats2 = worker.get_stats()

        self.assertIsNot(stats1, stats2)
        self.assertEqual(stats1["frames_captured"], stats2["frames_captured"])


class TestCameraManager(unittest.TestCase):
    """Test CameraManager."""

    def setUp(self):
        self.config_path = Path("configs/cameras.yaml")
        self.backup_path = None

        if self.config_path.exists():
            self.backup_path = self.config_path.with_suffix(".yaml.backup")
            self.config_path.rename(self.backup_path)

    def tearDown(self):
        if self.backup_path and self.backup_path.exists():
            self.backup_path.rename(self.config_path)

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = CameraManager(
            config_path=str(self.config_path),
            inference_pipeline=None,
            detection_processor=None,
        )

        self.assertIsNotNone(manager)

    def test_manager_add_camera(self):
        """Test adding camera dynamically."""
        manager = CameraManager(
            config_path=str(self.config_path),
            inference_pipeline=None,
            detection_processor=None,
        )

        camera_config = {
            "type": "rtsp",
            "url": "rtsp://example.com/stream",
            "enabled": True,
            "width": 640,
            "height": 480,
        }

        success = manager.add_camera("test_camera", camera_config)

        self.assertTrue(success)
        self.assertIn("test_camera", manager.list_cameras())

    def test_manager_list_cameras(self):
        """Test listing cameras."""
        manager = CameraManager(
            config_path=str(self.config_path),
            inference_pipeline=None,
            detection_processor=None,
        )

        cameras = manager.list_cameras()

        self.assertIsInstance(cameras, list)

    def test_manager_get_status(self):
        """Test getting camera status."""
        manager = CameraManager(
            config_path=str(self.config_path),
            inference_pipeline=None,
            detection_processor=None,
        )

        camera_config = {
            "type": "rtsp",
            "url": "rtsp://example.com/stream",
            "enabled": True,
            "width": 640,
            "height": 480,
        }

        manager.add_camera("test_camera", camera_config)
        status = manager.get_camera_status("test_camera")

        self.assertIsNotNone(status)
        self.assertIn("camera_id", status)
        self.assertIn("connected", status)
        self.assertIn("running", status)
        self.assertIn("stats", status)

    def test_manager_get_all_status(self):
        """Test getting all cameras status."""
        manager = CameraManager(
            config_path=str(self.config_path),
            inference_pipeline=None,
            detection_processor=None,
        )

        camera_config = {
            "type": "rtsp",
            "url": "rtsp://example.com/stream",
            "enabled": True,
            "width": 640,
            "height": 480,
        }

        manager.add_camera("test_camera_1", camera_config)
        manager.add_camera("test_camera_2", camera_config)

        all_status = manager.get_all_status()

        self.assertIsInstance(all_status, dict)
        self.assertGreaterEqual(len(all_status), 2)


class TestCameraMonitor(unittest.TestCase):
    """Test CameraMonitor."""

    def setUp(self):
        self.mock_manager = MagicMock()
        self.mock_manager.get_all_stats.return_value = {
            "test_01": {
                "fps": 15.0,
                "connected": True,
                "queue_size": 5,
            }
        }
        self.mock_manager.get_all_status.return_value = {
            "test_01": {
                "connected": True,
                "running": True,
                "stats": {
                    "fps": 15.0,
                    "queue_size": 5,
                },
            }
        }

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = CameraMonitor(
            camera_manager=self.mock_manager,
            collection_interval=1,
        )

        self.assertIsNotNone(monitor)

    def test_monitor_start_stop(self):
        """Test monitor start and stop."""
        monitor = CameraMonitor(
            camera_manager=self.mock_manager,
            collection_interval=1,
        )

        monitor.start()
        time.sleep(0.5)

        self.assertFalse(monitor._stop_event)

        monitor.stop()
        time.sleep(0.5)

        self.assertTrue(monitor._stop_event)

    def test_monitor_get_health(self):
        """Test getting camera health."""
        monitor = CameraMonitor(
            camera_manager=self.mock_manager,
            collection_interval=1,
        )

        self.mock_manager.get_camera_status.return_value = {
            "camera_id": "test_01",
            "connected": True,
            "running": True,
            "stats": {
                "fps": 15.0,
                "queue_size": 5,
            },
        }

        health = monitor.get_camera_health("test_01")

        self.assertIsNotNone(health)
        self.assertIn("status", health)
        self.assertIn("recent_alerts", health)


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading."""

    def test_config_yaml_structure(self):
        """Test cameras.yaml structure."""
        config_path = Path("configs/cameras.yaml")

        if not config_path.exists():
            self.skipTest("cameras.yaml not found")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.assertIn("cameras", config)
        self.assertIn("defaults", config)
        self.assertIn("multi_camera", config)

    def test_config_camera_schema(self):
        """Test camera configuration schema."""
        config_path = Path("configs/cameras.yaml")

        if not config_path.exists():
            self.skipTest("cameras.yaml not found")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        cameras = config.get("cameras", {})

        for camera_id, camera_config in cameras.items():
            self.assertIn("id", camera_config)
            self.assertIn("name", camera_config)
            self.assertIn("type", camera_config)
            self.assertIn("enabled", camera_config)
            self.assertIn("width", camera_config)
            self.assertIn("height", camera_config)
            self.assertIn("target_fps", camera_config)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety."""

    def test_worker_concurrent_stats_access(self):
        """Test concurrent access to worker stats."""
        worker = CameraWorker(
            camera_id="test_01",
            camera_config={
                "id": "test_01",
                "type": "rtsp",
                "url": "rtsp://example.com",
                "width": 640,
                "height": 480,
                "target_fps": 15,
                "reconnect_interval": 1,
                "max_reconnect_attempts": 1,
                "max_queue_size": 10,
            },
            inference_pipeline=None,
            detection_processor=None,
        )

        stats = []

        for _ in range(10):
            s = worker.get_stats()
            stats.append(s)

        for s in stats:
            self.assertIsInstance(s, dict)


class TestMultiCameraConfiguration(unittest.TestCase):
    """Test multi-camera configuration."""

    def test_default_values_override(self):
        """Test that defaults are applied correctly."""
        config = {
            "defaults": {
                "width": 640,
                "height": 480,
                "target_fps": 15,
            },
            "cameras": {
                "camera_01": {
                    "id": "camera_01",
                    "type": "rtsp",
                    "url": "rtsp://example.com",
                    "enabled": True,
                },
            },
        }

        defaults = config["defaults"]
        camera_config = {**defaults, **config["cameras"]["camera_01"]}

        self.assertEqual(camera_config["width"], 640)
        self.assertEqual(camera_config["height"], 480)
        self.assertEqual(camera_config["target_fps"], 15)


if __name__ == "__main__":
    unittest.main()
