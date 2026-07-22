"""
Unit tests for Phase 4 streaming optimization components.
"""

import time
import unittest

from streaming.camera_scheduler import CameraScheduler
from streaming.load_balancer import CameraLoadBalancer
from streaming.performance_optimizer import PerformanceOptimizer
from streaming.worker_pool import CameraWorkerPool


class TestCameraScheduler(unittest.TestCase):
    """Test CameraScheduler."""

    def test_scheduler_priority_and_scheduling(self):
        scheduler = CameraScheduler(default_priority=5)
        scheduler.register_camera("cam_low", priority=2)
        scheduler.register_camera("cam_high", priority=9)

        self.assertEqual(scheduler.get_priority("cam_low"), 2)
        self.assertEqual(scheduler.get_priority("cam_high"), 9)

        next_cam = scheduler.get_next_camera(["cam_low", "cam_high"])
        self.assertEqual(next_cam, "cam_high")

    def test_starvation_prevention(self):
        scheduler = CameraScheduler(starvation_threshold_seconds=0.01)
        scheduler.register_camera("cam1", priority=1)
        scheduler.register_camera("cam2", priority=10)

        # Schedule cam2 first
        scheduler.get_next_camera(["cam1", "cam2"])
        time.sleep(0.02)

        # cam1 should get starvation boost and be selected
        next_cam = scheduler.get_next_camera(["cam1", "cam2"])
        self.assertEqual(next_cam, "cam1")

    def test_dynamic_poll_interval(self):
        scheduler = CameraScheduler(min_poll_interval=0.01, max_poll_interval=0.1)
        low_poll = scheduler.calculate_dynamic_poll_interval(0.0)
        high_poll = scheduler.calculate_dynamic_poll_interval(1.0)

        self.assertEqual(low_poll, 0.01)
        self.assertEqual(high_poll, 0.1)


class TestCameraLoadBalancer(unittest.TestCase):
    """Test CameraLoadBalancer."""

    def test_assign_and_migrate(self):
        lb = CameraLoadBalancer(max_cameras_per_worker=5)
        lb.register_worker("w1")
        lb.register_worker("w2")

        w = lb.assign_camera("cam1")
        self.assertIsNotNone(w)

        success = lb.migrate_camera("cam1", "w2")
        self.assertTrue(success)
        stats = lb.get_assignment_stats()
        self.assertIn("cam1", stats["assignments"]["w2"])

    def test_auto_rebalance(self):
        lb = CameraLoadBalancer(imbalance_threshold=0.1)
        lb.register_worker("w1")
        lb.register_worker("w2")

        lb.assign_camera("cam1")
        lb.assign_camera("cam2")

        # Manually force both on w1
        lb.migrate_camera("cam2", "w1")
        rebalanced = lb.check_and_rebalance()
        self.assertIn("cam2", rebalanced)


class TestCameraWorkerPool(unittest.TestCase):
    """Test CameraWorkerPool."""

    def test_pool_management(self):
        pool = CameraWorkerPool(min_workers=1, max_workers=3)
        config = {
            "id": "cam1",
            "type": "rtsp",
            "url": "rtsp://example.com",
            "width": 640,
            "height": 480,
            "target_fps": 15,
            "reconnect_interval": 1,
            "max_reconnect_attempts": 1,
            "max_queue_size": 10,
        }

        w = pool.add_worker_for_camera("cam1", config)
        self.assertIsNotNone(w)

        health = pool.get_pool_health()
        self.assertEqual(health["total_workers"], 1)

        pool.remove_worker("cam1")
        self.assertEqual(pool.get_pool_health()["total_workers"], 0)


class TestPerformanceOptimizer(unittest.TestCase):
    """Test PerformanceOptimizer."""

    def test_profile_switching(self):
        opt = PerformanceOptimizer(profile_name="balanced")
        self.assertEqual(opt.get_optimal_queue_size(), 10)

        opt.set_profile("low_latency")
        self.assertEqual(opt.get_optimal_queue_size(), 3)
        self.assertEqual(opt.get_gpu_batch_size(), 1)

    def test_adaptive_skip_frame(self):
        opt = PerformanceOptimizer(profile_name="low_latency")
        # Near capacity queue or high latency -> skip frame
        self.assertTrue(opt.should_skip_frame(queue_fullness=0.9, latency_seconds=0.01))
        self.assertFalse(opt.should_skip_frame(queue_fullness=0.2, latency_seconds=0.01))


if __name__ == "__main__":
    unittest.main()
