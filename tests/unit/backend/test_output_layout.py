import tempfile
import unittest
from pathlib import Path

from evaluation.visualizer import EvaluationVisualizer
from intelligence.camera_topology_learner import CameraTopologyLearner
from intelligence.crowd_intelligence_system import CrowdIntelligenceSystem
from intelligence.missing_person_workflow import MissingPersonWorkflow
from monitoring.camera_monitor import CameraMonitor
from monitoring.logging_config import _DEFAULT_LOG_DIR, get_logger, init_logging
from security_layer.security_logger import SecurityLogger
from storage.evidence_manager import EvidenceManager
from storage.lineage_tracker import LineageTracker
from training.callbacks import TrainingLogger
from utils.alert_manager import AlertManager
from utils.detection_reporter import DetectionReporter, load_reporting_config
from utils.event_logger import EventLogger


class TestOutputLayoutHierarchy(unittest.TestCase):
    def test_default_paths_config(self) -> None:
        self.assertEqual(_DEFAULT_LOG_DIR, "outputs/logs/system")
        cfg = load_reporting_config()
        self.assertEqual(cfg["output_dir"], "outputs/media/detections")
        self.assertEqual(cfg["snapshot_dir"], "outputs/media/detections/snapshots")

    def test_camera_stats_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_dir = Path(tmpdir) / "outputs" / "monitoring" / "camera_stats"
            monitor = CameraMonitor(camera_manager=None, stats_dir=str(stats_dir))
            self.assertTrue(monitor.stats_dir.exists())

    def test_detection_media_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "outputs" / "media" / "detections"
            snap_dir = out_dir / "snapshots"
            cfg = {
                "enabled": True,
                "output_dir": str(out_dir),
                "snapshot_dir": str(snap_dir),
                "save_jsonl": True,
                "save_csv": True,
                "save_snapshots": True,
                "cooldown_seconds": 0,
            }
            _reporter = DetectionReporter(config=cfg, source_mode="test")
            self.assertTrue(out_dir.exists())
            self.assertTrue(snap_dir.exists())

            ev_mgr = EvidenceManager(base_evidence_dir=str(out_dir))
            self.assertTrue(ev_mgr.base_dir.exists())

    def test_evaluation_reports_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            charts_dir = Path(tmpdir) / "outputs" / "reports" / "evaluation" / "charts"
            viz = EvaluationVisualizer(output_dir=str(charts_dir))
            self.assertTrue(viz.output_dir.exists())

    def test_benchmark_reports_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "outputs" / "reports" / "benchmark" / "training_log.txt"
            t_logger = TrainingLogger(log_file=str(log_file))
            t_logger.write("epoch 1 loss 0.05")
            self.assertTrue(log_file.exists())

    def test_system_and_camera_log_output(self) -> None:
        init_logging()
        sys_logger = get_logger("system")
        cam_logger = get_logger("camera")
        self.assertEqual(sys_logger.name, "ARGUS.System")
        self.assertEqual(cam_logger.name, "ARGUS.Camera")

    def test_security_log_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "outputs" / "logs" / "security" / "security_events.csv"
            sec_logger = SecurityLogger(log_file=str(log_file))
            sec_logger.log(1, "person_1", 0.92, "HIGH", "ALLOW", "cam_01")
            self.assertTrue(log_file.exists())

    def test_event_log_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_file = Path(tmpdir) / "outputs" / "logs" / "events" / "recognition_log.csv"
            alert_file = Path(tmpdir) / "outputs" / "logs" / "events" / "alerts.csv"

            evt_logger = EventLogger(log_file=str(event_file))
            evt_logger.log(1, "person_1", 0.95, "cam_01")
            self.assertTrue(event_file.exists())

            alt_mgr = AlertManager(alert_file=str(alert_file))
            alt_mgr.create_alert(1, "p1", 0.90, "HIGH", "cam_1")
            self.assertTrue(alert_file.exists())

    def test_watchlist_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            w_dir = Path(tmpdir) / "outputs" / "watchlist"
            workflow = MissingPersonWorkflow(output_dir=str(w_dir))
            workflow.register_target("target_001")
            match_evt = workflow.process_match("target_001", 0.95, "cam_01")
            self.assertIsNotNone(match_evt)
            export_file = workflow.export_watchlist_events()
            self.assertTrue(export_file.exists())

    def test_explainable_and_timeline_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lineage_file = Path(tmpdir) / "outputs" / "reports" / "explainable" / "lineage.json"
            tracker = LineageTracker(output_file=str(lineage_file))
            tracker.add_record("MODEL_LOAD", "best_model.pth")
            self.assertTrue(lineage_file.exists())

    def test_exports_output(self) -> None:
        learner = CameraTopologyLearner()
        self.assertIn("outputs/reports/exports", learner.export_path)

        crowd_sys = CrowdIntelligenceSystem()
        self.assertIn("outputs/reports/exports", crowd_sys.topology_learner.export_path)

    def test_old_runtime_paths_inactive(self) -> None:
        old_defaults = [
            "outputs/camera_stats",
            "outputs/detection_reports",
            "outputs/eval_reports",
            "outputs/evaluation_charts",
            "outputs/events",
            "outputs/security_logs",
        ]
        cfg = load_reporting_config()
        for old in old_defaults:
            self.assertNotEqual(cfg["output_dir"], old)

    def test_disabled_features_preserve_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "should_not_exist"
            cfg = {"enabled": False, "output_dir": str(out_dir)}
            reporter = DetectionReporter(config=cfg)
            self.assertFalse(reporter._enabled)


if __name__ == "__main__":
    unittest.main()
