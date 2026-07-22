import unittest
from pathlib import Path

from monitoring.logging_config import get_logger, init_logging


class TestLoggingConfig(unittest.TestCase):
    def test_loggers_creation(self) -> None:
        init_logging()

        sys_logger = get_logger("system")
        cam_logger = get_logger("camera")
        det_logger = get_logger("detection")
        err_logger = get_logger("error")
        wd_logger = get_logger("watchdog")

        self.assertEqual(sys_logger.name, "ARGUS.System")
        self.assertEqual(cam_logger.name, "ARGUS.Camera")
        self.assertEqual(det_logger.name, "ARGUS.Detection")
        self.assertEqual(err_logger.name, "ARGUS.Error")
        self.assertEqual(wd_logger.name, "ARGUS.Watchdog")

    def test_log_file_creation(self) -> None:
        init_logging()

        logger = get_logger("system")
        logger.info("Test log entry for unit test verification.")

        log_file = Path("outputs/logs/system.log")
        self.assertTrue(log_file.exists())

        content = log_file.read_text(encoding="utf-8")
        self.assertIn("Test log entry for unit test verification.", content)


if __name__ == "__main__":
    unittest.main()
