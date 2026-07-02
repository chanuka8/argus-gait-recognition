from core.config import Config
from core.health_check import HealthCheck
from core.logger import setup_logger
from core.system_monitor import SystemMonitor


class BootManager:
    def __init__(self) -> None:
        self.logger = setup_logger("ARGUS.Boot")
        self.config = Config()
        self.health_check = HealthCheck()
        self.monitor = SystemMonitor()

    def boot(self) -> Config:
        self.logger.info("Starting boot sequence")

        self.health_check.assert_healthy()

        self.logger.info("Health check passed")
        self.logger.info(self.monitor.format_snapshot())
        self.logger.info("Boot sequence completed")

        return self.config