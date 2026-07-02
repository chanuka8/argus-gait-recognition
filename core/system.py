from core.boot import BootManager
from core.logger import setup_logger
from core.orchestrator import Orchestrator


class ArgusSystem:
    def __init__(self, mode: str = "inference") -> None:
        self.mode = mode
        self.logger = setup_logger("ARGUS.System")
        self.boot_manager = BootManager()
        self.orchestrator = Orchestrator(mode=self.mode)

    def start(self) -> None:
        self.logger.info(f"Starting ARGUS system in {self.mode} mode")

        config = self.boot_manager.boot()

        self.logger.info(
            f"Loaded project: {config.get('project_name', 'ARGUS')}"
        )

        self.orchestrator.start()

        self.logger.info("System startup and execution completed")