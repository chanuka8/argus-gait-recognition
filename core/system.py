from core.boot import BootManager
from core.logger import setup_logger
from core.orchestrator import Orchestrator
from deployment.backend_summary import BackendStartupSummary
from deployment.shutdown_manager import get_shutdown_manager
from deployment.startup_validator import DeploymentStartupValidator


class ArgusSystem:
    def __init__(self, mode: str = "inference") -> None:
        self.mode = mode
        self.logger = setup_logger("ARGUS.System")
        self.boot_manager = BootManager()
        self.orchestrator = Orchestrator(mode=self.mode)
        self.shutdown_manager = get_shutdown_manager()

    def start(self) -> None:
        self.shutdown_manager.register_signal_handlers()

        self.logger.info(f"Starting ARGUS system in {self.mode} mode")

        validator = DeploymentStartupValidator()
        startup_summary = validator.validate_startup(raise_on_failure=True)
        self.logger.info(f"Startup health validation status: {startup_summary['status']}")

        backend = startup_summary.get("backend")
        if backend is not None:
            summary_obj = BackendStartupSummary(
                backend=backend,
                startup_status=startup_summary.get("status", "READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING"),
            )
            summary_obj.emit(print_cli=True)

        config = self.boot_manager.boot()

        self.logger.info(f"Loaded project: {config.get('project_name', 'ARGUS')}")

        self.orchestrator.start()

        self.logger.info("System startup and execution completed")
