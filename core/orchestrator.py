from core.context import SystemContext
from core.logger import setup_logger
from core.system_monitor import SystemMonitor


class Orchestrator:
    def __init__(self, mode: str = "inference") -> None:
        self.mode = mode
        self.logger = setup_logger("ARGUS.Orchestrator")
        self.context = SystemContext()
        self.monitor = SystemMonitor()

    def initialize(self) -> None:
        self.logger.info("Initializing orchestrator")

        self.context.set("mode", self.mode)
        self.context.set("system_status", "initialized")

        self.logger.info(f"Runtime mode set to: {self.mode}")

    def start(self) -> None:
        self.initialize()

        self.logger.info("Starting ARGUS runtime controller")
        self.logger.info(self.monitor.format_snapshot())

        if self.mode == "inference":
            self._run_inference_mode()
        elif self.mode == "preprocess":
            self._run_preprocess_mode()
        elif self.mode == "train":
            self._run_train_mode()
        elif self.mode == "evaluate":
            self._run_evaluate_mode()
        else:
            self.logger.warning(f"Unknown mode: {self.mode}")

        self.context.set("system_status", "completed")
        self.logger.info("ARGUS runtime controller completed")

    def _run_inference_mode(self) -> None:
        self.logger.info("Inference mode selected")
        self.logger.info("Live inference pipeline will be connected in Step 6")

    def _run_preprocess_mode(self) -> None:
        self.logger.info("Preprocess mode selected")
        self.logger.info("Dataset preprocessing pipeline will be connected in Step 4")

    def _run_train_mode(self) -> None:
        self.logger.info("Train mode selected")
        self.logger.info("Training pipeline will be connected in Step 5")

    def _run_evaluate_mode(self) -> None:
        self.logger.info("Evaluate mode selected")
        self.logger.info("Evaluation pipeline will be connected in Step 5")