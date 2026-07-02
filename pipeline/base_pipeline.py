from abc import ABC, abstractmethod

from core.logger import setup_logger


class BasePipeline(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = setup_logger(f"ARGUS.Pipeline.{name}")
        self.initialized = False

    def initialize(self) -> None:
        self.logger.info(f"{self.name} pipeline initialized")
        self.initialized = True

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def shutdown(self) -> None:
        self.logger.info(f"{self.name} pipeline shutdown")
        self.initialized = False