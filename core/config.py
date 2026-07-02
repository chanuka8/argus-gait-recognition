from pathlib import Path
import yaml


class Config:
    def __init__(self) -> None:
        self.base_config = {}
        self.load()

    def load(self) -> None:
        config_file = Path("configs/base.yaml")

        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as file:
                self.base_config = yaml.safe_load(file) or {}
        else:
            self.base_config = {}

    def get(self, key: str, default=None):
        return self.base_config.get(key, default)