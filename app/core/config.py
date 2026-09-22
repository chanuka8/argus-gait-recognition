from pathlib import Path

import yaml

from app.core.paths import get_config_path, resolve_app_path


class Config:
    def __init__(self, config_file: str | Path | None = None) -> None:
        self.base_config = {}
        self.config_file = resolve_app_path(config_file) if config_file is not None else get_config_path("base.yaml")
        self.load()

    def load(self) -> None:
        if self.config_file.exists():
            with open(self.config_file, "r", encoding="utf-8") as file:
                self.base_config = yaml.safe_load(file) or {}
        else:
            self.base_config = {}

    def get(self, key: str, default=None):
        return self.base_config.get(key, default)
