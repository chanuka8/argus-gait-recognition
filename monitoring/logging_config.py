import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml


_DEFAULT_LOG_DIR = "outputs/logs"
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5
_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

_LOG_CHANNELS = {
    "ARGUS.System": "system.log",
    "ARGUS.Camera": "camera.log",
    "ARGUS.Detection": "detection.log",
    "ARGUS.Error": "error.log",
    "ARGUS.Watchdog": "watchdog.log",
}

_initialized = False


def _load_logging_config() -> dict:
    config_path = Path("configs/system.yaml")

    defaults = {
        "log_dir": _DEFAULT_LOG_DIR,
        "max_bytes": _DEFAULT_MAX_BYTES,
        "backup_count": _DEFAULT_BACKUP_COUNT,
        "level": _DEFAULT_LEVEL,
        "format": _DEFAULT_FORMAT,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except Exception:
        return defaults

    section = data.get("logging", {})

    if not isinstance(section, dict):
        return defaults

    for key, default_value in defaults.items():
        defaults[key] = section.get(key, default_value)

    return defaults


def init_logging() -> None:
    global _initialized

    if _initialized:
        return

    config = _load_logging_config()
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(config["level"]).upper(), logging.INFO)
    formatter = logging.Formatter(config["format"])

    for logger_name, filename in _LOG_CHANNELS.items():
        logger = logging.getLogger(logger_name)

        if logger.handlers:
            continue

        logger.setLevel(level)
        logger.propagate = False

        file_handler = RotatingFileHandler(
            log_dir / filename,
            maxBytes=int(config["max_bytes"]),
            backupCount=int(config["backup_count"]),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    error_logger = logging.getLogger("ARGUS.Error")
    error_handler = error_logger.handlers[0] if error_logger.handlers else None

    if error_handler is not None:
        for logger_name in _LOG_CHANNELS:
            if logger_name == "ARGUS.Error":
                continue

            other_logger = logging.getLogger(logger_name)
            error_filter_handler = RotatingFileHandler(
                log_dir / "error.log",
                maxBytes=int(config["max_bytes"]),
                backupCount=int(config["backup_count"]),
                encoding="utf-8",
            )
            error_filter_handler.setLevel(logging.WARNING)
            error_filter_handler.setFormatter(formatter)
            other_logger.addHandler(error_filter_handler)

    _initialized = True


def get_logger(channel: str) -> logging.Logger:
    init_logging()

    name_map = {
        "system": "ARGUS.System",
        "camera": "ARGUS.Camera",
        "detection": "ARGUS.Detection",
        "error": "ARGUS.Error",
        "watchdog": "ARGUS.Watchdog",
    }

    logger_name = name_map.get(channel.lower(), f"ARGUS.{channel}")

    return logging.getLogger(logger_name)
