import logging
from pathlib import Path

from security_layer.credentials import sanitize_rtsp_url


class SensitiveDataFilter(logging.Filter):
    """Logging filter that redacts RTSP credentials from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_rtsp_url(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(sanitize_rtsp_url(arg) if isinstance(arg, str) else arg for arg in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: sanitize_rtsp_url(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


def setup_logger(name: str = "ARGUS") -> logging.Logger:
    log_dir = Path("outputs/logs/system")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")

    sensitive_filter = SensitiveDataFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sensitive_filter)

    file_handler = logging.FileHandler(log_dir / "argus.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(sensitive_filter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
