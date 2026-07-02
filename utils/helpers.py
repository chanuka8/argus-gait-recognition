from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now().isoformat()


def safe_name(value: str) -> str:
    return value.strip().replace(" ", "_").lower()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()


def read_text(path: str | Path, default: str = "") -> str:
    file_path = Path(path)

    if not file_path.exists():
        return default

    return file_path.read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")