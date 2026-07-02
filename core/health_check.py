from pathlib import Path

from core.exceptions import BootError


class HealthCheck:
    REQUIRED_DIRECTORIES = [
        "core",
        "configs",
        "events",
        "models",
        "pipeline",
        "preprocessing",
        "training",
        "evaluation",
        "storage",
        "outputs",
        "data",
    ]

    REQUIRED_FILES = [
        "VERSION",
        "configs/base.yaml",
        "configs/inference.yaml",
        "configs/train.yaml",
        "requirements.txt",
    ]

    def check_directories(self) -> list[str]:
        missing = []

        for directory in self.REQUIRED_DIRECTORIES:
            if not Path(directory).exists():
                missing.append(directory)

        return missing

    def check_files(self) -> list[str]:
        missing = []

        for file_path in self.REQUIRED_FILES:
            if not Path(file_path).exists():
                missing.append(file_path)

        return missing

    def run(self) -> dict:
        missing_dirs = self.check_directories()
        missing_files = self.check_files()

        healthy = not missing_dirs and not missing_files

        return {
            "healthy": healthy,
            "missing_directories": missing_dirs,
            "missing_files": missing_files,
        }

    def assert_healthy(self) -> None:
        result = self.run()

        if not result["healthy"]:
            raise BootError(
                "ARGUS health check failed. "
                f"Missing directories: {result['missing_directories']}. "
                f"Missing files: {result['missing_files']}."
            )