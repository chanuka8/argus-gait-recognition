import platform
import sys
from pathlib import Path

import psutil


class SystemMonitor:
    def snapshot(self) -> dict:
        return {
            "python_version": sys.version.split()[0],
            "os": platform.platform(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_total_gb": round(
                psutil.virtual_memory().total / (1024 ** 3),
                2,
            ),
            "memory_percent": psutil.virtual_memory().percent,
            "project_root": str(Path.cwd()),
        }

    def format_snapshot(self) -> str:
        data = self.snapshot()

        return (
            f"Python={data['python_version']} | "
            f"OS={data['os']} | "
            f"CPU={data['cpu_percent']}% | "
            f"RAM={data['memory_percent']}% | "
            f"Root={data['project_root']}"
        )