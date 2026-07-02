import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import yaml


def check_path(path: str) -> str:
    return "OK" if Path(path).exists() else "MISSING"


def main() -> None:
    print("=" * 60)
    print("ARGUS SYSTEM CHECK")
    print("=" * 60)

    print(f"Python version     : {sys.version.split()[0]}")
    print(f"Operating system   : {platform.platform()}")
    print(f"CPU cores          : {psutil.cpu_count(logical=True)}")
    print(f"RAM usage          : {psutil.virtual_memory().percent}%")

    print("-" * 60)
    print("LIBRARIES")
    print("-" * 60)

    print(f"NumPy              : {np.__version__}")
    print(f"OpenCV             : {cv2.__version__}")
    print(f"PyTorch            : {torch.__version__}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    print(f"PyYAML             : {yaml.__version__}")

    print("-" * 60)
    print("PROJECT FILES")
    print("-" * 60)

    required_files = [
        "VERSION",
        "main.py",
        "core/system.py",
        "core/orchestrator.py",
        "core/boot.py",
        "configs/base.yaml",
        "configs/inference.yaml",
        "requirements.txt",
    ]

    for item in required_files:
        print(f"{item:<30}: {check_path(item)}")

    print("-" * 60)
    print("PROJECT DIRECTORIES")
    print("-" * 60)

    required_dirs = [
        "core",
        "configs",
        "events",
        "models",
        "preprocessing",
        "pipeline",
        "training",
        "evaluation",
        "storage",
        "outputs",
        "data",
    ]

    for item in required_dirs:
        print(f"{item:<30}: {check_path(item)}")

    print("=" * 60)
    print("System check completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()