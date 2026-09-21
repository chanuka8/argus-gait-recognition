#!/usr/bin/env python3
"""ARGUS AI - Export ByGaitLight PyTorch model to ONNX CLI Tool."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_platform.models.export.bygait_onnx import main

if __name__ == "__main__":
    main()
