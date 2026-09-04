#!/usr/bin/env python3
"""ARGUS AI - Backward-compatibility shim for export_silhouette_unet_onnx.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.export.silhouette_unet_onnx import *
from models.export.silhouette_unet_onnx import export_and_validate_onnx

if __name__ == "__main__":
    success, msg = export_and_validate_onnx()
    print(f"ONNX Export Result: {msg}")
    sys.exit(0 if success else 1)
