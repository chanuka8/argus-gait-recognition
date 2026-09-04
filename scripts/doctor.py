#!/usr/bin/env python3
"""ARGUS AI - Backward-compatibility shim for doctor.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.doctor import main

if __name__ == "__main__":
    main()

