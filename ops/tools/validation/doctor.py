#!/usr/bin/env python3
"""ARGUS AI - System Health & Environment Doctor CLI Tool."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.deployment.doctor import main

if __name__ == "__main__":
    main()
