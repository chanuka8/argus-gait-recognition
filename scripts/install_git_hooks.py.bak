#!/usr/bin/env python3
"""ARGUS AI - Backward-compatibility shim for scripts/install_git_hooks.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.maintenance.install_git_hooks import *
from tools.maintenance.install_git_hooks import main

if __name__ == "__main__":
    main()
