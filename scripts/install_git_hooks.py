"""Installs Git pre-commit hooks for automated ARGUS AI README synchronization.

Usage:
    python scripts/install_git_hooks.py
"""

import os
import stat
import sys
from pathlib import Path

HOOK_CONTENT = """#!/bin/sh
# ARGUS AI Automatic README Synchronization Pre-Commit Hook

echo "[pre-commit] Automatically synchronizing ARGUS AI package documentation..."

# Detect Python interpreter
if [ -f "./venv/Scripts/python.exe" ]; then
    PYTHON_CMD="./venv/Scripts/python.exe"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python executable not found. Cannot synchronize README files."
    exit 1
fi

# Execute sync script
$PYTHON_CMD scripts/sync_folder_readmes.py
SYNC_EXIT=$?

if [ $SYNC_EXIT -ne 0 ]; then
    echo "[ERROR] README synchronization failed. Aborting commit."
    exit 1
fi

# Automatically stage updated README files
git add README.md 2>/dev/null || true
git add */README.md 2>/dev/null || true
git add docs/README_INDEX.md 2>/dev/null || true

echo "[pre-commit] README documentation synchronized and staged automatically."
exit 0
"""


def install_pre_commit_hook(root_dir: Path) -> bool:
    git_hooks_dir = root_dir / ".git" / "hooks"
    if not git_hooks_dir.exists():
        print(f"[ERROR] .git/hooks directory not found at {git_hooks_dir}.")
        return False

    pre_commit_path = git_hooks_dir / "pre-commit"
    pre_commit_path.write_text(HOOK_CONTENT, encoding="utf-8")

    try:
        st = os.stat(pre_commit_path)
        os.chmod(pre_commit_path, st.st_mode | stat.S_IEXEC)
    except Exception:
        pass

    print(f"[SUCCESS] Pre-commit hook installed successfully at {pre_commit_path}")
    return True


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    if install_pre_commit_hook(root_dir):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
