import os
import stat
import subprocess
import sys
from pathlib import Path

HOOK_CONTENT = """#!/bin/sh
# ARGUS AI Automatic README Synchronization Pre-Commit Hook

echo "[pre-commit] Automatically synchronizing ARGUS AI package documentation..."

# Detect Python interpreter (prefer project-local .venv standard)
if [ -f "./.venv/Scripts/python.exe" ]; then
    PYTHON_CMD="./.venv/Scripts/python.exe"
elif [ -f "./.venv/bin/python" ]; then
    PYTHON_CMD="./.venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/Scripts/python.exe"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif [ -f "./venv/Scripts/python.exe" ]; then
    PYTHON_CMD="./venv/Scripts/python.exe"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python"
elif command -v py >/dev/null 2>&1; then
    PYTHON_CMD="py -3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python executable not found. Cannot synchronize README files."
    exit 1
fi

# Execute sync script
$PYTHON_CMD ops/tools/maintenance/sync_folder_readmes.py
SYNC_EXIT=$?

if [ $SYNC_EXIT -ne 0 ]; then
    echo "[ERROR] README synchronization failed. Aborting commit."
    exit 1
fi

# Stage only the README files the sync script actually modified, never an
# unrelated top-level folder's README that merely happens to exist (a broad
# `git add */README.md` would silently pick those up too).
CHANGED_READMES=$(git diff --name-only -- '*/README.md' 'README.md' 'docs/README_INDEX.md')
if [ -n "$CHANGED_READMES" ]; then
    echo "$CHANGED_READMES" | while IFS= read -r f; do
        git add -- "$f"
    done
fi

echo "[pre-commit] README documentation synchronized and staged automatically."
exit 0
"""


def _resolve_git_hooks_dir(root_dir: Path) -> Path:
    """Resolve the real hooks directory, including inside a linked git worktree.

    A worktree's `.git` is a file pointing at `<main-repo>/.git/worktrees/<name>`,
    not a directory - hooks always live under the shared common git dir, never
    per-worktree. `git rev-parse --git-common-dir` resolves that correctly in
    both a normal checkout and a worktree; falling back to `.git/hooks` keeps
    this working even if git itself isn't on PATH.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        common_dir = Path(result.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (root_dir / common_dir).resolve()
        return common_dir / "hooks"
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        return root_dir / ".git" / "hooks"


def install_pre_commit_hook(root_dir: Path) -> bool:
    git_hooks_dir = _resolve_git_hooks_dir(root_dir)
    if not git_hooks_dir.exists():
        print(f"[ERROR] .git/hooks directory not found at {git_hooks_dir}.")
        return False

    pre_commit_path = git_hooks_dir / "pre-commit"
    pre_commit_path.write_text(HOOK_CONTENT, encoding="utf-8")

    try:
        st = os.stat(pre_commit_path)
        os.chmod(pre_commit_path, st.st_mode | stat.S_IEXEC)
    except OSError:
        pass

    print(f"[SUCCESS] Pre-commit hook installed successfully at {pre_commit_path}")
    return True


def main() -> int:
    root_dir = Path(__file__).resolve().parents[3]
    if install_pre_commit_hook(root_dir):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
