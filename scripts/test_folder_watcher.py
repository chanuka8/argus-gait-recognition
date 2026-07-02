import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from enrollment.folder_watcher import (
    FolderWatcher,
)


def main():

    watcher = FolderWatcher(
        watch_dir="data/new_input",
        poll_interval=2,
    )

    watcher.watch()


if __name__ == "__main__":
    main()