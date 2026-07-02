import time
from pathlib import Path

from core.logger import setup_logger
from enrollment.enrollment_manager import EnrollmentManager


class FolderWatcher:
    def __init__(
        self,
        watch_dir: str = "data/new_input",
        poll_interval: int = 3,
        stable_wait_seconds: int = 5,
        min_images: int = 5,
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.poll_interval = poll_interval
        self.stable_wait_seconds = stable_wait_seconds
        self.min_images = min_images

        self.logger = setup_logger("ARGUS.FolderWatcher")
        self.enrollment_manager = EnrollmentManager()
        self.processed = set()

    def scan_existing(self) -> None:
        if not self.watch_dir.exists():
            self.watch_dir.mkdir(parents=True, exist_ok=True)

        for folder in self.watch_dir.iterdir():
            if folder.is_dir():
                self.processed.add(folder.name)

    def _count_images(self, folder: Path) -> int:
        total = 0

        for ext in ("*.png", "*.jpg", "*.jpeg"):
            total += len(list(folder.glob(ext)))

        return total

    def _wait_until_folder_ready(self, folder: Path) -> bool:
        last_count = -1
        stable_rounds = 0

        while True:
            current_count = self._count_images(folder)

            if current_count >= self.min_images and current_count == last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0

            if stable_rounds >= 2:
                return True

            last_count = current_count

            self.logger.info(
                f"Waiting for folder copy to finish: "
                f"{folder.name} | images={current_count}"
            )

            time.sleep(self.stable_wait_seconds)

    def watch(self) -> None:
        self.logger.info("Folder watcher started")

        self.scan_existing()

        while True:
            try:
                for folder in self.watch_dir.iterdir():
                    if not folder.is_dir():
                        continue

                    if folder.name in self.processed:
                        continue

                    self.logger.info(f"New folder detected: {folder.name}")

                    self._wait_until_folder_ready(folder)

                    result = self.enrollment_manager.enroll_person(str(folder))

                    self.logger.info(f"Enrollment result: {result}")

                    self.processed.add(folder.name)

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                self.logger.info("Folder watcher stopped")
                break

            except Exception as error:
                self.logger.error(f"Watcher error: {error}")
                time.sleep(self.poll_interval)