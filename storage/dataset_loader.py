from pathlib import Path


class DatasetLoader:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)

    def subject_folders(self):
        return sorted(
            [
                p for p in self.dataset_root.iterdir()
                if p.is_dir()
            ]
        )

    def image_files(self):
        return sorted(
            self.dataset_root.rglob("*.png")
        )

    def count_images(self) -> int:
        return len(self.image_files())

    def count_subjects(self) -> int:
        return len(self.subject_folders())