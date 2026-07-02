from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class GEIDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "data/casia_processed/gei",
        image_size: tuple[int, int] = (64, 128),
        max_classes: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.max_classes = max_classes
        self.max_samples = max_samples

        self.samples = []
        self.label_to_index = {}

        self._scan()

    def _scan(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root_dir}")

        person_dirs = sorted(
            [path for path in self.root_dir.iterdir() if path.is_dir()]
        )

        if self.max_classes is not None:
            person_dirs = person_dirs[: self.max_classes]

        for label_index, person_dir in enumerate(person_dirs):
            self.label_to_index[person_dir.name] = label_index

            for image_path in sorted(person_dir.glob("*.png")):
                self.samples.append(
                    {
                        "path": image_path,
                        "label": label_index,
                        "person_id": person_dir.name,
                    }
                )

        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

        if not self.samples:
            raise RuntimeError(f"No GEI images found in: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = cv2.imread(str(sample["path"]), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Failed to read image: {sample['path']}")

        image = cv2.resize(image, self.image_size)
        image = image.astype("float32") / 255.0

        tensor = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return tensor, label