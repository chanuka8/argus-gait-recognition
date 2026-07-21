from torch.utils.data import DataLoader, random_split

from training.dataset import GEIDataset


class AugmentedSubset:
    def __init__(self, subset, augment: bool = True):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        orig_augment = self.dataset.augment
        self.dataset.augment = self.augment
        try:
            return self.dataset[self.indices[idx]]
        finally:
            self.dataset.augment = orig_augment


def build_dataloaders(
    root_dir: str = "data/casia_processed/gei",
    batch_size: int = 16,
    train_ratio: float = 0.8,
    max_classes: int | None = None,
    max_samples: int | None = None,
):
    dataset = GEIDataset(
        root_dir=root_dir,
        max_classes=max_classes,
        max_samples=max_samples,
    )

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_sub, val_sub = random_split(
        dataset,
        [train_size, val_size],
    )

    train_dataset = AugmentedSubset(train_sub, augment=True)
    val_dataset = AugmentedSubset(val_sub, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, dataset