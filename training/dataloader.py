from torch.utils.data import DataLoader, random_split

from training.dataset import GEIDataset


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

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

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
