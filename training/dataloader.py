from pathlib import Path

from torch.utils.data import DataLoader, random_split

from evaluation.dataset_split import load_or_create_subject_split
from training.dataset import GEIDataset


def build_dataloaders(
    root_dir: str = "data/casia_processed/gei",
    batch_size: int = 16,
    train_ratio: float = 0.8,
    max_classes: int | None = None,
    max_samples: int | None = None,
    split_config_path: str | None = "configs/subject_split.json",
):
    split_path = Path(split_config_path) if split_config_path else None

    if split_path and split_path.exists():
        manifest = load_or_create_subject_split(config_path=str(split_path), data_dir=root_dir)
        train_subs = manifest["train_subjects"]
        val_subs = manifest["val_subjects"]
        root_p = Path(root_dir)

        has_train = any((root_p / s).exists() for s in train_subs)
        has_val = any((root_p / s).exists() for s in val_subs)

        if has_train and has_val:
            train_dataset = GEIDataset(
                root_dir=root_dir,
                max_classes=max_classes,
                max_samples=max_samples,
                subject_ids=train_subs,
            )

            val_dataset = GEIDataset(
                root_dir=root_dir,
                max_classes=max_classes,
                max_samples=max_samples,
                subject_ids=val_subs,
            )
        else:
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
            train_dataset.label_to_index = dataset.label_to_index
    else:
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
        train_dataset.label_to_index = dataset.label_to_index

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

    return train_loader, val_loader, train_dataset

