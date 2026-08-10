from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler, random_split

from evaluation.dataset_split import load_or_create_subject_split
from training.dataset import GEIDataset


class ConditionBalancedSampler(Sampler):
    """Sampler that balances NM (0), BG (1), CL (2) conditions within each batch epoch.

    Groups samples by condition code, then round-robins across conditions
    so that each condition is represented equally in the training stream.
    """

    def __init__(self, dataset: GEIDataset) -> None:
        super().__init__()
        self.dataset = dataset

        self.condition_indices: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for idx, sample in enumerate(dataset.samples):
            cond = sample.get("condition_code", 0)
            if cond in self.condition_indices:
                self.condition_indices[cond].append(idx)
            else:
                self.condition_indices[0].append(idx)

        self.condition_indices = {
            k: v for k, v in self.condition_indices.items() if len(v) > 0
        }

    def __iter__(self):
        shuffled = {}
        for cond, indices in self.condition_indices.items():
            perm = torch.randperm(len(indices)).tolist()
            shuffled[cond] = [indices[i] for i in perm]

        max_len = max(len(v) for v in shuffled.values())
        result = []
        cond_keys = sorted(shuffled.keys())

        for i in range(max_len):
            for cond in cond_keys:
                idx_list = shuffled[cond]
                if len(idx_list) > 0:
                    result.append(idx_list[i % len(idx_list)])

        return iter(result)

    def __len__(self) -> int:
        if not self.condition_indices:
            return 0
        max_len = max(len(v) for v in self.condition_indices.values())
        return max_len * len(self.condition_indices)


def build_dataloaders(
    root_dir: str = "data/casia_processed/gei",
    batch_size: int = 16,
    train_ratio: float = 0.8,
    max_classes: int | None = None,
    max_samples: int | None = None,
    split_config_path: str | None = "configs/subject_split.json",
    condition_balanced: bool = False,
    return_condition: bool = False,
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
                return_condition=return_condition,
            )

            val_dataset = GEIDataset(
                root_dir=root_dir,
                max_classes=max_classes,
                max_samples=max_samples,
                subject_ids=val_subs,
                return_condition=return_condition,
            )
        else:
            dataset = GEIDataset(
                root_dir=root_dir,
                max_classes=max_classes,
                max_samples=max_samples,
                return_condition=return_condition,
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
            return_condition=return_condition,
        )

        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
        )
        train_dataset.label_to_index = dataset.label_to_index

    sampler = None
    shuffle = True
    if condition_balanced and isinstance(train_dataset, GEIDataset):
        sampler = ConditionBalancedSampler(train_dataset)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, train_dataset

