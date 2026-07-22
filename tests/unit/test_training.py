import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.dataloader import build_dataloaders


def test_gei_dataloader() -> None:
    dataset_dir = ROOT / "data" / "casia_processed" / "gei"
    if not dataset_dir.exists() or not any(dataset_dir.rglob("*.png")):
        pytest.skip(f"Processed CASIA GEI dataset not found in {dataset_dir}")

    train_loader, val_loader, dataset = build_dataloaders(
        root_dir="data/casia_processed/gei",
        batch_size=8,
    )

    images, labels = next(iter(train_loader))

    assert images.ndim == 4
    assert images.shape[1:] == (1, 128, 64)
    assert labels.ndim == 1

    print("PASS - Dataset size:", len(dataset))
    print("PASS - Train batches:", len(train_loader))
    print("PASS - Val batches:", len(val_loader))
    print("PASS - Image batch shape:", images.shape)
    print("PASS - Label batch shape:", labels.shape)


if __name__ == "__main__":
    test_gei_dataloader()
