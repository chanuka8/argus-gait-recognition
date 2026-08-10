import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.dataloader import build_dataloaders
from training.trainer import GaitClassifier, Trainer


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


def test_gait_classifier_validation_and_forward() -> None:
    with pytest.raises(ValueError, match="num_classes must be greater than 0"):
        GaitClassifier(num_classes=0)

    with pytest.raises(ValueError, match="Invalid loss_mode"):
        GaitClassifier(num_classes=5, loss_mode="invalid")

    model_ce = GaitClassifier(num_classes=10, loss_mode="ce")
    dummy_x = torch.randn(4, 1, 128, 64)
    loss_logits, pred_logits, embeddings = model_ce(dummy_x)
    assert loss_logits.shape == (4, 10)
    assert pred_logits.shape == (4, 10)
    assert embeddings.shape == (4, 256)

    model_arc = GaitClassifier(num_classes=10, loss_mode="ce_arcface")
    labels = torch.tensor([0, 1, 2, 3])
    loss_logits, pred_logits, embeddings = model_arc(dummy_x, labels=labels)
    assert loss_logits.shape == (4, 10)
    assert pred_logits.shape == (4, 10)
    assert embeddings.shape == (4, 256)

    loss_logits_no_label, pred_logits_no_label, embeddings_no_label = model_arc(dummy_x, labels=None)
    assert loss_logits_no_label.shape == (4, 10)


def test_trainer_validation_and_training() -> None:
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        Trainer(epochs=0)

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        Trainer(batch_size=-1)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        Trainer(learning_rate=0.0)

    with pytest.raises(ValueError, match="Invalid loss_mode"):
        Trainer(loss_mode="unknown")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = tmp_path / "gei"
        p1 = data_dir / "001"
        p2 = data_dir / "002"
        p1.mkdir(parents=True)
        p2.mkdir(parents=True)

        img = np.zeros((128, 64), dtype=np.uint8)
        cv2.imwrite(str(p1 / "001-bg-01-000.png"), img)
        cv2.imwrite(str(p1 / "001-bg-01-018.png"), img)
        cv2.imwrite(str(p2 / "002-bg-01-000.png"), img)
        cv2.imwrite(str(p2 / "002-bg-01-018.png"), img)

        run_dir = tmp_path / "run"
        trainer = Trainer(
            data_dir=str(data_dir),
            run_dir=str(run_dir),
            batch_size=2,
            epochs=1,
            loss_mode="ce",
            device="cpu",
        )
        history = trainer.train()
        assert "epochs" in history
        assert len(history["epochs"]) == 1
        assert (run_dir / "last_model.pth").exists()
        assert (run_dir / "metrics.json").exists()


if __name__ == "__main__":
    test_gei_dataloader()
    test_gait_classifier_validation_and_forward()
    test_trainer_validation_and_training()
    print("ALL TRAINING TESTS PASSED SUCCESSFULLY!")

