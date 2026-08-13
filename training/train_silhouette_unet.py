import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from models.architectures.silhouette_unet import SilhouetteUNet
from scripts.export_silhouette_unet_onnx import export_and_validate_onnx
from training.silhouette_dataset import SilhouetteSegmentationDataset


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy + Dice Loss for Silhouette Segmentation."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        intersection = (pred * target).sum(dim=(2, 3))
        cardinality = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return bce_loss + dice_loss.mean()


def calculate_metrics(pred_binary: torch.Tensor, target_binary: torch.Tensor) -> dict[str, float]:
    """Calculates Dice, IoU, Precision, and Recall on binary masks."""
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1.0 - target_binary)).sum().item()
    fn = ((1.0 - pred_binary) * target_binary).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def train_and_export_silhouette_unet(
    zip_path: str = "data/casia_b_raw.zip",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    output_dir: str = "models/weights",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models/engines", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training UNet Silhouette Segmenter on device: {device}")

    # Datasets
    train_ds = SilhouetteSegmentationDataset(
        zip_path=zip_path, subject_range=(1, 62), max_samples=250, seed=42
    )
    val_ds = SilhouetteSegmentationDataset(
        zip_path=zip_path, subject_range=(63, 74), max_samples=50, seed=101
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SilhouetteUNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_dice = 0.0
    best_pth_path = Path(output_dir) / "silhouette_segmenter.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        metrics_list = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item() * imgs.size(0)

                preds_binary = (preds > 0.5).float()
                m = calculate_metrics(preds_binary, masks)
                metrics_list.append(m)

        val_loss /= len(val_ds)
        avg_dice = np.mean([m["dice"] for m in metrics_list])
        avg_iou = np.mean([m["iou"] for m in metrics_list])
        avg_prec = np.mean([m["precision"] for m in metrics_list])
        avg_rec = np.mean([m["recall"] for m in metrics_list])

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}"
        )

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), best_pth_path)
            print(f"  -> Saved best model checkpoint to {best_pth_path}")

    # Load best weights
    model.load_state_dict(torch.load(best_pth_path, map_location=device))
    model.eval()

    # Measure latency & FPS on CPU/GPU
    dummy_in = torch.randn(1, 3, 256, 256, device=device)
    for _ in range(10):
        _ = model(dummy_in)

    start_time = time.perf_counter()
    num_runs = 50
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_in)
    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / num_runs) * 1000.0
    fps = num_runs / elapsed

    # Export & validate ONNX using helper function
    onnx_valid, onnx_msg = export_and_validate_onnx(
        pth_path=str(best_pth_path),
        output_onnx_path=str(Path(output_dir) / "silhouette_segmenter.onnx"),
        engine_onnx_path="models/engines/silhouette_segmenter.onnx",
    )

    if not onnx_valid:
        raise RuntimeError(f"ONNX export validation failed: {onnx_msg}")

    summary = {
        "epochs": epochs,
        "best_checkpoint": str(best_pth_path),
        "onnx_weights_path": str(Path(output_dir) / "silhouette_segmenter.onnx"),
        "onnx_engines_path": "models/engines/silhouette_segmenter.onnx",
        "dice": float(avg_dice),
        "iou": float(avg_iou),
        "precision": float(avg_prec),
        "recall": float(avg_rec),
        "latency_ms": float(avg_latency_ms),
        "fps": float(fps),
    }

    return summary


if __name__ == "__main__":
    res = train_and_export_silhouette_unet()
    print("\n--- TRAINING & EXPORT SUMMARY ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
