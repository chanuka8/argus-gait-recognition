"""
3D Pose Gait Model Trainer for ARGUS AI EXP-007.

Supports model architecture selection (TCN, ST-GCN, CTR-GCN), configurable sequence lengths (30/60/90),
ArcFace + Triplet loss tuning, and subject-disjoint validation early stopping.
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logger import setup_logger
from evaluation.dataset_split import load_or_create_subject_split
from models.architectures.losses import ArcMarginProduct, JointGaitLoss
from models.architectures.pose_gait_3d import (
    CTRGCNGait3DNet,
    PoseGait3DNet,
    PoseLifter3D,
    STGCNGait3DNet,
)
from training.gait_3d_dataset import Gait3DSkeletonDataset


def get_gait3d_model(encoder_type: str = "tcn", embedding_dim: int = 256) -> torch.nn.Module:
    enc_type = encoder_type.lower()
    if enc_type == "stgcn":
        return STGCNGait3DNet(embedding_dim=embedding_dim)
    elif enc_type == "ctrgcn":
        return CTRGCNGait3DNet(embedding_dim=embedding_dim)
    else:
        return PoseGait3DNet(embedding_dim=embedding_dim)


class Gait3DTrainer:
    """
    Trainer for 3D Pose Gait Backbone and Pose Lifter.
    """

    def __init__(
        self,
        data_dir: str = "data/casia_processed/skeletons",
        run_dir: str = "runs/exp_007_3d",
        encoder_type: str = "tcn",
        batch_size: int = 32,
        epochs: int = 25,
        learning_rate: float = 1e-3,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.50,
        triplet_weight: float = 0.25,
        triplet_margin: float = 0.20,
        sequence_length: int = 30,
        split_config_path: str = "configs/subject_split.json",
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.logger = setup_logger("ARGUS.Gait3DTrainer")
        self.data_dir = Path(data_dir)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.encoder_type = encoder_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.arcface_scale = arcface_scale
        self.arcface_margin = arcface_margin
        self.triplet_weight = triplet_weight
        self.triplet_margin = triplet_margin
        self.sequence_length = sequence_length
        self.split_config_path = split_config_path
        self.seed = seed

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self) -> Dict[str, Any]:
        self.logger.info(f"Loading subject split from {self.split_config_path}")
        split_manifest = load_or_create_subject_split(
            config_path=self.split_config_path,
            data_dir="data/casia_processed/gei",
        )
        train_subs = split_manifest["train_subjects"]
        val_subs = split_manifest["val_subjects"]

        self.logger.info(f"Train subjects: {len(train_subs)} (001-062) | Val subjects: {len(val_subs)} (063-074)")

        train_ds = Gait3DSkeletonDataset(
            subjects=train_subs,
            data_dir=str(self.data_dir),
            sequence_length=self.sequence_length,
            split_config_path=self.split_config_path,
        )

        val_ds = Gait3DSkeletonDataset(
            subjects=val_subs,
            data_dir=str(self.data_dir),
            sequence_length=self.sequence_length,
            split_config_path=self.split_config_path,
        )

        if len(train_ds) == 0:
            raise RuntimeError(f"No 3D skeleton samples found in {self.data_dir} for train subjects!")

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        num_classes = len(train_ds.label_to_index)
        self.logger.info(f"Encoder: {self.encoder_type.upper()} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Classes: {num_classes}")

        lifter = PoseLifter3D().to(self.device)
        gait_net = get_gait3d_model(encoder_type=self.encoder_type, embedding_dim=256).to(self.device)
        arcface = ArcMarginProduct(in_features=256, out_features=num_classes, s=self.arcface_scale, m=self.arcface_margin).to(self.device)

        criterion = JointGaitLoss(triplet_margin=self.triplet_margin, triplet_weight=self.triplet_weight)

        optimizer = torch.optim.AdamW(
            list(lifter.parameters()) + list(gait_net.parameters()) + list(arcface.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        best_val_acc = 0.0
        metrics_history = []

        for epoch in range(1, self.epochs + 1):
            lifter.train()
            gait_net.train()
            arcface.train()

            total_train_loss = 0.0
            train_correct = 0
            train_total = 0

            for kpts_2d, labels, _conds in train_loader:
                kpts_2d = kpts_2d.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                joints_3d = lifter(kpts_2d)
                embeddings = gait_net(joints_3d)

                logits, _unscaled = arcface(embeddings, labels)
                loss, _ce, _triplet = criterion(logits, embeddings, labels)

                loss.backward()
                optimizer.step()

                total_train_loss += float(loss.item()) * len(labels)
                preds = torch.argmax(logits, dim=1)
                train_correct += int((preds == labels).sum().item())
                train_total += len(labels)

            scheduler.step()

            train_loss = total_train_loss / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            lifter.eval()
            gait_net.eval()
            arcface.eval()

            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for kpts_2d, labels, _conds in val_loader:
                    kpts_2d = kpts_2d.to(self.device)
                    labels = labels.to(self.device)

                    joints_3d = lifter(kpts_2d)
                    embeddings = gait_net(joints_3d)

                    logits = arcface(embeddings, None)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += int((preds == labels).sum().item())
                    val_total += len(labels)

            val_acc = val_correct / max(val_total, 1)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
            }
            metrics_history.append(epoch_record)

            self.logger.info(f"[{self.encoder_type.upper()}] Epoch {epoch:02d}/{self.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

            if val_acc >= best_val_acc or epoch == 1:
                best_val_acc = val_acc
                torch.save(
                    {
                        "lifter": lifter.state_dict(),
                        "gait_net": gait_net.state_dict(),
                        "arcface": arcface.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "embedding_dim": 256,
                        "encoder_type": self.encoder_type,
                        "sequence_length": self.sequence_length,
                    },
                    self.run_dir / "best_model.pth",
                )

            torch.save(
                {
                    "lifter": lifter.state_dict(),
                    "gait_net": gait_net.state_dict(),
                    "arcface": arcface.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "embedding_dim": 256,
                    "encoder_type": self.encoder_type,
                    "sequence_length": self.sequence_length,
                },
                self.run_dir / "last_model.pth",
            )

        with open(self.run_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics_history, "best_val_acc": round(best_val_acc, 4)}, f, indent=4)

        with open(self.run_dir / "pose_lifter_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "PoseLifter3D",
                "in_channels": 3,
                "num_joints": 17,
                "trained_epochs": self.epochs,
                "version": "1.1.0",
                "checksum": "sha256_pose_lifter_v11",
            }, f, indent=4)

        with open(self.run_dir / "gait3d_model_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": f"{self.encoder_type.upper()}Gait3DNet",
                "encoder_type": self.encoder_type,
                "embedding_dim": 256,
                "sequence_length": self.sequence_length,
                "arcface_s": self.arcface_scale,
                "arcface_m": self.arcface_margin,
                "triplet_weight": self.triplet_weight,
                "best_val_acc": round(best_val_acc, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=4)

        with open(self.run_dir / "split_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=4)

        self.logger.info(f"Training completed. Saved best model to {self.run_dir / 'best_model.pth'}")
        return {"best_val_acc": best_val_acc}
