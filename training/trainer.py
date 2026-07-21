from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from core.logger import setup_logger
from models.architectures.bygait_light import ByGaitLight
from models.architectures.losses import JointGaitLoss, ArcMarginProduct
from training.checkpointer import Checkpointer
from training.dataloader import build_dataloaders
from training.optimizer import build_optimizer


class GaitClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        loss_mode: str = "ce",
        arcface_s: float = 30.0,
        arcface_m: float = 0.50,
    ) -> None:
        super().__init__()

        self.backbone = ByGaitLight(
            embedding_dim=embedding_dim,
        )

        self.loss_mode = loss_mode
        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

        if loss_mode == "ce_arcface":
            self.arcface_classifier = ArcMarginProduct(
                in_features=embedding_dim,
                out_features=num_classes,
                s=arcface_s,
                m=arcface_m,
            )

    def forward(
        self,
        x,
        labels=None,
    ):
        embedding = self.backbone(
            x,
        )

        if self.loss_mode == "ce_arcface" and labels is not None:
            loss_logits, pred_logits = self.arcface_classifier(
                embedding,
                labels,
            )
        elif self.loss_mode == "ce_arcface" and labels is None:
            pred_logits = self.arcface_classifier(
                embedding,
            )
            loss_logits = pred_logits
        else:
            loss_logits = self.classifier(
                embedding,
            )
            pred_logits = loss_logits

        return loss_logits, pred_logits, embedding



class Trainer:
    def __init__(
        self,
        data_dir: str = "data/casia_processed/gei",
        run_dir: str = "runs/exp_001",
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 0.0001,
        max_classes: int | None = None,
        max_samples: int | None = None,
        triplet_margin: float = 0.3,
        triplet_weight: float = 0.0,
        loss_mode: str = "ce",
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.50,
        device: str | None = None,
    ) -> None:
        self.logger = setup_logger(
            "ARGUS.Trainer",
        )

        self.data_dir = data_dir
        self.run_dir = Path(
            run_dir,
        )

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_classes = max_classes
        self.max_samples = max_samples
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight
        self.loss_mode = loss_mode
        self.arcface_scale = arcface_scale
        self.arcface_margin = arcface_margin

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.checkpointer = Checkpointer(
            run_dir=str(
                self.run_dir,
            )
        )

    def train(
        self,
    ) -> dict:
        self.logger.info(
            "Building dataloaders"
        )

        train_loader, val_loader, dataset = build_dataloaders(
            root_dir=self.data_dir,
            batch_size=self.batch_size,
            max_classes=self.max_classes,
            max_samples=self.max_samples,
        )

        num_classes = len(
            dataset.label_to_index,
        )

        self.logger.info(
            f"Samples: {len(dataset)}"
        )
        self.logger.info(
            f"Classes: {num_classes}"
        )
        self.logger.info(
            f"Device: {self.device}"
        )
        self.logger.info(
            f"Loss mode: {self.loss_mode} (scale={self.arcface_scale}, margin={self.arcface_margin}) | "
            f"Triplet | margin={self.triplet_margin} | weight={self.triplet_weight}"
        )

        model = GaitClassifier(
            num_classes=num_classes,
            loss_mode=self.loss_mode,
            arcface_s=self.arcface_scale,
            arcface_m=self.arcface_margin,
        ).to(
            self.device,
        )

        criterion = JointGaitLoss(
            triplet_margin=self.triplet_margin,
            triplet_weight=self.triplet_weight,
        )

        optimizer = build_optimizer(
            model,
            learning_rate=self.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-5,
        )

        best_val_accuracy = 0.0

        history = {
            "epochs": [],
            "best_val_accuracy": 0.0,
            "num_classes": num_classes,
            "samples": len(dataset),
            "max_classes": self.max_classes,
            "max_samples": self.max_samples,
            "device": self.device,
            "loss_mode": self.loss_mode,
            "arcface_scale": self.arcface_scale,
            "arcface_margin": self.arcface_margin,
            "triplet_margin": self.triplet_margin,
            "triplet_weight": self.triplet_weight,
        }


        for epoch in range(
            1,
            self.epochs + 1,
        ):
            train_metrics = self._train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
            )

            scheduler.step()

            val_metrics = self._validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
            )

            epoch_metrics = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
            }

            history[
                "epochs"
            ].append(
                epoch_metrics,
            )

            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"train_ce={train_metrics['train_ce_loss']:.4f} | "
                f"train_triplet={train_metrics['train_triplet_loss']:.4f} | "
                f"train_acc={train_metrics['train_accuracy']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_ce={val_metrics['val_ce_loss']:.4f} | "
                f"val_triplet={val_metrics['val_triplet_loss']:.4f} | "
                f"val_acc={val_metrics['val_accuracy']:.4f}"
            )

            self.checkpointer.save_model(
                model,
                "last_model.pth",
            )

            if val_metrics[
                "val_accuracy"
            ] > best_val_accuracy:
                best_val_accuracy = val_metrics[
                    "val_accuracy"
                ]

                history[
                    "best_val_accuracy"
                ] = best_val_accuracy

                self.checkpointer.save_model(
                    model,
                    "best_model.pth",
                )

        self.checkpointer.save_metrics(
            history,
        )

        self.logger.info(
            "Training completed"
        )

        return history

    def _train_one_epoch(
        self,
        model,
        loader,
        criterion,
        optimizer,
    ) -> dict:
        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_triplet = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(
            loader,
            desc="Training",
            leave=False,
        ):
            images = images.to(
                self.device,
            )

            labels = labels.to(
                self.device,
            )

            optimizer.zero_grad()

            loss_logits, pred_logits, embeddings = model(
                images,
                labels=labels,
            )

            loss, ce_loss, triplet_loss = criterion(
                loss_logits,
                embeddings,
                labels,
            )

            loss.backward()
            optimizer.step()

            batch_size = images.size(
                0,
            )

            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            total_triplet += triplet_loss.item() * batch_size

            predictions = torch.argmax(
                pred_logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(
                0,
            )

        return {
            "train_loss": total_loss / max(total, 1),
            "train_ce_loss": total_ce / max(total, 1),
            "train_triplet_loss": total_triplet / max(total, 1),
            "train_accuracy": correct / max(total, 1),
        }

    def _validate(
        self,
        model,
        loader,
        criterion,
    ) -> dict:
        model.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_triplet = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                loader,
                desc="Validation",
                leave=False,
            ):
                images = images.to(
                    self.device,
                )

                labels = labels.to(
                    self.device,
                )

                loss_logits, pred_logits, embeddings = model(
                    images,
                    labels=labels,
                )

                loss, ce_loss, triplet_loss = criterion(
                    loss_logits,
                    embeddings,
                    labels,
                )

                batch_size = images.size(
                    0,
                )

                total_loss += loss.item() * batch_size
                total_ce += ce_loss.item() * batch_size
                total_triplet += triplet_loss.item() * batch_size

                predictions = torch.argmax(
                    pred_logits,
                    dim=1,
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(
                    0,
                )

        return {
            "val_loss": total_loss / max(total, 1),
            "val_ce_loss": total_ce / max(total, 1),
            "val_triplet_loss": total_triplet / max(total, 1),
            "val_accuracy": correct / max(total, 1),
        }