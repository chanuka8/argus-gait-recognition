from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.logger import setup_logger
from models.architectures.bygait_light import ByGaitLight
from models.architectures.losses import ArcMarginProduct, JointGaitLoss
from training.checkpointer import Checkpointer
from training.dataloader import build_dataloaders
from training.optimizer import build_optimizer


class GaitClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        part_bins: int = 4,
        loss_mode: str = "ce",
        arcface_s: float = 30.0,
        arcface_m: float = 0.50,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be greater than 0, got {num_classes}")

        if loss_mode not in ("ce", "ce_arcface"):
            raise ValueError(f"Invalid loss_mode: '{loss_mode}'. Supported options are 'ce' and 'ce_arcface'.")

        self.backbone = ByGaitLight(
            embedding_dim=embedding_dim,
            part_bins=part_bins,
        )

        self.loss_mode = loss_mode

        if loss_mode == "ce_arcface":
            self.arcface_classifier: ArcMarginProduct | None = ArcMarginProduct(
                in_features=embedding_dim,
                out_features=num_classes,
                s=arcface_s,
                m=arcface_m,
            )
            self.classifier: nn.Linear | None = None
        else:
            self.classifier = nn.Linear(
                embedding_dim,
                num_classes,
            )
            self.arcface_classifier = None

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.backbone(x)

        if self.loss_mode == "ce_arcface":
            if self.arcface_classifier is None:
                raise RuntimeError("ArcMarginProduct classifier is not initialized.")
            if labels is not None:
                loss_logits, pred_logits = self.arcface_classifier(
                    embedding,
                    labels,
                )
            else:
                pred_logits = self.arcface_classifier(
                    embedding,
                )
                loss_logits = pred_logits
        else:
            if self.classifier is None:
                raise RuntimeError("Linear classifier is not initialized.")
            loss_logits = self.classifier(
                embedding,
            )
            pred_logits = loss_logits

        return loss_logits, pred_logits, embedding


class Trainer:
    def __init__(
        self,
        data_dir: str = "data/casia_processed/gei",
        run_dir: str = "runs/exp_002_hpp_arcface",
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
        part_bins: int = 4,
        split_config_path: str | None = "configs/subject_split.json",
        device: str | None = None,
        condition_balanced: bool = False,
        cross_condition_triplet: bool = False,
    ) -> None:
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if loss_mode not in ("ce", "ce_arcface"):
            raise ValueError(f"Invalid loss_mode: '{loss_mode}'. Supported options are 'ce' and 'ce_arcface'.")
        if triplet_margin < 0:
            raise ValueError(f"triplet_margin cannot be negative, got {triplet_margin}")
        if triplet_weight < 0:
            raise ValueError(f"triplet_weight cannot be negative, got {triplet_weight}")

        self.logger = setup_logger(
            "ARGUS.Trainer",
        )

        self.data_dir = data_dir
        self.run_dir = Path(
            run_dir,
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

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
        self.part_bins = part_bins
        self.split_config_path = split_config_path
        self.condition_balanced = condition_balanced
        self.cross_condition_triplet = cross_condition_triplet

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA device requested but not available. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = device

        self.checkpointer = Checkpointer(
            run_dir=str(
                self.run_dir,
            )
        )

    def train(
        self,
    ) -> dict:
        self.logger.info("Building dataloaders")

        use_return_condition = self.cross_condition_triplet or self.condition_balanced

        train_loader, val_loader, dataset = build_dataloaders(
            root_dir=self.data_dir,
            batch_size=self.batch_size,
            max_classes=self.max_classes,
            max_samples=self.max_samples,
            split_config_path=self.split_config_path,
            condition_balanced=self.condition_balanced,
            return_condition=use_return_condition,
        )

        num_classes = len(
            dataset.label_to_index,
        )

        if num_classes <= 0:
            raise ValueError(f"Dataset at '{self.data_dir}' contains 0 valid classes.")
        if len(dataset) <= 0:
            raise ValueError(f"Dataset at '{self.data_dir}' contains 0 samples.")

        self.logger.info(f"Samples: {len(dataset)}")
        self.logger.info(f"Classes: {num_classes}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(
            f"Loss mode: {self.loss_mode} (scale={self.arcface_scale}, margin={self.arcface_margin}) | "
            f"Triplet | margin={self.triplet_margin} | weight={self.triplet_weight}"
        )

        model = GaitClassifier(
            num_classes=num_classes,
            part_bins=self.part_bins,
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

        exp_config = {
            "data_dir": self.data_dir,
            "run_dir": str(self.run_dir),
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "max_classes": self.max_classes,
            "max_samples": self.max_samples,
            "triplet_margin": self.triplet_margin,
            "triplet_weight": self.triplet_weight,
            "loss_mode": self.loss_mode,
            "arcface_scale": self.arcface_scale,
            "arcface_margin": self.arcface_margin,
            "part_bins": self.part_bins,
            "embedding_dim": 256,
            "device": str(self.device),
            "split_config_path": self.split_config_path,
            "condition_balanced": self.condition_balanced,
            "cross_condition_triplet": self.cross_condition_triplet,
        }
        with open(self.run_dir / "experiment_config.json", "w", encoding="utf-8") as f:
            import json

            json.dump(exp_config, f, indent=4)

        if self.split_config_path and Path(self.split_config_path).exists():
            import shutil

            shutil.copy(self.split_config_path, self.run_dir / "subject_split.json")

        model_meta = {
            "architecture": "ByGaitLight",
            "part_bins": self.part_bins,
            "embedding_dim": 256,
            "l2_normalized": True,
            "num_training_classes": num_classes,
        }
        with open(self.run_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            import json

            json.dump(model_meta, f, indent=4)

        history = {
            "epochs": [],
            "best_val_accuracy": 0.0,
            "num_classes": num_classes,
            "samples": len(dataset),
            "max_classes": self.max_classes,
            "max_samples": self.max_samples,
            "device": str(self.device),
            "loss_mode": self.loss_mode,
            "arcface_scale": self.arcface_scale,
            "arcface_margin": self.arcface_margin,
            "triplet_margin": self.triplet_margin,
            "triplet_weight": self.triplet_weight,
            "part_bins": self.part_bins,
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

            history["epochs"].append(
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

            if val_metrics["val_accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["val_accuracy"]

                history["best_val_accuracy"] = best_val_accuracy

                self.checkpointer.save_model(
                    model,
                    "best_model.pth",
                )

        self.checkpointer.save_metrics(
            history,
        )

        self.logger.info("Training completed")

        return history

    def _train_one_epoch(
        self,
        model: GaitClassifier,
        loader: DataLoader,
        criterion: JointGaitLoss,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_triplet = 0.0
        correct = 0
        total = 0

        for batch in tqdm(
            loader,
            desc="Training",
            leave=False,
        ):
            if len(batch) == 3:
                images, labels, condition_labels = batch
                condition_labels = condition_labels.to(self.device)
            else:
                images, labels = batch
                condition_labels = None

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

            cond_for_loss = condition_labels if self.cross_condition_triplet else None
            loss, ce_loss, triplet_loss = criterion(
                loss_logits,
                embeddings,
                labels,
                condition_labels=cond_for_loss,
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

            correct += (predictions == labels).sum().item()

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
        model: GaitClassifier,
        loader: DataLoader,
        criterion: JointGaitLoss,
    ) -> dict[str, float]:
        model.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_triplet = 0.0
        total = 0

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Validation",
                leave=False,
            ):
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(
                    self.device,
                )

                labels = labels.to(
                    self.device,
                )

                loss_logits, _pred_logits, embeddings = model(
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

                total += batch_size

                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

        if all_embeddings:
            concat_emb = torch.cat(all_embeddings, dim=0)
            concat_lbl = torch.cat(all_labels, dim=0)
            N = concat_emb.size(0)

            sim_matrix = torch.mm(concat_emb, concat_emb.t())
            sim_matrix.fill_diagonal_(-1.0)

            top1_indices = torch.argmax(sim_matrix, dim=1)
            correct_nn = (concat_lbl[top1_indices] == concat_lbl).sum().item()
            val_acc = correct_nn / max(N, 1)
        else:
            val_acc = 0.0

        return {
            "val_loss": total_loss / max(total, 1),
            "val_ce_loss": total_ce / max(total, 1),
            "val_triplet_loss": total_triplet / max(total, 1),
            "val_accuracy": val_acc,
        }
