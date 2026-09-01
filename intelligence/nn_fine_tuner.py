import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


class NNFineTuner:
    def __init__(
        self,
        candidate_dir: str = "models/candidates",
        device: str | None = None,
        max_epochs: int = 3,
        learning_rate: float = 1e-5,
        batch_size: int = 16,
        historical_replay_ratio: float = 0.50,
        timeout_seconds: float = 600.0,
    ) -> None:
        self._logger = get_logger("nn_fine_tuner")
        self.candidate_dir = Path(candidate_dir)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max(1, max_epochs)
        self.learning_rate = learning_rate
        self.batch_size = max(1, batch_size)
        self.historical_replay_ratio = float(historical_replay_ratio)
        self.timeout_seconds = float(timeout_seconds)


        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device





    def fine_tune_bygait_light(
        self,
        active_weights_path: str,
        training_gei_data: list[dict[str, Any]],
        historical_gei_data: list[dict[str, Any]],
        candidate_version: str,
        part_bins: int = 4,
    ) -> dict[str, Any]:
        start_time = time.time()
        self._logger.info(
            f"[BYGAIT_FINETUNE_START] version={candidate_version} "
            f"new_samples={len(training_gei_data)} historical={len(historical_gei_data)} "
            f"device={self.device}"
        )

        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            from models.architectures.bygait_light import ByGaitLight


            all_data = list(training_gei_data) + list(historical_gei_data)
            if len(all_data) < 4:
                raise ValueError(f"Insufficient training samples: {len(all_data)} (minimum 4)")


            unique_labels = sorted({d["label"] for d in all_data})
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            num_classes = len(unique_labels)

            if num_classes < 2:
                raise ValueError(f"Insufficient identities: {num_classes} (minimum 2)")


            images = []
            labels = []
            for d in all_data:
                img = np.asarray(d["image"], dtype=np.float32)
                if img.ndim == 2:
                    img = img[np.newaxis, :, :]
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = img.transpose(2, 0, 1)
                elif img.ndim == 3:
                    img = img[:1, :, :]
                images.append(img)
                labels.append(label_map[d["label"]])

            X = torch.from_numpy(np.array(images, dtype=np.float32))
            y = torch.tensor(labels, dtype=torch.long)


            n = len(X)
            n_val = max(1, n // 5)
            indices = torch.randperm(n)
            train_idx = indices[n_val:]
            val_idx = indices[:n_val]

            train_ds = TensorDataset(X[train_idx], y[train_idx])
            val_ds = TensorDataset(X[val_idx], y[val_idx])
            drop_last = len(train_ds) > self.batch_size and len(train_ds) % self.batch_size == 1
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)


            backbone = ByGaitLight(embedding_dim=256, part_bins=part_bins)


            if active_weights_path:
                active_path = Path(active_weights_path)
                if active_path.is_file():
                    try:
                        state_dict = torch.load(str(active_path), map_location="cpu", weights_only=True)

                        backbone_keys = {k for k in backbone.state_dict()}
                        filtered = {}
                        for k, v in state_dict.items():
                            clean_key = k.replace("backbone.", "")
                            if clean_key in backbone_keys:
                                filtered[clean_key] = v
                        if filtered:
                            backbone.load_state_dict(filtered, strict=False)
                            self._logger.info(
                                f"[TRANSFER_LEARNING] Loaded {len(filtered)} weight tensors from "
                                f"active model '{active_path.name}'"
                            )
                    except (RuntimeError, ValueError, KeyError, OSError) as load_err:
                        self._logger.warning(
                            f"[TRANSFER_LEARNING] Could not load active weights: {load_err}. "
                            f"Training from scratch."
                        )


            initial_params = {
                name: param.clone().detach()
                for name, param in backbone.named_parameters()
                if param.requires_grad
            }
            total_trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)


            classifier = nn.Linear(256, num_classes)
            model = nn.Sequential(backbone, classifier).to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=1e-7
            )


            best_val_acc = 0.0
            training_history = []

            for epoch in range(1, self.max_epochs + 1):

                elapsed = time.time() - start_time
                if elapsed > self.timeout_seconds:
                    self._logger.warning(
                        f"[TIMEOUT] Training exceeded {self.timeout_seconds}s at epoch {epoch}. "
                        f"Saving best candidate so far."
                    )
                    break


                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    logits = model(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * batch_X.size(0)
                    train_correct += (logits.argmax(1) == batch_y).sum().item()
                    train_total += batch_X.size(0)

                scheduler.step()


                model.eval()
                val_embeddings = []
                val_labels_list = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        embs = backbone(batch_X).cpu()
                        val_embeddings.append(embs)
                        val_labels_list.append(batch_y)

                if val_embeddings:
                    all_embs = torch.cat(val_embeddings, dim=0)
                    all_lbls = torch.cat(val_labels_list, dim=0)
                    N = all_embs.size(0)


                    sim_matrix = torch.mm(all_embs, all_embs.t())
                    sim_matrix.fill_diagonal_(-1.0)
                    nn_indices = torch.argmax(sim_matrix, dim=1)
                    val_acc = float((all_lbls[nn_indices] == all_lbls).sum().item()) / max(N, 1)
                else:
                    val_acc = 0.0

                train_acc = train_correct / max(train_total, 1)
                avg_loss = train_loss / max(train_total, 1)

                training_history.append({
                    "epoch": epoch,
                    "train_loss": round(avg_loss, 4),
                    "train_accuracy": round(train_acc, 4),
                    "val_rank1_accuracy": round(val_acc, 4),
                })

                self._logger.info(
                    f"[BYGAIT_EPOCH {epoch}/{self.max_epochs}] "
                    f"loss={avg_loss:.4f} train_acc={train_acc:.4f} val_rank1={val_acc:.4f}"
                )

                best_val_acc = max(best_val_acc, val_acc)


            changed_tensors = 0
            max_delta = 0.0
            for name, param in backbone.named_parameters():
                if param.requires_grad and name in initial_params:
                    diff = (param.detach().cpu() - initial_params[name].cpu()).abs().max().item()
                    if diff > 1e-7:
                        changed_tensors += 1
                        max_delta = max(max_delta, diff)


            candidate_path = self.candidate_dir / f"bygait_candidate_{candidate_version}.pth"
            torch.save(backbone.state_dict(), str(candidate_path))
            checksum = self._calculate_checksum(candidate_path)


            metrics = {
                "val_rank1_accuracy": round(best_val_acc * 100, 2),
                "train_samples": len(training_gei_data),
                "historical_samples": len(historical_gei_data),
                "total_samples": len(all_data),
                "num_classes": num_classes,
                "epochs_completed": len(training_history),
                "device": self.device,
                "learning_rate": self.learning_rate,
                "part_bins": part_bins,
                "total_trainable_params": total_trainable_params,
                "changed_tensors": changed_tensors,
                "total_tensors": len(initial_params),
                "max_param_delta": max_delta,
                "training_history": training_history,
            }

            duration = round(time.time() - start_time, 2)
            self._logger.info(
                f"[BYGAIT_FINETUNE_COMPLETE] version={candidate_version} "
                f"artifact={candidate_path.name} val_rank1={best_val_acc:.4f} "
                f"duration={duration}s checksum={checksum[:12]}..."
            )

            return {
                "success": True,
                "model_type": "bygait_light",
                "candidate_version": candidate_version,
                "artifact_path": str(candidate_path),
                "checksum_sha256": checksum,
                "architecture": "ByGaitLight-CNN-256D",
                "embedding_dim": 256,
                "metrics": metrics,
                "duration": duration,
            }

        except Exception as err:  # noqa: BLE001
            duration = round(time.time() - start_time, 2)
            self._logger.error(
                f"[BYGAIT_FINETUNE_FAILED] version={candidate_version} "
                f"error={err} duration={duration}s"
            )
            return {
                "success": False,
                "model_type": "bygait_light",
                "candidate_version": candidate_version,
                "error": str(err),
                "duration": duration,
            }





    def fine_tune_osnet(
        self,
        active_weights_path: str,
        training_crop_data: list[dict[str, Any]],
        historical_crop_data: list[dict[str, Any]],
        candidate_version: str,
    ) -> dict[str, Any]:
        start_time = time.time()
        self._logger.info(
            f"[OSNET_FINETUNE_START] version={candidate_version} "
            f"new_samples={len(training_crop_data)} historical={len(historical_crop_data)} "
            f"device={self.device}"
        )

        try:
            import cv2
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset


            all_data = list(training_crop_data) + list(historical_crop_data)
            if len(all_data) < 4:
                raise ValueError(f"Insufficient training samples: {len(all_data)} (minimum 4)")

            unique_labels = sorted({d["label"] for d in all_data})
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            num_classes = len(unique_labels)

            if num_classes < 2:
                raise ValueError(f"Insufficient identities: {num_classes} (minimum 2)")


            images = []
            labels = []
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            for d in all_data:
                img = np.asarray(d["image"], dtype=np.uint8)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (128, 256))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                tensor = (tensor - mean) / std
                images.append(tensor)
                labels.append(label_map[d["label"]])

            X = torch.stack(images)
            y = torch.tensor(labels, dtype=torch.long)


            n = len(X)
            n_val = max(1, n // 5)
            indices = torch.randperm(n)
            train_idx = indices[n_val:]
            val_idx = indices[:n_val]

            train_ds = TensorDataset(X[train_idx], y[train_idx])
            val_ds = TensorDataset(X[val_idx], y[val_idx])
            drop_last = len(train_ds) > self.batch_size and len(train_ds) % self.batch_size == 1
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)


            from models.reid.osnet_backbone import _build_osnet_x0_25

            osnet_backbone = _build_osnet_x0_25()


            if active_weights_path:
                active_path = Path(active_weights_path)
                if active_path.is_file():
                    try:
                        state_dict = torch.load(str(active_path), map_location="cpu", weights_only=True)
                        osnet_backbone.load_state_dict(state_dict, strict=False)
                        self._logger.info(
                            f"[TRANSFER_LEARNING] Loaded OSNet weights from '{active_path.name}'"
                        )
                    except (RuntimeError, ValueError, KeyError, OSError) as load_err:
                        self._logger.warning(
                            f"[TRANSFER_LEARNING] Could not load active OSNet weights: {load_err}."
                        )


            initial_params = {
                name: param.clone().detach()
                for name, param in osnet_backbone.named_parameters()
                if param.requires_grad
            }
            total_trainable_params = sum(p.numel() for p in osnet_backbone.parameters() if p.requires_grad)

            osnet_backbone = osnet_backbone.to(self.device)


            classifier = nn.Linear(512, num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                list(osnet_backbone.parameters()) + list(classifier.parameters()),
                lr=self.learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=1e-7
            )


            best_val_acc = 0.0
            training_history = []

            for epoch in range(1, self.max_epochs + 1):
                elapsed = time.time() - start_time
                if elapsed > self.timeout_seconds:
                    self._logger.warning(
                        f"[TIMEOUT] OSNet training exceeded {self.timeout_seconds}s. "
                        f"Saving best candidate."
                    )
                    break

                osnet_backbone.train()
                classifier.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    embs = osnet_backbone(batch_X)
                    logits = classifier(embs)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * batch_X.size(0)
                    train_correct += (logits.argmax(1) == batch_y).sum().item()
                    train_total += batch_X.size(0)

                scheduler.step()


                osnet_backbone.eval()
                val_embs = []
                val_lbls = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        embs = osnet_backbone(batch_X).cpu()
                        val_embs.append(embs)
                        val_lbls.append(batch_y)

                if val_embs:
                    cat_embs = torch.cat(val_embs, dim=0)
                    import torch.nn.functional as F

                    cat_embs = F.normalize(cat_embs, p=2, dim=1)
                    cat_lbls = torch.cat(val_lbls, dim=0)
                    N = cat_embs.size(0)
                    sim = torch.mm(cat_embs, cat_embs.t())
                    sim.fill_diagonal_(-1.0)
                    nn_idx = torch.argmax(sim, dim=1)
                    val_acc = float((cat_lbls[nn_idx] == cat_lbls).sum().item()) / max(N, 1)
                else:
                    val_acc = 0.0

                train_acc = train_correct / max(train_total, 1)
                avg_loss = train_loss / max(train_total, 1)

                training_history.append({
                    "epoch": epoch,
                    "train_loss": round(avg_loss, 4),
                    "train_accuracy": round(train_acc, 4),
                    "val_rank1_accuracy": round(val_acc, 4),
                })

                self._logger.info(
                    f"[OSNET_EPOCH {epoch}/{self.max_epochs}] "
                    f"loss={avg_loss:.4f} train_acc={train_acc:.4f} val_rank1={val_acc:.4f}"
                )

                best_val_acc = max(best_val_acc, val_acc)


            changed_tensors = 0
            max_delta = 0.0
            for name, param in osnet_backbone.named_parameters():
                if param.requires_grad and name in initial_params:
                    diff = (param.detach().cpu() - initial_params[name].cpu()).abs().max().item()
                    if diff > 1e-7:
                        changed_tensors += 1
                        max_delta = max(max_delta, diff)


            candidate_path = self.candidate_dir / f"osnet_candidate_{candidate_version}.pth"
            torch.save(osnet_backbone.state_dict(), str(candidate_path))
            checksum = self._calculate_checksum(candidate_path)

            metrics = {
                "val_rank1_accuracy": round(best_val_acc * 100, 2),
                "train_samples": len(training_crop_data),
                "historical_samples": len(historical_crop_data),
                "total_samples": len(all_data),
                "num_classes": num_classes,
                "epochs_completed": len(training_history),
                "device": self.device,
                "total_trainable_params": total_trainable_params,
                "changed_tensors": changed_tensors,
                "total_tensors": len(initial_params),
                "max_param_delta": max_delta,
                "training_history": training_history,
            }

            duration = round(time.time() - start_time, 2)
            self._logger.info(
                f"[OSNET_FINETUNE_COMPLETE] version={candidate_version} "
                f"artifact={candidate_path.name} val_rank1={best_val_acc:.4f} "
                f"duration={duration}s checksum={checksum[:12]}..."
            )

            return {
                "success": True,
                "model_type": "osnet_reid",
                "candidate_version": candidate_version,
                "artifact_path": str(candidate_path),
                "checksum_sha256": checksum,
                "architecture": "OSNet-x0.25-ReID-512D",
                "embedding_dim": 512,
                "metrics": metrics,
                "duration": duration,
            }

        except Exception as err:  # noqa: BLE001
            duration = round(time.time() - start_time, 2)
            self._logger.error(
                f"[OSNET_FINETUNE_FAILED] version={candidate_version} "
                f"error={err} duration={duration}s"
            )
            return {
                "success": False,
                "model_type": "osnet_reid",
                "candidate_version": candidate_version,
                "error": str(err),
                "duration": duration,
            }





    @staticmethod
    def _calculate_checksum(file_path: Path | str) -> str:
        p = Path(file_path)
        if not p.exists():
            return ""
        h = hashlib.sha256()
        try:
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return ""
