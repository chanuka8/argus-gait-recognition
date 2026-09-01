# Training

The `training` package handles deep neural network model training, loss function computation, learning rate optimization, training dataset loading, callback execution, and checkpoint persistence for ARGUS AI.

## Responsibilities

- Training PyTorch CNN gait feature extraction models (`ByGaitLight`).
- Computing Triplet Loss, Cross-Entropy Loss, and Center Loss over batch embeddings.
- Executing early stopping, learning rate decay, and logging callbacks during training.
- Saving trained model weights into candidate checkpoints (`models/candidates/` / `runs/`).
- Boundaries: Does not run real-time RTSP video decoding or camera management services.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [callbacks.py](callbacks.py) | Early stopping, model checkpointer, and `TrainingLogger` callbacks |
| [checkpointer.py](checkpointer.py) | Checkpoint saver and model state dictionary restorer |
| [dataloader.py](dataloader.py) | PyTorch DataLoader factory supporting balanced identity batch sampling |
| [dataset.py](dataset.py) | PyTorch Dataset class loading 64x128 GEI images and subject labels |
| [gait_3d_dataset.py](gait_3d_dataset.py) | Module/resource file gait_3d_dataset.py |
| [gait_3d_trainer.py](gait_3d_trainer.py) | Module/resource file gait_3d_trainer.py |
| [loss_functions.py](loss_functions.py) | Combined loss implementations: Triplet Loss with hard mining, Cross-Entropy, Margin Loss |
| [optimizer.py](optimizer.py) | Adam / SGD optimizer builder and Cosine Annealing learning rate schedulers |
| [silhouette_dataset.py](silhouette_dataset.py) | Module/resource file silhouette_dataset.py |
| [train_silhouette_unet.py](train_silhouette_unet.py) | Module/resource file train_silhouette_unet.py |
| [trainer.py](trainer.py) | Main training loop coordinator executing forward/backward passes and validation checks |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

GEI Training Dataset → `training/dataset.py` & `training/dataloader.py` → `models/architectures/bygait_light.py` → `training/loss_functions.py` → `training/trainer.py` → `models/candidates/best_model.pth`.

## Configuration

- [configs/train.yaml](../configs/train.yaml): hyper-parameters, batch size, learning rate, epoch count
- [configs/auto_train.yaml](../configs/auto_train.yaml): automated re-training parameters

## Public Interfaces

- `GaitTrainer`: Training coordinator in [training/trainer.py](trainer.py).
- `GaitDataset`: PyTorch Dataset in [training/dataset.py](dataset.py).
- `TrainingLogger`: Logging callback in [training/callbacks.py](callbacks.py).
- `CombinedLoss`: Loss function in [training/loss_functions.py](loss_functions.py).

## Tests

- [tests/unit/test_output_layout.py](../tests/unit/test_output_layout.py)
- [scripts/evaluate_model.py](../scripts/evaluate_model.py)

## Related Documentation

- [Root README](../README.md)
- [Models Documentation](../models/README.md)
- [Evaluation Documentation](../evaluation/README.md)
