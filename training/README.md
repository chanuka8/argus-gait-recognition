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
|---|---|
| [callbacks.py](file:///E:/ARGUS_AI/training/callbacks.py) | Early stopping, model checkpointer, and `TrainingLogger` callbacks |
| [checkpointer.py](file:///E:/ARGUS_AI/training/checkpointer.py) | Checkpoint saver and model state dictionary restorer |
| [dataloader.py](file:///E:/ARGUS_AI/training/dataloader.py) | PyTorch DataLoader factory supporting balanced identity batch sampling |
| [dataset.py](file:///E:/ARGUS_AI/training/dataset.py) | PyTorch Dataset class loading 64x128 GEI images and subject labels |
| [loss_functions.py](file:///E:/ARGUS_AI/training/loss_functions.py) | Combined loss implementations: Triplet Loss with hard mining, Cross-Entropy, Margin Loss |
| [optimizer.py](file:///E:/ARGUS_AI/training/optimizer.py) | Adam / SGD optimizer builder and Cosine Annealing learning rate schedulers |
| [trainer.py](file:///E:/ARGUS_AI/training/trainer.py) | Main training loop coordinator executing forward/backward passes and validation checks |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

GEI Training Dataset → `training/dataset.py` & `training/dataloader.py` → `models/architectures/bygait_light.py` → `training/loss_functions.py` → `training/trainer.py` → `models/candidates/best_model.pth`.

## Configuration

- [configs/train.yaml](file:///e:/ARGUS_AI/configs/train.yaml): hyper-parameters, batch size, learning rate, epoch count
- [configs/auto_train.yaml](file:///e:/ARGUS_AI/configs/auto_train.yaml): automated re-training parameters

## Public Interfaces

- `GaitTrainer`: Training coordinator in [training/trainer.py](file:///e:/ARGUS_AI/training/trainer.py).
- `GaitDataset`: PyTorch Dataset in [training/dataset.py](file:///e:/ARGUS_AI/training/dataset.py).
- `TrainingLogger`: Logging callback in [training/callbacks.py](file:///e:/ARGUS_AI/training/callbacks.py).
- `CombinedLoss`: Loss function in [training/loss_functions.py](file:///e:/ARGUS_AI/training/loss_functions.py).

## Tests

- [tests/unit/test_output_layout.py](file:///e:/ARGUS_AI/tests/unit/test_output_layout.py)
- [scripts/evaluate_model.py](file:///e:/ARGUS_AI/scripts/evaluate_model.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Models Documentation](file:///e:/ARGUS_AI/models/README.md)
- [Evaluation Documentation](file:///e:/ARGUS_AI/evaluation/README.md)
