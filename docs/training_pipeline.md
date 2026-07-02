# Training Pipeline

This document details the neural network architecture, data loaders, training configs, and the steps to execute model training.

---

## 1. Model Architecture: ByGaitLight

ARGUS uses **ByGaitLight**, a lightweight deep convolutional neural network optimized for extracting discriminative features from Gait Energy Images (GEI). Designed for real-time edge processing, it minimizes parameters while maintaining high identification accuracy.

### Layer Specifications

| Layer | Type | Configuration Details | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **Input** | Grayscale GEI | 1 Channel, Normalized Grayscale | `[Batch, 1, 128, 64]` |
| **Conv Block 1** | Conv2d -> BN -> ReLU -> MaxPool | 32 Filters, $3\times3$ Kernel, Pool $2\times2$ | `[Batch, 32, 64, 32]` |
| **Conv Block 2** | Conv2d -> BN -> ReLU -> MaxPool | 64 Filters, $3\times3$ Kernel, Pool $2\times2$ | `[Batch, 64, 32, 16]` |
| **Conv Block 3** | Conv2d -> BN -> ReLU -> MaxPool | 128 Filters, $3\times3$ Kernel, Pool $2\times2$ | `[Batch, 128, 16, 8]` |
| **Global Pool** | AdaptiveAvgPool2d | Average pooling down to 1x1 | `[Batch, 128, 1, 1]` |
| **Fully Connected** | Linear | Dense projections to output size | `[Batch, 256]` |
| **L2 Norm** | Vector Normalization | Force output to lie on unit hypersphere | `[Batch, 256]` |

The resulting **256-dimensional embedding** is L2-normalized, allowing similarity comparisons to be executed via cosine similarity.

---

## 2. Dataloaders & Data Splitting

The training pipeline loads images via [training/dataset.py](file:///e:/ARGUS_AI/training/dataset.py#L9) and structures batches using [training/dataloader.py](file:///e:/ARGUS_AI/training/dataloader.py#L8).
- **Target Images:** Grayscale GEI PNG files ($128 \times 64$).
- **Partition Splits:** By default, data is split subject-wise or sequence-wise (e.g., sequences 1-4 for training, sequences 5-6 for validation) to prevent data leakage during testing.
- **Labels mapping:** Subdirectory names (e.g., `001`, `034`) serve as the class labels.

---

## 3. Training Configurations

Training parameters are configured in [configs/train.yaml](file:///e:/ARGUS_AI/configs/train.yaml) or passed directly as command line options:
- **Optimizer:** Adam with $L_2$ weight decay ($10^{-4}$).
- **Initial Learning Rate:** $10^{-3}$ with adaptive decay policies.
- **Loss Function:** Cross-entropy loss over output class predictions.
- **Checkpoints:** The checkpointer tracks the validation loss and saves the best model checkpoint weights to `runs/exp_001/best_model.pth`.

---

## 4. Training Command

To run the training process:
```bash
python cli.py --mode train --epochs 20 --batch-size 32 --max-classes 100
```
- `--epochs`: Number of training iterations (default: 20).
- `--batch-size`: Images processed per batch (default: 32).
- `--max-classes`: Truncate dataset to $N$ subject identities for faster validation or staged training.
- Checkpoints, logs, and loss graphs are saved to `runs/exp_001/`.
