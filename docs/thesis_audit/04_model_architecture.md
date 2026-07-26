# Phase 3 — Model Architecture Analysis

## 4.1 Model Identity

| Property | Value |
|---|---|
| **Model Name** | ByGaitLight |
| **Type** | Lightweight 2D CNN Feature Extractor |
| **Framework** | PyTorch |
| **File** | `models/architectures/bygait_light.py` |
| **Input Shape** | `(batch, 1, 128, 64)` — single-channel 128×64 GEI image |
| **Output** | 256-dimensional L2-normalized embedding vector |
| **Embedding Dimension** | 256 |

## 4.2 Layer-by-Layer Architecture

| # | Layer | Type | In Channels | Out Channels | Kernel | Padding | Output Shape |
|---|---|---|---|---|---|---|---|
| 1 | `features.0` | Conv2d | 1 | 32 | 3×3 | 1 | (B, 32, 128, 64) |
| 2 | `features.1` | BatchNorm2d | 32 | 32 | — | — | (B, 32, 128, 64) |
| 3 | `features.2` | ReLU (inplace) | — | — | — | — | (B, 32, 128, 64) |
| 4 | `features.3` | MaxPool2d | — | — | 2×2 | — | (B, 32, 64, 32) |
| 5 | `features.4` | Conv2d | 32 | 64 | 3×3 | 1 | (B, 64, 64, 32) |
| 6 | `features.5` | BatchNorm2d | 64 | 64 | — | — | (B, 64, 64, 32) |
| 7 | `features.6` | ReLU (inplace) | — | — | — | — | (B, 64, 64, 32) |
| 8 | `features.7` | MaxPool2d | — | — | 2×2 | — | (B, 64, 32, 16) |
| 9 | `features.8` | Conv2d | 64 | 128 | 3×3 | 1 | (B, 128, 32, 16) |
| 10 | `features.9` | BatchNorm2d | 128 | 128 | — | — | (B, 128, 32, 16) |
| 11 | `features.10` | ReLU (inplace) | — | — | — | — | (B, 128, 32, 16) |
| 12 | `features.11` | MaxPool2d | — | — | 2×2 | — | (B, 128, 16, 8) |
| 13 | `pool` | AdaptiveAvgPool2d | — | — | → (1,1) | — | (B, 128, 1, 1) |
| 14 | flatten | torch.flatten | — | — | — | — | (B, 128) |
| 15 | `embedding` | Linear | 128 | 256 | — | — | (B, 256) |
| 16 | L2-Normalize | F.normalize(p=2, dim=1) | — | — | — | — | (B, 256) |

## 4.3 Tensor Shape Flow

```
Input:           (B, 1, 128, 64)    [GEI grayscale image]
After Conv1+BN+ReLU:   (B, 32, 128, 64)
After MaxPool1:        (B, 32, 64, 32)
After Conv2+BN+ReLU:   (B, 64, 64, 32)
After MaxPool2:        (B, 64, 32, 16)
After Conv3+BN+ReLU:   (B, 128, 32, 16)
After MaxPool3:        (B, 128, 16, 8)
After AdaptiveAvgPool: (B, 128, 1, 1)
After Flatten:         (B, 128)
After Linear:          (B, 256)
After L2-Normalize:    (B, 256)   [unit-norm embedding]
```

## 4.4 Parameter Count

| Component | Parameters | Trainable |
|---|---|---|
| `features.0` (Conv2d 1→32, 3×3) | 32 × (1×3×3 + 1) = 320 | Yes |
| `features.1` (BN 32) | 32 × 2 = 64 | Yes |
| `features.4` (Conv2d 32→64, 3×3) | 64 × (32×3×3 + 1) = 18,496 | Yes |
| `features.5` (BN 64) | 64 × 2 = 128 | Yes |
| `features.8` (Conv2d 64→128, 3×3) | 128 × (64×3×3 + 1) = 73,856 | Yes |
| `features.9` (BN 128) | 128 × 2 = 256 | Yes |
| `embedding` (Linear 128→256) | 128 × 256 + 256 = 33,024 | Yes |
| **Total (backbone only)** | **~126,144** | **All** |

## 4.5 Training Configuration (from metrics.json)

| Property | Value |
|---|---|
| **Loss Mode** | `ce_arcface` (CrossEntropy + ArcFace) |
| **ArcFace Scale (s)** | 64.0 |
| **ArcFace Margin (m)** | 0.35 |
| **Triplet Margin** | 0.3 |
| **Triplet Weight** | 0.0 (triplet loss was disabled) |
| **Optimizer** | Adam (from `training/optimizer.py`) |
| **Learning Rate** | 0.0001 |
| **Scheduler** | CosineAnnealingLR (T_max=epochs, eta_min=1e-5) |
| **Batch Size** | 16 |
| **Epochs Trained** | 50 |
| **Device** | CPU |
| **Number of Classes** | 124 |
| **Total Samples** | 13,544 |
| **Max Classes** | None (all) |
| **Max Samples** | None (all) |

## 4.6 Training Wrapper: GaitClassifier

The training wrapper (`training/trainer.py::GaitClassifier`) adds:
- A standard `nn.Linear(256, num_classes)` classifier head
- An optional `ArcMarginProduct(256, num_classes, s=64, m=0.35)` for angular-margin softmax
- The backbone weights are extracted from `GaitClassifier.backbone.*` for inference

**ArcFace Mathematical Formulation:**

```
ArcMarginProduct:
  cosθ = Linear(normalize(x), normalize(W))
  φ = cos(θ + m)     [additive angular margin]
  output = s × (one_hot × φ + (1 - one_hot) × cosθ)
```

Where `s=64` is the scaling factor and `m=0.35` is the angular margin.

## 4.7 Loss Function: JointGaitLoss

```
Total Loss = CE(loss_logits, labels) + triplet_weight × TripletLoss(embeddings, labels)
```

Since `triplet_weight = 0.0`, the effective loss was **ArcFace CrossEntropy only**.

**BatchHardTripletLoss** (implemented but not active):
- Computes pairwise L2 distance matrix
- For each anchor: selects hardest positive (max distance) and hardest negative (min distance)
- `loss = max(0, d_pos - d_neg + margin)`
- Uses `margin = 0.3`

## 4.8 Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 26.950 | 1.32% | 26.390 | 2.33% |
| 10 | 7.478 | 18.20% | 7.577 | 17.02% |
| 20 | 3.746 | 49.73% | 5.247 | 38.28% |
| 30 | 3.283 | 68.55% | 4.684 | 55.72% |
| 40 | 6.116 | 83.15% | 7.035 | 72.98% |
| 50 | 9.718 | 88.87% | 9.860 | **80.14%** |

**Best Validation Accuracy: 80.14%** (epoch 50)

> [!CAUTION]
> **Subject Leakage Warning:** This training used ALL 124 subjects (`num_classes: 124`). The validation accuracy of 80.14% is computed on a random 80/20 split of ALL samples, meaning test subjects 075-124 appeared in training data. This accuracy cannot be reported as a subject-disjoint result. See Phase 4 for full leakage analysis.

## 4.9 Checkpoint Details

| Checkpoint | Size | Description | Leakage Status |
|---|---|---|---|
| `best_model.pth` | 770 KB | Best validation accuracy (all 124 classes) | **Contains test subjects in training** |
| `last_model.pth` | 770 KB | Final epoch (same run) | **Contains test subjects in training** |
| `best_model_ce_726.pth` | 567 KB | Legacy CE-only run | Unknown |

## 4.10 Model Inference Path

```python
# At inference time, only the backbone is loaded (no classifier head):
model = ByGaitLight()
checkpoint = torch.load("runs/exp_001/best_model.pth")
filtered = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if k.startswith("backbone.")}
model.load_state_dict(filtered, strict=True)
model.eval()

# Forward pass:
gei_tensor = torch.from_numpy(gei_float).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 64)
embedding = model(gei_tensor).cpu().numpy().flatten()  # (256,)
```

## 4.11 Model Complexity Summary

| Metric | Value |
|---|---|
| **Total Parameters** | ~126,144 (backbone) |
| **All Trainable** | Yes |
| **Approximate Model Size** | ~770 KB (full GaitClassifier state_dict) |
| **Backbone-only Size** | ~504 KB |
| **Convolution Blocks** | 3 |
| **Pooling Strategy** | MaxPool2d (×3) + AdaptiveAvgPool2d |
| **Activation** | ReLU (inplace) |
| **Normalization** | BatchNorm2d (per block) |
| **Dropout** | None |
| **Early Stopping** | Not implemented (runs all epochs) |
| **Mixed Precision** | Not used |
| **Random Seed Control** | Not set in training code |
| **Estimated FLOPs** | ~12.5 MFLOPs (approximate, single inference) |

## 4.12 Thesis-Ready Model Description

> The ARGUS system employs **ByGaitLight**, a custom lightweight convolutional neural network designed for gait embedding extraction. The architecture consists of three convolutional blocks, each containing a 3×3 convolution layer, batch normalisation, ReLU activation, and 2×2 max pooling. The convolution blocks progressively increase the channel depth from 1 to 32, 64, and 128 respectively. An adaptive average pooling layer reduces the spatial dimensions to 1×1, producing a 128-dimensional feature vector. A fully connected layer projects this to a 256-dimensional embedding, which is L2-normalised to produce unit-norm vectors suitable for cosine similarity matching.
>
> During training, the backbone is paired with an ArcFace classification head (s=64, m=0.35) that applies an additive angular margin to the softmax loss, encouraging greater inter-class separation in the embedding space. The model was trained for 50 epochs using the Adam optimiser with a learning rate of 1×10⁻⁴ and cosine annealing scheduling. At inference time, only the backbone is loaded, and the classification head is discarded. The total backbone parameter count is approximately 126,000, making it suitable for CPU-based real-time deployment.
