# ARGUS AI — Model Architecture Audit Report

**Report Generated:** 2026-07-30T10:37:00+05:30  
**Repository Version / Commit Hash:** `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`  
**Working Tree Status:** Dirty (Uncommitted documentation/report suite additions)  
**Audit Policy:** Zero False Positive Evidence-Based Reporting Policy  
**Target Path:** `docs/reports/MODEL_ARCHITECTURE_REPORT.md`

---

## 1. Overview & Specification

The core embedding architecture of ARGUS AI is **ByGaitLight**, a lightweight convolutional neural network designed for fast, CPU-friendly gait feature extraction from Gait Energy Images (GEI).

| Property | Value | Type | Evidence Source / File Reference | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture Name** | `ByGaitLight` | Configured | [bygait_light.py](../../models/architectures/bygait_light.py) | **Verified** |
| **Input Tensor Shape** | `(1, 1, 64, 64)` | Configured | [model_config.yaml](../../configs/model_config.yaml) | **Verified** |
| **Output Tensor Shape** | `(1, 128)` | Configured | [bygait_light.py](../../models/architectures/bygait_light.py) | **Verified** |
| **Embedding Dimension** | `128` | Configured | [bygait_light.py](../../models/architectures/bygait_light.py) | **Verified** |
| **Output Normalization** | `L2 Normalization` (`F.normalize(x, p=2, dim=1)`) | Configured | [bygait_light.py](../../models/architectures/bygait_light.py) | **Verified** |
| **Embedding Storage Size** | `512 bytes` (128 x Float32 values) | Derived | $128 \times 4\text{ bytes}$ | **Verified** |

---

## 2. Layer & Module Inventory

The `ByGaitLight` backbone consists of three sequential convolutional blocks followed by adaptive global average pooling and a dense projection layer.

```
Input Tensor (1, 1, 64, 64)
  │
  ├── Block 1: Conv2d(1 -> 32, kernel=3, padding=1) ──> BatchNorm2d(32) ──> ReLU ──> MaxPool2d(2, 2)  [Output: 32 x 32 x 32]
  ├── Block 2: Conv2d(32 -> 64, kernel=3, padding=1) ──> BatchNorm2d(64) ──> ReLU ──> MaxPool2d(2, 2)  [Output: 64 x 16 x 16]
  ├── Block 3: Conv2d(64 -> 128, kernel=3, padding=1) ──> BatchNorm2d(128) ──> ReLU                     [Output: 128 x 16 x 16]
  ├── Global Pool: AdaptiveAvgPool2d((1, 1))                                                            [Output: 128 x 1 x 1]
  ├── Flatten: Flatten(start_dim=1)                                                                    [Output: 128]
  └── Projection: Linear(128 -> 128) ──> L2 Normalization                                              [Output: 128]
```

---

## 3. Parameter Breakdown & Storage Metrics

Values were verified directly by inspecting the PyTorch checkpoint (`runs/exp_001/best_model.pth`) and ONNX engine file (`models/engines/bygait_light.onnx`).

| Parameter Category | Value | Type | Verification Source | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone Embedder Params** | **126,144** (`0.126 M`) | Measured | Checkpoint `backbone.*` tensor elements | **Verified** |
| **Classifier Head Params** | **63,612** (`0.064 M`) | Measured | Checkpoint `classifier.*` & `arcface.*` | **Verified** |
| **Trainable Model Params** | **189,756** (`0.190 M`) | Measured | Sum of trainable parameters | **Verified** |
| **Non-Trainable State Buffers** | **451** | Measured | BatchNorm `running_mean`, `running_var`, `num_batches_tracked` | **Verified** |
| **Total Checkpoint Elements** | **190,207** | Measured | `torch.load('runs/exp_001/best_model.pth')` | **Verified** |
| **PyTorch Checkpoint Size** | **0.73 MB** (`763,893 bytes`) | Measured | `runs/exp_001/best_model.pth` file size | **Verified** |
| **ONNX File Size** | **0.48 MB** (`505,589 bytes`) | Measured | `models/engines/bygait_light.onnx` file size | **Verified** |
| **Compression Ratio** | **1.51x** | Derived | Checkpoint size / ONNX size ($763,893 / 505,589$) | **Verified** |

---

## 4. FLOPs & MACs Computation

Profiling was conducted using the installed `thop` package.

| Computational Metric | Value | Type | Profiling Tool & Environment | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Profiling Tool** | `thop` v2.0.20 | Measured | `venv/Scripts/python.exe` | **Verified** |
| **Input Shape** | `(1, 1, 64, 64)` | Configured | Synthetic batch size 1 GEI tensor | **Verified** |
| **MACs (Multiply-Accumulate)** | **39,886,976** (`39.89 M`) | Measured | Executed twice with identical output | **Verified** |
| **FLOPs (Floating Point Ops)** | **79,773,952** (`79.77 M`) | Derived | Defined as $2 \times \text{MACs}$ | **Verified** |
| **Profiler Limitations** | Skips zero-ops | Observation | THOP registers Conv2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, and skips ReLU/MaxPool2d zero-ops | **Verified** |

---

## 5. Training Strategy & Loss Function

| Training Setting | Value | Type | Source File | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Loss Function** | `ArcFace` (Additive Angular Margin Loss) | Configured | [losses.py](../../training/losses.py) | **Verified** |
| **Scale Parameter ($s$)** | `30.0` | Configured | [losses.py](../../training/losses.py) | **Verified** |
| **Margin Parameter ($m$)** | `0.50` | Configured | [losses.py](../../training/losses.py) | **Verified** |
| **Subject Split Protocol** | Subject-Disjoint (62 Train / 12 Val / 50 Test) | Configured | [subject_split.json](../../configs/subject_split.json) | **Verified** |
| **Checkpoint Path** | `runs/exp_001/best_model.pth` | Measured | File system check | **Verified** |

---

## 6. Architecture Limitations

- **Resolution Sensitivity:** Fixed input size of $64 \times 64$; silhouettes must be cropped and resized prior to embedding.
- **Single Silhouette Input:** Takes a 2D Gait Energy Image (GEI) frame rather than a 3D video sequence tensor.
- **Feature Capacity:** Compact 128-dimensional embedding space optimized for low latency rather than extreme multi-million identity galleries.

---
**Status:** `VERIFIED - MODEL ARCHITECTURE AUDITED`
