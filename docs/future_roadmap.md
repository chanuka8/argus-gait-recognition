# Future Roadmap

This document outlines the strategic engineering upgrades and research directions planned to transition the ARGUS prototype to an enterprise-grade surveillance solution.

---

## 1. Keypoint-Based Pose Extraction

To address background subtraction noise and clothing changes, the system will integrate keypoint-based pose estimation:
- **Component:** Complete [preprocessing/skeleton_extractor.py](file:///e:/ARGUS_AI/preprocessing/skeleton_extractor.py).
- **Technology:** Leverage YOLOv8-Pose or MediaPipe to extract 2D/3D joint coordinates (ankles, knees, hips, shoulders) from target crops.
- **Goal:** Build keypoint trajectory models that focus purely on joint motion dynamics. This approach is invariant to clothing outlines, carrying items, and background noise.

---

## 2. Advanced Deep Metric Learning

Improve the discriminative power of the gait feature encoder:
- **Component:** Wire the existing [TripletLoss](file:///e:/ARGUS_AI/models/architectures/losses.py) and [gait_encoder.py](file:///e:/ARGUS_AI/models/architectures/gait_encoder.py) stubs into the training pipeline.
- **Technology:** Transition from cross-entropy classification loss to triplet margin loss. This forces embeddings of the same subject closer together while pushing embeddings of different subjects further apart in the vector space.
- **Goal:** Improve generalizability to unseen classes in open-set scenarios.

---

## 3. Distributed Vector Indexes

Support large-scale biometric databases with millions of profiles:
- **Component:** Upgrade [storage/vector_store.py](file:///e:/ARGUS_AI/storage/vector_store.py).
- **Technology:** Replace flat NumPy scans with **Faiss (Facebook AI Similarity Search)** or **Milvus**.
- **Goal:** Implement hierarchical navigable small world (HNSW) graphs and product quantization (PQ) to maintain sub-millisecond query latency at scale.

---

## 4. AutoML Retraining & Feedback Loops

Implement automated self-learning loops:
- **Component:** Implement [automation/auto_trainer.py](file:///e:/ARGUS_AI/automation/auto_trainer.py).
- **Technology:** Set up triggers that monitor new enrollments. Once a threshold of new profiles is met, the system schedules background retraining, validates the new candidate model weights against safety gates, and deploys it without downtime.

---

## 5. Edge Hardware Optimization

Optimize model inference for deployment on resource-constrained devices:
- **Technology:** Convert PyTorch model weights to **ONNX** formats, compile via **TensorRT** for NVIDIA Jetson platforms, or target **OpenVINO** for Intel hardware.
- **Goal:** Reduce inference latency below $30$ ms, enabling $30$ FPS real-time tracking on edge surveillance devices.
