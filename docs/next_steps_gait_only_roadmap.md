# ARGUS Gait Module: Next Steps & Gait-Only Development Roadmap

**Date:** 2026-06-13  
**Status:** Completed

This document outlines the roadmap and next steps for the ARGUS Gait Recognition Module as a specialized, gait-only final-year research prototype.

---

## 1. Context & Present State
The ARGUS active runtime has been successfully restricted to a pure video/camera-based gait pipeline. Silhouettes are extracted from camera feeds or video files, averaged into Gait Energy Images (GEIs), mapped into 256-dimensional embeddings by `ByGaitLight`, and matched against `models/live_gallery` with a cosine similarity threshold of `0.85`.

Photo-only matching has been identified as a runtime limitation for gait-only systems due to the lack of motion profile features in static imagery. The codebase retains `AppearanceFeatureExtractionStep` and `AppearanceMatchingStep` as legacy components/stubs, but they are disconnected from active execution.

---

## 2. Next Steps & Recommended Research Upgrades

For future police-level or high-security deployments, the following academic research and engineering improvements are recommended:

### 1. Integrate Deep Person Re-Identification (Person ReID)
- **Problem:** Static photo matching currently relies on simple deep feature extraction, which struggles under variable angles, shadows, or occlusions.
- **Solution:** Integrate a specialized Person ReID model such as **OSNet** (Omni-Scale Network), **FastReID**, or **TorchReID**. This will enable robust static-to-video appearance matching to work alongside the gait module.

### 2. Transition from Silhouette (MOG2) to Pose-Estimation Keypoints
- **Problem:** Silhouette segmentation using Mixture of Gaussians (MOG2) is highly sensitive to background noise, shadows, lighting changes, and clothing variations.
- **Solution:** Transition to keypoint-based pose estimation using YOLOv8-Pose or HRNet. Extracting coordinate trajectories of joints (knees, ankles, hips) makes the walking signature invariant to clothing, shadows, and color variations.

### 3. Add Dynamic Viewpoint Invariance (GaitGAN / View Synthesis)
- **Problem:** Walking silhouettes vary dramatically depending on the camera angle (e.g., side profile vs. frontal profile).
- **Solution:** Integrate a Gait Generative Adversarial Network (GaitGAN) or a View Synthesis network to transform any oblique walking view into a standardized 90-degree side profile view before classification.

### 4. Implement Scalable Indexing (FAISS)
- **Problem:** Performing linear cosine scanning (via NumPy) slows down when scaling to thousands of registered identities.
- **Solution:** Integrate **FAISS** (Facebook AI Similarity Search) to build Hierarchical Navigable Small World (HNSW) index graphs for sub-millisecond similarity lookups.
