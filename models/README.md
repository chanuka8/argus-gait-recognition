# Models

The `models` package defines PyTorch neural network architectures, pre-trained weights, candidate checkpoints, and identity feature galleries for ARGUS AI.

## Responsibilities

- Defining deep learning architecture models for GEI gait feature extraction (`ByGaitLight`).
- Managing active model checkpoints (`models/active/`) and rollback backups (`models/rollback/`).
- Storing enrolled gait identity templates (`models/gallery/`) and live enrolled templates (`models/live_gallery/`).
- Boundaries: Does not train models directly (training code lives in `training/`).

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| `active/` | Module/resource file active/ |
| `appearance_gallery/` | Module/resource file appearance_gallery/ |
| [architectures/bygait_light.py](file:///E:/ARGUS_AI/models/architectures/bygait_light.py) | ByGaitLight 3-block lightweight CNN architecture mapping GEIs to 256-dim embeddings |
| `candidates/` | Module/resource file candidates/ |
| `gallery/` | Module/resource file gallery/ |
| `live_gallery/` | Module/resource file live_gallery/ |
| `reid/` | Module/resource file reid/ |
| `rollback/` | Module/resource file rollback/ |
| `weights/` | Module/resource file weights/ |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Input GEI Tensor (1×1×128×64) → `models/architectures/bygait_light.py` (`ByGaitLight.forward()`) → 256-dim L2-Normalized Embedding Vector.

## Configuration

- [configs/system.yaml](file:///e:/ARGUS_AI/configs/system.yaml): `recognition.model_path`, `recognition.gallery_dir`
- [configs/inference.yaml](file:///e:/ARGUS_AI/configs/inference.yaml): model inference parameters

## Public Interfaces

- `ByGaitLight`: PyTorch neural network model in [models/architectures/bygait_light.py](file:///e:/ARGUS_AI/models/architectures/bygait_light.py).

## Tests

- [tests/integration/test_dual_modal_pipeline.py](file:///e:/ARGUS_AI/tests/integration/test_dual_modal_pipeline.py)
- [scripts/evaluate_model.py](file:///e:/ARGUS_AI/scripts/evaluate_model.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Training Documentation](file:///e:/ARGUS_AI/training/README.md)
