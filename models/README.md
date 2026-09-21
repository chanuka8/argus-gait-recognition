# Models

The `models` package defines PyTorch neural network architectures, pre-trained weights, candidate checkpoints, and identity feature galleries for ARGUS AI.

## Responsibilities

- Defining deep learning architecture models for GEI gait feature extraction (`ByGaitLight`).
- Managing active model checkpoints (`models/model_store/active/`) and rollback backups (`models/model_store/rollback/`).
- Storing enrolled gait identity templates (`models/galleries/gallery/`) and live enrolled templates (`models/galleries/live_gallery/`).
- Boundaries: Does not train models directly (training code lives in `training/`).

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [architectures/bygait_light.py](architectures/bygait_light.py) | ByGaitLight 3-block lightweight CNN architecture mapping GEIs to 256-dim embeddings |
| `reid/` | Module/resource file reid/ |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Input GEI Tensor (1×1×128×64) → `models/architectures/bygait_light.py` (`ByGaitLight.forward()`) → 256-dim L2-Normalized Embedding Vector.

## Configuration

- [configs/system.yaml](../configs/system.yaml): `recognition.model_path`, `recognition.gallery_dir`
- [configs/inference.yaml](../configs/inference.yaml): model inference parameters

## Public Interfaces

- `ByGaitLight`: PyTorch neural network model in [models/architectures/bygait_light.py](architectures/bygait_light.py).
- `get_inference_backend`: Pluggable inference backend factory (PyTorch, ONNX, TensorRT) in [models/inference/backend.py](inference/backend.py).

## Tests

- [tests/integration/test_dual_modal_pipeline.py](../tests/integration/test_dual_modal_pipeline.py)
- [evaluation/experiments/evaluate_model.py](../evaluation/experiments/evaluate_model.py)

## Related Documentation

- [Root README](../README.md)
- [Training Documentation](../training/README.md)
