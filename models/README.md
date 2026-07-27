# Models

The `models` package defines PyTorch neural network architectures, pre-trained weights, candidate checkpoints, and identity feature galleries for ARGUS AI.

## Responsibilities

- Defining deep learning architecture models for GEI gait feature extraction (`ByGaitLight`).
- Managing active model checkpoints (`models/active/`) and rollback backups (`models/rollback/`).
- Storing enrolled gait identity templates (`models/gallery/`) and live enrolled templates (`models/live_gallery/`).
- Boundaries: Does not train models directly (training code lives in `training/`).

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module / Directory | Purpose |
|---|---|
| [architectures/bygait_light.py](file:///e:/ARGUS_AI/models/architectures/bygait_light.py) | ByGaitLight 3-block lightweight CNN architecture mapping GEIs to 256-dim embeddings |
| `active/` | Directory containing currently active production model weights |
| `appearance_gallery/` | Directory storing enrolled person appearance ReID feature embeddings |
| `candidates/` | Directory holding candidate model checkpoints undergoing validation |
| `gallery/` | Directory storing baseline enrolled gait GEI feature matrix and identity metadata |
| `live_gallery/` | Directory holding live enrolled identity templates |
| `reid/` | Directory containing ReID appearance model checkpoints |
| `rollback/` | Directory containing backup model weights for instant rollback |
| `weights/` | Directory containing object detector weights (e.g. `yolov8n.pt`) |
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
