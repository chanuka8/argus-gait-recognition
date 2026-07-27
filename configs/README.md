# Configs

The `configs` package contains declarative YAML and JSON configuration manifests that control pipeline parameters, thresholds, system options, hardware profiles, and logging settings for ARGUS AI.

## Responsibilities

- Storing declarative parameters for inference, training, system logging, and multi-camera setup.
- Providing environment profile presets for GPU execution and subject-disjoint evaluation protocols.
- Decoupling algorithmic hyper-parameters from Python codebase implementation.
- Boundaries: Does not parse or mutate files at runtime; modules in `core/config.py` read these manifests.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [base.yaml](file:///e:/ARGUS_AI/configs/base.yaml) | Base system-wide default settings and path definitions |
| [system.yaml](file:///e:/ARGUS_AI/configs/system.yaml) | Camera, logging, watchdog, and service configuration |
| [inference.yaml](file:///e:/ARGUS_AI/configs/inference.yaml) | Gait and appearance recognition thresholds, ReID, and reporting parameters |
| [logging.yaml](file:///e:/ARGUS_AI/configs/logging.yaml) | Log output targets and log level rules |
| [cameras.yaml](file:///e:/ARGUS_AI/configs/cameras.yaml) | RTSP and USB camera stream source definitions |
| [detection.yaml](file:///e:/ARGUS_AI/configs/detection.yaml) | YOLO person detector parameters and confidence bounds |
| [gei.yaml](file:///e:/ARGUS_AI/configs/gei.yaml) | Silhouette extraction and GEI temporal window parameters |
| [train.yaml](file:///e:/ARGUS_AI/configs/train.yaml) | CNN training hyper-parameters, learning rates, and loss weights |
| [auto_train.yaml](file:///e:/ARGUS_AI/configs/auto_train.yaml) | Automated model re-training trigger rules |
| [gpu_profiles.yaml](file:///e:/ARGUS_AI/configs/gpu_profiles.yaml) | Memory limits and batch size settings per GPU tier |
| [mode_config.yaml](file:///e:/ARGUS_AI/configs/mode_config.yaml) | Mode settings for live, video, and multi-camera execution |
| [subject_split.json](file:///e:/ARGUS_AI/configs/subject_split.json) | Train/test subject-disjoint partition definitions |
| [gallery_probe_manifest.json](file:///e:/ARGUS_AI/configs/gallery_probe_manifest.json) | Evaluation gallery and probe set split definitions |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

YAML/JSON Manifests → `core/config.py` / `utils/detection_reporter.py` → In-memory Config Dictionaries → Pipeline Engine.

## Configuration

- Key files: [system.yaml](file:///e:/ARGUS_AI/configs/system.yaml), [inference.yaml](file:///e:/ARGUS_AI/configs/inference.yaml), [base.yaml](file:///e:/ARGUS_AI/configs/base.yaml).

## Public Interfaces

- Standard YAML files loaded via `PyYAML` (`yaml.safe_load`).
- Standard JSON files loaded via `json.load`.

## Tests

- [tests/test_audit_verification.py](file:///e:/ARGUS_AI/tests/test_audit_verification.py)
- [tests/unit/test_output_layout.py](file:///e:/ARGUS_AI/tests/unit/test_output_layout.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Core Documentation](file:///e:/ARGUS_AI/core/README.md)
