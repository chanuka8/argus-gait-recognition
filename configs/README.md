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
| --- | --- |
| [auto_train.yaml](auto_train.yaml) | Automated model re-training trigger rules |
| [base.yaml](base.yaml) | Base system-wide default settings and path definitions |
| [cameras.yaml](cameras.yaml) | RTSP and USB camera stream source definitions |
| [continuous_learning.yaml](continuous_learning.yaml) | Module/resource file continuous_learning.yaml |
| [detection.yaml](detection.yaml) | YOLO person detector parameters and confidence bounds |
| [gallery_probe_manifest.json](gallery_probe_manifest.json) | Evaluation gallery and probe set split definitions |
| [gei.yaml](gei.yaml) | Silhouette extraction and GEI temporal window parameters |
| [gpu_profiles.yaml](gpu_profiles.yaml) | Memory limits and batch size settings per GPU tier |
| [inference.yaml](inference.yaml) | Gait and appearance recognition thresholds, ReID, and reporting parameters |
| [logging.yaml](logging.yaml) | Log output targets and log level rules |
| [mode_config.yaml](mode_config.yaml) | Mode settings for live, video, and multi-camera execution |
| [production.yaml](production.yaml) | Module/resource file production.yaml |
| [subject_split.json](subject_split.json) | Train/test subject-disjoint partition definitions |
| [system.yaml](system.yaml) | Camera, logging, watchdog, and service configuration |
| [train.yaml](train.yaml) | CNN training hyper-parameters, learning rates, and loss weights |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

YAML/JSON Manifests → `core/config.py` / `utils/detection_reporter.py` → In-memory Config Dictionaries → Pipeline Engine.

## Configuration

- Key files: [system.yaml](system.yaml), [inference.yaml](inference.yaml), [base.yaml](base.yaml).

## Public Interfaces

- Standard YAML files loaded via `PyYAML` (`yaml.safe_load`).
- Standard JSON files loaded via `json.load`.

## Tests

- [tests/test_audit_verification.py](../tests/test_audit_verification.py)
- [tests/unit/test_output_layout.py](../tests/unit/test_output_layout.py)

## Related Documentation

- [Root README](../README.md)
- [Core Documentation](../core/README.md)
