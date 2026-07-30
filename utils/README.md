# Utils

The `utils` package provides general-purpose helpers for bounding box stabilization, prediction smoothing, detection event reporting, heads-up display (HUD) rendering, threat alert management, and CSV logging in ARGUS AI.

## Responsibilities

- Stabilizing raw YOLO bounding boxes using Exponential Moving Average (EMA) filtering (`BoxStabilizer`).
- Smoothing temporal identity predictions over a sliding vote window (`PredictionSmoother`).
- Generating structured JSONL/CSV detection reports and cropping event snapshots (`DetectionReporter`).
- Rendering bounding boxes, identity labels, confidence scores, and status overlays (`DetectionDisplayRenderer`).
- Managing threat alerts and CSV recognition logs (`AlertManager`, `EventLogger`).
- Boundaries: Does not run deep neural network forward passes or manage camera worker threads.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [alert_manager.py](alert_manager.py) | Manages threat alerts, cooldown deduplication, and `outputs/logs/events/alerts.csv` logging |
| [box_stabilizer.py](box_stabilizer.py) | Exponential Moving Average (EMA) bounding box coordinate stabilizer |
| [config_validator.py](config_validator.py) | Module/resource file config_validator.py |
| [detection_reporter.py](detection_reporter.py) | Thread-safe reporter generating JSONL/CSV detection logs and snapshots in `outputs/media/detections/` |
| [display_renderer.py](display_renderer.py) | Open-CV image renderer drawing bounding boxes, status color codes, and identity HUD labels |
| [event_logger.py](event_logger.py) | Thread-safe CSV recognition logger writing events to `outputs/logs/events/recognition_log.csv` |
| [helpers.py](helpers.py) | Miscellaneous helper functions for string formatting and filesystem operations |
| [io_utils.py](io_utils.py) | File I/O helpers for JSON, YAML, and image file loading/saving |
| [math_utils.py](math_utils.py) | Mathematical helper functions for IoU, vector normalization, and distance metrics |
| [prediction_smoother.py](prediction_smoother.py) | Temporal sliding-window prediction smoother preventing identity flickering |
| [queue_utils.py](queue_utils.py) | Queue utilities for safe multi-threaded data passing |
| [video.py](video.py) | Video file reader/writer helper functions |
| [zip_streamer.py](zip_streamer.py) | Utilities for zipping and streaming archived dataset/report artifacts |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Raw Detection Boxes & Matches → `utils/box_stabilizer.py` & `utils/prediction_smoother.py` → `utils/display_renderer.py` (Screen HUD) & `utils/detection_reporter.py` (`outputs/media/detections/`).

## Configuration

- [configs/inference.yaml](../configs/inference.yaml): `reporting` section (`output_dir`, `snapshot_dir`, `cooldown_seconds`)

## Public Interfaces

- `DetectionReporter`: Event reporter in [utils/detection_reporter.py](detection_reporter.py).
- `DetectionDisplayRenderer`: Visual renderer in [utils/display_renderer.py](display_renderer.py).
- `BoxStabilizer`: EMA stabilizer in [utils/box_stabilizer.py](box_stabilizer.py).
- `PredictionSmoother`: Prediction smoother in [utils/prediction_smoother.py](prediction_smoother.py).
- `AlertManager`: Alert manager in [utils/alert_manager.py](alert_manager.py).
- `EventLogger`: Event logger in [utils/event_logger.py](event_logger.py).

## Tests

- [tests/integration/test_dual_modal_pipeline.py](../tests/integration/test_dual_modal_pipeline.py)
- [tests/unit/test_output_layout.py](../tests/unit/test_output_layout.py)

## Related Documentation

- [Root README](../README.md)
- [Pipeline Documentation](../pipeline/README.md)
