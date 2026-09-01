# Pipeline

The `pipeline` package implements high-level recognition execution pipelines for live camera streams, pre-recorded video files, and multi-camera CCTV networks in ARGUS AI.

## Responsibilities

- Orchestrating frame-by-frame person detection, ByteTrack tracking, silhouette extraction, GEI synthesis, embedding match search, and decision output.
- Supporting execution modes: single live camera, video file batch processing, and multi-camera CCTV networks.
- Providing modular pipeline steps in `pipeline/steps/`.
- Boundaries: Does not handle OS background service registration or raw network socket IO.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [base_pipeline.py](base_pipeline.py) | Abstract base class defining pipeline lifecycle, hook interfaces, and step execution |
| [cache_engine.py](cache_engine.py) | High-speed feature and silhouette caching engine for pipeline steps |
| `camera/` | Module/resource file camera/ |
| `detection/` | Module/resource file detection/ |
| [folder_recognition.py](folder_recognition.py) | Executes recognition over folders of input video files or GEI images |
| `gei/` | Module/resource file gei/ |
| [inference_pipeline.py](inference_pipeline.py) | Core inference engine running end-to-end gait feature extraction |
| [live_recognition.py](live_recognition.py) | Real-time live camera recognition pipeline handling frame queues and display overlay |
| [multi_camera_recognition.py](multi_camera_recognition.py) | Multi-camera recognition pipeline orchestrating cross-camera tracking and evidence fusion |
| [pipeline_factory.py](pipeline_factory.py) | Factory instantiating pipeline implementations based on source mode |
| `silhouette/` | Module/resource file silhouette/ |
| [speed_controller.py](speed_controller.py) | Controls frame processing rate to align with target FPS constraints |
| `steps/` | Module/resource file steps/ |
| `tracking/` | Module/resource file tracking/ |
| [video_recognition.py](video_recognition.py) | Offline video file recognition pipeline for recorded video analysis |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Video Frame Stream → Detection & ByteTrack → Silhouette Extraction → GEI Builder → CNN Feature Embedding → Cosine Matching → Adaptive Decision Policy → Detection Reporter.

## Configuration

- [configs/inference.yaml](../configs/inference.yaml): thresholds, GEI settings, ReID, reporting
- [configs/mode_config.yaml](../configs/mode_config.yaml): pipeline mode options

## Public Interfaces

- `LiveRecognitionPipeline`: Live webcam/RTSP pipeline in [pipeline/live_recognition.py](live_recognition.py).
- `VideoRecognitionPipeline`: Video file pipeline in [pipeline/video_recognition.py](video_recognition.py).
- `MultiCameraRecognitionPipeline`: Multi-camera pipeline in [pipeline/multi_camera_recognition.py](multi_camera_recognition.py).
- `PipelineFactory`: Instantiator in [pipeline/pipeline_factory.py](pipeline_factory.py).

## Tests

- [tests/integration/test_dual_modal_pipeline.py](../tests/integration/test_dual_modal_pipeline.py)
- [tests/test_detector.py](../tests/test_detector.py)
- [tests/test_tracker.py](../tests/test_tracker.py)
- [tests/test_silhouette.py](../tests/test_silhouette.py)
- [tests/test_gei_stream.py](../tests/test_gei_stream.py)

## Related Documentation

- [Root README](../README.md)
- [Intelligence Documentation](../intelligence/README.md)
