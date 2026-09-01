# Streaming

The `streaming` package provides high-throughput video stream ingestion, thread-safe ring buffering, camera workload scheduling, worker thread pool execution, and frame dropping controls for ARGUS AI.

## Responsibilities

- Capturing video frames continuously from RTSP streams, USB webcams, and video files without UI thread blocking.
- Managing thread-safe ring buffer queues (`BufferQueue`) with overflow protection and frame dropping.
- Scheduling multi-camera workloads across worker thread pools (`WorkerPool`, `CameraScheduler`).
- Boundaries: Does not run deep neural network feature extraction or match vector similarity.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [buffer_queue.py](buffer_queue.py) | Thread-safe bounded ring buffer queue with non-blocking put/get operations |
| [camera_scheduler.py](camera_scheduler.py) | Round-robin and priority scheduler distributing camera stream frame processing |
| [deployment_readiness.py](deployment_readiness.py) | Phase 5 production deployment readiness, hardware capability discovery, dynamic system profiling, capacity estimation, and admission control |
| [frame_dropper.py](frame_dropper.py) | Intelligent frame dropper maintaining target processing FPS under CPU congestion |
| [load_balancer.py](load_balancer.py) | Balances processing workload across multi-stream worker threads |
| [multi_stream_engine.py](multi_stream_engine.py) | Multi-stream acquisition engine handling concurrent CCTV streams |
| [performance_optimizer.py](performance_optimizer.py) | Dynamic performance optimizer adjusting frame queue sizes and capture rates |
| [person_track_scheduler.py](person_track_scheduler.py) | Module/resource file person_track_scheduler.py |
| [production_multicamera_engine.py](production_multicamera_engine.py) | Production-grade multi-camera scalability and hardware-agnostic inference engine |
| [production_runtime.py](production_runtime.py) | Phase 4 production surveillance hardening runtime with camera lifecycle, reconnect, failure isolation, adaptive resources, model hot-swap, and graceful shutdown |
| [stream_engine.py](stream_engine.py) | Core single-stream acquisition engine wrapping OpenCV `VideoCapture` |
| [worker_pool.py](worker_pool.py) | Thread worker pool executing concurrent stream capture tasks |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

RTSP / USB / File Stream → `streaming/stream_engine.py` → `streaming/buffer_queue.py` → `streaming/frame_dropper.py` → Pipeline Capture Hook.

## Configuration

- [configs/system.yaml](../configs/system.yaml): `camera.target_fps`, `camera.max_queue_size`, `camera.type`, `camera.url`
- [configs/cameras.yaml](../configs/cameras.yaml): RTSP camera streams

## Public Interfaces

- `ProductionMultiCameraEngine`: Production multi-camera scalability engine in [streaming/production_multicamera_engine.py](production_multicamera_engine.py).
- `DeploymentReadinessManager`: Phase 5 deployment readiness manager in [streaming/deployment_readiness.py](deployment_readiness.py).
- `HardwareCapabilityDetector`: Hardware capability discovery engine in [streaming/deployment_readiness.py](deployment_readiness.py).
- `CameraAdmissionController`: Pre-flight camera admission controller in [streaming/deployment_readiness.py](deployment_readiness.py).
- `ProductionSurveillanceRuntime`: Phase 4 production surveillance runtime in [streaming/production_runtime.py](production_runtime.py).
- `CameraStateMachine`: Camera lifecycle state machine in [streaming/production_runtime.py](production_runtime.py).
- `ReconnectEngine`: Exponential backoff reconnect engine in [streaming/production_runtime.py](production_runtime.py).
- `SafeModelSwapper`: Atomic model hot-swap with rollback in [streaming/production_runtime.py](production_runtime.py).
- `StreamEngine`: Single stream capture engine in [streaming/stream_engine.py](stream_engine.py).
- `MultiStreamEngine`: Multi-stream engine in [streaming/multi_stream_engine.py](multi_stream_engine.py).
- `BufferQueue`: Ring buffer queue in [streaming/buffer_queue.py](buffer_queue.py).
- `WorkerPool`: Thread worker pool in [streaming/worker_pool.py](worker_pool.py).

## Tests

- [tests/test_phase4_streaming.py](../tests/test_phase4_streaming.py)
- [tests/test_camera_service.py](../tests/test_camera_service.py)

## Related Documentation

- [Root README](../README.md)
- [Services Documentation](../services/README.md)
