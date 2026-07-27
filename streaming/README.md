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
|---|---|
| [buffer_queue.py](buffer_queue.py) | Thread-safe bounded ring buffer queue with non-blocking put/get operations |
| [camera_scheduler.py](camera_scheduler.py) | Round-robin and priority scheduler distributing camera stream frame processing |
| [frame_dropper.py](frame_dropper.py) | Intelligent frame dropper maintaining target processing FPS under CPU congestion |
| [load_balancer.py](load_balancer.py) | Balances processing workload across multi-stream worker threads |
| [multi_stream_engine.py](multi_stream_engine.py) | Multi-stream acquisition engine handling concurrent CCTV streams |
| [performance_optimizer.py](performance_optimizer.py) | Dynamic performance optimizer adjusting frame queue sizes and capture rates |
| [stream_engine.py](stream_engine.py) | Core single-stream acquisition engine wrapping OpenCV `VideoCapture` |
| [worker_pool.py](worker_pool.py) | Thread worker pool executing concurrent stream capture tasks |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

RTSP / USB / File Stream → `streaming/stream_engine.py` → `streaming/buffer_queue.py` → `streaming/frame_dropper.py` → Pipeline Capture Hook.

## Configuration

- [configs/system.yaml](../configs/system.yaml): `camera.target_fps`, `camera.max_queue_size`, `camera.type`, `camera.url`
- [configs/cameras.yaml](../configs/cameras.yaml): RTSP camera streams

## Public Interfaces

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
