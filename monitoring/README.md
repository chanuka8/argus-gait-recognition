# Monitoring

The `monitoring` package provides camera health monitoring, process watchdog failover, multi-channel rotating log file routing, GPU performance tuning, and metrics collection for ARGUS AI.

## Responsibilities

- Monitoring RTSP/USB camera health, FPS performance, and frame drop rates.
- Periodically dumping camera statistics to `outputs/monitoring/camera_stats/`.
- Managing system watchdog auto-restart loops and multi-channel log initialization (`outputs/logs/system/`).
- Boundaries: Does not run object detection or biometric vector matching directly.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [camera_monitor.py](camera_monitor.py) | Periodically gathers camera stream stats and writes JSON summaries to `outputs/monitoring/camera_stats/` |
| [crash_guard.py](crash_guard.py) | Process crash interception and graceful recovery guard |
| [gpu_tuner.py](gpu_tuner.py) | Monitors CUDA memory utilization and tunes batch allocation parameters |
| [logging_config.py](logging_config.py) | Multi-channel log router supporting 5 rotating log files (`system`, `camera`, `detection`, `error`, `watchdog`) |
| [metrics_collector.py](metrics_collector.py) | Aggregates system resource utilization metrics (CPU, RAM, GPU, Queue Size, FPS) |
| [performance_profiler.py](performance_profiler.py) | Measures execution latencies across pipeline steps |
| [watchdog.py](watchdog.py) | Background watchdog checking component health and triggering worker auto-restarts |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Component Execution → `monitoring/logging_config.py` & `monitoring/metrics_collector.py` → `monitoring/watchdog.py` → `outputs/logs/system/` & `outputs/monitoring/camera_stats/`.

## Configuration

- [configs/system.yaml](../configs/system.yaml): `logging` and `watchdog` sections

## Public Interfaces

- `CameraMonitor`: Camera health tracker in [monitoring/camera_monitor.py](camera_monitor.py).
- `SystemWatchdog`: Watchdog coordinator in [monitoring/watchdog.py](watchdog.py).
- `init_logging()`, `get_logger(channel: str)`: Logger initialization in [monitoring/logging_config.py](logging_config.py).

## Tests

- [tests/test_logging.py](../tests/test_logging.py)
- [tests/test_watchdog.py](../tests/test_watchdog.py)
- [tests/unit/test_output_layout.py](../tests/unit/test_output_layout.py)

## Related Documentation

- [Root README](../README.md)
- [Core Documentation](../core/README.md)
