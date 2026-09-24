# Core

The `core` package provides system initialization, logging setup, process lifecycle management, orchestrators, and system-wide health checking for ARGUS AI.

## Responsibilities

- Bootstrapping application execution and verifying hardware resources.
- Centralizing logging initialization and error handling exceptions.
- Managing overall pipeline orchestration and system component status checks.
- Boundaries: Does not implement computer vision algorithms or model architectures directly.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [boot.py](boot.py) | Application startup sequence, dependency checks, and environment setup |
| [config.py](config.py) | Configuration file loader and setting merge utilities |
| [context.py](context.py) | Thread-safe global system context and runtime state holder |
| [exceptions.py](exceptions.py) | Custom exception hierarchy for ARGUS AI error handling |
| [health_check.py](health_check.py) | Hardware, CUDA, disk space, and memory health verifiers |
| [logger.py](logger.py) | Logger creation and formatting utilities |
| [orchestrator.py](orchestrator.py) | Top-level coordinator managing streaming, recognition, and security layers |
| [paths.py](paths.py) | Module/resource file paths.py |
| [resource_profile.py](resource_profile.py) | Torch-free RAM/CPU/GPU profile (LOW_RESOURCE/BALANCED/PERFORMANCE) used to size thread budgets and defer background-optional ML components |
| [system.py](system.py) | Primary application container and lifecycle coordinator |
| [system_monitor.py](system_monitor.py) | Background monitoring for CPU, RAM, and GPU resource usage |
| [thread_limits.py](thread_limits.py) | Hardware-aware OMP/MKL/OpenBLAS/PyTorch/OpenCV/ONNXRuntime thread-pool limits, configured once before any ML library is imported |
| [threshold_manager.py](threshold_manager.py) | Authoritative recognition threshold manager and calibration resolution |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

`main.py` → `app/core/boot.py` → `app/core/system.py` → `app/core/orchestrator.py` → Pipeline Execution Loop.

## Configuration

- [configs/system.yaml](../configs/system.yaml): `logging`, `watchdog`, `service` sections
- [configs/base.yaml](../configs/base.yaml): path defaults and system flags

## Public Interfaces

- `setup_logger(name: str) -> Logger`: Log setup in [core/logger.py](logger.py).
- `ArgusSystem`: Primary application container in [core/system.py](system.py).
- `HealthCheck`: System diagnostic verifier in [core/health_check.py](health_check.py).

## Tests

- [tests/test_audit_verification.py](../tests/test_audit_verification.py)
- [tests/test_logging.py](../tests/test_logging.py)

## Related Documentation

- [Root README](../README.md)
- [Monitoring Documentation](../monitoring/README.md)
