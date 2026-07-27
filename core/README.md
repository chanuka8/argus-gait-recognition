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
|---|---|
| [boot.py](file:///E:/ARGUS_AI/core/boot.py) | Application startup sequence, dependency checks, and environment setup |
| [config.py](file:///E:/ARGUS_AI/core/config.py) | Configuration file loader and setting merge utilities |
| [context.py](file:///E:/ARGUS_AI/core/context.py) | Thread-safe global system context and runtime state holder |
| [exceptions.py](file:///E:/ARGUS_AI/core/exceptions.py) | Custom exception hierarchy for ARGUS AI error handling |
| [health_check.py](file:///E:/ARGUS_AI/core/health_check.py) | Hardware, CUDA, disk space, and memory health verifiers |
| [logger.py](file:///E:/ARGUS_AI/core/logger.py) | Logger creation and formatting utilities |
| [orchestrator.py](file:///E:/ARGUS_AI/core/orchestrator.py) | Top-level coordinator managing streaming, recognition, and security layers |
| [system.py](file:///E:/ARGUS_AI/core/system.py) | Primary application container and lifecycle coordinator |
| [system_monitor.py](file:///E:/ARGUS_AI/core/system_monitor.py) | Background monitoring for CPU, RAM, and GPU resource usage |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

`main.py` → `core/boot.py` → `core/system.py` → `core/orchestrator.py` → Pipeline Execution Loop.

## Configuration

- [configs/system.yaml](file:///e:/ARGUS_AI/configs/system.yaml): `logging`, `watchdog`, `service` sections
- [configs/base.yaml](file:///e:/ARGUS_AI/configs/base.yaml): path defaults and system flags

## Public Interfaces

- `setup_logger(name: str) -> Logger`: Log setup in [core/logger.py](file:///e:/ARGUS_AI/core/logger.py).
- `ArgusSystem`: Primary application container in [core/system.py](file:///e:/ARGUS_AI/core/system.py).
- `HealthCheck`: System diagnostic verifier in [core/health_check.py](file:///e:/ARGUS_AI/core/health_check.py).

## Tests

- [tests/test_audit_verification.py](file:///e:/ARGUS_AI/tests/test_audit_verification.py)
- [tests/test_logging.py](file:///e:/ARGUS_AI/tests/test_logging.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Monitoring Documentation](file:///e:/ARGUS_AI/monitoring/README.md)
