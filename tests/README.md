# Tests

The `tests` package contains unit, integration, and verification tests for validating computer vision algorithms, pipelines, streaming stability, logging, security logic, and output layouts in ARGUS AI.

## Responsibilities

- Providing deterministic unit tests (`tests/unit/`) and end-to-end integration tests (`tests/integration/`).
- Verifying person detection, ByteTrack tracking, Otsu silhouette extraction, GEI generation, and matching algorithms.
- Validating zero identity leakage between gallery and probe datasets.
- Testing output directory layout, logging channels, security logger CSV generation, and watchdog auto-restart logic.
- Boundaries: Must use temporary directories (`tmp_path`, `tempfile.TemporaryDirectory`) for test outputs to prevent polluting the repository.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [conftest.py](conftest.py) | Pytest root configuration and shared test fixtures |
| `integration/` | Module/resource file integration/ |
| `performance/` | Module/resource file performance/ |
| `system/` | Module/resource file system/ |
| `unit/` | Module/resource file unit/ |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Pytest CLI Command → `tests/conftest.py` → Unit/Integration Test Modules → Target Codebase Modules → Temporary Assertions.

## Configuration

- Root pytest configuration and `tests/conftest.py` shared fixtures

## Public Interfaces

- Running complete test suite: `pytest -q`
- Running unit tests only: `pytest -q tests/unit/`
- Running integration tests only: `pytest -q tests/integration/`

## Tests

- Self-testing package executing all repository tests.

## Related Documentation

- [Root README](../README.md)
