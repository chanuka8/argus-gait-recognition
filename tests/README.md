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
|---|---|
| [conftest.py](file:///E:/ARGUS_AI/tests/conftest.py) | Pytest root configuration and shared test fixtures |
| `integration/` | Module/resource file integration/ |
| [test_audit_verification.py](file:///E:/ARGUS_AI/tests/test_audit_verification.py) | Verifies audit logging, security event logging, and CSV log output |
| [test_camera_service.py](file:///E:/ARGUS_AI/tests/test_camera_service.py) | Tests camera acquisition service, stream reconnects, and status reporting |
| [test_camera_transition_model.py](file:///E:/ARGUS_AI/tests/test_camera_transition_model.py) | Tests cross-camera topology travel-time model calculations |
| [test_detector.py](file:///E:/ARGUS_AI/tests/test_detector.py) | Unit tests for YOLO person detector bounding box outputs |
| [test_gei_stream.py](file:///E:/ARGUS_AI/tests/test_gei_stream.py) | Tests rolling window silhouette accumulation and GEI generation |
| [test_leakage_prevention.py](file:///E:/ARGUS_AI/tests/test_leakage_prevention.py) | Asserts zero identity overlap between evaluation dataset partitions |
| [test_logging.py](file:///E:/ARGUS_AI/tests/test_logging.py) | Tests multi-channel rotating log file initialization and logging output |
| [test_multi_camera.py](file:///E:/ARGUS_AI/tests/test_multi_camera.py) | Tests multi-camera recognition pipeline and cross-camera evidence fusion |
| [test_phase4_streaming.py](file:///E:/ARGUS_AI/tests/test_phase4_streaming.py) | Tests stream engine buffering, queueing, and frame dropper behavior |
| [test_phase5_cctv.py](file:///E:/ARGUS_AI/tests/test_phase5_cctv.py) | Tests CCTV camera manager and worker thread pool operations |
| [test_phase6_intelligence.py](file:///E:/ARGUS_AI/tests/test_phase6_intelligence.py) | Tests intelligence modules: open-set recognition, decision engine, and score normalizers |
| [test_rtsp_credentials.py](file:///E:/ARGUS_AI/tests/test_rtsp_credentials.py) | Tests RTSP stream credential parsing and secure encryption storage |
| [test_silhouette.py](file:///E:/ARGUS_AI/tests/test_silhouette.py) | Tests silhouette extraction, Otsu thresholding, morphology, and contour filtering |
| [test_tracker.py](file:///E:/ARGUS_AI/tests/test_tracker.py) | Tests ByteTrack multi-object tracking and track ID assignment |
| [test_watchdog.py](file:///E:/ARGUS_AI/tests/test_watchdog.py) | Tests system watchdog health checks and component auto-restart logic |
| [test_watchlist_integration.py](file:///E:/ARGUS_AI/tests/test_watchlist_integration.py) | Tests real-time missing person watchlist workflow and match event processing |
| `unit/` | Module/resource file unit/ |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Pytest CLI Command → `tests/conftest.py` → Unit/Integration Test Modules → Target Codebase Modules → Temporary Assertions.

## Configuration

- [pytest.ini](file:///e:/ARGUS_AI/pytest.ini) / root pytest configuration

## Public Interfaces

- Running complete test suite: `pytest -q`
- Running unit tests only: `pytest -q tests/unit/`
- Running integration tests only: `pytest -q tests/integration/`

## Tests

- Self-testing package executing all repository tests.

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
