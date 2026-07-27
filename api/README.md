# API

The `api` package provides RESTful HTTP endpoints and web server infrastructure for controlling cameras, querying system status, retrieving recognition events, and streaming health telemetry in ARGUS AI.

## Responsibilities

- Exposing HTTP REST endpoints for camera status, health metrics, and recognition events.
- Providing structured Pydantic data schemas for request and response models.
- Separating HTTP network transport concerns from core recognition and streaming engines.
- Boundaries: Does not perform direct inference, video decoding, or gallery state mutation.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [schemas.py](file:///e:/ARGUS_AI/api/schemas.py) | Pydantic request and response schemas for API data validation |
| [server.py](file:///e:/ARGUS_AI/api/server.py) | FastAPI application factory, server lifecycle, and route mounting |
| [routes/camera.py](file:///e:/ARGUS_AI/api/routes/camera.py) | HTTP endpoints for camera state, configuration, and stream statistics |
| [routes/enrollment.py](file:///e:/ARGUS_AI/api/routes/enrollment.py) | HTTP endpoints for target identity enrollment requests |
| [routes/health.py](file:///e:/ARGUS_AI/api/routes/health.py) | HTTP endpoints for system health, CPU/RAM/GPU telemetry, and watchdog status |
| [routes/inference.py](file:///e:/ARGUS_AI/api/routes/inference.py) | HTTP endpoints for model inference and feature extraction triggers |
| [routes/recognition.py](file:///e:/ARGUS_AI/api/routes/recognition.py) | HTTP endpoints for real-time recognition alerts, events, and query results |
| [routes/status.py](file:///e:/ARGUS_AI/api/routes/status.py) | HTTP endpoints for operational status and system metrics |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

HTTP Client Request → `api/server.py` (FastAPI router) → `api/routes/*.py` → `services/camera_manager.py` / `monitoring/watchdog.py` → JSON Response (`api/schemas.py`).

## Configuration

- [configs/system.yaml](file:///e:/ARGUS_AI/configs/system.yaml): `service.name`, `service.headless`
- [configs/base.yaml](file:///e:/ARGUS_AI/configs/base.yaml): system host and port parameters

## Public Interfaces

- `create_app() -> FastAPI`: Application factory in [api/server.py](file:///e:/ARGUS_AI/api/server.py).
- `start_api_server(host, port)`: Launches Uvicorn server instance.
- Endpoints: `GET /health`, `GET /cameras`, `GET /recognition/events`.

## Tests

- [tests/integration/test_dual_modal_pipeline.py](file:///e:/ARGUS_AI/tests/integration/test_dual_modal_pipeline.py)
- [tests/unit/test_output_layout.py](file:///e:/ARGUS_AI/tests/unit/test_output_layout.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Services Documentation](file:///e:/ARGUS_AI/services/README.md)
