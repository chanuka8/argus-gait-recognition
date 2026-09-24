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
| --- | --- |
| [health.py](health.py) | Module/resource file health.py |
| `legacy/` | Module/resource file legacy/ |
| [schemas.py](schemas.py) | Pydantic request and response schemas for API data validation |
| [server.py](server.py) | FastAPI application factory, server lifecycle, and route mounting |
| [v1/auth_router.py](v1/auth_router.py) | Module/resource file v1/auth_router.py |
| [v1/router.py](v1/router.py) | Module/resource file v1/router.py |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

HTTP Client Request → `app/api/server.py` (FastAPI router) → `app/api/v1/*.py` / `app/api/health.py` → `app/services/camera_manager.py` / `app/monitoring/watchdog.py` → JSON Response (`app/api/schemas.py`). `app/api/legacy/*.py` holds pre-versioning routes kept unmounted for regression testing only.

## Configuration

- [configs/system.yaml](../../configs/system.yaml): `service.name`, `service.headless`
- [configs/base.yaml](../../configs/base.yaml): system host and port parameters

## Public Interfaces

- `create_app() -> FastAPI`: Application factory in [api/server.py](server.py).
- `start_api_server(host, port)`: Launches Uvicorn server instance.
- Endpoints: `GET /health`, `GET /cameras`, `GET /recognition/events`.

## Tests

- [tests/integration/test_dual_modal_pipeline.py](../../tests/integration/recognition/test_dual_modal_pipeline.py)
- [tests/unit/test_output_layout.py](../../tests/unit/backend/test_output_layout.py)

## Related Documentation

- [Root README](../../README.md)
- [Services Documentation](../services/README.md)
