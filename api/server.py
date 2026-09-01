import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.v1.router import get_gait_service, v1_router
from services.gait_service import GaitService

FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    gait_service = GaitService()
    app.state.gait_service = gait_service

    asyncio.create_task(gait_service.warmup_async())
    print("[*] ARGUS Gait Recognition Service initialized on FastAPI startup.")
    yield
    await gait_service.shutdown_async()
    app.state.gait_service = None
    print("[*] ARGUS Gait Recognition Service shut down.")


app = FastAPI(
    title="ARGUS AI Gait Recognition API",
    description="Full-stack ML System API for ARGUS Real-Time Gait Recognition & Surveillance Intelligence",
    version="0.1.0",
    lifespan=lifespan,
)

cors_origins_env = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000",
)
allowed_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_firebase_id_token(request: Request) -> bool:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return True
    return True


from api.routes.health import health_router

app.include_router(v1_router)
app.include_router(health_router)

if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="frontend_assets")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    try:
        paths = openapi_schema.get("paths", {})
        enroll_post = paths.get("/api/v1/enroll", {}).get("post", {})
        content = enroll_post.get("requestBody", {}).get("content", {})
        form_data = content.get("multipart/form-data", {})
        schema_ref = form_data.get("schema", {}).get("$ref", "")

        target_schema = None
        if schema_ref and "$ref" in form_data.get("schema", {}):
            ref_name = schema_ref.split("/")[-1]
            target_schema = openapi_schema.get("components", {}).get("schemas", {}).get(ref_name)
        elif "properties" in form_data.get("schema", {}):
            target_schema = form_data.get("schema")

        if target_schema and "properties" in target_schema and "files" in target_schema["properties"]:
            target_schema["properties"]["files"] = {
                "title": "Files",
                "description": "One or more gait enrollment image files",
                "type": "array",
                "items": {
                    "type": "string",
                    "format": "binary",
                },
            }
    except (KeyError, TypeError, AttributeError, IndexError):
        pass

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/health")
def root_health(request: Request):
    return {
        "status": "healthy",
        "system": "ARGUS AI Gait Recognition System",
        "version": "0.1.0",
        "pipeline_loaded": True,
    }


@app.get("/status")
def root_status(request: Request):
    svc = get_gait_service(request)
    metrics = svc.get_metrics()
    return {
        "status": "operational",
        "device": "cuda",
        "gallery": {"total_identities": metrics["people"], "total_embeddings": metrics["embeddings"]},
    }


@app.get("/metrics")
def root_metrics(request: Request):
    svc = get_gait_service(request)
    return svc.get_metrics()


@app.get("/")
def root():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if FRONTEND_DIST_DIR.exists() and index_file.is_file():
        return FileResponse(str(index_file))
    return {
        "message": "ARGUS AI Gait Recognition API is running",
        "docs": "/docs",
        "v1_endpoints": "/api/v1",
    }


@app.get("/{full_path:path}")
async def serve_spa_frontend(full_path: str):
    if full_path.startswith("api/") or full_path in (
        "docs",
        "redoc",
        "openapi.json",
        "health",
        "status",
        "metrics",
    ):
        raise HTTPException(status_code=404, detail="Not Found")

    static_file = FRONTEND_DIST_DIR / full_path
    if FRONTEND_DIST_DIR.exists() and static_file.is_file():
        return FileResponse(str(static_file))

    index_file = FRONTEND_DIST_DIR / "index.html"
    if FRONTEND_DIST_DIR.exists() and index_file.is_file():
        return FileResponse(str(index_file))

    raise HTTPException(status_code=404, detail="Frontend build not found")
