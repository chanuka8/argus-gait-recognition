from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.v1.router import v1_router, get_gait_service
from services.gait_service import GaitService

FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize single-instance GaitService on app.state
    app.state.gait_service = GaitService()
    print("[*] ARGUS Gait Recognition Service initialized on FastAPI startup.")
    yield
    # Shutdown
    app.state.gait_service = None
    print("[*] ARGUS Gait Recognition Service shut down.")


app = FastAPI(
    title="ARGUS AI Gait Recognition API",
    description="Full-stack ML System API for ARGUS Real-Time Gait Recognition & Surveillance Intelligence",
    version="0.1.0",
    lifespan=lifespan,
)

# Configurable CORS Origins for React/Vite Frontend
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


# Security boundary placeholder: Firebase ID token verification middleware / helper
async def verify_firebase_id_token(request: Request) -> bool:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        # Development pass-through if unauthenticated
        return True
    # Placeholder for production firebase_admin.auth.verify_id_token(token)
    return True


app.include_router(v1_router)

# Mount frontend production assets if dist exists
if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="frontend_assets")


def custom_openapi():
    """Custom OpenAPI schema generator ensuring multipart files format=binary for Swagger file picker."""
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
    except Exception:
        pass

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Backward-compatibility aliases for root endpoints
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
    """
    Catch-all route to serve the React Single Page Application (SPA).
    Supports client-side routing while preserving /api, /docs, /redoc, /health, /status, and /metrics.
    """
    if full_path.startswith("api/") or full_path in (
        "docs",
        "redoc",
        "openapi.json",
        "health",
        "status",
        "metrics",
    ):
        raise HTTPException(status_code=404, detail="Not Found")

    # Serve static files in dist root (e.g. logo.png, favicon.ico)
    static_file = FRONTEND_DIST_DIR / full_path
    if FRONTEND_DIST_DIR.exists() and static_file.is_file():
        return FileResponse(str(static_file))

    # SPA index.html fallback for client-side routes (e.g. /dashboard, /cctv-network, /admin/dashboard)
    index_file = FRONTEND_DIST_DIR / "index.html"
    if FRONTEND_DIST_DIR.exists() and index_file.is_file():
        return FileResponse(str(index_file))

    raise HTTPException(status_code=404, detail="Frontend build not found")
