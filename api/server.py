from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from api.v1.router import v1_router, get_gait_service
from services.gait_service import GaitService


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
cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000")
allowed_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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
@app.get("/")
def root():
    return {
        "message": "ARGUS AI Gait Recognition API is running",
        "docs": "/docs",
        "v1_endpoints": "/api/v1",
    }


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
