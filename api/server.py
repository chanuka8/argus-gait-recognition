from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.v1.router import v1_router, get_gait_service
from services.gait_service import GaitService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize single-instance GaitService on app.state
    app.state.gait_service = GaitService()
    print("[*] ARGUS Gait Recognition Service initialized on FastAPI startup.")
    yield
    # Shutdown
    if hasattr(app.state, "gait_service") and app.state.gait_service:
        app.state.gait_service.shutdown()
        app.state.gait_service = None
    print("[*] ARGUS Gait Recognition Service shut down cleanly.")


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
    return True


app.include_router(v1_router)


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
