from fastapi import FastAPI

from api.routes.enrollment import router as enrollment_router
from api.routes.inference import router as inference_router
from api.routes.status import router as status_router

app = FastAPI(
    title="ARGUS API",
    description="Demo API for ARGUS gait recognition system",
    version="0.1.0",
)

app.include_router(status_router)
app.include_router(inference_router)
app.include_router(enrollment_router)


@app.get("/")
def root():
    return {
        "message": "ARGUS API is running",
        "docs": "/docs",
    }