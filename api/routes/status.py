from pathlib import Path

from fastapi import APIRouter

from api.schemas import HealthResponse, MetricsResponse
from storage.vector_store import VectorStore

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    version = "0.1.0"

    version_file = Path("VERSION")
    if version_file.exists():
        version = version_file.read_text(encoding="utf-8").strip()

    return {
        "status": "healthy",
        "system": "ARGUS",
        "version": version,
    }


@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    gallery = VectorStore().load()

    if gallery is None:
        return {
            "people": 0,
            "embeddings": 0,
            "labels": 0,
        }

    features, labels, metadata = gallery

    return {
        "people": len(metadata),
        "embeddings": len(features),
        "labels": len(labels),
    }