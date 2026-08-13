from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    system: str = "ARGUS AI Gait Recognition System"
    version: str = "0.1.0"
    pipeline_loaded: bool = True
    active_backend: str = "pytorch"
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "person_detector": "active",
            "silhouette_extractor": "active",
            "gait_encoder": "active",
        }
    )


class StatusResponse(BaseModel):
    status: str = "operational"
    device: str = "cuda"
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "known_threshold": 0.85,
            "unknown_threshold": 0.70,
            "margin_threshold": 0.05,
        }
    )
    gallery: dict[str, int] = Field(
        default_factory=lambda: {
            "total_identities": 0,
            "total_embeddings": 0,
        }
    )
    active_cameras: int = 0


class MetricsResponse(BaseModel):
    people: int = 0
    embeddings: int = 0
    labels: int = 0
    processed_images: int = 0
    processed_videos: int = 0
    total_events: int = 0


class RecognitionEvent(BaseModel):
    event_id: str
    camera_id: str
    track_id: int = 1
    identity: str
    decision: str  # KNOWN | UNCERTAIN | UNKNOWN
    confidence: float
    quality: float = 0.85
    bbox: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    recognition_branch: str = "2D_GEI"
    timestamp: str


class CameraStartRequest(BaseModel):
    camera_id: str
    source: str
    location: Optional[str] = "Surveillance Zone"


class CameraStopRequest(BaseModel):
    camera_id: str


class CameraInfoResponse(BaseModel):
    camera_id: str
    source: str
    location: str
    status: str
    fps: float
    processed_frames: int
    active_tracks: int


class EnrollResponse(BaseModel):
    success: bool
    person_id: str
    message: str
    embeddings_added: int
