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
    source: Optional[str] = "auto"
    location: Optional[str] = "Surveillance Zone"
    zone_id: Optional[str] = None
    credential_id: Optional[str] = None


class CameraStopRequest(BaseModel):
    camera_id: str


class CameraInfoResponse(BaseModel):
    camera_id: str
    source: str
    location: str
    status: str
    fps: float = 0.0
    processed_frames: int = 0
    active_tracks: int = 0
    zone_id: Optional[str] = None
    requested_source: Optional[str] = None
    resolved_source: Optional[str] = None
    resolved_source_type: Optional[str] = None
    resolved_source_label: Optional[str] = None
    preview_url: Optional[str] = None
    started_at: Optional[str] = None
    last_frame_at: Optional[str] = None
    credential_id: Optional[str] = None
    credential_configured: Optional[bool] = False


class CredentialCreateRequest(BaseModel):
    username: str
    password: str
    credential_id: Optional[str] = None
    description: Optional[str] = None
    shared_user_ids: Optional[List[str]] = Field(default_factory=list)


class CredentialResponse(BaseModel):
    credential_id: str
    owner_user_id: str
    username: str = "***"
    password: str = "***"
    description: Optional[str] = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    shared_user_ids: List[str] = Field(default_factory=list)
    credential_configured: bool = True
    is_owner: Optional[bool] = True


class CredentialShareRequest(BaseModel):
    target_user_id: str


class EnrollResponse(BaseModel):
    success: bool
    person_id: str
    message: str
    embeddings_added: int
