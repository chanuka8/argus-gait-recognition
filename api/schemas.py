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
    zone_id: Optional[str] = "Z01"
    source_type: Optional[str] = "auto"
    capture_backend: Optional[str] = "auto"
    location: Optional[str] = "Surveillance Zone"
    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: Optional[float] = 30.0


class CameraProbeRequest(BaseModel):
    source: str
    source_type: Optional[str] = "auto"
    capture_backend: Optional[str] = "auto"


class CameraStopRequest(BaseModel):
    camera_id: str


class CameraSourceResponse(BaseModel):
    source_id: str
    display_name: str
    source_type: str
    device_index: Optional[int] = None
    source_url: Optional[str] = ""
    sanitized_source: str
    capture_backend_requested: str = "auto"
    capture_backend_used: str = "auto"
    available: bool = False
    actual_width: int = 0
    actual_height: int = 0
    actual_fps: float = 0.0
    last_probe_at: Optional[str] = None
    error: Optional[str] = None


class CameraInfoResponse(BaseModel):
    camera_id: str
    zone_id: str = "Z01"
    source: str
    source_type: str = "auto"
    location: str = "Surveillance Zone"
    status: str = "ACTIVE"
    capture_backend: str = "auto"
    fps: float = 0.0
    capture_fps: float = 0.0
    processing_fps: float = 0.0
    captured_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    active_tracks: int = 0
    processing_latency_ms: float = 0.0
    started_at: Optional[str] = None
    last_frame_at: Optional[str] = None
    error: Optional[str] = None


class EnrollResponse(BaseModel):
    success: bool
    person_id: str
    message: str
    embeddings_added: int
