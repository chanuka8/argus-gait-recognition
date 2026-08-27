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


class ComputeInfo(BaseModel):
    backend: str = "cpu"
    device: str = "cpu"
    gpu: str | None = None
    vram_mb: float = 0.0
    cuda_available: bool = False
    pytorch_version: str | None = None
    cuda_version: str | None = None
    onnx_provider: str = "CPUExecutionProvider"


class StatusResponse(BaseModel):
    status: str = "operational"
    device: str = "cuda"
    compute: ComputeInfo | None = None
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
    decision: str
    confidence: float
    quality: float = 0.85
    bbox: list[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    recognition_branch: str = "2D_GEI"
    timestamp: str


class CameraStartRequest(BaseModel):
    camera_id: str
    source: str | None = "auto"
    location: str | None = "Surveillance Zone"
    zone_id: str | None = None
    credential_id: str | None = None


class CameraStopRequest(BaseModel):
    camera_id: str


class CameraInfoResponse(BaseModel):
    camera_id: str
    source: str
    source_type: str | None = "webcam"
    location: str
    status: str
    fps: float = 0.0
    processed_frames: int = 0
    active_tracks: int = 0
    recognition_active: bool | None = False
    last_recognition_at: str | None = None
    active_clients: int | None = 0
    recognized_identities: list[str] | None = Field(default_factory=list)
    zone_id: str | None = None
    requested_source: str | None = None
    resolved_source: str | None = None
    resolved_source_type: str | None = None
    resolved_source_label: str | None = None
    preview_url: str | None = None
    started_at: str | None = None
    last_frame_at: str | None = None
    credential_id: str | None = None
    credential_configured: bool | None = False


class CredentialCreateRequest(BaseModel):
    username: str
    password: str
    credential_id: str | None = None
    description: str | None = None
    shared_user_ids: list[str] | None = Field(default_factory=list)


class CredentialResponse(BaseModel):
    credential_id: str
    owner_user_id: str
    username: str = "***"
    password: str = "***"
    description: str | None = ""
    created_at: str | None = None
    updated_at: str | None = None
    shared_user_ids: list[str] = Field(default_factory=list)
    credential_configured: bool = True
    is_owner: bool | None = True


class CredentialShareRequest(BaseModel):
    target_user_id: str


class EnrollRequest(BaseModel):
    folder_path: str


class EnrollResponse(BaseModel):
    success: bool
    person_id: str
    message: str
    embeddings_added: int


class IdentifyRequest(BaseModel):
    image_path: str


class IdentifyResponse(BaseModel):
    identity: str
    score: float
