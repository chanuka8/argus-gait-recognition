from pydantic import BaseModel


class IdentifyRequest(BaseModel):
    image_path: str


class IdentifyResponse(BaseModel):
    identity: str
    score: float


class EnrollRequest(BaseModel):
    folder_path: str


class EnrollResponse(BaseModel):
    success: bool
    person_id: str
    message: str
    embeddings_added: int


class HealthResponse(BaseModel):
    status: str
    system: str
    version: str


class MetricsResponse(BaseModel):
    people: int
    embeddings: int
    labels: int