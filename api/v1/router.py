import tempfile
from typing import List

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from api.schemas import (
    CameraInfoResponse,
    CameraStartRequest,
    CameraStopRequest,
    EnrollResponse,
    HealthResponse,
    MetricsResponse,
    RecognitionEvent,
    StatusResponse,
)
from services.gait_service import GaitService


def get_gait_service(request: Request = None) -> GaitService:
    if request and hasattr(request.app.state, "gait_service") and request.app.state.gait_service:
        return request.app.state.gait_service
    return GaitService()


v1_router = APIRouter(prefix="/api/v1", tags=["v1"])


@v1_router.get("/health", response_model=HealthResponse)
def get_health(service: GaitService = Depends(get_gait_service)):
    return {
        "status": "healthy",
        "system": "ARGUS AI Gait Recognition System",
        "version": "0.1.0",
        "pipeline_loaded": True,
        "active_backend": getattr(service.extractor.backend, "backend_name", "pytorch"),
        "models": {
            "person_detector": "active" if service.detector else "optional",
            "silhouette_extractor": "active",
            "gait_encoder": "active",
        },
    }


@v1_router.get("/status", response_model=StatusResponse)
def get_status(service: GaitService = Depends(get_gait_service)):
    metrics = service.get_metrics()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "operational",
        "device": device,
        "thresholds": {
            "known_threshold": 0.85,
            "unknown_threshold": 0.70,
            "margin_threshold": 0.05,
        },
        "gallery": {
            "total_identities": metrics["people"],
            "total_embeddings": metrics["embeddings"],
        },
        "active_cameras": len(service.active_cameras),
    }


@v1_router.get("/metrics", response_model=MetricsResponse)
def get_metrics(service: GaitService = Depends(get_gait_service)):
    return service.get_metrics()


@v1_router.post("/identify/image", response_model=RecognitionEvent)
async def identify_image(
    file: UploadFile = File(...),
    camera_id: str = Form("upload-image"),
    service: GaitService = Depends(get_gait_service),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        # Accept image/jpeg, image/png, etc.
        pass

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        event = service.process_image_bytes(content, camera_id=camera_id)
        return event
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Image identification failed: {err}")


@v1_router.post("/analyze/video", response_model=List[RecognitionEvent])
async def analyze_video(
    file: UploadFile = File(...),
    service: GaitService = Depends(get_gait_service),
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            tmp_vid.write(content)
            tmp_path = tmp_vid.name

        service.stats["processed_videos"] += 1
        # Extract key frame for video analysis
        import cv2
        cap = cv2.VideoCapture(tmp_path)
        events = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 15 == 0:  # Sample every 15th frame
                _, img_buf = cv2.imencode(".jpg", frame)
                evt = service.process_image_bytes(img_buf.tobytes(), camera_id="upload-video")
                events.append(evt)
                if len(events) >= 10:
                    break
            frame_idx += 1
        cap.release()

        if not events:
            # Fallback single frame event
            _, img_buf = cv2.imencode(".jpg", np.zeros((200, 100, 3), dtype=np.uint8))
            events.append(service.process_image_bytes(img_buf.tobytes(), camera_id="upload-video"))

        return events
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {err}")


@v1_router.post("/cameras/start", response_model=CameraInfoResponse)
def start_camera(
    body: CameraStartRequest,
    service: GaitService = Depends(get_gait_service),
):
    if not body.camera_id or not body.source:
        raise HTTPException(status_code=400, detail="camera_id and source are required")

    cam_info = service.start_camera(body.camera_id, body.source, body.location or "Surveillance Zone")
    return cam_info


@v1_router.post("/cameras/stop")
def stop_camera(
    body: CameraStopRequest,
    service: GaitService = Depends(get_gait_service),
):
    stopped = service.stop_camera(body.camera_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"Camera {body.camera_id} not found or active")
    return {"success": True, "message": f"Camera {body.camera_id} stopped"}


@v1_router.get("/cameras", response_model=List[CameraInfoResponse])
def list_cameras(service: GaitService = Depends(get_gait_service)):
    return list(service.active_cameras.values())


@v1_router.post("/enroll", response_model=EnrollResponse)
async def enroll_subject(
    person_id: str = Form(...),
    files: List[UploadFile] = File(...),
    service: GaitService = Depends(get_gait_service),
):
    if not person_id:
        raise HTTPException(status_code=400, detail="person_id is required")

    image_bytes_list = []
    for f in files:
        b = await f.read()
        if b:
            image_bytes_list.append(b)

    if not image_bytes_list:
        raise HTTPException(status_code=400, detail="No valid image files supplied")

    res = service.enroll_images(person_id, image_bytes_list)
    return res


@v1_router.get("/events", response_model=List[RecognitionEvent])
def get_events(service: GaitService = Depends(get_gait_service)):
    return service.events_log


@v1_router.websocket("/ws/recognition")
async def websocket_recognition(
    websocket: WebSocket,
):
    service = get_gait_service(request=None)
    await service.ws_manager.connect(websocket)
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        service.ws_manager.disconnect(websocket)
    except Exception:
        service.ws_manager.disconnect(websocket)
