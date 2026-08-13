import asyncio
import tempfile
from typing import List

import cv2
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
from fastapi.responses import StreamingResponse

from api.schemas import (
    CameraInfoResponse,
    CameraProbeRequest,
    CameraSourceResponse,
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
        cap = cv2.VideoCapture(tmp_path)
        events = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 15 == 0:
                _, img_buf = cv2.imencode(".jpg", frame)
                evt = service.process_image_bytes(img_buf.tobytes(), camera_id="upload-video")
                events.append(evt)
                if len(events) >= 10:
                    break
            frame_idx += 1
        cap.release()

        if not events:
            _, img_buf = cv2.imencode(".jpg", np.zeros((200, 100, 3), dtype=np.uint8))
            events.append(service.process_image_bytes(img_buf.tobytes(), camera_id="upload-video"))

        return events
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {err}")


# CAMERA SOURCE DISCOVERY AND PROBING ENDPOINTS
@v1_router.get("/camera-sources", response_model=List[CameraSourceResponse])
def get_camera_sources(service: GaitService = Depends(get_gait_service)):
    sources = service.discover_sources(max_index=5, force_refresh=False)
    return sources


@v1_router.post("/camera-sources/discover", response_model=List[CameraSourceResponse])
def discover_camera_sources(service: GaitService = Depends(get_gait_service)):
    sources = service.discover_sources(max_index=5, force_refresh=True)
    return sources


@v1_router.post("/cameras/probe", response_model=CameraSourceResponse)
def probe_camera(
    body: CameraProbeRequest,
    service: GaitService = Depends(get_gait_service),
):
    try:
        res = service.probe_camera(source=body.source, source_type=body.source_type or "auto", backend=body.capture_backend or "auto")
        return res
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Camera probe failed: {err}")


# CAMERA WORKER LIFECYCLE ENDPOINTS
@v1_router.post("/cameras/start", response_model=CameraInfoResponse)
def start_camera(
    body: CameraStartRequest,
    service: GaitService = Depends(get_gait_service),
):
    if not body.camera_id or not body.source:
        raise HTTPException(status_code=400, detail="camera_id and source are required.")

    try:
        cam_info = service.start_camera(
            camera_id=body.camera_id,
            source=body.source,
            zone_id=body.zone_id or "Z01",
            source_type=body.source_type or "auto",
            capture_backend=body.capture_backend or "auto",
            location=body.location or "Surveillance Zone",
            width=body.width or 640,
            height=body.height or 480,
            fps=body.fps or 30.0,
        )
        return cam_info
    except ValueError as val_err:
        err_str = str(val_err)
        if err_str.startswith("CONFLICT:"):
            raise HTTPException(status_code=409, detail=err_str)
        elif "Invalid requested source_type" in err_str or "detected as" in err_str or "Ambiguous" in err_str:
            raise HTTPException(status_code=422, detail=err_str)
        else:
            raise HTTPException(status_code=400, detail=err_str)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to start camera worker: {err}")


@v1_router.post("/cameras/stop")
def stop_camera(
    body: CameraStopRequest,
    service: GaitService = Depends(get_gait_service),
):
    stopped = service.stop_camera(body.camera_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"Camera worker '{body.camera_id}' not found or active.")
    return {"success": True, "message": f"Camera worker '{body.camera_id}' stopped successfully."}


@v1_router.get("/cameras", response_model=List[CameraInfoResponse])
def list_cameras(service: GaitService = Depends(get_gait_service)):
    return list(service.active_cameras.values())


# MJPEG STREAM PREVIEW ENDPOINT
@v1_router.get("/cameras/{camera_id}/stream")
def stream_camera_preview(
    camera_id: str,
    service: GaitService = Depends(get_gait_service),
):
    worker = service.zone_registry.get_worker(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"Active camera worker '{camera_id}' not found.")

    def frame_generator():
        while True:
            jpeg_bytes = worker.get_jpeg_frame()
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
            time_delay = 1.0 / max(1.0, worker.processing_fps or 15.0)
            asyncio.run(asyncio.sleep(time_delay))

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@v1_router.post("/enroll", response_model=EnrollResponse)
async def enroll_subject(
    person_id: str = Form(...),
    files: List[UploadFile] = File(...),
    service: GaitService = Depends(get_gait_service),
):
    if not person_id:
        raise HTTPException(status_code=400, detail="person_id is required.")

    image_bytes_list = []
    for f in files:
        b = await f.read()
        if b:
            image_bytes_list.append(b)

    if not image_bytes_list:
        raise HTTPException(status_code=400, detail="No valid image files supplied.")

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
