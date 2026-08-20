from fastapi.responses import StreamingResponse, Response
import asyncio
"""Version 1 API routes for the ARGUS gait recognition backend."""

import tempfile
from typing import Annotated

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
    CredentialCreateRequest,
    CredentialResponse,
    CredentialShareRequest,
    EnrollResponse,
    HealthResponse,
    MetricsResponse,
    RecognitionEvent,
    StatusResponse,
)
from security_layer.credentials import CredentialManager, sanitize_rtsp_url
from services.gait_service import GaitService


def get_current_user_id(request: Request) -> str:
    """Extract authenticated user ID from request headers or default to development identity."""
    user_header = request.headers.get("X-User-ID")
    if user_header:
        return user_header.strip()

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token.split(".")[0] if "." in token else token

    return "default_user"


def get_gait_service(request: Request = None) -> GaitService:
    """Return the initialized application-level gait service."""

    if (
        request
        and hasattr(request.app.state, "gait_service")
        and request.app.state.gait_service
    ):
        return request.app.state.gait_service

    return GaitService()


v1_router = APIRouter(
    prefix="/api/v1",
    tags=["v1"],
)


@v1_router.get(
    "/health",
    response_model=HealthResponse,
)
def get_health(
    service: GaitService = Depends(get_gait_service),
):
    """Return the health status of the ARGUS gait service."""

    return {
        "status": "healthy",
        "system": "ARGUS AI Gait Recognition System",
        "version": "0.1.0",
        "pipeline_loaded": True,
        "active_backend": getattr(
            service.extractor.backend,
            "backend_name",
            "pytorch",
        ),
        "models": {
            "person_detector": (
                "active"
                if service.detector
                else "optional"
            ),
            "silhouette_extractor": "active",
            "gait_encoder": "active",
        },
    }


@v1_router.get(
    "/status",
    response_model=StatusResponse,
)
def get_status(
    service: GaitService = Depends(get_gait_service),
):
    """Return the operational status of the ARGUS backend."""

    import torch

    metrics = service.get_metrics()
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


@v1_router.get(
    "/metrics",
    response_model=MetricsResponse,
)
def get_metrics(
    service: GaitService = Depends(get_gait_service),
):
    """Return runtime metrics."""

    return service.get_metrics()


@v1_router.post(
    "/identify/image",
    response_model=RecognitionEvent,
)
async def identify_image(
    file: Annotated[
        UploadFile,
        File(description="Image file used for gait identification"),
    ],
    camera_id: Annotated[
        str,
        Form(description="Camera or upload source identifier"),
    ] = "upload-image",
    service: GaitService = Depends(get_gait_service),
):
    """Identify a person from an uploaded image."""

    if (
        file.content_type
        and not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. "
                "Upload a valid image file."
            ),
        )

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        return service.process_image_bytes(
            content,
            camera_id=camera_id,
        )

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image identification failed: {error}",
        ) from error

    finally:
        await file.close()


@v1_router.post(
    "/analyze/video",
    response_model=list[RecognitionEvent],
)
async def analyze_video(
    file: Annotated[
        UploadFile,
        File(description="Video file used for gait analysis"),
    ],
    service: GaitService = Depends(get_gait_service),
):
    """Analyze sampled frames from an uploaded video."""

    temporary_path: str | None = None

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video file is empty",
            )

        suffix = ".mp4"

        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", maxsplit=1)[-1]

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_video:
            temporary_video.write(content)
            temporary_path = temporary_video.name

        service.stats["processed_videos"] += 1

        import cv2

        capture = cv2.VideoCapture(temporary_path)
        events: list[RecognitionEvent] = []
        frame_index = 0

        try:
            while capture.isOpened():
                success, frame = capture.read()

                if not success:
                    break

                # Process every fifteenth frame.
                if frame_index % 15 == 0:
                    encoded, image_buffer = cv2.imencode(
                        ".jpg",
                        frame,
                    )

                    if encoded:
                        event = service.process_image_bytes(
                            image_buffer.tobytes(),
                            camera_id="upload-video",
                        )
                        events.append(event)

                    if len(events) >= 10:
                        break

                frame_index += 1

        finally:
            capture.release()

        if not events:
            fallback_frame = np.zeros(
                (200, 100, 3),
                dtype=np.uint8,
            )

            encoded, image_buffer = cv2.imencode(
                ".jpg",
                fallback_frame,
            )

            if not encoded:
                raise HTTPException(
                    status_code=500,
                    detail="Unable to create fallback video frame",
                )

            events.append(
                service.process_image_bytes(
                    image_buffer.tobytes(),
                    camera_id="upload-video",
                )
            )

        return events

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {error}",
        ) from error

    finally:
        await file.close()

        if temporary_path:
            try:
                import os

                os.remove(temporary_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# RTSP Credential Management Endpoints
# ---------------------------------------------------------------------------


@v1_router.post(
    "/credentials",
    response_model=CredentialResponse,
)
def create_credential(
    body: CredentialCreateRequest,
    request: Request,
):
    """Store an encrypted RTSP credential associated with the authenticated user."""
    user_id = get_current_user_id(request)
    cm = CredentialManager()
    try:
        meta = cm.store_credential(
            owner_user_id=user_id,
            username=body.username,
            password=body.password,
            credential_id=body.credential_id,
            description=body.description or "",
            shared_user_ids=body.shared_user_ids or [],
        )
        return meta
    except PermissionError as err:
        raise HTTPException(status_code=403, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to store credential: {err}") from err


@v1_router.get(
    "/credentials",
    response_model=list[CredentialResponse],
)
def list_credentials(
    request: Request,
):
    """List all credentials accessible by the authenticated user with masked secrets."""
    user_id = get_current_user_id(request)
    cm = CredentialManager()
    return cm.list_credentials_for_user(user_id=user_id)


@v1_router.get(
    "/credentials/{credential_id}",
    response_model=CredentialResponse,
)
def get_credential(
    credential_id: str,
    request: Request,
):
    """Retrieve metadata for a specific credential with masked secrets."""
    user_id = get_current_user_id(request)
    cm = CredentialManager()
    meta = cm.get_credential_metadata(credential_id, user_id=user_id)
    if not meta:
        raise HTTPException(
            status_code=404,
            detail=f"Credential '{credential_id}' not found or unauthorized",
        )
    return meta


@v1_router.delete("/credentials/{credential_id}")
def delete_credential(
    credential_id: str,
    request: Request,
):
    """Delete a credential owned by the authenticated user."""
    user_id = get_current_user_id(request)
    cm = CredentialManager()
    try:
        deleted = cm.delete_credential(credential_id, user_id=user_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Credential '{credential_id}' not found",
            )
        return {
            "success": True,
            "message": f"Credential '{credential_id}' deleted successfully",
        }
    except PermissionError as err:
        raise HTTPException(status_code=403, detail=str(err)) from err


@v1_router.post("/credentials/{credential_id}/share")
def share_credential(
    credential_id: str,
    body: CredentialShareRequest,
    request: Request,
):
    """Grant another user access to a shared credential."""
    user_id = get_current_user_id(request)
    cm = CredentialManager()
    try:
        shared = cm.grant_access(
            credential_id,
            owner_user_id=user_id,
            target_user_id=body.target_user_id,
        )
        if not shared:
            raise HTTPException(
                status_code=404,
                detail=f"Credential '{credential_id}' not found",
            )
        return {
            "success": True,
            "message": f"Credential '{credential_id}' shared with '{body.target_user_id}'",
        }
    except PermissionError as err:
        raise HTTPException(status_code=403, detail=str(err)) from err


@v1_router.post(
    "/cameras/{camera_id}/credentials",
    response_model=CredentialResponse,
)
def set_camera_credentials(
    camera_id: str,
    body: CredentialCreateRequest,
    request: Request,
):
    """Store credentials scoped for a specific camera_id."""
    user_id = get_current_user_id(request)
    cm = CredentialManager()
    cred_id = body.credential_id or f"cred_{camera_id}"
    try:
        meta = cm.store_credential(
            owner_user_id=user_id,
            username=body.username,
            password=body.password,
            credential_id=cred_id,
            description=body.description or f"Credential for camera {camera_id}",
            shared_user_ids=body.shared_user_ids or [],
        )
        return meta
    except PermissionError as err:
        raise HTTPException(status_code=403, detail=str(err)) from err


# ---------------------------------------------------------------------------
# Camera Lifecycle Endpoints
# ---------------------------------------------------------------------------


@v1_router.post(
    "/cameras/start",
    response_model=CameraInfoResponse,
)
def start_camera(
    body: CameraStartRequest,
    request: Request,
    service: GaitService = Depends(get_gait_service),
):
    """Start a camera recognition worker with automatic or explicit source resolution."""

    if not body.camera_id:
        raise HTTPException(
            status_code=400,
            detail="camera_id is required",
        )

    user_id = get_current_user_id(request)

    try:
        return service.start_camera(
            camera_id=body.camera_id,
            source=body.source or "auto",
            location=body.location or "Surveillance Zone",
            zone_id=body.zone_id,
            user_id=user_id,
            credential_id=body.credential_id,
        )
    except RuntimeError as err:
        raise HTTPException(
            status_code=400,
            detail=sanitize_rtsp_url(str(err)),
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Camera worker startup failed: {sanitize_rtsp_url(str(err))}",
        ) from err


@v1_router.post("/cameras/stop")
def stop_camera(
    body: CameraStopRequest,
    service: GaitService = Depends(get_gait_service),
):
    """Stop an active camera recognition worker."""

    stopped = service.stop_camera(body.camera_id)

    if not stopped:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Camera {body.camera_id} "
                "was not found or is not active"
            ),
        )

    return {
        "success": True,
        "message": f"Camera {body.camera_id} stopped",
    }


@v1_router.get(
    "/cameras",
    response_model=list[CameraInfoResponse],
)
def list_cameras(
    service: GaitService = Depends(get_gait_service),
):
    """Return all active camera workers with live telemetry metrics."""

    return service.list_all_cameras()


@v1_router.get(
    "/cameras/{camera_id}/stream",
)
async def stream_camera(
    camera_id: str,
    service: GaitService = Depends(get_gait_service),
):
    """
    Stream live camera frames in multipart/x-mixed-replace MJPEG format.
    Reuses the active CameraWorker instance without opening additional captures.
    """
    if camera_id not in service.active_cameras:
        raise HTTPException(
            status_code=404,
            detail=f"Camera worker {camera_id} is not active",
        )

    async def frame_generator():
        try:
            while camera_id in service.active_cameras:
                worker = service.get_camera_worker(camera_id)
                if not worker:
                    break

                jpeg_bytes = worker.get_latest_jpeg()
                if jpeg_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n"
                        + jpeg_bytes + b"\r\n"
                    )
                    await asyncio.sleep(0.066)  # ~15 FPS max yield
                else:
                    await asyncio.sleep(0.1)
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@v1_router.get(
    "/cameras/{camera_id}/snapshot",
)
def get_camera_snapshot(
    camera_id: str,
    service: GaitService = Depends(get_gait_service),
):
    """Return the latest single JPEG snapshot frame from the camera worker."""
    worker = service.get_camera_worker(camera_id)
    if not worker or camera_id not in service.active_cameras:
        raise HTTPException(
            status_code=404,
            detail=f"Camera worker {camera_id} is not active",
        )

    jpeg_bytes = worker.get_latest_jpeg()
    if jpeg_bytes is None:
        raise HTTPException(
            status_code=503,
            detail="No frame captured yet from camera",
        )

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@v1_router.post(
    "/enroll",
    response_model=EnrollResponse,
)
async def enroll_subject(
    person_id: Annotated[
        str,
        Form(description="Unique identity assigned to the subject"),
    ],
    files: Annotated[
        list[UploadFile],
        File(description="One or more gait enrollment image files"),
    ],
    service: GaitService = Depends(get_gait_service),
):
    """Enroll a person using one or more gait images."""

    normalized_person_id = person_id.strip()

    if not normalized_person_id:
        raise HTTPException(
            status_code=400,
            detail="person_id is required",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one image file is required",
        )

    image_bytes_list: list[bytes] = []

    try:
        for upload in files:
            if (
                upload.content_type
                and not upload.content_type.startswith("image/")
            ):
                raise HTTPException(
                    status_code=415,
                    detail=(
                        f"Unsupported file type for "
                        f"{upload.filename or 'uploaded file'}: "
                        f"{upload.content_type}"
                    ),
                )

            content = await upload.read()

            if content:
                image_bytes_list.append(content)

        if not image_bytes_list:
            raise HTTPException(
                status_code=400,
                detail="No valid image files supplied",
            )

        result = service.enroll_images(
            normalized_person_id,
            image_bytes_list,
        )

        return result

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Enrollment failed: {error}",
        ) from error

    finally:
        for upload in files:
            await upload.close()


@v1_router.get(
    "/events",
    response_model=list[RecognitionEvent],
)
def get_events(
    service: GaitService = Depends(get_gait_service),
):
    """Return recent gait recognition events."""

    return service.events_log


@v1_router.websocket("/ws/recognition")
async def websocket_recognition(
    websocket: WebSocket,
):
    """Provide real-time recognition events through WebSocket."""

    if (
        hasattr(websocket.app.state, "gait_service")
        and websocket.app.state.gait_service
    ):
        service = websocket.app.state.gait_service
    else:
        service = GaitService()

    await service.ws_manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        service.ws_manager.disconnect(websocket)

    except Exception:
        service.ws_manager.disconnect(websocket)
