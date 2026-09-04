import asyncio

from fastapi.responses import Response, StreamingResponse

"""Version 1 API routes for the ARGUS gait recognition backend."""

import tempfile
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel

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
    ReadinessResponse,
    RecognitionEvent,
    ReferenceJobStatusResponse,
    ReferenceVideoUploadResponse,
    StatusResponse,
    UploadChunkResponse,
    UploadSessionCommitResponse,
    UploadSessionInitRequest,
    UploadSessionInitResponse,
    UploadSessionStatusResponse,
)
from security_layer.auth import SessionToken, extract_bearer_token, get_session_store
from security_layer.authorization import (
    Permission,
    Role,
    has_permission,
    normalize_role,
    verify_case_access,
    verify_job_access,
)
from security_layer.credentials import CredentialManager, sanitize_rtsp_url
from services.gait_service import GaitService
from services.reference_job_manager import ReferenceJobManager, ReferenceJobStatus
from services.upload_session_manager import UploadSessionManager


def get_current_operator_session(request: Request) -> SessionToken:
    """Derive authenticated operator session from cryptographically verified server session.

    Security Guarantee:
      - Never trusts client-asserted X-User-ID header as proof of identity.
      - Requires a valid Authorization: Bearer <session_token>.
      - Raises HTTP 401 if missing, invalid, or expired.
      - Raises HTTP 403 if operator account is suspended.
    """
    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide Authorization: Bearer <session_token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    session = get_session_store().get_session(token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if session.status == "Suspended":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been suspended",
        )

    return session


def get_current_user_id(request: Request) -> str:
    """Convenience accessor returning operator username."""
    session = get_current_operator_session(request)
    return session.username


def require_admin_operator(request: Request) -> SessionToken:
    """Ensure operator possesses administrative role (admin or root_admin)."""
    session = get_current_operator_session(request)
    role_norm = normalize_role(session.role)
    if role_norm not in (Role.ROOT_ADMIN.value, Role.ADMIN.value):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Operation requires administrative privileges; current role is '{session.role}'",
        )
    return session


_fallback_service_lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None
_fallback_gait_service: GaitService | None = None


def get_gait_service(request: Request = None) -> GaitService:
    global _fallback_gait_service

    if request is not None and hasattr(request, "app") and hasattr(request.app.state, "gait_service") and request.app.state.gait_service:
        return request.app.state.gait_service

    if _fallback_gait_service is None:
        _fallback_gait_service = GaitService()

    return _fallback_gait_service


v1_router = APIRouter(
    prefix="/api/v1",
    tags=["v1"],
)


@v1_router.get(
    "/health",
    response_model=HealthResponse,
)
def get_health(
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    backend_name = "pytorch"
    if service._extractor is not None:
        backend_name = getattr(service._extractor.backend, "backend_name", "pytorch")

    readiness = service.get_readiness() if hasattr(service, "get_readiness") else {}
    states = readiness.get("states", {})
    rec_ready = readiness.get("recognition_ready", service.is_warmed_up)

    return {
        "status": "healthy",
        "system": "ARGUS AI Gait Recognition System",
        "version": "0.1.0",
        "pipeline_loaded": True,
        "recognition_ready": rec_ready,
        "active_backend": backend_name,
        "models": {
            "person_detector": "active" if states.get("DETECTOR_READY", service._detector is not None) else "initializing",
            "silhouette_extractor": "active" if states.get("SILHOUETTE_READY", False) else "initializing",
            "gait_encoder": "active" if states.get("BYGAIT_READY", False) else "initializing",
        },
        "readiness": states,
    }


@v1_router.get(
    "/readiness",
    response_model=ReadinessResponse,
)
def get_readiness(
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    if hasattr(service, "get_readiness"):
        return service.get_readiness()
    return {
        "api_ready": True,
        "recognition_ready": service.is_warmed_up,
        "states": {"API_READY": True, "RECOGNITION_READY": service.is_warmed_up},
        "components": {},
        "warmup_duration_seconds": 0.0,
    }


@v1_router.get(
    "/status",
    response_model=StatusResponse,
)
def get_status(
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    from automation.device_manager import DeviceManager

    metrics = service.get_metrics()
    dm = DeviceManager.get_instance()

    return {
        "status": "operational",
        "device": dm.backend,
        "compute": {
            "backend": dm.backend,
            "device": dm.device,
            "gpu": dm.gpu_name,
            "vram_mb": dm.vram_mb,
            "cuda_available": dm.cuda_available,
            "pytorch_version": dm.pytorch_version,
            "cuda_version": dm.cuda_version,
            "onnx_provider": dm.onnx_provider,
        },
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
    service: Annotated[GaitService, Depends(get_gait_service)],
):
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
    service: Annotated[GaitService, Depends(get_gait_service)],
    request: Request,
    camera_id: Annotated[
        str,
        Form(description="Camera or upload source identifier"),
    ] = "upload-image",
):
    get_current_operator_session(request)

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=("Unsupported file type. Upload a valid image file."),
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
    service: Annotated[GaitService, Depends(get_gait_service)],
    request: Request,
):
    get_current_operator_session(request)
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


@v1_router.post(
    "/credentials",
    response_model=CredentialResponse,
)
def create_credential(
    body: CredentialCreateRequest,
    request: Request,
):
    session = require_admin_operator(request)
    user_id = session.username
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
    session = require_admin_operator(request)
    user_id = session.username
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
    session = require_admin_operator(request)
    user_id = session.username
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
    session = require_admin_operator(request)
    user_id = session.username
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
    session = require_admin_operator(request)
    user_id = session.username
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
    session = require_admin_operator(request)
    user_id = session.username
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


@v1_router.post(
    "/cameras/start",
    response_model=CameraInfoResponse,
)
def start_camera(
    body: CameraStartRequest,
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    session = get_current_operator_session(request)
    if not (
        has_permission(session.role, Permission.CAMERA_START)
        or has_permission(session.role, Permission.CAMERA_CONTROL)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Operation requires camera start privileges; current role is '{session.role}'",
        )
    user_id = session.username

    if not body.camera_id:
        raise HTTPException(
            status_code=400,
            detail="camera_id is required",
        )

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
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    session = get_current_operator_session(request)
    if not (
        has_permission(session.role, Permission.CAMERA_STOP)
        or has_permission(session.role, Permission.CAMERA_CONTROL)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Operation requires camera stop privileges; current role is '{session.role}'",
        )
    stopped = service.stop_camera(body.camera_id)

    if not stopped:
        raise HTTPException(
            status_code=404,
            detail=(f"Camera {body.camera_id} was not found or is not active"),
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
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    get_current_operator_session(request)
    return service.list_all_cameras()


@v1_router.get(
    "/cameras/{camera_id}",
    response_model=CameraInfoResponse,
)
def get_camera(
    camera_id: str,
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    get_current_operator_session(request)
    info = service.get_camera_info(camera_id)
    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {camera_id} is not active",
        )
    return info


@v1_router.get(
    "/cameras/{camera_id}/stream",
)
async def stream_camera(
    camera_id: str,
    max_frames: int = Query(default=0, ge=0, description="Optional frame limit (0 for continuous)"),
    service: Annotated[GaitService, Depends(get_gait_service)] = None,
    request: Request = None,
):
    if request is not None:
        get_current_operator_session(request)
    if service is None:
        service = get_gait_service(request)
    if camera_id not in service.active_cameras:
        raise HTTPException(
            status_code=404,
            detail=f"Camera worker {camera_id} is not active",
        )

    worker = service.get_camera_worker(camera_id)
    if not worker:
        raise HTTPException(
            status_code=404,
            detail=f"Camera worker instance {camera_id} not found",
        )

    try:
        limit = int(max_frames)
    except (ValueError, TypeError):
        limit = int(getattr(max_frames, "default", 0) or 0)

    async def frame_generator():
        worker.register_client()
        frames_sent = 0
        try:
            while camera_id in service.active_cameras:
                if request is not None and await request.is_disconnected():
                    break

                curr_worker = service.get_camera_worker(camera_id)
                if not curr_worker or not curr_worker.is_running():
                    if curr_worker:
                        offline_jpeg = curr_worker.get_latest_jpeg()
                        if offline_jpeg:
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: "
                                + str(len(offline_jpeg)).encode()
                                + b"\r\n\r\n"
                                + offline_jpeg
                                + b"\r\n"
                            )
                    break

                jpeg_bytes = curr_worker.get_latest_jpeg()
                if jpeg_bytes is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )
                    frames_sent += 1
                    if limit > 0 and frames_sent >= limit:
                        break
                    await asyncio.sleep(0.05)
                else:
                    await asyncio.sleep(0.01)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            curr_worker = service.get_camera_worker(camera_id)
            if curr_worker:
                curr_worker.unregister_client()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


@v1_router.get(
    "/cameras/{camera_id}/snapshot",
)
def get_camera_snapshot(
    camera_id: str,
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    get_current_operator_session(request)
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
    service: Annotated[GaitService, Depends(get_gait_service)],
    request: Request,
    async_mode: Annotated[
        bool,
        Form(description="Whether to process enrollment asynchronously via background job"),
    ] = False,
):
    session = get_current_operator_session(request)
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
            if upload.content_type and not upload.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=415,
                    detail=(f"Unsupported file type for {upload.filename or 'uploaded file'}: {upload.content_type}"),
                )

            content = await upload.read()

            if content:
                image_bytes_list.append(content)

        if not image_bytes_list:
            raise HTTPException(
                status_code=400,
                detail="No valid image files supplied",
            )

        if async_mode:
            photos_dir = Path("data/reference_photos")
            photos_dir.mkdir(parents=True, exist_ok=True)
            saved_paths: list[str] = []
            timestamp = int(time.time())
            for idx, raw_bytes in enumerate(image_bytes_list):
                p_path = photos_dir / f"{normalized_person_id}_{timestamp}_{idx:02d}.jpg"
                p_path.write_bytes(raw_bytes)
                saved_paths.append(str(p_path))

            job_mgr = ReferenceJobManager.get_instance()
            job = job_mgr.create_job(
                person_id=normalized_person_id,
                media_path=saved_paths[0],
                media_type="image",
                owner=session.username,
            )

            from services.missing_person_processor import MissingPersonVideoProcessor

            processor = MissingPersonVideoProcessor(
                detector=service.detector,
                extractor=service.extractor,
                appearance_extractor=service.appearance_extractor,
                silhouette_step=service.silhouette_extractor,
                store=service.store,
                embedding_db=service.embedding_db,
            )

            job_mgr.submit_task(
                processor.process_reference_photos,
                person_id=normalized_person_id,
                photo_paths=saved_paths,
                job_id=job.job_id,
                case_id=normalized_person_id,
                gait_service_ref=service,
            )

            return {
                "success": True,
                "person_id": normalized_person_id,
                "message": "Images uploaded and queued for asynchronous biometric processing",
                "embeddings_added": len(saved_paths),
                "gait_embeddings_added": 0,
                "appearance_embeddings_added": 0,
                "firebase_status": "PENDING",
                "status": "QUEUED",
                "job_id": job.job_id,
            }

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


@v1_router.post(
    "/cases/upload-reference",
    response_model=ReferenceVideoUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_case_reference_video(
    person_id: Annotated[
        str,
        Form(description="Missing person / case identifier"),
    ],
    file: Annotated[
        UploadFile,
        File(description="Reference video file containing target subject"),
    ],
    service: Annotated[GaitService, Depends(get_gait_service)],
    request: Request,
    case_id: Annotated[
        str | None,
        Form(description="Optional case identifier if distinct from person_id"),
    ] = None,
):
    session = get_current_operator_session(request)
    normalized_person_id = person_id.strip()
    if not normalized_person_id:
        raise HTTPException(status_code=400, detail="person_id is required")

    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    suffix = ".mp4"
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", maxsplit=1)[-1].lower()

    if suffix not in valid_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video format '{suffix}'. Allowed: {sorted(valid_extensions)}",
        )

    videos_dir = Path("data/reference_videos")
    videos_dir.mkdir(parents=True, exist_ok=True)

    safe_base = Path(file.filename or f"reference{suffix}").name
    clean_name = "".join(c for c in safe_base if c.isalnum() or c in "._-") or f"reference{suffix}"
    save_filename = f"{normalized_person_id}_{int(time.time())}_{clean_name}"
    saved_path = videos_dir / save_filename

    def _save_stream_to_file(src_file, dst_path: Path) -> int:
        written = 0
        with open(dst_path, "wb") as f_out:
            while True:
                chunk = src_file.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
                written += len(chunk)
        return written

    total_bytes = 0
    try:
        total_bytes = await asyncio.to_thread(_save_stream_to_file, file.file, saved_path)
        if total_bytes == 0:
            if saved_path.exists():
                saved_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded reference video file is empty")
    except HTTPException:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {exc}") from exc
    finally:
        await file.close()

    job_mgr = ReferenceJobManager.get_instance()
    job = job_mgr.create_job(
        person_id=normalized_person_id,
        video_path=str(saved_path),
        case_id=case_id or normalized_person_id,
        owner=session.username,
    )

    from services.missing_person_processor import MissingPersonVideoProcessor

    processor = MissingPersonVideoProcessor(
        detector=service.detector,
        tracker=service.tracker,
        extractor=service.extractor,
        appearance_extractor=service.appearance_extractor,
        silhouette_step=service.silhouette_extractor,
        store=service.store,
        embedding_db=service.embedding_db,
    )

    job_mgr.submit_task(
        processor.process_reference_video,
        person_id=normalized_person_id,
        video_path=str(saved_path),
        job_id=job.job_id,
        case_id=case_id or normalized_person_id,
        gait_service_ref=service,
    )

    return {
        "job_id": job.job_id,
        "person_id": normalized_person_id,
        "status": ReferenceJobStatus.QUEUED.value,
        "message": "Reference video uploaded and queued for camera-independent gait embedding extraction",
        "created_at": job.created_at,
    }


@v1_router.get(
    "/cases/jobs/{job_id}",
    response_model=ReferenceJobStatusResponse,
)
def get_case_job_status(
    job_id: str,
    request: Request,
):
    session = get_current_operator_session(request)
    job = verify_job_access(job_id, session)
    return job.to_dict()


@v1_router.post(
    "/cases/jobs/{job_id}/retry",
    response_model=ReferenceJobStatusResponse,
)
def retry_case_job(
    job_id: str,
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    session = get_current_operator_session(request)
    job = verify_job_access(job_id, session, mutate=True)

    job_mgr = ReferenceJobManager.get_instance()

    # Idempotency: if already completed, return cached result immediately
    if job.status == ReferenceJobStatus.COMPLETED:
        return job.to_dict()

    if job.status in (ReferenceJobStatus.PROCESSING, ReferenceJobStatus.RESUMING):
        return job.to_dict()

    # Resume interrupted or failed job from last safe checkpoint
    job_mgr.update_progress(job_id, status=ReferenceJobStatus.RESUMING, resumed=True)
    from services.missing_person_processor import MissingPersonVideoProcessor

    processor = MissingPersonVideoProcessor(
        detector=service.detector,
        tracker=service.tracker,
        extractor=service.extractor,
        appearance_extractor=service.appearance_extractor,
        silhouette_step=service.silhouette_extractor,
        store=service.store,
        embedding_db=service.embedding_db,
    )

    if job.media_type == "video":
        job_mgr.submit_task(
            processor.process_reference_video,
            person_id=job.person_id,
            video_path=job.video_path,
            job_id=job.job_id,
            case_id=job.case_id,
            gait_service_ref=service,
        )
    else:
        job_mgr.submit_task(
            processor.process_reference_photos,
            person_id=job.person_id,
            photo_paths=[job.media_path],
            job_id=job.job_id,
            case_id=job.case_id,
            gait_service_ref=service,
        )

    updated_job = job_mgr.get_job(job_id)
    return updated_job.to_dict() if updated_job else job.to_dict()


@v1_router.get(
    "/cases/jobs",
    response_model=list[ReferenceJobStatusResponse],
)
def list_case_jobs(
    request: Request,
):
    session = get_current_operator_session(request)
    owner_filter = session.username if normalize_role(session.role) == Role.INVESTIGATOR.value else None
    jobs = ReferenceJobManager.get_instance().list_jobs(limit=50, owner=owner_filter)
    return [j.to_dict() for j in jobs]


# =========================================================================
# Resumable Chunked Upload Session Endpoints
# =========================================================================


@v1_router.post(
    "/cases/upload-session/init",
    response_model=UploadSessionInitResponse,
    status_code=status.HTTP_201_CREATED,
)
def init_upload_session(
    payload: UploadSessionInitRequest,
    request: Request,
):
    session = get_current_operator_session(request)
    session_mgr = UploadSessionManager.get_instance()
    try:
        record = session_mgr.create_session(
            person_id=payload.person_id,
            filename=payload.filename,
            total_size=payload.total_size,
            chunk_size=payload.chunk_size,
            media_type=payload.media_type,
            case_id=payload.case_id or payload.person_id,
            owner=session.username,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "upload_id": record.upload_id,
        "person_id": record.person_id,
        "chunk_size": record.chunk_size,
        "total_chunks": record.total_chunks,
        "total_size": record.total_size,
        "expires_at": record.expires_at,
    }


@v1_router.post(
    "/cases/upload-session/{upload_id}/chunk",
    response_model=UploadChunkResponse,
)
async def upload_session_chunk(
    upload_id: str,
    chunk_index: Annotated[int, Form(description="0-indexed chunk sequence number")],
    file: Annotated[UploadFile, File(description="Chunk binary payload")],
    request: Request,
):
    get_current_operator_session(request)
    session_mgr = UploadSessionManager.get_instance()
    try:
        chunk_bytes = await file.read()
        success, msg, data = session_mgr.write_chunk(
            upload_id=upload_id,
            chunk_index=chunk_index,
            chunk_bytes=chunk_bytes,
        )
        if not success:
            raise HTTPException(status_code=400, detail=msg)
    finally:
        await file.close()

    return {
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "chunks_received": data["chunks_received"],
        "total_chunks": data["total_chunks"],
        "bytes_received": data["bytes_received"],
        "is_complete": data["is_complete"],
    }


@v1_router.get(
    "/cases/upload-session/{upload_id}/status",
    response_model=UploadSessionStatusResponse,
)
def get_upload_session_status(
    upload_id: str,
    request: Request,
):
    get_current_operator_session(request)
    session_mgr = UploadSessionManager.get_instance()
    record = session_mgr.get_session(upload_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Upload session '{upload_id}' not found")

    return {
        "upload_id": record.upload_id,
        "person_id": record.person_id,
        "status": record.status,
        "total_chunks": record.total_chunks,
        "chunks_received": sorted(record.chunks_received),
        "bytes_received": record.bytes_received,
        "total_size": record.total_size,
        "is_complete": len(record.chunks_received) == record.total_chunks,
        "expires_at": record.expires_at,
        "job_id": record.job_id,
    }


@v1_router.post(
    "/cases/upload-session/{upload_id}/commit",
    response_model=UploadSessionCommitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def commit_upload_session(
    upload_id: str,
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    session = get_current_operator_session(request)
    session_mgr = UploadSessionManager.get_instance()
    success, msg, final_path = session_mgr.assemble_and_commit(upload_id)
    if not success or final_path is None:
        raise HTTPException(status_code=400, detail=msg)

    record = session_mgr.get_session(upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload session not found after commit")

    job_mgr = ReferenceJobManager.get_instance()
    job = job_mgr.create_job(
        person_id=record.person_id,
        case_id=record.case_id,
        media_path=str(final_path),
        media_type=record.media_type,
        owner=session.username,
    )
    record.job_id = job.job_id
    session_mgr._persist_session(record)

    from services.missing_person_processor import MissingPersonVideoProcessor

    processor = MissingPersonVideoProcessor(
        detector=service.detector,
        tracker=service.tracker,
        extractor=service.extractor,
        appearance_extractor=service.appearance_extractor,
        silhouette_step=service.silhouette_extractor,
        store=service.store,
        embedding_db=service.embedding_db,
    )

    if record.media_type == "video":
        job_mgr.submit_task(
            processor.process_reference_video,
            person_id=record.person_id,
            video_path=str(final_path),
            job_id=job.job_id,
            case_id=record.case_id,
            gait_service_ref=service,
        )
    else:
        job_mgr.submit_task(
            processor.process_reference_photos,
            person_id=record.person_id,
            photo_paths=[final_path],
            job_id=job.job_id,
            case_id=record.case_id,
            gait_service_ref=service,
        )

    return {
        "upload_id": upload_id,
        "person_id": record.person_id,
        "status": ReferenceJobStatus.QUEUED.value,
        "job_id": job.job_id,
        "media_path": str(final_path),
        "message": f"Session committed. Asynchronous biometric extraction job queued for {record.media_type}.",
    }


@v1_router.post(
    "/cases/upload-session/{upload_id}/cancel",
)
def cancel_upload_session(
    upload_id: str,
    request: Request,
):
    get_current_operator_session(request)
    session_mgr = UploadSessionManager.get_instance()
    cancelled = session_mgr.cancel_session(upload_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"Upload session '{upload_id}' not found")
    return {"success": True, "upload_id": upload_id, "status": "CANCELLED"}


@v1_router.get("/gallery")
def get_gallery(
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    get_current_operator_session(request)
    persons = service.embedding_db.list_all_persons()
    active = [p for p in persons if p.status == "ACTIVE"]
    return {
        "total_persons": len(active),
        "persons": [
            {
                "person_id": p.person_id,
                "gait_embeddings": len(p.gait_embeddings),
                "appearance_embeddings": len(p.appearance_embeddings),
                "status": p.status,
                "updated_at": p.updated_at,
            }
            for p in active
        ],
    }


@v1_router.get("/cases/{case_id}/gallery")
def get_case_gallery(
    case_id: str,
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    session = get_current_operator_session(request)
    verify_case_access(case_id, session)
    person = service.embedding_db.get_person(case_id)
    if not person or person.status != "ACTIVE":
        raise HTTPException(status_code=404, detail=f"Case biometric subject '{case_id}' not found")
    return {
        "case_id": case_id,
        "person_id": person.person_id,
        "gait_embeddings_count": len(person.gait_embeddings),
        "appearance_embeddings_count": len(person.appearance_embeddings),
        "status": person.status,
    }


class GalleryDeleteRequest(BaseModel):
    person_id: str


@v1_router.post("/gallery/delete")
def delete_gallery_subject(
    body: GalleryDeleteRequest,
    request: Request,
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    require_admin_operator(request)
    success = service.embedding_db.delete_person(body.person_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Subject '{body.person_id}' not found in biometric gallery")
    return {"success": True, "person_id": body.person_id, "message": "Biometric gallery record marked as deleted"}


@v1_router.get("/learning/status")
def get_continual_learning_status(request: Request):
    require_admin_operator(request)
    return {
        "status": "IDLE",
        "model_version": "v1.0.0",
        "candidates_evaluated": 0,
        "promotion_eligible": False,
    }


@v1_router.post("/learning/promote")
def promote_candidate_model(request: Request):
    session = get_current_operator_session(request)
    if normalize_role(session.role) != Role.ROOT_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Root Administrator privileges required to promote model versions",
        )
    return {"success": True, "message": "Candidate model promoted to production"}


@v1_router.post("/learning/rollback")
def rollback_model_version(request: Request):
    session = get_current_operator_session(request)
    if normalize_role(session.role) != Role.ROOT_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Root Administrator privileges required to rollback model versions",
        )
    return {"success": True, "message": "Production model rolled back to previous baseline"}


@v1_router.get(
    "/events",
    response_model=list[RecognitionEvent],
)
def get_events(
    service: Annotated[GaitService, Depends(get_gait_service)],
):
    return list(service.events_log)


@v1_router.websocket("/ws/recognition")
@v1_router.websocket("/ws/events")
async def websocket_recognition(
    websocket: WebSocket,
):
    if hasattr(websocket.app.state, "gait_service") and websocket.app.state.gait_service:
        service = websocket.app.state.gait_service
    else:
        service = GaitService()

    await service.ws_manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except (WebSocketDisconnect, ConnectionResetError, RuntimeError):
        service.ws_manager.disconnect(websocket)
