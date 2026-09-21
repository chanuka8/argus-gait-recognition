import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import EnrollRequest, EnrollResponse
from app.enrollment.enrollment_manager import EnrollmentManager

logger = logging.getLogger("ARGUS.EnrollmentAPI")

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
def enroll(request: EnrollRequest):
    try:
        manager = EnrollmentManager()
        result = manager.enroll_person(request.folder_path)
        return result

    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        logger.exception("Enrollment failed")
        raise HTTPException(
            status_code=500,
            detail="Enrollment failed",
        ) from error
