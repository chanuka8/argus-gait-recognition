from fastapi import APIRouter, HTTPException

from api.schemas import EnrollRequest, EnrollResponse
from enrollment.enrollment_manager import EnrollmentManager

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
def enroll(request: EnrollRequest):
    try:
        manager = EnrollmentManager()
        result = manager.enroll_person(request.folder_path)
        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )