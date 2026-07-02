from fastapi import APIRouter, HTTPException

from api.schemas import IdentifyRequest, IdentifyResponse
from pipeline.inference_pipeline import InferencePipeline

router = APIRouter()


@router.post("/identify", response_model=IdentifyResponse)
def identify(request: IdentifyRequest):
    try:
        pipeline = InferencePipeline()
        result = pipeline.predict(request.image_path)

        return {
            "identity": str(result["identity"]),
            "score": float(result["score"]),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )