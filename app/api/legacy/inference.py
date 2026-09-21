import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import IdentifyRequest, IdentifyResponse
from app.pipeline.inference_pipeline import InferencePipeline

logger = logging.getLogger("ARGUS.InferenceAPI")

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

    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        logger.exception("Image identification failed")
        raise HTTPException(
            status_code=500,
            detail="Image identification failed",
        ) from error
