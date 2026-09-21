from app.pipeline.inference_pipeline import InferencePipeline
from app.pipeline.live_recognition import LiveRecognitionPipeline


class PipelineFactory:
    @staticmethod
    def create(mode: str):
        normalized = mode.strip().lower()

        if normalized == "inference":
            return InferencePipeline()

        if normalized in {"live", "live_recognition", "surveillance"}:
            return LiveRecognitionPipeline()

        raise ValueError(f"Unsupported pipeline mode: {mode}")
