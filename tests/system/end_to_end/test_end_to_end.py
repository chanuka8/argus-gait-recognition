from pathlib import Path

import pytest

from app.pipeline.inference_pipeline import InferencePipeline
from app.storage.vector_store import VectorStore


def test_sample_gei_exists(sample_gei_path):
    gei_file = Path(sample_gei_path)
    if not gei_file.exists():
        pytest.skip(f"Required sample GEI asset not found: {sample_gei_path}")

    assert gei_file.exists()


def test_inference_pipeline(sample_gei_path):
    gei_file = Path(sample_gei_path)
    if not gei_file.exists():
        pytest.skip(f"Required sample GEI asset not found: {sample_gei_path}")

    model_path = Path("runs/exp_001/best_model.pth")
    if not model_path.exists():
        pytest.skip(f"Required model checkpoint not found: {model_path}")

    gallery = VectorStore(gallery_dir="ml_platform/models/galleries/live_gallery").load()
    if gallery is None:
        pytest.skip("Required gallery vector store files not found in models/galleries/live_gallery")

    pipeline = InferencePipeline()

    result = pipeline.predict(str(sample_gei_path))

    assert result is not None

    assert "identity" in result
    assert "score" in result

    assert result["identity"] != ""

    assert isinstance(
        result["score"],
        (int, float),
    )
