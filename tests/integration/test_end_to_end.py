from pathlib import Path

from pipeline.inference_pipeline import InferencePipeline


def test_sample_gei_exists(sample_gei_path):
    assert Path(sample_gei_path).exists()


def test_inference_pipeline(sample_gei_path):
    pipeline = InferencePipeline()

    result = pipeline.predict(
        str(sample_gei_path)
    )

    assert result is not None

    assert "identity" in result
    assert "score" in result

    assert result["identity"] != ""

    assert isinstance(
        result["score"],
        (int, float),
    )