"""
Unit tests for Inference Backends abstraction and fallback mechanisms.
"""

from pathlib import Path
import numpy as np
import pytest

from models.inference.backend import get_inference_backend
from models.inference.pytorch_backend import PyTorchBackend
from models.inference.tensorrt_backend import TensorRTBackend
from scripts.export_bygait_onnx import export_onnx


def test_pytorch_backend_predict_shape_and_l2_normalization():
    backend = PyTorchBackend(config={"backend": "pytorch", "device": "cpu", "precision": "fp32", "warmup_iterations": 0})
    dummy_input = np.random.randn(1, 1, 64, 128).astype(np.float32)
    embedding = backend.predict(dummy_input)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 256)

    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_backend_selection_factory():
    backend_pytorch = get_inference_backend(config={"backend": "pytorch", "device": "cpu"})
    assert isinstance(backend_pytorch, PyTorchBackend)


def test_missing_engine_tensorrt_fallback(tmp_path: Path):
    cfg = {
        "backend": "tensorrt",
        "engine_path": str(tmp_path / "non_existent.engine"),
        "allow_fallback": True,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    backend = get_inference_backend(config=cfg)
    # Should safely fall back to PyTorch backend without crashing
    assert isinstance(backend, PyTorchBackend)

    dummy_input = np.zeros((1, 1, 64, 128), dtype=np.float32)
    embedding = backend.predict(dummy_input)
    assert embedding.shape == (1, 256)


def test_onnx_fallback_when_file_missing(tmp_path: Path):
    cfg = {
        "backend": "onnxruntime",
        "onnx_path": str(tmp_path / "non_existent.onnx"),
        "allow_fallback": True,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    backend = get_inference_backend(config=cfg)
    assert isinstance(backend, PyTorchBackend)


def test_invalid_backend_rejection():
    cfg = {"backend": "unknown_backend", "allow_fallback": False}
    with pytest.raises(ValueError, match="Unsupported inference backend"):
        get_inference_backend(config=cfg)


def test_onnx_export_script_execution(tmp_path: Path):
    onnx_file = tmp_path / "bygait_light_test.onnx"
    # Export dummy initialized model
    success = export_onnx(
        model_path="non_existent_ckpt.pth",
        output_onnx_path=str(onnx_file),
        precision="fp32",
    )
    assert isinstance(success, bool)
    if success:
        assert onnx_file.exists()


def test_tensorrt_backend_instantiation_without_tensorrt_package(tmp_path: Path):
    cfg = {
        "backend": "tensorrt",
        "engine_path": str(tmp_path / "fake.engine"),
        "allow_fallback": True,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    # Test that when tensorrt is absent, fallback PyTorch backend works seamlessly
    trt_backend = TensorRTBackend(config=cfg)
    assert not trt_backend.is_available()

    dummy_input = np.ones((1, 1, 64, 128), dtype=np.float32)
    output = trt_backend.predict(dummy_input)
    assert output.shape == (1, 256)
    assert np.isclose(np.linalg.norm(output), 1.0, atol=1e-5)
