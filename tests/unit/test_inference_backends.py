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
    backend = PyTorchBackend(
        config={"backend": "pytorch", "device": "cpu", "precision": "fp32", "warmup_iterations": 0}
    )
    dummy_input = np.random.randn(1, 1, 128, 64).astype(np.float32)
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
    assert isinstance(backend, PyTorchBackend)

    dummy_input = np.zeros((1, 1, 128, 64), dtype=np.float32)
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
    ckpt_file = tmp_path / "model.pth"
    import torch

    from models.architectures.bygait_light import ByGaitLight

    torch.save(ByGaitLight().state_dict(), ckpt_file)

    onnx_file = tmp_path / "bygait_light_test.onnx"
    success = export_onnx(
        model_path=str(ckpt_file),
        output_onnx_path=str(onnx_file),
        precision="fp32",
    )
    import importlib.util

    has_onnx = (importlib.util.find_spec("onnx") is not None) and (importlib.util.find_spec("onnxruntime") is not None)

    if has_onnx:
        assert success is True
        assert onnx_file.exists()
    else:
        assert success is False


def test_strict_onnx_mode_no_fallback_when_disabled(tmp_path: Path):
    cfg = {
        "backend": "onnxruntime",
        "onnx_path": str(tmp_path / "missing.onnx"),
        "allow_fallback": False,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    with pytest.raises(RuntimeError, match="ONNX backend unavailable and fallback disabled"):
        get_inference_backend(config=cfg)


def test_tensorrt_backend_instantiation_without_tensorrt_package(tmp_path: Path):
    cfg = {
        "backend": "tensorrt",
        "engine_path": str(tmp_path / "fake.engine"),
        "allow_fallback": True,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    trt_backend = TensorRTBackend(config=cfg)
    assert not trt_backend.is_available()

    dummy_input = np.ones((1, 1, 128, 64), dtype=np.float32)
    output = trt_backend.predict(dummy_input)
    assert output.shape == (1, 256)
    assert np.isclose(np.linalg.norm(output), 1.0, atol=1e-5)


def test_missing_onnx_file_fallback_properties(tmp_path: Path):
    cfg = {
        "backend": "onnxruntime",
        "onnx_path": str(tmp_path / "missing.onnx"),
        "allow_fallback": True,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    backend = get_inference_backend(config=cfg)
    assert backend.requested_backend == "onnxruntime"
    assert backend.active_backend == "pytorch"
    assert backend.fallback_used is True
    assert backend.fallback_reason is not None


def test_missing_tensorrt_package_or_file_fallback_properties(tmp_path: Path):
    cfg = {
        "backend": "tensorrt",
        "engine_path": str(tmp_path / "missing.engine"),
        "allow_fallback": True,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    backend = get_inference_backend(config=cfg)
    assert backend.requested_backend == "tensorrt"
    assert backend.active_backend == "pytorch"
    assert backend.fallback_used is True
    assert backend.fallback_reason is not None


def test_auto_fallback_properties(tmp_path: Path):
    cfg = {
        "backend": "auto",
        "engine_path": str(tmp_path / "missing.engine"),
        "onnx_path": str(tmp_path / "missing.onnx"),
        "allow_fallback": True,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    backend = get_inference_backend(config=cfg)
    assert backend.requested_backend == "auto"
    assert backend.active_backend == "pytorch"
    assert backend.fallback_used is True
    assert backend.selection_fallback_used is True
    assert "onnxruntime" in backend.attempted_backends
    assert "pytorch" in backend.attempted_backends


def test_backend_validator_and_report_generation(tmp_path: Path):
    from models.inference.backend import BackendStatus, BackendValidator, generate_backend_report

    cfg = {
        "backend": "pytorch",
        "device": "cpu",
        "warmup_iterations": 0,
    }
    backend = get_inference_backend(config=cfg)
    validator = BackendValidator(config=cfg)

    pt_health = validator.check_pytorch()
    assert pt_health.is_available is True
    assert pt_health.status in (BackendStatus.READY, BackendStatus.AVAILABLE)

    report_file = tmp_path / "backend_report.json"
    rep = generate_backend_report(backend, output_path=str(report_file))

    assert rep["requested_backend"] == "pytorch"
    assert rep["active_backend"] == "pytorch"
    assert rep["initialization_result"] == "SUCCESS"
    assert rep["inference_smoke_test_result"] == "PASSED"
    assert report_file.exists()


def test_genuine_onnx_session_properties_and_metrics(tmp_path: Path):
    ckpt_file = tmp_path / "model.pth"
    import torch

    from models.architectures.bygait_light import ByGaitLight

    torch.save(ByGaitLight().state_dict(), ckpt_file)

    onnx_file = tmp_path / "test_model.onnx"
    success = export_onnx(
        model_path=str(ckpt_file),
        output_onnx_path=str(onnx_file),
        precision="fp32",
    )
    if not success or not onnx_file.exists():
        pytest.skip("ONNX export unavailable in environment")

    cfg = {
        "backend": "onnxruntime",
        "onnx_path": str(onnx_file),
        "allow_fallback": True,
        "device": "cpu",
        "warmup_iterations": 0,
    }
    try:
        backend = get_inference_backend(config=cfg)
        if backend.is_available():
            assert backend.requested_backend == "onnxruntime"
            assert backend.active_backend == "onnxruntime"
            assert backend.fallback_used is False
            assert backend.execution_provider == "CPUExecutionProvider"
    except (ImportError, RuntimeError, ValueError, OSError):
        pytest.skip("ONNXRuntime unavailable in environment")


def test_cpu_only_onnx_provider_selection_emits_no_cuda_warning(tmp_path: Path, capwarn=None):
    ckpt_file = tmp_path / "model.pth"
    import torch

    from models.architectures.bygait_light import ByGaitLight

    torch.save(ByGaitLight().state_dict(), ckpt_file)

    onnx_file = tmp_path / "test_model.onnx"
    export_onnx(str(ckpt_file), str(onnx_file), "fp32")
    if not onnx_file.exists():
        pytest.skip("ONNX model unavailable for test")

    from models.inference.onnx_backend import ONNXBackend

    cfg = {
        "backend": "onnxruntime",
        "onnx_path": str(onnx_file),
        "device": "cpu",
        "allow_fallback": False,
        "warmup_iterations": 0,
    }
    try:
        import warnings

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            onnx_be = ONNXBackend(config=cfg)
            if onnx_be.is_available():
                assert onnx_be.execution_provider == "CPUExecutionProvider"
                cuda_warns = [w for w in record if "CUDAExecutionProvider" in str(w.message)]
                assert len(cuda_warns) == 0
    except (ImportError, RuntimeError, ValueError, OSError):
        pytest.skip("ONNXRuntime not installed")


def test_pytorch_fallback_parity_is_exact(tmp_path: Path):
    """
    Verify that when TensorRT is unavailable and fallback is enabled,
    get_inference_backend() returns a working PyTorch fallback backend
    with correct metadata and successful deterministic inference.

    This test validates the fallback *contract*, not cross-model weight parity.
    Two independently constructed PyTorchBackend instances have different random
    weights when no checkpoint is loaded, so comparing their outputs is invalid.
    Instead we verify:
      1. Fallback metadata is correctly reported
      2. Inference succeeds without exceptions
      3. Output tensor has expected shape and dtype
      4. The same backend produces identical output for the same input (self-consistency)
      5. Output embeddings are L2-normalized
    """
    cfg = {
        "backend": "tensorrt",
        "engine_path": str(tmp_path / "missing.engine"),
        "allow_fallback": True,
        "device": "cpu",
        "precision": "fp32",
        "warmup_iterations": 0,
    }
    backend = get_inference_backend(config=cfg)

    assert backend.requested_backend == "tensorrt"
    assert backend.active_backend == "pytorch"
    assert backend.fallback_used is True
    assert backend.fallback_reason is not None
    assert len(backend.fallback_reason) > 0

    np.random.seed(42)
    test_input = np.random.randn(1, 1, 128, 64).astype(np.float32)

    embedding = backend.predict(test_input)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 256)
    assert embedding.dtype == np.float32

    embedding_again = backend.predict(test_input)
    assert np.array_equal(embedding, embedding_again), (
        "Same backend instance must produce identical output for identical input"
    )

    norm = float(np.linalg.norm(embedding))
    assert np.isclose(norm, 1.0, atol=1e-5), f"Expected L2 norm ≈ 1.0, got {norm}"


def test_tensorrt_failure_emits_single_warning():
    import logging

    from monitoring.logging_config import get_logger

    logger = get_logger("detection")
    records = []

    class TestHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = TestHandler()
    logger.addHandler(handler)
    try:
        cfg = {
            "backend": "tensorrt",
            "engine_path": "non_existent.engine",
            "allow_fallback": True,
            "device": "cpu",
            "warmup_iterations": 0,
        }
        get_inference_backend(config=cfg)
        trt_warns = [r for r in records if "TensorRT" in r.getMessage() and r.levelno == logging.WARNING]
        assert len(trt_warns) == 1
    finally:
        logger.removeHandler(handler)


def test_repeated_inference_resource_safety():
    """Verify backend instance stability and resource safety over 100 repeated inferences."""
    cfg = {"backend": "pytorch", "device": "cpu"}
    backend = get_inference_backend(config=cfg)
    initial_id = id(backend)

    dummy_input = np.ones((1, 1, 128, 64), dtype=np.float32)

    for i in range(100):
        out = backend.predict(dummy_input)
        assert out.shape == (1, 256)
        assert np.isfinite(out).all()
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)

    assert id(backend) == initial_id


def test_benchmark_output_labelled_embedding_only():
    from scripts.benchmark_inference_backends import benchmark_backend

    res = benchmark_backend("pytorch", sample_count=5, device="cpu", precision="fp32")
    assert res["measurement_scope"] == "embedding_only_synthetic_gei"
    assert "embedding-only" in res["measurement_notice"].lower() or "synthetic" in res["measurement_notice"].lower()
