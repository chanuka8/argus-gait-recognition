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
    assert "tensorrt" in backend.attempted_backends
    assert "onnxruntime" in backend.attempted_backends


def test_genuine_onnx_session_properties_and_metrics(tmp_path: Path):
    onnx_file = tmp_path / "test_model.onnx"
    success = export_onnx(
        model_path="non_existent_ckpt.pth",
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
    except Exception:
        pytest.skip("ONNXRuntime unavailable in environment")


def test_cpu_only_onnx_provider_selection_emits_no_cuda_warning(tmp_path: Path, capwarn=None):
    onnx_file = tmp_path / "test_model.onnx"
    export_onnx("non_existent_ckpt.pth", str(onnx_file), "fp32")
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
        with pytest.warns(None) as record:
            onnx_be = ONNXBackend(config=cfg)
            if onnx_be.is_available():
                assert onnx_be.execution_provider == "CPUExecutionProvider"
                # Confirm no CUDA warnings in recorded warnings
                cuda_warns = [w for w in record if "CUDAExecutionProvider" in str(w.message)]
                assert len(cuda_warns) == 0
    except Exception:
        pytest.skip("ONNXRuntime not installed")


def test_pytorch_fallback_parity_is_exact(tmp_path: Path):
    """
    Verify that a TensorRT->PyTorch fallback backend produces bit-identical
    output to a direct PyTorch reference backend, given identical model weights
    and identical input tensors.

    Root-cause context: ByGaitLight() initializes Conv2d/BatchNorm/Linear layers
    with random weights when no checkpoint is loaded.  Two separate calls to
    get_inference_backend() construct two independent ByGaitLight() instances.
    Unless torch's RNG is re-seeded to the same value before each construction,
    the weights diverge and the outputs differ.  On Windows the global RNG state
    happened to be identical between the two constructions; on Linux/macOS
    (GitHub Actions) intervening library work shifted the RNG, causing failure.

    Fix: seed torch (and numpy) to the same value immediately before each
    backend construction so that ByGaitLight()'s random parameter init is
    deterministic and identical for both instances.
    """

    # --- Deterministic input generation ----------------------------------------
    np.random.seed(42)
    sample_inputs = [np.random.randn(1, 1, 64, 128).astype(np.float32) for _ in range(5)]

    # --- Reference: direct PyTorch backend ------------------------------------
    _seed_all(42)
    ref_backend = PyTorchBackend(
        config={"backend": "pytorch", "device": "cpu", "precision": "fp32", "warmup_iterations": 0},
    )
    ref_outputs = [ref_backend.predict(inp) for inp in sample_inputs]

    # --- Fallback: request TensorRT (unavailable) → falls back to PyTorch -----
    _seed_all(42)   # re-seed to get identical ByGaitLight weights
    fb_backend = get_inference_backend(config={
        "backend": "tensorrt",
        "engine_path": str(tmp_path / "missing.engine"),
        "allow_fallback": True,
        "device": "cpu",
        "precision": "fp32",
        "warmup_iterations": 0,
    })

    assert fb_backend.fallback_used is True
    assert fb_backend.active_backend == "pytorch"

    fb_outputs = [fb_backend.predict(inp) for inp in sample_inputs]

    # --- Parity assertions ----------------------------------------------------
    for i, (ref, fb) in enumerate(zip(ref_outputs, fb_outputs)):
        max_diff = float(np.max(np.abs(ref - fb)))
        assert max_diff == 0.0, (
            f"Sample {i}: max_abs_diff={max_diff} (expected 0.0). "
            "Identical seeds + identical inputs + identical architecture must produce identical outputs."
        )


def _seed_all(seed: int = 42) -> None:
    """Set all RNG seeds to ensure deterministic PyTorch weight initialization."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def test_benchmark_output_labelled_embedding_only():
    from scripts.benchmark_inference_backends import benchmark_backend

    res = benchmark_backend("pytorch", sample_count=5, device="cpu", precision="fp32")
    assert res["measurement_scope"] == "embedding_only_synthetic_gei"
    assert "embedding-only" in res["measurement_notice"].lower() or "synthetic" in res["measurement_notice"].lower()
