import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from deployment.doctor import run_doctor
from models.export.bygait_onnx import export_onnx
from models.inference.backend import BackendStatus, BackendValidator, get_inference_backend
from utils.config_validator import ConfigValidator, sanitize_rtsp_url


def _has_onnx_pkgs() -> bool:
    return (importlib.util.find_spec("onnx") is not None) and (importlib.util.find_spec("onnxruntime") is not None)


def test_valid_onnx_export_metadata(tmp_path: Path):
    ckpt_file = tmp_path / "model.pth"
    import torch

    from models.architectures.bygait_light import ByGaitLight

    torch.save(ByGaitLight().state_dict(), ckpt_file)

    target_onnx = tmp_path / "bygait.onnx"
    json_report = tmp_path / "onnx_report.json"
    md_report = tmp_path / "onnx_report.md"

    has_onnx = _has_onnx_pkgs()

    success = export_onnx(
        model_path=str(ckpt_file),
        output_onnx_path=str(target_onnx),
        report_json_path=str(json_report),
        report_md_path=str(md_report),
    )

    assert json_report.exists()
    with open(json_report, encoding="utf-8") as f:
        data = json.load(f)

    assert data["checkpoint_exists"] is True
    if has_onnx:
        assert success is True
        assert data["export_succeeded"] is True
        assert data["input_shape"] == [1, 1, 128, 64]
        assert data["output_shape"] == [1, 256]
    else:
        assert success is False
        assert data["export_succeeded"] is False


def test_missing_onnx_file_handling(tmp_path: Path):
    cfg = {
        "backend": "onnxruntime",
        "onnx_path": str(tmp_path / "non_existent.onnx"),
        "allow_fallback": True,
    }
    backend = get_inference_backend(config=cfg)

    meta = backend.metadata
    assert meta["requested_backend"] == "onnxruntime"
    assert meta["active_backend"] == "pytorch"
    assert meta["fallback_used"] is True


def test_invalid_onnx_model_handling(tmp_path: Path):
    bad_onnx = tmp_path / "bad.onnx"
    bad_onnx.write_bytes(b"INVALID_ONNX_BYTES")

    cfg = {
        "backend": "onnxruntime",
        "onnx_path": str(bad_onnx),
        "allow_fallback": True,
    }
    backend = get_inference_backend(config=cfg)

    meta = backend.metadata
    assert meta["requested_backend"] == "onnxruntime"
    assert meta["active_backend"] == "pytorch"
    assert meta["fallback_used"] is True


def test_onnx_runtime_unavailable_simulation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    validator = BackendValidator(config={"onnx_path": str(tmp_path / "model.onnx")})
    health = validator.check_onnxruntime()

    assert health.is_available is False
    assert health.status == BackendStatus.NOT_INSTALLED


def test_onnx_numerical_parity_pass_and_fail(tmp_path: Path):
    ckpt_file = tmp_path / "model.pth"
    import torch

    from models.architectures.bygait_light import ByGaitLight

    torch.save(ByGaitLight().state_dict(), ckpt_file)

    if not _has_onnx_pkgs():
        pytest.skip("ONNX / ONNX Runtime package not available in test environment")

    target_onnx = tmp_path / "bygait.onnx"
    json_report = tmp_path / "onnx_report.json"
    md_report = tmp_path / "onnx_report.md"

    pass_ok = export_onnx(
        model_path=str(ckpt_file),
        output_onnx_path=str(target_onnx),
        rtol=1e-2,
        atol=1e-2,
        report_json_path=str(json_report),
        report_md_path=str(md_report),
    )
    assert pass_ok is True

    fail_ok = export_onnx(
        model_path=str(ckpt_file),
        output_onnx_path=str(target_onnx),
        rtol=0.0,
        atol=0.0,
        report_json_path=str(json_report),
        report_md_path=str(md_report),
    )
    assert fail_ok is False


def test_pytorch_backend_readiness():
    backend = get_inference_backend(config={"backend": "pytorch", "device": "cpu"})
    meta = backend.metadata
    assert meta["requested_backend"] == "pytorch"
    assert meta["active_backend"] == "pytorch"
    assert meta["fallback_used"] is False

    dummy = np.zeros((1, 1, 128, 64), dtype=np.float32)
    out = backend.predict(dummy)

    assert out.shape == (1, 256)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_onnx_backend_readiness(tmp_path: Path):
    ckpt_file = tmp_path / "model.pth"
    import torch

    from models.architectures.bygait_light import ByGaitLight

    torch.save(ByGaitLight().state_dict(), ckpt_file)

    target_onnx = tmp_path / "model.onnx"
    export_onnx(model_path=str(ckpt_file), output_onnx_path=str(target_onnx))

    cfg = {"backend": "onnxruntime", "onnx_path": str(target_onnx), "allow_fallback": True}
    backend = get_inference_backend(config=cfg)

    meta = backend.metadata
    assert meta["requested_backend"] == "onnxruntime"
    if meta["active_backend"] == "onnxruntime":
        assert meta["fallback_used"] is False
    else:
        assert meta["active_backend"] == "pytorch"
        assert meta["fallback_used"] is True

    dummy = np.zeros((1, 1, 128, 64), dtype=np.float32)
    out = backend.predict(dummy)
    assert out.shape == (1, 256)


def test_onnx_to_pytorch_fallback(tmp_path: Path):
    cfg = {"backend": "onnxruntime", "onnx_path": str(tmp_path / "missing.onnx"), "allow_fallback": True}
    backend = get_inference_backend(config=cfg)

    assert backend.requested_backend == "onnxruntime"
    assert backend.active_backend == "pytorch"
    assert backend.fallback_used is True


def test_invalid_yaml_handling(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed list", encoding="utf-8")

    validator = ConfigValidator()
    data, err = validator.load_yaml(bad_yaml)

    assert data is None
    assert err is not None
    assert "YAML syntax error" in err


def test_missing_checkpoint_handling():
    backend = get_inference_backend(config={"backend": "pytorch"}, model_path="non_existent_model.pth")
    assert backend.active_backend == "pytorch"
    out = backend.predict(np.zeros((1, 1, 128, 64), dtype=np.float32))
    assert out.shape == (1, 256)


def test_sanitized_rtsp_errors():
    secret_url = "rtsp://admin:super_secret_password@192.168.1.50:554/stream"
    sanitized = sanitize_rtsp_url(secret_url)

    assert "super_secret_password" not in sanitized
    assert "admin:***@" in sanitized


def test_health_report_generation_and_exit_code(tmp_path: Path):
    j_path = tmp_path / "health_report.json"
    m_path = tmp_path / "health_report.md"

    code, data = run_doctor(json_path=str(j_path), md_path=str(m_path))

    assert code in (0, 1)
    assert j_path.exists()
    assert m_path.exists()
    assert "overall_status" in data


def test_doctor_no_repository_modification(tmp_path: Path):
    models_dir = Path("models")
    mtimes_before = {f: f.stat().st_mtime for f in models_dir.rglob("*") if f.is_file()}

    run_doctor(json_path=str(tmp_path / "j.json"), md_path=str(tmp_path / "m.md"))

    mtimes_after = {f: f.stat().st_mtime for f in models_dir.rglob("*") if f.is_file()}
    assert mtimes_before == mtimes_after
