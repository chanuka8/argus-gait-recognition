import hashlib
import json
from pathlib import Path

import pytest

from models.export.bygait_onnx import export_onnx


def test_missing_checkpoint_refuses_export(tmp_path: Path):
    target_onnx = tmp_path / "bygait_light.onnx"
    json_report = tmp_path / "onnx_validation.json"
    md_report = tmp_path / "onnx_validation.md"

    success = export_onnx(
        model_path="non_existent_checkpoint_file.pth",
        output_onnx_path=str(target_onnx),
        report_json_path=str(json_report),
        report_md_path=str(md_report),
    )

    assert success is False
    assert not target_onnx.exists()
    assert json_report.exists()

    with open(json_report, encoding="utf-8") as f:
        data = json.load(f)

    assert data["checkpoint_exists"] is False
    assert data["export_succeeded"] is False
    assert "not found" in data["error_message"]


def test_onnx_export_atomic_replacement_and_reports(tmp_path: Path):
    ckpt_file = tmp_path / "model.pth"
    import torch

    from models.architectures.bygait_light import ByGaitLight

    torch.save(ByGaitLight().state_dict(), ckpt_file)

    target_onnx = tmp_path / "bygait_light.onnx"
    json_report = tmp_path / "onnx_validation.json"
    md_report = tmp_path / "onnx_validation.md"

    import importlib.util

    has_onnx = (importlib.util.find_spec("onnx") is not None) and (importlib.util.find_spec("onnxruntime") is not None)

    success = export_onnx(
        model_path=str(ckpt_file),
        output_onnx_path=str(target_onnx),
        report_json_path=str(json_report),
        report_md_path=str(md_report),
    )

    assert json_report.exists()
    assert md_report.exists()

    with open(json_report, encoding="utf-8") as f:
        data = json.load(f)

    assert data["checkpoint_exists"] is True

    if has_onnx:
        assert success is True
        assert target_onnx.exists()
        assert data["export_succeeded"] is True
        assert data["onnx_model_valid"] is True
        assert data["numerical_parity_passed"] is True
        assert data["input_shape"] == [1, 1, 128, 64]
        assert data["output_shape"] == [1, 256]
    else:
        assert success is False
        assert data["export_succeeded"] is False

    tmp_file = target_onnx.with_name(f"{target_onnx.name}.tmp")
    assert not tmp_file.exists()


def test_onnx_export_failure_preserves_sha256_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ckpt_file = tmp_path / "model.pth"
    import torch

    from models.architectures.bygait_light import ByGaitLight

    torch.save(ByGaitLight().state_dict(), ckpt_file)

    target_onnx = tmp_path / "existing.onnx"
    original_bytes = b"ORIGINAL_SAFE_ONNX_BYTES_PRESERVED"
    target_onnx.write_bytes(original_bytes)
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()

    json_report = tmp_path / "onnx_validation.json"
    md_report = tmp_path / "onnx_validation.md"

    def fake_export(*args, **kwargs):
        raise RuntimeError("Simulated export failure")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    success = export_onnx(
        model_path=str(ckpt_file),
        output_onnx_path=str(target_onnx),
        report_json_path=str(json_report),
        report_md_path=str(md_report),
    )

    assert success is False
    current_bytes = target_onnx.read_bytes()
    current_sha256 = hashlib.sha256(current_bytes).hexdigest()

    assert current_sha256 == original_sha256
    assert current_bytes == original_bytes
    assert json_report.exists()

    tmp_file = target_onnx.with_name(f"{target_onnx.name}.tmp")
    assert not tmp_file.exists()
