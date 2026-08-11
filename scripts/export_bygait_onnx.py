"""
Export PyTorch ByGaitLight model checkpoint to ONNX format and verify numerical parity.

Generates outputs/reports/onnx_validation.json and outputs/reports/onnx_validation.md.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Union


import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.architectures.bygait_light import ByGaitLight


def _to_rel_path(path: Union[str, Path]) -> str:
    """Format path relative to repository ROOT using forward slashes."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def export_onnx(
    model_path: str = "runs/exp_001/best_model.pth",
    output_onnx_path: str = "models/engines/bygait_light.onnx",
    precision: str = "fp32",
    rtol: float = 1e-3,
    atol: float = 1e-4,
    report_json_path: str = "outputs/reports/onnx_validation.json",
    report_md_path: str = "outputs/reports/onnx_validation.md",
) -> bool:
    """Export PyTorch checkpoint to ONNX atomically and verify numerical output parity."""
    torch.manual_seed(42)
    np.random.seed(42)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    model_file = Path(model_path)
    output_file = Path(output_onnx_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_output_file = output_file.with_name(f"{output_file.name}.tmp")

    report_status = {
        "checkpoint_path": _to_rel_path(model_file),
        "checkpoint_exists": model_file.exists(),
        "output_onnx_path": _to_rel_path(output_file),
        "export_succeeded": False,
        "onnx_model_valid": False,
        "onnx_runtime_available": False,
        "numerical_parity_passed": False,
        "numerical_parity_failed": False,
        "unable_to_verify": False,
        "input_shape": [1, 1, 128, 64],
        "output_shape": [1, 256],
        "max_absolute_diff": None,
        "rtol": rtol,
        "atol": atol,
        "error_message": None,
    }

    if not model_file.exists():
        err = f"Model checkpoint file not found: {_to_rel_path(model_file)}"
        print(f"[ERROR] {err}")
        report_status["error_message"] = err
        _write_reports(report_status, report_json_path, report_md_path)
        return False

    print("[INFO] Initializing ByGaitLight model...")
    model = ByGaitLight()
    try:
        checkpoint = torch.load(model_file, map_location="cpu")
        filtered = {}
        for key, value in checkpoint.items():
            if key.startswith("backbone."):
                filtered[key.replace("backbone.", "")] = value
            elif key in model.state_dict():
                filtered[key] = value
        model.load_state_dict(filtered, strict=False)
        print(f"[INFO] Loaded checkpoint weights from {_to_rel_path(model_file)}")
    except Exception as e:
        err = f"Could not load model checkpoint: {e}"
        print(f"[ERROR] {err}")
        report_status["error_message"] = err
        _write_reports(report_status, report_json_path, report_md_path)
        return False

    model.eval()


    dummy_input = torch.randn(1, 1, 128, 64, dtype=torch.float32)

    print(f"[INFO] Exporting model atomically to ONNX: {output_file}")
    try:
        import warnings

        batch_dim = torch.export.Dim("batch_size", min=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            torch.onnx.export(
                model,
                (dummy_input,),
                str(temp_output_file),
                export_params=True,
                dynamic_shapes=({0: batch_dim},),
                dynamo=True,
            )
        report_status["export_succeeded"] = True
        print("[INFO] ONNX export completed successfully.")

    except Exception as e:
        report_status["export_succeeded"] = False
        report_status["error_message"] = f"Export failed: {e}"
        print(f"[WARNING] ONNX export failed: {e}")
        if temp_output_file.exists():
            temp_output_file.unlink()
        _write_reports(report_status, report_json_path, report_md_path)
        return False

    # Structural ONNX validation
    try:
        import onnx

        onnx_model = onnx.load(str(temp_output_file))
        onnx.checker.check_model(onnx_model)
        report_status["onnx_model_valid"] = True
        print("[INFO] ONNX model structural check PASSED.")
    except Exception as e:
        report_status["onnx_model_valid"] = False
        report_status["error_message"] = f"Structural validation failed: {e}"
        print(f"[WARNING] ONNX structural check failed: {e}")
        if temp_output_file.exists():
            temp_output_file.unlink()
        _write_reports(report_status, report_json_path, report_md_path)
        return False

    # Validate parity if onnxruntime is available
    parity_passed = False
    try:
        import onnxruntime as ort

        report_status["onnx_runtime_available"] = True
        session = ort.InferenceSession(str(temp_output_file), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()

        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

        if pytorch_output.shape != (1, 256) or onnx_output.shape != (1, 256):
            raise ValueError(f"Output shape mismatch. PyTorch: {pytorch_output.shape}, ONNX: {onnx_output.shape}")

        max_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
        report_status["max_absolute_diff"] = max_diff

        parity_passed = bool(np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol))
        if parity_passed:
            report_status["numerical_parity_passed"] = True
            report_status["numerical_parity_failed"] = False
            print(f"[SUCCESS] Numerical parity check PASSED (max absolute diff: {max_diff:.6f}).")
        else:
            report_status["numerical_parity_passed"] = False
            report_status["numerical_parity_failed"] = True
            print(f"[WARNING] Parity check failed! Max diff: {max_diff:.6f}")

    except ImportError:
        report_status["onnx_runtime_available"] = False
        report_status["unable_to_verify"] = True
        report_status["error_message"] = "onnxruntime not installed"
        print("[INFO] onnxruntime not installed. Skipping numerical parity validation.")
        parity_passed = True
    except Exception as e:
        report_status["numerical_parity_passed"] = False
        report_status["numerical_parity_failed"] = True
        report_status["error_message"] = f"Inference/parity check failed: {e}"
        print(f"[WARNING] ONNXRuntime validation failed: {e}")
        parity_passed = False

    if report_status["export_succeeded"] and report_status["onnx_model_valid"] and parity_passed:
        temp_output_file.replace(output_file)
        print(f"[INFO] Successfully replaced target ONNX file: {output_file}")
    else:
        if temp_output_file.exists():
            temp_output_file.unlink()
        print(f"[WARNING] Export or validation failed. Preserved target file: {output_file}")

    _write_reports(report_status, report_json_path, report_md_path)
    return parity_passed


def _write_reports(status: dict, json_path: str, md_path: str) -> None:
    """Write JSON and Markdown ONNX validation reports."""
    j_file = Path(json_path)
    j_file.parent.mkdir(parents=True, exist_ok=True)

    with open(j_file, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=4)

    m_file = Path(md_path)
    m_file.parent.mkdir(parents=True, exist_ok=True)

    md_content = f"""# ONNX Export & Validation Report

## Executive Summary

- **Export Succeeded**: {"PASS" if status["export_succeeded"] else "FAIL"}
- **ONNX Model Valid**: {"PASS" if status["onnx_model_valid"] else "FAIL"}
- **ONNX Runtime Available**: {"YES" if status["onnx_runtime_available"] else "NO"}
- **Numerical Parity Passed**: {"PASS" if status["numerical_parity_passed"] else ("FAIL" if status["numerical_parity_failed"] else "N/A")}
- **Unable to Verify**: {"YES" if status["unable_to_verify"] else "NO"}

## Verification Metrics

| Metric | Value |
| :--- | :--- |
| Checkpoint Path | `{status["checkpoint_path"]}` |
| Output ONNX Path | `{status["output_onnx_path"]}` |
| Input Shape | `{status["input_shape"]}` |
| Output Shape | `{status["output_shape"]}` |
| Max Absolute Diff | `{status["max_absolute_diff"]}` |
| Relative Tolerance (`rtol`) | `{status["rtol"]}` |
| Absolute Tolerance (`atol`) | `{status["atol"]}` |
| Error Message | `{status["error_message"] or "None"}` |
"""
    with open(m_file, "w", encoding="utf-8") as f:
        f.write(md_content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ARGUS ByGaitLight PyTorch model to ONNX.")
    parser.add_argument("--model-path", type=str, default="runs/exp_001/best_model.pth", help="Path to PyTorch checkpoint")
    parser.add_argument("--output-path", type=str, default="models/engines/bygait_light.onnx", help="Output ONNX file path")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"], help="Model precision")

    args = parser.parse_args()
    success = export_onnx(
        model_path=args.model_path,
        output_onnx_path=args.output_path,
        precision=args.precision,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
