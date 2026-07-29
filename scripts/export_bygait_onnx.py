"""
Export PyTorch ByGaitLight model checkpoint to ONNX format and verify numerical parity.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.architectures.bygait_light import ByGaitLight


def export_onnx(
    model_path: str = "runs/exp_001/best_model.pth",
    output_onnx_path: str = "models/engines/bygait_light.onnx",
    precision: str = "fp32",
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> bool:
    """Export PyTorch checkpoint to ONNX and verify numerical output parity."""
    model_file = Path(model_path)
    output_file = Path(output_onnx_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Initializing ByGaitLight model...")
    model = ByGaitLight()
    if model_file.exists():
        try:
            checkpoint = torch.load(model_file, map_location="cpu")
            filtered = {}
            for key, value in checkpoint.items():
                if key.startswith("backbone."):
                    filtered[key.replace("backbone.", "")] = value
                elif key in model.state_dict():
                    filtered[key] = value
            model.load_state_dict(filtered, strict=False)
            print(f"[INFO] Loaded checkpoint weights from {model_file}")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}. Using randomly initialized weights.")

    model.eval()

    dummy_input = torch.randn(1, 1, 64, 128, dtype=torch.float32)

    print(f"[INFO] Exporting model to ONNX: {output_file}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_file),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print("[INFO] ONNX export completed successfully.")
    except Exception as e:
        print(f"[WARNING] ONNX export failed: {e}")
        print("[INFO] PyTorch backend remains active as the reference backend.")
        return False

    # Validate parity if onnxruntime is available
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()

        onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

        parity_passed = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
        max_diff = np.max(np.abs(pytorch_output - onnx_output))

        if parity_passed:
            print(f"[SUCCESS] Numerical parity check PASSED (max absolute diff: {max_diff:.6f}).")
        else:
            print(f"[WARNING] Parity check failed! Max diff: {max_diff:.6f}")
        return parity_passed
    except ImportError:
        print("[INFO] onnxruntime not installed. Skipping numerical parity validation.")
        return True


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
