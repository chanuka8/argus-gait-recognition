import sys
from pathlib import Path

# Fix Windows console UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import torch

from models.architectures.silhouette_unet import SilhouetteUNet


def export_and_validate_onnx(
    pth_path: str = "models/weights/silhouette_segmenter.pth",
    output_onnx_path: str = "models/weights/silhouette_segmenter.onnx",
    engine_onnx_path: str = "models/engines/silhouette_segmenter.onnx",
) -> tuple[bool, str]:
    pth_file = Path(pth_path)
    onnx_file = Path(output_onnx_path)
    engine_file = Path(engine_onnx_path)

    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    engine_file.parent.mkdir(parents=True, exist_ok=True)

    model = SilhouetteUNet()
    if pth_file.exists():
        state_dict = torch.load(pth_file, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[*] Loaded trained weights from: {pth_file}")
    else:
        print(f"[!] PyTorch weight file {pth_file} not found; exporting initialized model structure.")

    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )
    except TypeError:
        # If dynamo param is not supported in this torch version
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    print(f"[SUCCESS] ONNX model exported to: {onnx_file}")

    # Copy to engines path
    with open(onnx_file, "rb") as f_in, open(engine_file, "wb") as f_out:
        f_out.write(f_in.read())
    print(f"[SUCCESS] Mirrored ONNX asset to: {engine_file}")

    # Validate ONNX Runtime inference vs PyTorch
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    np_input = dummy_input.numpy()
    ort_outs = session.run([out_name], {in_name: np_input})[0]

    with torch.no_grad():
        pytorch_outs = model(dummy_input).numpy()

    if ort_outs.shape != (1, 1, 256, 256):
        return False, f"ONNX output shape mismatch: expected (1, 1, 256, 256), got {ort_outs.shape}"

    if not np.all(np.isfinite(ort_outs)):
        return False, "ONNX output contains non-finite values (NaN/Inf)"

    max_diff = float(np.max(np.abs(ort_outs - pytorch_outs)))
    if max_diff > 1e-4:
        return False, f"PyTorch vs ONNX max diff too high: {max_diff}"

    return True, f"ONNX export and validation successful! Max diff: {max_diff:.6f}"


if __name__ == "__main__":
    success, msg = export_and_validate_onnx()
    print(f"ONNX Export Result: {msg}")
    sys.exit(0 if success else 1)
