import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_engine_python(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",
) -> bool:
    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"[ERROR] ONNX Parser: {parser.get_error(error)}")
                return False

        config = builder.create_builder_config()
        if precision == "fp16" and builder.platform_has_tf32:
            config.set_flag(trt.BuilderFlag.FP16)

        print(f"[INFO] Building TensorRT engine for precision={precision}...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[ERROR] Failed to build TensorRT serialized engine.")
            return False

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"[SUCCESS] TensorRT engine saved to {engine_path}")
        return True
    except ImportError:
        print("[WARNING] tensorrt Python package not installed.")
        return False
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        print(f"[ERROR] TensorRT build failed: {e}")
        return False


def build_engine_trtexec(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",
) -> bool:
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    if precision == "fp16":
        cmd.append("--fp16")

    print(f"[INFO] Executing command: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            print("[SUCCESS] trtexec engine build complete.")
            return True
        print(f"[ERROR] trtexec failed:\n{res.stderr}")
        return False
    except FileNotFoundError:
        print("[WARNING] trtexec command not found in PATH.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model.")
    parser.add_argument("--onnx-path", type=str, default="models/engines/bygait_light.onnx", help="Path to ONNX file")
    parser.add_argument(
        "--engine-path", type=str, default="models/engines/bygait_light_fp16.engine", help="Output engine path"
    )
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16"], help="Precision mode")

    args = parser.parse_args()
    onnx_file = Path(args.onnx_path)
    engine_file = Path(args.engine_path)

    if not onnx_file.exists():
        print(f"[ERROR] Input ONNX file does not exist: {onnx_file}")
        print("[HINT] Run 'python scripts/export_bygait_onnx.py' first.")
        sys.exit(1)

    success = build_engine_python(onnx_file, engine_file, precision=args.precision)
    if not success:
        success = build_engine_trtexec(onnx_file, engine_file, precision=args.precision)

    if not success:
        print("\n[CONCLUSION] TensorRT tools are not available on this environment.")
        print("System will safely fall back to PyTorch backend at runtime when allow_fallback=True.")
        sys.exit(1)


if __name__ == "__main__":
    main()
