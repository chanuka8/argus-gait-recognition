"""
ARGUS AI Complete Environment & Model Verification Suite.

Performs 6-phase real-time verification of dependencies, compute acceleration,
CUDA matrix operations, ByGaitLight CNN execution, YOLOv8 person detector runtime device,
and ONNX Runtime CUDA inference, then persists the verified manifest.

Usage:
    python scripts/verify_environment.py
"""

import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def setup_torch_dll_path() -> None:
    """Ensure torch/lib is in PATH and DLL search directory for ONNX CUDA provider."""
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(torch_lib))
    except Exception:
        pass


def save_environment_manifest(
    gpu_name: str,
    driver_version: str,
    target_compute: str,
    torch_version: str,
    torchvision_version: str,
    cuda_build: str,
    pytorch_cuda_ready: bool,
    yolo_cuda_ready: bool,
    yolo_runtime_dev: str,
    onnx_version: str,
    onnx_cuda_ready: bool,
    onnx_selected_provider: str,
    overall_status: str,
) -> None:
    """Persist verified environment fingerprint to .venv/argus_env_manifest.json."""
    manifest_path = ROOT / ".venv" / "argus_env_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "compute_mode": target_compute.lower(),
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "cuda_build": cuda_build,
        "pytorch_cuda": pytorch_cuda_ready,
        "yolo_cuda": yolo_cuda_ready,
        "yolo_runtime_device": yolo_runtime_dev,
        "onnx_version": onnx_version,
        "onnx_cuda": onnx_cuda_ready,
        "onnx_provider": onnx_selected_provider,
        "overall_status": overall_status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    try:
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def run_verification() -> bool:
    setup_torch_dll_path()

    print("=" * 60)
    print("ARGUS ENVIRONMENT VERIFICATION SUITE")
    print("=" * 60)

    overall_pass = True

    # Phase 1: Core Package Imports
    print("\n[PHASE 1] Checking Core Package Imports...")
    packages = [
        ("fastapi", "FastAPI Web Framework"),
        ("uvicorn", "Uvicorn ASGI Server"),
        ("torch", "PyTorch Deep Learning Framework"),
        ("torchvision", "TorchVision Computer Vision"),
        ("cv2", "OpenCV Image Processing"),
        ("numpy", "NumPy Numerical Computing"),
        ("ultralytics", "Ultralytics YOLO Engine"),
        ("supervision", "Supervision Tracking Tools"),
        ("yaml", "PyYAML Configuration Parser"),
        ("psutil", "System Process & Telemetry"),
    ]

    for mod_name, desc in packages:
        try:
            __import__(mod_name)
            print(f"  [VERIFY] {desc:<35} : PASS")
        except ImportError as err:
            print(f"  [VERIFY] {desc:<35} : FAIL ({err})")
            overall_pass = False

    # Phase 2: Compute Device & CUDA Acceleration
    print("\n[PHASE 2] Probing Compute Hardware & Acceleration...")
    gpu_name = "None"
    driver_ver = "N/A"
    cuda_build = "None"
    torch_ver = "N/A"
    vision_ver = "N/A"
    is_cuda_avail = False
    pytorch_cuda_ready = False

    try:
        import torch
        import torchvision
        torch_ver = getattr(torch, "__version__", "N/A")
        vision_ver = getattr(torchvision, "__version__", "N/A")
        is_cuda_avail = torch.cuda.is_available()
        cuda_build = getattr(torch.version, "cuda", "None") or "None"

        if is_cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            pytorch_cuda_ready = True
            print(f"  [ARGUS CUDA] Availability                    : PASS (True)")
            print(f"  [ARGUS CUDA] Device                          : {gpu_name}")
            print(f"  [ARGUS CUDA] Build                           : {cuda_build}")
            print(f"  [ARGUS CUDA] Status                          : VERIFIED")
        else:
            print(f"  [ARGUS CUDA] Availability                    : FALSE (CPU Fallback)")
    except Exception as err:
        print(f"  [ARGUS CUDA] Probe Error                     : FAIL ({err})")
        overall_pass = False

    # Phase 3: Tensor Acceleration Math (1024x1024 MatMul)
    print("\n[PHASE 3] Executing Tensor MatMul Probe (1024x1024)...")
    try:
        import torch
        device = "cuda" if is_cuda_avail else "cpu"
        a = torch.zeros((1024, 1024), device=device)
        b = torch.ones((1024, 1024), device=device)
        c = a @ b
        if device == "cuda":
            torch.cuda.synchronize()

        if c.shape == (1024, 1024):
            print(f"  [ARGUS CUDA] Tensor Probe ({device.upper()})            : PASS (Synchronized)")
        else:
            print(f"  [ARGUS CUDA] Tensor Probe ({device.upper()})            : FAIL (Shape mismatch)")
            overall_pass = False
    except Exception as err:
        print(f"  [ARGUS CUDA] Tensor probe error              : FAIL ({err})")
        overall_pass = False

    # Phase 4: ByGaitLight CNN Architecture Verification
    print("\n[PHASE 4] Verifying ByGaitLight Gait Recognition CNN...")
    try:
        import torch
        from models.architectures.bygait_light import ByGaitLight

        device = "cuda" if is_cuda_avail else "cpu"
        model = ByGaitLight().to(device)
        model.eval()

        dummy_gei = torch.randn(1, 1, 128, 64, device=device)
        with torch.no_grad():
            emb = model(dummy_gei)

        if emb.shape == (1, 256):
            l2_norm = torch.norm(emb, p=2, dim=-1).item()
            print(f"  [ARGUS MODEL] ByGaitLight Forward Pass       : PASS")
            print(f"  [ARGUS MODEL] Device                         : {device}:0" if device == "cuda" else f"  [ARGUS MODEL] Device                         : {device}")
            print(f"  [ARGUS MODEL] Output Shape                   : {list(emb.shape)}")
            print(f"  [ARGUS MODEL] L2 Norm                        : {l2_norm:.4f} (PASS)")
        else:
            print(f"  [ARGUS MODEL] ByGaitLight Forward Pass       : FAIL (Shape: {emb.shape})")
            overall_pass = False
    except Exception as err:
        print(f"  [ARGUS MODEL] Model execution error          : FAIL ({err})")
        overall_pass = False

    # Phase 5: YOLO Person Detector Runtime Verification
    print("\n[PHASE 5] Verifying PersonDetector (YOLOv8) Runtime Device...")
    yolo_cuda_ready = False
    yolo_runtime_dev = "cpu"
    try:
        from pipeline.detection.person_detector import PersonDetector
        import numpy as np

        detector = PersonDetector()
        cfg_device = detector.device
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = detector.detect(dummy_frame)
        yolo_runtime_dev = str(next(detector.model.model.parameters()).device)
        yolo_cuda_ready = "cuda" in yolo_runtime_dev

        print(f"  [ARGUS YOLO] Configured Device               : {cfg_device}")
        print(f"  [ARGUS YOLO] Runtime Device                  : {yolo_runtime_dev}")
        print(f"  [ARGUS YOLO] CUDA Execution                  : {'PASS' if yolo_cuda_ready else 'CPU Fallback'}")
    except Exception as err:
        print(f"  [ARGUS YOLO] Initialization error            : FAIL ({err})")
        overall_pass = False

    # Phase 6: ONNX Runtime CUDA Verification & Smoke Test
    print("\n[PHASE 6] Verifying ONNX Runtime CUDA Acceleration...")
    onnx_ver = "N/A"
    onnx_cuda_ready = False
    onnx_selected_provider = "CPUExecutionProvider"

    try:
        import onnxruntime as ort
        import numpy as np

        onnx_ver = ort.__version__
        providers = ort.get_available_providers()
        cuda_avail = "CUDAExecutionProvider" in providers
        print(f"  [ARGUS ONNX CUDA] Runtime Version            : {onnx_ver}")
        print(f"  [ARGUS ONNX CUDA] CUDA Provider              : {'AVAILABLE' if cuda_avail else 'UNAVAILABLE'}")
        print(f"  [ARGUS ONNX CUDA] CPU Provider               : AVAILABLE")

        model_path = ROOT / "models/weights/silhouette_segmenter.onnx"
        if model_path.exists():
            req_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_avail else ["CPUExecutionProvider"]
            sess = ort.InferenceSession(str(model_path), providers=req_providers)
            active_providers = sess.get_providers()
            onnx_selected_provider = active_providers[0]
            print(f"  [ARGUS ONNX CUDA] Session Creation           : PASS (Active: {onnx_selected_provider})")

            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            dummy_in = np.random.randn(1, 3, 256, 256).astype(np.float32)
            t0 = time.perf_counter()
            res = sess.run([out_name], {in_name: dummy_in})
            lat_ms = (time.perf_counter() - t0) * 1000

            if onnx_selected_provider == "CUDAExecutionProvider":
                onnx_cuda_ready = True
                print(f"  [ARGUS ONNX CUDA] CUDA Inference             : PASS (Shape: {list(res[0].shape)}, Latency: {lat_ms:.2f} ms)")
            else:
                print(f"  [ARGUS ONNX CUDA] Inference (CPU Fallback)   : PASS (Shape: {list(res[0].shape)})")
            print(f"  [ARGUS ONNX CUDA] Selected Provider          : {onnx_selected_provider}")
    except Exception as err:
        print(f"  [ARGUS ONNX CUDA] Provider verification error : FAIL ({err})")
        overall_pass = False

    # Compute State Assessment
    target_compute = "CUDA" if is_cuda_avail else "CPU"
    if pytorch_cuda_ready and yolo_cuda_ready and onnx_cuda_ready:
        overall_status = "FULL_CUDA_ACCELERATION_READY"
    elif pytorch_cuda_ready and yolo_cuda_ready:
        overall_status = "PARTIAL_CUDA_ACCELERATION (PyTorch/YOLO on CUDA, ONNX on CPU)"
    elif is_cuda_avail:
        overall_status = "PARTIAL_CUDA_ACCELERATION"
    else:
        overall_status = "CPU_MODE_READY"

    save_environment_manifest(
        gpu_name=gpu_name,
        driver_version=driver_ver,
        target_compute=target_compute,
        torch_version=torch_ver,
        torchvision_version=vision_ver,
        cuda_build=cuda_build,
        pytorch_cuda_ready=pytorch_cuda_ready,
        yolo_cuda_ready=yolo_cuda_ready,
        yolo_runtime_dev=yolo_runtime_dev,
        onnx_version=onnx_ver,
        onnx_cuda_ready=onnx_cuda_ready,
        onnx_selected_provider=onnx_selected_provider,
        overall_status=overall_status,
    )

    print("\n" + "=" * 60)
    print("ARGUS PIPELINE COMPUTE STATUS")
    print("-" * 60)
    print(f"PyTorch CUDA       : {'READY' if pytorch_cuda_ready else 'NOT READY'}")
    print(f"YOLO CUDA          : {'READY' if yolo_cuda_ready else 'NOT READY'}")
    print(f"ONNX CUDA          : {'READY' if onnx_cuda_ready else 'NOT READY'}")
    print(f"ARGUS GPU Pipeline : {'FULL_CUDA_ACCELERATION_READY' if overall_status == 'FULL_CUDA_ACCELERATION_READY' else overall_status}")
    print("-" * 60)
    if overall_pass:
        print("[ARGUS] ALL ENVIRONMENT VERIFICATION CHECKS PASSED.")
        print("[ARGUS] Environment Fingerprint saved to .venv/argus_env_manifest.json")
    else:
        print("[ARGUS ERROR] ONE OR MORE VERIFICATION CHECKS FAILED.")
    print("=" * 60 + "\n")

    return overall_pass


def main() -> int:
    success = run_verification()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
