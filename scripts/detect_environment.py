"""
ARGUS AI Hardware & Compute Environment Detector CLI.

Scans host hardware (CPU, RAM, NVIDIA GPU, driver, VRAM, CUDA) and compares it
against the installed PyTorch/CUDA runtime, YOLOv8 runtime device, and ONNX
execution providers in .venv to determine if the environment is healthy or requires repair.

Usage:
    python scripts/detect_environment.py [--json]
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    except (ImportError, AttributeError, OSError):
        pass


def get_system_hardware() -> dict[str, Any]:
    """Retrieve OS, CPU core count, and total RAM in GB."""
    os_name = f"{platform.system()} {platform.release()}"
    cpu_count = os.cpu_count() or 1
    ram_gb = 0.0

    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
    except (ImportError, AttributeError, OSError):
        pass

    return {
        "os": os_name,
        "python_version": sys.version.split()[0],
        "cpu_cores": cpu_count,
        "ram_gb": round(ram_gb, 1),
    }


def get_nvidia_smi_info() -> tuple[bool, str | None, str | None, float | None, str | None]:
    """
    Query nvidia-smi for GPU presence, name, driver version, VRAM (MB), and max supported CUDA.
    """
    smi_path = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    if not os.path.exists(smi_path) and not shutil.which("nvidia-smi"):
        return False, None, None, None, None

    try:
        cmd = [
            smi_path,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                driver = parts[1]
                vram_mb = float(parts[2])

                cuda_driver = "12.x"
                try:
                    smi_banner = subprocess.run([smi_path], capture_output=True, text=True, timeout=5, check=False)
                    if "CUDA Version:" in smi_banner.stdout:
                        cuda_driver = smi_banner.stdout.split("CUDA Version:")[1].split()[0]
                except (subprocess.SubprocessError, OSError, ValueError):
                    pass

                return True, name, driver, vram_mb, cuda_driver
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    return False, None, None, None, None


def get_onnx_providers() -> tuple[list[str], str, bool]:
    """Inspect available ONNX Runtime execution providers."""
    setup_torch_dll_path()
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        cuda_avail = "CUDAExecutionProvider" in providers
        selected = "CUDAExecutionProvider" if cuda_avail else "CPUExecutionProvider"
        return providers, selected, cuda_avail
    except (ImportError, RuntimeError, ValueError, TypeError, OSError):
        return [], "CPUExecutionProvider", False


def probe_pytorch_cuda() -> dict[str, Any]:
    """Inspect active PyTorch build in the current environment."""
    info = {
        "installed": False,
        "version": None,
        "cuda_in_build": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "tensor_probe_passed": False,
        "error": None,
    }

    try:
        import torch

        info["installed"] = True
        info["version"] = getattr(torch, "__version__", None)
        info["cuda_in_build"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = torch.cuda.is_available()
        info["device_count"] = torch.cuda.device_count()

        if info["cuda_available"] and info["device_count"] > 0:
            info["device_name"] = torch.cuda.get_device_name(0)
            try:
                a = torch.zeros((128, 128), device="cuda")
                b = torch.ones((128, 128), device="cuda")
                c = a @ b
                torch.cuda.synchronize()
                if c.shape == (128, 128):
                    info["tensor_probe_passed"] = True
            except (RuntimeError, ValueError, TypeError, OSError) as probe_err:
                info["error"] = f"Tensor probe failed: {probe_err}"
    except ImportError as imp_err:
        info["error"] = f"PyTorch not imported: {imp_err}"
    except (RuntimeError, ValueError, TypeError, OSError) as general_err:
        info["error"] = str(general_err)

    return info


def probe_yolo_runtime_device() -> tuple[str, str, bool]:
    """Inspect YOLOv8 configured and resolved runtime execution device."""
    try:
        from pipeline.detection.person_detector import PersonDetector

        detector = PersonDetector()
        cfg_dev = detector.device
        runtime_dev = detector.runtime_device
        is_cuda = "cuda" in runtime_dev
        return cfg_dev, runtime_dev, is_cuda
    except (ImportError, RuntimeError, ValueError, TypeError, OSError):
        return "auto", "cpu", False


def detect_environment() -> dict[str, Any]:
    """Aggregate hardware state, PyTorch, YOLO, and ONNX compute requirements."""
    sys_hw = get_system_hardware()
    has_gpu, gpu_name, driver_ver, vram_mb, cuda_driver = get_nvidia_smi_info()
    torch_info = probe_pytorch_cuda()
    onnx_providers, onnx_selected, onnx_cuda_avail = get_onnx_providers()
    yolo_cfg_dev, yolo_runtime_dev, yolo_cuda_avail = probe_yolo_runtime_device()

    cuda_capable_hardware = has_gpu and (driver_ver is not None)
    is_healthy = False
    action_required = "NONE"
    target_compute = "CPU"

    pytorch_cuda_ready = torch_info["installed"] and torch_info["cuda_available"] and torch_info["tensor_probe_passed"]
    yolo_cuda_ready = yolo_cuda_avail
    onnx_cuda_ready = onnx_cuda_avail

    if cuda_capable_hardware:
        target_compute = "CUDA"
        if pytorch_cuda_ready and onnx_cuda_ready:
            is_healthy = True
            action_required = "NONE"
            pipeline_status = "FULL_CUDA_ACCELERATION_READY"
        elif pytorch_cuda_ready and not onnx_cuda_ready:
            is_healthy = True
            action_required = "INSTALL_ONNX_GPU"
            pipeline_status = "PARTIAL_CUDA_ACCELERATION (PyTorch/YOLO on CUDA, ONNX on CPU)"
        else:
            is_healthy = False
            action_required = "INSTALL_CUDA_PYTORCH"
            pipeline_status = "REPAIR_REQUIRED"
    else:
        target_compute = "CPU"
        if torch_info["installed"]:
            is_healthy = True
            action_required = "NONE"
            pipeline_status = "CPU_MODE_READY"
        else:
            is_healthy = False
            action_required = "INSTALL_CPU_PYTORCH"
            pipeline_status = "REPAIR_REQUIRED"

    return {
        "python_executable": sys.executable,
        "system": sys_hw,
        "hardware": {
            "has_nvidia_gpu": has_gpu,
            "gpu_name": gpu_name,
            "driver_version": driver_ver,
            "cuda_driver_version": cuda_driver,
            "vram_mb": vram_mb,
            "cuda_capable": cuda_capable_hardware,
        },
        "pytorch": torch_info,
        "yolo": {
            "configured_device": yolo_cfg_dev,
            "runtime_device": yolo_runtime_dev,
            "cuda_execution": yolo_cuda_ready,
        },
        "onnx": {
            "available_providers": onnx_providers,
            "selected_provider": onnx_selected,
            "cuda_available": onnx_cuda_avail,
        },
        "assessment": {
            "target_compute": target_compute,
            "is_healthy": is_healthy,
            "pipeline_status": pipeline_status,
            "action_required": action_required,
            "download_required": not is_healthy,
            "installation_required": not is_healthy,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS Hardware & Environment Detector")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    env_data = detect_environment()

    if args.json:
        print(json.dumps(env_data, indent=2))
        return 0 if env_data["assessment"]["is_healthy"] else 1

    sys_info = env_data["system"]
    hw = env_data["hardware"]
    pt = env_data["pytorch"]
    yolo = env_data["yolo"]
    onnx = env_data["onnx"]
    asm = env_data["assessment"]

    print("=" * 60)
    print("ARGUS ENVIRONMENT DETECTOR")
    print("=" * 60)
    print(f"OS              : {sys_info['os']}")
    print(f"Python          : {sys_info['python_version']}")
    print(f"CPU             : {sys_info['cpu_cores']} Cores")
    print(f"RAM             : {sys_info['ram_gb']} GB")

    if hw["has_nvidia_gpu"]:
        print(f"GPU             : {hw['gpu_name']}")
        print(f"VRAM            : {hw['vram_mb']:.0f} MB")
        print(f"Driver          : {hw['driver_version']}")
        print(f"CUDA Driver     : {hw['cuda_driver_version'] or '12.x'}")
    else:
        print("GPU             : None detected (CPU Mode)")

    if pt["installed"]:
        print(f"PyTorch         : {pt['version']}")
        print(f"PyTorch CUDA    : {pt['cuda_in_build'] or 'None (CPU build)'}")
        print(f"CUDA Available  : {'TRUE' if pt['cuda_available'] else 'FALSE'}")
        print(f"CUDA Probe      : {'PASS' if pt['tensor_probe_passed'] else 'FAIL'}")
    else:
        print(f"PyTorch Status  : NOT INSTALLED ({pt['error']})")

    print(f"YOLO Runtime    : {yolo['runtime_device']} (Config: {yolo['configured_device']})")
    print(
        f"ONNX Provider   : {onnx['selected_provider']} (CUDA: {'AVAILABLE' if onnx['cuda_available'] else 'UNAVAILABLE'})"
    )
    print("-" * 60)
    print(f"TARGET COMPUTE  : {asm['target_compute']}")
    print(f"PIPELINE STATUS : {asm['pipeline_status']}")
    print(f"ENVIRONMENT     : {'HEALTHY' if asm['is_healthy'] else 'REPAIR NEEDED'}")
    print("=" * 60)

    return 0 if asm["is_healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
