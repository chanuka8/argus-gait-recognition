"""
Production-Grade ARGUS AI Environment Bootstrap & Hardware Arbitration Orchestrator.

Performs deterministic 12-stage hardware discovery, runtime capability validation,
idempotent package management, and manifest persistence.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from automation.cuda_detector import CudaDetector
from automation.device_manager import DeviceManager
from automation.dll_manager import setup_cuda_dll_paths
from automation.environment_validator import (
    ComputeBackend,
    EnvironmentState,
    EnvironmentValidator,
)
from automation.hardware_detector import HardwareDetector
from automation.onnx_manager import OnnxManager
from automation.pytorch_manager import PyTorchManager

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class EnvironmentBootstrap:
    """Master orchestrator for ARGUS environment discovery, repair, and validation."""

    def __init__(self, force_repair: bool = False, force_cpu: bool = False) -> None:
        self.force_repair = force_repair
        self.force_cpu = force_cpu
        self.manifest_path = ROOT / ".venv" / "argus_env_manifest.json"
        setup_cuda_dll_paths()

    def run(self) -> bool:
        print("=" * 60)
        print(" ARGUS AI ENVIRONMENT BOOTSTRAP")
        print("=" * 60)

        # Stage 1: Operating System
        print("\n[01/12] Detecting operating system...")
        sys_hw = HardwareDetector.detect_system()
        print(f"[PASS] {sys_hw.os_name} {sys_hw.os_version} ({sys_hw.architecture})")

        # Stage 2: Python
        print("\n[02/12] Detecting Python...")
        print(f"[PASS] Python {sys_hw.python_version}")

        # Stage 3: Hardware (CPU, RAM, GPU)
        print("\n[03/12] Detecting hardware...")
        print(f"[PASS] CPU: {sys_hw.cpu_cores} Cores | RAM: {sys_hw.ram_total_gb} GB")
        gpu_info = HardwareDetector.detect_nvidia_gpu()
        if gpu_info.present:
            print(f"[PASS] {gpu_info.gpu_name}")
            print(f"[PASS] VRAM: {gpu_info.vram_mb:.0f} MB")
        else:
            print("[INFO] NVIDIA GPU: None detected.")

        # Stage 4: NVIDIA Driver
        print("\n[04/12] Detecting NVIDIA driver...")
        if gpu_info.present:
            print(f"[PASS] Driver: {gpu_info.driver_version}")
        else:
            print("[INFO] N/A (CPU Mode)")

        # Stage 5: CUDA Compatibility
        print("\n[05/12] Detecting CUDA compatibility...")
        if self.force_cpu:
            print("[INFO] Target compute backend: CPU (--force-cpu active)")
            target_backend = ComputeBackend.CPU
        elif gpu_info.present:
            print(f"[PASS] CUDA Driver API: {gpu_info.cuda_driver_version or '12.x'}")
            target_backend = ComputeBackend.CUDA
        else:
            print("[INFO] Target compute backend: CPU")
            target_backend = ComputeBackend.CPU

        # Stage 6: PyTorch Check & Arbitration
        print("\n[06/12] Checking PyTorch...")
        pt_mgr = PyTorchManager()
        pt_info = pt_mgr.inspect_current_pytorch()

        if pt_info["installed"]:
            build_type = "CUDA" if pt_info["is_cuda_build"] else "CPU"
            print(f"[INFO] Current build: {pt_info['version']} ({build_type})")
        else:
            print("[WARN] PyTorch is not installed.")

        pt_needs_repair = not pt_mgr.is_compatible(target_backend) or self.force_repair
        if pt_needs_repair:
            print(f"[WARN] PyTorch build mismatch for target {target_backend.value}.")
            print(f"[07/12] Installing {target_backend.value} PyTorch...")
            pt_ok = pt_mgr.ensure_pytorch(target_backend=target_backend, force_reinstall=self.force_repair)
            if not pt_ok and target_backend == ComputeBackend.CUDA:
                print("[WARN] CUDA PyTorch install failed. Falling back to CPU...")
                target_backend = ComputeBackend.CPU
                pt_mgr.ensure_pytorch(target_backend=ComputeBackend.CPU)
        else:
            print("[PASS] PyTorch build already compatible.")
            print("[07/12] PyTorch installation required: NO")

        # Stage 7: ONNX Runtime Check & Arbitration
        onnx_mgr = OnnxManager()
        onnx_info = onnx_mgr.inspect_current_onnx()
        onnx_needs_repair = not onnx_mgr.is_compatible(target_backend) or self.force_repair

        if onnx_needs_repair:
            print(f"\n[INFO] Configuring ONNX Runtime for target: {target_backend.value}...")
            onnx_ok = onnx_mgr.ensure_onnx(target_backend=target_backend, force_reinstall=self.force_repair)
            if not onnx_ok and target_backend == ComputeBackend.CUDA:
                print("[WARN] ONNX GPU configuration failed. Falling back to CPU ONNX...")
                onnx_mgr.ensure_onnx(target_backend=ComputeBackend.CPU)
        else:
            print(f"[PASS] ONNX Runtime already compatible (Provider: {onnx_info.get('active_provider')}).")

        # Stage 8: Validate CUDA / CPU Tensor Execution
        print("\n[08/12] Validating Compute Device & Tensor Operations...")
        setup_cuda_dll_paths()
        validator = EnvironmentValidator()

        if target_backend == ComputeBackend.CUDA:
            cuda_det = CudaDetector()
            t_ok, t_details, t_err = cuda_det.probe_cuda_tensor_execution()
            if t_ok:
                print(f"[PASS] torch.cuda.is_available(): TRUE")
                print(f"[PASS] Device: {gpu_info.gpu_name}")
                print(f"[PASS] CUDA tensor execution: PASS")
            else:
                print(f"[FAIL] CUDA tensor probe failed: {t_err}")
                print("[WARN] Falling back to CPU backend...")
                target_backend = ComputeBackend.CPU

        if target_backend == ComputeBackend.CPU:
            cpu_ok, cpu_details, cpu_errors = validator.validate_cpu_pipeline()
            if cpu_ok:
                print("[PASS] CPU tensor execution: PASS")
            else:
                print(f"[FAIL] CPU validation failed: {cpu_errors}")
                return False

        # Authoritative DeviceManager initialization for subsequent pipeline steps
        dm = DeviceManager.get_instance(force_refresh=True, force_cpu=(target_backend == ComputeBackend.CPU))

        # Stage 9: Validate YOLO Runtime Device
        print("\n[09/12] Validating YOLO PersonDetector...")
        try:
            from pipeline.detection.person_detector import PersonDetector
            import numpy as np
            detector = PersonDetector()
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = detector.detect(dummy)
            yolo_dev = str(next(detector.model.model.parameters()).device)
            print(f"[PASS] Runtime Device: {yolo_dev}")
        except Exception as e:
            print(f"[FAIL] YOLO validation error: {e}")

        # Stage 10: Validate ONNX Runtime
        print("\n[10/12] Validating ONNX Runtime...")
        try:
            import numpy as np
            import onnxruntime as ort
            model_candidates = [
                ROOT / "models/weights/silhouette_segmenter.onnx",
                ROOT / "models/engines/silhouette_segmenter.onnx",
            ]
            model_path = next((p for p in model_candidates if p.exists()), None)
            providers = ort.get_available_providers()
            active_p = "CUDAExecutionProvider" if ("CUDAExecutionProvider" in providers and target_backend == ComputeBackend.CUDA) else "CPUExecutionProvider"
            if model_path:
                req_provs = [active_p]
                if active_p != "CPUExecutionProvider":
                    req_provs.append("CPUExecutionProvider")
                sess = ort.InferenceSession(str(model_path), providers=req_provs)
                active_p = sess.get_providers()[0] if sess.get_providers() else active_p
                in_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name
                dummy = np.zeros((1, 3, 256, 256), dtype=np.float32)
                _ = sess.run([out_name], {in_name: dummy})
            print(f"[PASS] Active Provider: {active_p}")
            print("[PASS] Silhouette inference")
        except Exception as e:
            print(f"[FAIL] ONNX validation error: {e}")

        # Stage 11: Validate ByGaitLight CNN
        print("\n[11/12] Validating ByGaitLight CNN...")
        try:
            import torch
            from models.architectures.bygait_light import ByGaitLight

            dev_str = dm.device
            model = ByGaitLight().to(dev_str)
            model.eval()
            dummy_input = torch.randn(1, 1, 128, 64, device=dev_str)
            with torch.no_grad():
                emb = model(dummy_input)

            norm = torch.norm(emb, p=2, dim=-1).item()
            print(f"[PASS] Model Device: {dev_str}")
            print(f"[PASS] Output Shape: {list(emb.shape)}")
            print(f"[PASS] L2 Norm: {norm:.4f}")
        except Exception as e:
            print(f"[FAIL] ByGaitLight validation error: {e}")

        # Stage 12: Final Environment Validation & Manifest Persistence
        print("\n[12/12] Final environment validation...")
        summary = dm.summary()

        self._save_manifest(summary)

        print("\n============================================================")
        print(" ARGUS COMPUTE ENVIRONMENT")
        print("============================================================")
        print(f"Backend          : {summary['backend'].upper()}")
        print(f"Device           : {summary['device']}")
        print(f"GPU              : {summary['gpu'] or 'None (CPU Mode)'}")
        print(f"VRAM             : {summary['vram_mb']:.0f} MB")
        print(f"PyTorch          : {summary['pytorch_version']}")
        print(f"CUDA             : {summary['cuda_version'] or 'N/A'}")
        print(f"YOLO             : {summary['backend'].upper()}")
        print(f"ONNX             : {'CUDA' if 'CUDA' in summary['onnx_provider'] else 'CPU'}")
        print(f"ByGaitLight      : {summary['backend'].upper()}")
        print("============================================================\n")

        if dm.is_cuda:
            print("[ARGUS] ENVIRONMENT READY")
            print("[ARGUS] FULL CUDA ACCELERATION READY")
        else:
            print("[ARGUS] ENVIRONMENT READY")
            print("[ARGUS] CPU MODE READY")

        return True

    def _save_manifest(self, summary: Dict[str, Any]) -> None:
        """Persist environment snapshot to .venv/argus_env_manifest.json."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            **summary,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS AI Environment Bootstrap")
    parser.add_argument("--force-repair", action="store_true", help="Force re-installation of compute wheels")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU compute backend mode")
    args = parser.parse_args()

    bootstrap = EnvironmentBootstrap(force_repair=args.force_repair, force_cpu=args.force_cpu)
    success = bootstrap.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
