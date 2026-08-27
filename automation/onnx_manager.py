"""
ONNX Runtime Environment & Provider Management Subsystem for ARGUS AI.

Manages onnxruntime vs onnxruntime-gpu packages. Prevents conflicting duplicate
installations, configures centralized Windows DLL paths, and validates actual
InferenceSession execution with the silhouette segmentation model.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

from automation.dll_manager import setup_cuda_dll_paths
from automation.environment_validator import ComputeBackend


class OnnxManager:
    """Idempotent manager for ONNX Runtime CPU / GPU execution stacks."""

    def __init__(self, weights_dir: str = "models/weights") -> None:
        self.weights_dir = Path(weights_dir)
        self.python_exe = sys.executable
        setup_cuda_dll_paths()

    def inspect_current_onnx(self) -> dict[str, Any]:
        """Inspect installed ONNX Runtime version and available providers."""
        setup_cuda_dll_paths()
        info = {
            "installed": False,
            "version": None,
            "providers": [],
            "cuda_available": False,
            "session_probe_passed": False,
            "active_provider": "None",
            "is_gpu_package": False,
        }

        try:
            import onnxruntime as ort

            info["installed"] = True
            info["version"] = getattr(ort, "__version__", None)
            providers = ort.get_available_providers()
            info["providers"] = providers
            info["cuda_available"] = "CUDAExecutionProvider" in providers

            # Check if package is onnxruntime-gpu
            try:
                import importlib.metadata

                dist_names = [d.metadata["Name"].lower() for d in importlib.metadata.distributions()]
                info["is_gpu_package"] = "onnxruntime-gpu" in dist_names
            except (ImportError, KeyError, AttributeError, OSError):
                info["is_gpu_package"] = info["cuda_available"]

            # Test real InferenceSession creation & inference
            model_candidates = [
                self.weights_dir / "silhouette_segmenter.onnx",
                Path("models/weights/silhouette_segmenter.onnx"),
                Path("models/engines/silhouette_segmenter.onnx"),
            ]
            model_path = next((p for p in model_candidates if p.exists()), None)

            if model_path:
                req_provs = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if info["cuda_available"]
                    else ["CPUExecutionProvider"]
                )
                sess = ort.InferenceSession(str(model_path), providers=req_provs)
                active = sess.get_providers()
                info["active_provider"] = active[0] if active else "CPUExecutionProvider"
                info["session_probe_passed"] = True
        except (ImportError, RuntimeError, ValueError, OSError):
            pass

        return info

    def is_compatible(self, target_backend: ComputeBackend) -> bool:
        """Check if active ONNX Runtime installation matches target compute."""
        info = self.inspect_current_onnx()
        if not info["installed"]:
            return False

        if target_backend == ComputeBackend.CUDA:
            return bool(
                info["cuda_available"]
                and info["session_probe_passed"]
                and info["active_provider"] == "CUDAExecutionProvider"
            )
        else:
            return bool(info["session_probe_passed"])

    def _run_pip_unbuffered(self, args: list[str]) -> bool:
        """Execute pip commands with live streaming output."""
        cmd = [self.python_exe, "-u", "-m", "pip"] + args
        print(f"[PIP] Running: {' '.join(cmd)}")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    sys.stdout.write(f"[PIP] {line}")
                    sys.stdout.flush()
            process.wait()
            return process.returncode == 0
        except (subprocess.SubprocessError, OSError) as e:
            print(f"[PIP ERROR] Execution failed: {e}")
            return False

    def clean_conflicting_packages(self) -> bool:
        """Remove any conflicting CPU/GPU ONNX Runtime packages."""
        print("[ARGUS] Cleaning ONNX Runtime packages to avoid conflicts...")
        return self._run_pip_unbuffered(["uninstall", "-y", "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"])

    def ensure_onnx(self, target_backend: ComputeBackend, force_reinstall: bool = False) -> bool:
        """
        Idempotently ensure the correct ONNX Runtime variant is installed.
        Skips if already healthy.
        """
        current = self.inspect_current_onnx()

        if not force_reinstall and self.is_compatible(target_backend):
            print(f"[ARGUS] ONNX Runtime already compatible (Provider: {current.get('active_provider')}).")
            print("[ARGUS] Installation required: NO")
            print("[ARGUS] Download required: NO")
            return True

        print(f"\n[ARGUS] Configuring ONNX Runtime for target: {target_backend.value}")

        if target_backend == ComputeBackend.CUDA:
            self.clean_conflicting_packages()
            print("[ARGUS] Installing onnxruntime-gpu (1.20.0)...")
            ok = self._run_pip_unbuffered(["install", "onnxruntime-gpu==1.20.0"])
            if not ok:
                print("[ARGUS WARN] onnxruntime-gpu installation failed. Falling back to CPU ONNX...")
                return self._install_cpu_onnx()
            return ok
        else:
            return self._install_cpu_onnx()

    def _install_cpu_onnx(self) -> bool:
        """Install standard CPU ONNX Runtime build."""
        self.clean_conflicting_packages()
        print("[ARGUS] Installing CPU onnxruntime...")
        return self._run_pip_unbuffered(["install", "onnxruntime>=1.15.0"])
