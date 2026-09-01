import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from automation.download_manager import DownloadManager
from automation.environment_validator import ComputeBackend


@dataclass
class PyTorchInstallSpec:
    backend: ComputeBackend
    torch_version: str = "2.5.1"
    vision_version: str = "0.20.1"
    cuda_tag: str = "cu121"
    python_tag: str = "cp311-cp311-win_amd64"
    torch_wheel_url: str = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"
    vision_wheel_url: str = (
        "https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp311-cp311-win_amd64.whl"
    )
    torch_wheel_name: str = "torch-2.5.1+cu121-cp311-cp311-win_amd64.whl"
    vision_wheel_name: str = "torchvision-0.20.1+cu121-cp311-cp311-win_amd64.whl"


class PyTorchManager:
    def __init__(self, cache_dir: str = ".venv/wheel_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.python_exe = sys.executable

    def inspect_current_pytorch(self) -> dict[str, Any]:
        info = {
            "installed": False,
            "version": None,
            "cuda_in_build": None,
            "cuda_available": False,
            "tensor_probe_passed": False,
            "is_cuda_build": False,
            "is_cpu_build": False,
        }

        try:
            import torch

            info["installed"] = True
            info["version"] = getattr(torch, "__version__", None)
            cuda_tag = getattr(torch.version, "cuda", None)
            info["cuda_in_build"] = cuda_tag
            info["is_cuda_build"] = bool(cuda_tag)
            info["is_cpu_build"] = not bool(cuda_tag)
            info["cuda_available"] = torch.cuda.is_available()


            dev = "cuda" if info["cuda_available"] else "cpu"
            a = torch.zeros((128, 128), device=dev)
            b = torch.ones((128, 128), device=dev)
            c = a @ b
            if dev == "cuda":
                torch.cuda.synchronize()
            if c.shape == (128, 128):
                info["tensor_probe_passed"] = True
        except (RuntimeError, ValueError, TypeError, AttributeError):
            pass

        return info

    def is_compatible(self, target_backend: ComputeBackend) -> bool:
        info = self.inspect_current_pytorch()
        if not info["installed"] or not info["tensor_probe_passed"]:
            return False

        if target_backend == ComputeBackend.CUDA:
            return bool(info["is_cuda_build"] and info["cuda_available"])
        else:
            return bool(info["installed"] and info["tensor_probe_passed"])

    def _run_pip_unbuffered(self, args: list[str]) -> bool:
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

    def uninstall_pytorch(self) -> bool:
        print("\n[ARGUS] Removing conflicting PyTorch build to ensure clean environment...")
        return self._run_pip_unbuffered(["uninstall", "-y", "torch", "torchvision", "torchaudio"])

    def ensure_pytorch(self, target_backend: ComputeBackend, force_reinstall: bool = False) -> bool:
        current = self.inspect_current_pytorch()

        if not force_reinstall and self.is_compatible(target_backend):
            print(f"[ARGUS] PyTorch already compatible ({current.get('version')}).")
            print("[ARGUS] Installation required: NO")
            print("[ARGUS] Download required: NO")
            return True

        print(f"\n[ARGUS] PyTorch installation required for target: {target_backend.value}")
        spec = PyTorchInstallSpec(backend=target_backend)


        if current["installed"]:
            self.uninstall_pytorch()


        if target_backend == ComputeBackend.CUDA:
            print("[ARGUS] Installing PyTorch CUDA build (2.5.1+cu121)...")
            torch_wheel_path = self.cache_dir / spec.torch_wheel_name
            vision_wheel_path = self.cache_dir / spec.vision_wheel_name


            if not torch_wheel_path.exists():
                ok = DownloadManager.download_file(
                    url=spec.torch_wheel_url,
                    dest_path=torch_wheel_path,
                    package_name=f"PyTorch CUDA ({spec.torch_wheel_name})",
                )
                if not ok:
                    print("[ARGUS ERROR] Failed to download PyTorch CUDA wheel. Falling back to CPU index...")
                    return self._install_cpu_pytorch()

            if not vision_wheel_path.exists():
                ok = DownloadManager.download_file(
                    url=spec.vision_wheel_url,
                    dest_path=vision_wheel_path,
                    package_name=f"TorchVision CUDA ({spec.vision_wheel_name})",
                )
                if not ok:
                    print("[ARGUS ERROR] Failed to download TorchVision CUDA wheel. Falling back to CPU index...")
                    return self._install_cpu_pytorch()


            install_ok = self._run_pip_unbuffered(
                [
                    "install",
                    "--no-cache-dir",
                    str(torch_wheel_path),
                    str(vision_wheel_path),
                ]
            )

            if not install_ok:
                print("[ARGUS WARN] Local wheel install failed. Attempting direct index install...")
                install_ok = self._run_pip_unbuffered(
                    [
                        "install",
                        "torch==2.5.1+cu121",
                        "torchvision==0.20.1+cu121",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu121",
                    ]
                )

            return install_ok

        else:
            return self._install_cpu_pytorch()

    def _install_cpu_pytorch(self) -> bool:
        print("[ARGUS] Installing official CPU PyTorch build (2.5.1 / 0.20.1)...")
        return self._run_pip_unbuffered(
            [
                "install",
                "torch==2.5.1",
                "torchvision==0.20.1",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
