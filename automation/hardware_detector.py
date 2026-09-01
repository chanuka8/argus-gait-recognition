import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class HostSystemInfo:
    os_name: str
    os_version: str
    python_version: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    architecture: str


@dataclass
class NvidiaGpuInfo:
    present: bool
    gpu_name: str | None = None
    driver_version: str | None = None
    vram_mb: float = 0.0
    cuda_driver_version: str | None = None
    query_success: bool = False
    error: str | None = None


@dataclass
class HardwareProfile:
    system: HostSystemInfo
    gpu: NvidiaGpuInfo

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": f"{self.system.os_name} {self.system.os_version}",
            "python_version": self.system.python_version,
            "cpu_cores": self.system.cpu_cores,
            "ram_gb": self.system.ram_total_gb,
            "architecture": self.system.architecture,
            "has_nvidia_gpu": self.gpu.present,
            "gpu_name": self.gpu.gpu_name,
            "driver_version": self.gpu.driver_version,
            "vram_mb": self.gpu.vram_mb,
            "cuda_driver_version": self.gpu.cuda_driver_version,
        }


class HardwareDetector:
    @staticmethod
    def detect_system() -> HostSystemInfo:
        os_name = platform.system()
        os_version = platform.release()
        py_ver = sys.version.split()[0]
        cpu_count = os.cpu_count() or 1
        arch = platform.machine()
        total_ram = 0.0
        avail_ram = 0.0

        try:
            import psutil

            vm = psutil.virtual_memory()
            total_ram = round(vm.total / (1024**3), 1)
            avail_ram = round(vm.available / (1024**3), 1)
        except (ImportError, AttributeError, OSError):
            pass

        return HostSystemInfo(
            os_name=os_name,
            os_version=os_version,
            python_version=py_ver,
            cpu_cores=cpu_count,
            ram_total_gb=total_ram,
            ram_available_gb=avail_ram,
            architecture=arch,
        )

    @staticmethod
    def detect_nvidia_gpu() -> NvidiaGpuInfo:
        smi_candidates = [
            shutil.which("nvidia-smi"),
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            "/usr/bin/nvidia-smi",
            "/usr/local/cuda/bin/nvidia-smi",
        ]

        smi_path = next((p for p in smi_candidates if p and os.path.exists(p)), None)
        if not smi_path:
            return NvidiaGpuInfo(
                present=False,
                error="nvidia-smi utility not found on host system",
            )

        try:

            cmd = [
                smi_path,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )

            if result.returncode != 0 or not result.stdout.strip():
                err_msg = result.stderr.strip() or "nvidia-smi returned non-zero exit code"
                return NvidiaGpuInfo(present=False, error=err_msg)

            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                return NvidiaGpuInfo(present=False, error="Malformed nvidia-smi output")

            name = parts[0]
            driver = parts[1]
            try:
                vram_mb = float(parts[2])
            except ValueError:
                vram_mb = 0.0


            cuda_driver_ver: str | None = None
            try:
                smi_banner = subprocess.run(
                    [smi_path],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if "CUDA Version:" in smi_banner.stdout:
                    cuda_driver_ver = smi_banner.stdout.split("CUDA Version:")[1].split()[0].strip()
            except (subprocess.SubprocessError, OSError, UnicodeDecodeError, IndexError):
                pass

            return NvidiaGpuInfo(
                present=True,
                gpu_name=name,
                driver_version=driver,
                vram_mb=vram_mb,
                cuda_driver_version=cuda_driver_ver or "12.x",
                query_success=True,
            )
        except (subprocess.SubprocessError, OSError, ValueError, IndexError) as exc:
            return NvidiaGpuInfo(present=False, error=str(exc))

    @classmethod
    def detect(cls) -> HardwareProfile:
        sys_info = cls.detect_system()
        gpu_info = cls.detect_nvidia_gpu()
        return HardwareProfile(system=sys_info, gpu=gpu_info)
