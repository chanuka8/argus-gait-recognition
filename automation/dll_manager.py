import os
import sys
from pathlib import Path

_DLL_DIRECTORIES_REGISTERED: list[str] = []
_INITIALIZED: bool = False


def setup_cuda_dll_paths() -> list[str]:
    global _INITIALIZED, _DLL_DIRECTORIES_REGISTERED

    if _INITIALIZED:
        return list(_DLL_DIRECTORIES_REGISTERED)

    added_paths: list[str] = []

    if sys.platform != "win32":
        _INITIALIZED = True
        return added_paths

    candidate_paths: list[Path] = []


    try:
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            candidate_paths.append(torch_lib)
    except (ImportError, AttributeError, OSError):
        pass


    cuda_path_env = os.environ.get("CUDA_PATH")
    if cuda_path_env:
        cuda_bin = Path(cuda_path_env) / "bin"
        if cuda_bin.exists():
            candidate_paths.append(cuda_bin)


    system_nv = Path(r"C:\Windows\System32")
    if system_nv.exists():
        candidate_paths.append(system_nv)


    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep)

    for p in candidate_paths:
        resolved_str = str(p.resolve())
        if resolved_str not in path_entries:
            os.environ["PATH"] = resolved_str + os.pathsep + os.environ.get("PATH", "")
            path_entries.insert(0, resolved_str)

        if hasattr(os, "add_dll_directory") and p.is_dir():
            try:
                os.add_dll_directory(resolved_str)
                added_paths.append(resolved_str)
            except (OSError, ValueError):
                pass

    _DLL_DIRECTORIES_REGISTERED = added_paths
    _INITIALIZED = True
    return added_paths
