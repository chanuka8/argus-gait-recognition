"""
Build and Version Metadata Extractor and Container for ARGUS AI.

Extracts application version, Git commit/branch details, Python environment metadata,
model reference identifiers, and configuration fingerprints without network requests.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Optional


@dataclass
class BuildMetadata:
    """Dataclass holding build and version metadata for deployment artifacts."""

    application_name: str
    application_version: str
    git_commit: str
    git_branch: str
    python_version: str
    model_reference: str
    backend_requested: str
    backend_active: str
    configuration_fingerprint: str

    def to_dict(self) -> dict:
        """Export metadata as dictionary, verifying no absolute paths or credentials exist."""
        data = asdict(self)
        raw_str = str(data)
        if "C:\\Users" in raw_str or "/home/" in raw_str:
            raise ValueError("Absolute user-home path detected in build metadata")
        if "SECRET" in raw_str.upper() and "PATTERNS" not in raw_str.upper():
            raise ValueError("Credential leak detected in build metadata")
        return data


def get_git_commit(repo_dir: str = ".") -> str:
    """Safely fetch short git commit hash or return UNKNOWN on failure."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def get_git_branch(repo_dir: str = ".") -> str:
    """Safely fetch git branch name or return UNKNOWN on failure."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def get_application_version(version_path: str = "VERSION") -> str:
    """Read application version string from VERSION file or return 0.0.0-dev."""
    path = Path(version_path)
    if path.exists():
        try:
            ver = path.read_text(encoding="utf-8").strip()
            if ver:
                return ver
        except Exception:
            pass
    return "0.0.0-dev"


def compute_configuration_fingerprint(configs_dir: str = "configs") -> str:
    """Compute deterministic SHA-256 fingerprint hash of configuration files in configs/."""
    cdir = Path(configs_dir)
    if not cdir.exists() or not cdir.is_dir():
        return "NO_CONFIG_DIR"

    hasher = hashlib.sha256()
    yaml_files = sorted(cdir.glob("*.yaml"))

    for yfile in yaml_files:
        hasher.update(yfile.name.encode("utf-8"))
        try:
            hasher.update(yfile.read_bytes())
        except Exception:
            pass

    return hasher.hexdigest()[:12]


def extract_build_metadata(
    repo_dir: str = ".",
    version_file: str = "VERSION",
    configs_dir: str = "configs",
    backend=None,
    model_reference: Optional[str] = None,
) -> BuildMetadata:
    """
    Extract authoritative BuildMetadata structure.

    Args:
        repo_dir: Root repository path.
        version_file: Path to VERSION file.
        configs_dir: Directory containing YAML configuration files.
        backend: Optional initialized inference backend.
        model_reference: Optional model path string override.

    Returns:
        Populated BuildMetadata instance.
    """
    app_ver = get_application_version(version_path=version_file)
    git_commit = get_git_commit(repo_dir=repo_dir)
    git_branch = get_git_branch(repo_dir=repo_dir)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    req_be = "pytorch"
    act_be = "pytorch"
    m_ref = model_reference or "runs/exp_001/best_model.pth"

    if backend is not None:
        req_be = getattr(backend, "requested_backend", "pytorch")
        act_be = getattr(backend, "active_backend", "pytorch")
        if not model_reference:
            if act_be == "onnxruntime":
                m_ref = str(getattr(backend, "config", {}).get("onnx_path", "models/engines/bygait_light.onnx"))
            else:
                m_ref = str(getattr(backend, "config", {}).get("model_path", "runs/exp_001/best_model.pth"))

    m_ref_posix = Path(m_ref).as_posix()
    cfg_fp = compute_configuration_fingerprint(configs_dir=configs_dir)

    return BuildMetadata(
        application_name="ARGUS AI",
        application_version=app_ver,
        git_commit=git_commit,
        git_branch=git_branch,
        python_version=py_ver,
        model_reference=m_ref_posix,
        backend_requested=req_be,
        backend_active=act_be,
        configuration_fingerprint=cfg_fp,
    )
