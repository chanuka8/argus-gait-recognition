"""ARGUS AI Centralized Runtime Path Resolver.

Provides deterministic, CWD-independent path resolution distinguishing:
1. Application Root (immutable/bundled application assets, configs, models, manifests)
2. Runtime Root (writable runtime state, outputs, reports, logs)

Invariants:
- Never defaults to os.getcwd(), Path.cwd(), or Path(".")
- Never mutates process CWD or calls os.chdir()
- Preserves explicit absolute paths without modification
- Normalizes paths safely using pathlib.Path.resolve()
"""

from __future__ import annotations

import os
from pathlib import Path

# Module-level overrides for programmatic / testing control
_APP_ROOT_OVERRIDE: Path | None = None
_RUNTIME_ROOT_OVERRIDE: Path | None = None


def set_app_root(path: str | Path | None) -> None:
    """Set or clear the global programmatic application root override."""
    global _APP_ROOT_OVERRIDE
    _APP_ROOT_OVERRIDE = Path(path).resolve() if path is not None else None


def set_runtime_root(path: str | Path | None) -> None:
    """Set or clear the global programmatic runtime root override."""
    global _RUNTIME_ROOT_OVERRIDE
    _RUNTIME_ROOT_OVERRIDE = Path(path).resolve() if path is not None else None


def get_app_root(override: str | Path | None = None) -> Path:
    """Resolve the canonical application root directory.

    Precedence:
    1. Explicit function override argument
    2. Global programmatic override (set_app_root)
    3. ARGUS_APP_ROOT environment variable
    4. Deterministic repository/application root derived from core/paths.py __file__
    """
    if override is not None:
        return Path(override).resolve()

    if _APP_ROOT_OVERRIDE is not None:
        return _APP_ROOT_OVERRIDE

    env_root = os.environ.get("ARGUS_APP_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Derived deterministically from this file location: <repo_root>/core/paths.py -> parents[1]
    return Path(__file__).resolve().parent.parent


def get_runtime_root(override: str | Path | None = None) -> Path:
    """Resolve the canonical runtime/writable data root directory.

    Precedence:
    1. Explicit function override argument
    2. Global programmatic override (set_runtime_root)
    3. ARGUS_RUNTIME_ROOT environment variable
    4. Canonical application root (for backward compatibility)
    """
    if override is not None:
        return Path(override).resolve()

    if _RUNTIME_ROOT_OVERRIDE is not None:
        return _RUNTIME_ROOT_OVERRIDE

    env_runtime = os.environ.get("ARGUS_RUNTIME_ROOT")
    if env_runtime:
        return Path(env_runtime).resolve()

    return get_app_root()


def resolve_app_path(path: str | Path, *, app_root: str | Path | None = None) -> Path:
    """Resolve an application asset path against the canonical application root.

    - If path is already absolute, it is returned resolved and unmodified.
    - If path is relative, it is anchored to get_app_root(app_root).
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()

    root = get_app_root(app_root)
    return (root / p).resolve()


def resolve_runtime_path(path: str | Path, *, runtime_root: str | Path | None = None) -> Path:
    """Resolve a runtime writable path against the canonical runtime root.

    - If path is already absolute, it is returned resolved and unmodified.
    - If path is relative, it is anchored to get_runtime_root(runtime_root).
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()

    root = get_runtime_root(runtime_root)
    return (root / p).resolve()


def get_config_path(config_name: str | Path = "configs", *, app_root: str | Path | None = None) -> Path:
    """Resolve a configuration directory or file path against the application root."""
    p = Path(config_name)
    if p.is_absolute():
        return p.resolve()

    s = p.as_posix()
    if s == "configs" or s.startswith("configs/"):
        return resolve_app_path(p, app_root=app_root)

    return resolve_app_path(Path("configs") / p, app_root=app_root)


def get_model_path(model_path: str | Path, *, app_root: str | Path | None = None) -> Path:
    """Resolve a model weight, checkpoint, or engine path against the application root."""
    return resolve_app_path(model_path, app_root=app_root)
