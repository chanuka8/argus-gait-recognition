"""Centralized, hardware-aware thread limits for the ML stack.

Without this, PyTorch, OpenCV, ONNX Runtime, and the OpenMP/MKL/OpenBLAS
layers beneath them each default to spawning one thread per logical CPU
core. Loaded independently, several of those pools end up alive at once
(e.g. an ONNX silhouette session next to a PyTorch gait/appearance model),
each trying to claim every core, starving the OS, the browser, and
FastAPI's own event loop.

Two-stage design, because importing torch is itself expensive (tens of
seconds when the OS file cache is cold - see Phase-3A profiling) and must
not block `/health`:

- `configure_env()` only sets OMP_NUM_THREADS/MKL_NUM_THREADS/
  OPENBLAS_NUM_THREADS/NUMEXPR_NUM_THREADS. It imports nothing and costs
  microseconds. These env vars are read once when a native library is
  first loaded, so this must still run before torch/cv2/numpy are
  imported anywhere in the process - every real entrypoint
  (app/api/server.py, main.py) calls it as its first statement.
- `configure_native()` actually imports torch/cv2 and calls their
  explicit thread-count APIs (env vars alone don't fully control torch's
  intra-op pool). This is deferred to GaitService.warmup(), which already
  runs off the FastAPI event loop, so the one-time torch import cost
  lands during background warmup instead of blocking health-ready.
"""

import os

_budget: int | None = None
_env_configured = False
_native_configured = False


def _compute_budget() -> int:
    cores = os.cpu_count() or 4
    if cores <= 2:
        return 1
    if cores <= 4:
        return cores - 1
    # Leave a quarter of cores (min 2) for the OS, browser/UI, the FastAPI
    # event loop, and camera ingestion threads.
    reserved = max(2, cores // 4)
    return max(2, cores - reserved)


def configure_env() -> int:
    """Set BLAS/OpenMP env vars only. No imports. Idempotent. Cheap."""
    global _budget, _env_configured
    if _budget is None:
        _budget = int(os.environ.get("ARGUS_THREAD_BUDGET", _compute_budget()))
    if not _env_configured:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, str(_budget))
        _env_configured = True
    return _budget


def configure_native() -> int:
    """Import torch/cv2 and apply explicit thread-count APIs. Idempotent.

    Expensive on first call (imports torch). Call this only from a path
    that's already off the request/startup critical path, such as
    GaitService.warmup().
    """
    global _native_configured
    budget = configure_env()
    if _native_configured:
        return budget

    # RAM-headroom check (psutil + nvidia-smi, a couple seconds) is deferred
    # to here rather than configure_env(), so it never delays /health.
    try:
        from app.core.resource_profile import get_runtime_parameters

        params = get_runtime_parameters()
        if params.profile_name == "LOW_RESOURCE":
            budget = min(budget, 2)
    except Exception:  # noqa: BLE001 - thread limiting must never block warmup
        pass

    try:
        import torch

        torch.set_num_threads(budget)
        try:
            torch.set_num_interop_threads(max(1, min(2, budget)))
        except RuntimeError:
            pass  # already in use elsewhere in this process; intra-op limit above still applies
    except ImportError:
        pass

    try:
        import cv2

        cv2.setNumThreads(budget)
    except ImportError:
        pass

    _native_configured = True
    return budget


def current_budget() -> int:
    return configure_env()


def onnx_session_options(intra_op_threads: int | None = None):
    """SessionOptions for onnxruntime.InferenceSession, capped to the configured budget."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads or current_budget()
    opts.inter_op_num_threads = 1
    return opts
