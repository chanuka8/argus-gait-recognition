"""Centralized, hardware-aware thread limits for the ML stack.

Without this, PyTorch, OpenCV, ONNX Runtime, and the OpenMP/MKL/OpenBLAS
layers beneath them each default to spawning one thread per logical CPU
core. Loaded independently, several of those pools end up alive at once
(e.g. an ONNX silhouette session next to a PyTorch gait/appearance model),
each trying to claim every core, starving the OS, the browser, and
FastAPI's own event loop.

`configure()` must run before torch/cv2/numpy are imported anywhere in the
process, since OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS are
read once when those native libraries are loaded. Every real process
entrypoint (app/api/server.py, main.py) calls it as its first statement,
before any other import.
"""

import os

_budget: int | None = None


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


def configure() -> int:
    """Set BLAS/OpenMP env vars and native thread pools. Idempotent."""
    global _budget
    if _budget is not None:
        return _budget

    budget = int(os.environ.get("ARGUS_THREAD_BUDGET", _compute_budget()))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(budget))

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

    _budget = budget
    return budget


def current_budget() -> int:
    return _budget if _budget is not None else configure()


def onnx_session_options(intra_op_threads: int | None = None):
    """SessionOptions for onnxruntime.InferenceSession, capped to the configured budget."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra_op_threads or current_budget()
    opts.inter_op_num_threads = 1
    return opts
