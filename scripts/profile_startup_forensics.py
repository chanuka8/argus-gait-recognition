"""
Forensic Startup Profiling for ARGUS AI Backend Server.

Measures all individual startup stages with microsecond precision:
1. Python process overhead & stdlib imports
2. Third-party library imports (torch, cv2, numpy, fastapi, uvicorn, etc.)
3. Subsystem imports
4. Model loadings (ByGaitLight, OSNet, Silhouette, YOLO, OpenSet)
5. Gallery loading & VectorStore
6. Database & Continual Learning inits
7. App & Router assembly
8. End-to-end Uvicorn launch until /health HTTP 200 OK
"""

import gc
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def time_block(name: str, fn):
    gc.collect()
    t0 = time.perf_counter()
    res = fn()
    t1 = time.perf_counter()
    dur = t1 - t0
    print(f"  {name:<50} : {dur:7.4f} s", flush=True)
    return res, dur


def run_stage_profiling() -> dict:
    print("=" * 70, flush=True)
    print("STAGE 1: ISOLATED IMPORT & INITIALIZATION PROFILING", flush=True)
    print("=" * 70, flush=True)

    timings = {}

    # 1. Stdlib & Lightweight imports
    _, timings["import_yaml_json_pathlib"] = time_block(
        "1. import yaml, json, pathlib",
        lambda: (__import__("yaml"), __import__("json"), __import__("pathlib")),
    )
    _, timings["import_numpy"] = time_block("2. import numpy", lambda: __import__("numpy"))
    _, timings["import_cv2"] = time_block("3. import cv2", lambda: __import__("cv2"))
    _, timings["import_fastapi"] = time_block("4. import fastapi", lambda: __import__("fastapi"))
    _, timings["import_uvicorn"] = time_block("5. import uvicorn", lambda: __import__("uvicorn"))
    _, timings["import_torch"] = time_block("6. import torch", lambda: __import__("torch"))
    _, timings["import_torchvision"] = time_block("7. import torchvision", lambda: __import__("torchvision"))
    _, timings["import_ultralytics"] = time_block("8. import ultralytics", lambda: __import__("ultralytics"))

    # 2. Config & Logging
    _, timings["logging_and_config_init"] = time_block(
        "9. logging and configs init",
        lambda: __import__("monitoring.logging_config"),
    )

    # 3. Model Subsystem Imports
    _, timings["import_bygait_extractor"] = time_block(
        "10. import ByGaitLight extraction step",
        lambda: __import__("pipeline.steps.feature_extraction"),
    )
    _, timings["import_silhouette_extractor"] = time_block(
        "11. import SilhouetteExtractor",
        lambda: __import__("pipeline.silhouette.extractor"),
    )
    _, timings["import_osnet_extractor"] = time_block(
        "12. import AppearanceEmbeddingExtractor (OSNet)",
        lambda: __import__("intelligence.appearance_embedding"),
    )
    _, timings["import_person_detector"] = time_block(
        "13. import PersonDetector",
        lambda: __import__("pipeline.detection.person_detector"),
    )
    _, timings["import_open_set_recognizer"] = time_block(
        "14. import OpenSetRecognizer",
        lambda: __import__("intelligence.open_set_recognizer"),
    )
    _, timings["import_vector_store"] = time_block(
        "15. import VectorStore",
        lambda: __import__("storage.vector_store"),
    )
    _, timings["import_model_registry"] = time_block(
        "16. import ModelRegistry",
        lambda: __import__("models.model_registry"),
    )
    _, timings["import_continual_learning"] = time_block(
        "17. import ContinuousImprovementEngine",
        lambda: __import__("intelligence.continuous_improvement_engine"),
    )
    _, timings["import_camera_resolver"] = time_block(
        "18. import CameraSourceResolver",
        lambda: __import__("services.camera_source_resolver"),
    )

    # 4. Component Instantiations & Model Weights Loading
    from pipeline.steps.feature_extraction import FeatureExtractionStep

    _, timings["init_bygait_light"] = time_block(
        "19. Model Load: ByGaitLight (FeatureExtractionStep)",
        lambda: FeatureExtractionStep(),
    )

    from pipeline.silhouette.extractor import SilhouetteExtractor

    _, timings["init_silhouette_extractor"] = time_block(
        "20. Model Load: SilhouetteExtractor",
        lambda: SilhouetteExtractor(target_size=(64, 128)),
    )

    from intelligence.appearance_embedding import AppearanceEmbeddingExtractor

    _, timings["init_osnet"] = time_block(
        "21. Model Load: OSNet (AppearanceEmbeddingExtractor)",
        lambda: AppearanceEmbeddingExtractor(),
    )

    from pipeline.detection.person_detector import PersonDetector

    _, timings["init_person_detector"] = time_block(
        "22. Model Load: PersonDetector (YOLO)",
        lambda: PersonDetector(),
    )

    from intelligence.open_set_recognizer import OpenSetRecognizer

    _, timings["init_open_set_recognizer"] = time_block(
        "23. Model Load: OpenSetRecognizer",
        lambda: OpenSetRecognizer(),
    )

    from storage.vector_store import VectorStore

    _, timings["init_vector_stores"] = time_block(
        "24. Storage: VectorStore (gait + appearance)",
        lambda: (
            VectorStore(gallery_dir="models/live_gallery"),
            VectorStore(gallery_dir="models/appearance_gallery"),
        ),
    )

    from models.model_registry import ModelRegistry
    from storage.embedding_database import EmbeddingDatabase

    _, timings["init_embedding_db"] = time_block(
        "25. Storage: EmbeddingDatabase & ModelRegistry",
        lambda: (
            EmbeddingDatabase(
                gait_gallery_dir="models/live_gallery",
                appearance_gallery_dir="models/appearance_gallery",
            ),
            ModelRegistry(),
        ),
    )

    from intelligence.continuous_improvement_engine import ContinuousImprovementEngine

    _, timings["init_continual_learning_engine"] = time_block(
        "26. Intelligence: ContinuousImprovementEngine",
        lambda: ContinuousImprovementEngine(),
    )

    from services.camera_source_resolver import CameraSourceResolver

    _, timings["init_camera_source_resolver"] = time_block(
        "27. Services: CameraSourceResolver",
        lambda: CameraSourceResolver(),
    )

    # 5. Full GaitService Instantiation
    from services.gait_service import GaitService

    _, timings["init_full_gait_service"] = time_block(
        "28. Full GaitService() instantiation",
        lambda: GaitService(),
    )

    # 6. App & Routers Assembly
    _, timings["import_app_and_routers"] = time_block(
        "29. Import api.server & build routers",
        lambda: __import__("api.server"),
    )

    return timings


def measure_server_ready(port: int = 8888, timeout_s: float = 60.0) -> float:
    """
    Launch Uvicorn backend server in a fresh isolated subprocess.
    Polls http://127.0.0.1:{port}/health until 200 OK is received.
    Returns the exact measured time in seconds.
    """
    env = dict(os.environ)
    env["PORT"] = str(port)
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-access-log",
    ]

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}/health"

    ready_time = None
    deadline = t0 + timeout_s

    while time.perf_counter() < deadline:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status == 200:
                    ready_time = time.perf_counter() - t0
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)

    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()

    if ready_time is None:
        raise TimeoutError(f"Server on port {port} failed to reach READY state within {timeout_s}s")

    return ready_time


if __name__ == "__main__":
    timings = run_stage_profiling()

    print("\n" + "=" * 70, flush=True)
    print("STAGE 2: END-TO-END SERVER READY BENCHMARK (5 REPEATED RUNS)", flush=True)
    print("=" * 70, flush=True)

    server_ready_runs = []
    base_port = 8890
    for run_idx in range(1, 6):
        port = base_port + run_idx
        print(f"  Run {run_idx}/5 (port {port})...", end="", flush=True)
        dur = measure_server_ready(port=port, timeout_s=90.0)
        server_ready_runs.append(dur)
        print(f" READY in {dur:6.3f} s", flush=True)
        time.sleep(0.5)

    import statistics

    print("\n" + "=" * 70, flush=True)
    print("SERVER READY SUMMARY STATISTICS:", flush=True)
    print(f"  Min   : {min(server_ready_runs):6.3f} s", flush=True)
    print(f"  Max   : {max(server_ready_runs):6.3f} s", flush=True)
    print(f"  Mean  : {statistics.mean(server_ready_runs):6.3f} s", flush=True)
    print(f"  Median: {statistics.median(server_ready_runs):6.3f} s", flush=True)
    print("=" * 70, flush=True)

    results = {
        "stage_timings": timings,
        "server_ready_runs": server_ready_runs,
        "stats": {
            "min": min(server_ready_runs),
            "max": max(server_ready_runs),
            "mean": statistics.mean(server_ready_runs),
            "median": statistics.median(server_ready_runs),
        },
    }
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    with open("outputs/reports/startup_profiling_optimized.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved profiling report to outputs/reports/startup_profiling_optimized.json", flush=True)
