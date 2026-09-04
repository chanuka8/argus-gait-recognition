import argparse
import json
import os
import sys
import time
from typing import Any

import cv2
import numpy as np
import psutil
import torch

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def compute_distribution(samples: list[float]) -> dict[str, float]:
    """Compute mean, median, p95, p99, min, max for a list of measurement samples."""
    if not samples:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    arr = np.array(samples, dtype=np.float64)
    return {
        "mean": float(round(float(np.mean(arr)), 4)),
        "median": float(round(float(np.median(arr)), 4)),
        "p95": float(round(float(np.percentile(arr, 95)), 4)),
        "p99": float(round(float(np.percentile(arr, 99)), 4)),
        "min": float(round(float(np.min(arr)), 4)),
        "max": float(round(float(np.max(arr)), 4)),
        "count": len(samples),
    }


def run_benchmark(
    num_iterations: int = 50,
    output_path: str = "outputs/reports/baseline_performance.json",
) -> dict[str, Any]:
    print("=" * 80)
    print("ARGUS AI — PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report: dict[str, Any] = {
        "timestamp": time.time(),
        "iterations": num_iterations,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "metrics": {},
    }

    # -------------------------------------------------------------
    # 1. Backend Startup Time (instantiating GaitService + dependencies)
    # -------------------------------------------------------------
    print("[1/30] Measuring Backend Startup Time...")
    startup_samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        from services.gait_service import GaitService

        gs = GaitService()
        startup_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["1_backend_startup_time_ms"] = compute_distribution(startup_samples)

    # -------------------------------------------------------------
    # 2. API Request Latency (health endpoint)
    # -------------------------------------------------------------
    print("[2/30] Measuring API Request Latency...")
    from starlette.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    api_samples = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        resp = client.get("/api/v1/health")
        if resp.status_code == 200:
            api_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["2_api_request_latency_ms"] = compute_distribution(api_samples)

    # -------------------------------------------------------------
    # 3. Authentication Latency (Argon2id password verification)
    # -------------------------------------------------------------
    print("[3/30] Measuring Authentication Latency...")
    from security_layer.password_hasher import PasswordHasher

    hasher = PasswordHasher()
    dummy_hash = hasher.hash("BenchmarkAdminPassword123!")
    auth_samples = []
    for _ in range(15):  # Argon2id is intentionally CPU-heavy
        t0 = time.perf_counter()
        ok, _ = hasher.verify("BenchmarkAdminPassword123!", dummy_hash)
        if ok:
            auth_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["3_authentication_latency_ms"] = compute_distribution(auth_samples)

    # -------------------------------------------------------------
    # 4. Camera Initialization / Open Latency
    # -------------------------------------------------------------
    print("[4/30] Measuring Camera Open Latency...")
    cam_open_samples = []
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    for _ in range(3):
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cam_open_samples.append((time.perf_counter() - t0) * 1000.0)
            cap.release()
        time.sleep(0.3)
    report["metrics"]["4_camera_init_latency_ms"] = compute_distribution(cam_open_samples)

    # Open persistent capture for frame capture benchmarks
    cap = cv2.VideoCapture(0, backend)
    sample_frame = None
    first_frame_latency = 0.0
    if cap.isOpened():
        t0 = time.perf_counter()
        ret, sample_frame = cap.read()
        first_frame_latency = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------
    # 5. First-frame Latency
    # -------------------------------------------------------------
    print("[5/30] Measuring First-Frame Latency...")
    report["metrics"]["5_first_frame_latency_ms"] = compute_distribution([first_frame_latency] if cap.isOpened() else [])

    # -------------------------------------------------------------
    # 6. Frame Capture Latency
    # -------------------------------------------------------------
    print("[6/30] Measuring Frame Capture Latency...")
    capture_samples = []
    if cap.isOpened():
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if ret and frame is not None:
                capture_samples.append((time.perf_counter() - t0) * 1000.0)
                sample_frame = frame
        cap.release()
    else:
        # Fallback synthetic frame if no physical camera
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    report["metrics"]["6_frame_capture_latency_ms"] = compute_distribution(capture_samples)

    # -------------------------------------------------------------
    # 7. JPEG Encode Latency
    # -------------------------------------------------------------
    print("[7/30] Measuring JPEG Encode Latency...")
    jpeg_encode_samples = []
    encoded_bytes = None
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", sample_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            jpeg_encode_samples.append((time.perf_counter() - t0) * 1000.0)
            encoded_bytes = buf
    report["metrics"]["7_jpeg_encode_latency_ms"] = compute_distribution(jpeg_encode_samples)

    # -------------------------------------------------------------
    # 8. JPEG Decode Latency
    # -------------------------------------------------------------
    print("[8/30] Measuring JPEG Decode Latency...")
    jpeg_decode_samples = []
    if encoded_bytes is not None:
        raw_bytes = encoded_bytes.tobytes()
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            jpeg_decode_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["8_jpeg_decode_latency_ms"] = compute_distribution(jpeg_decode_samples)

    # -------------------------------------------------------------
    # 9. Person Detection Latency
    # -------------------------------------------------------------
    print("[9/30] Measuring Person Detection Latency...")
    from pipeline.detection.person_detector import PersonDetector

    detector = PersonDetector()
    detect_samples = []
    for _ in range(min(15, num_iterations)):
        t0 = time.perf_counter()
        _ = detector.detect(sample_frame)
        detect_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["9_person_detection_latency_ms"] = compute_distribution(detect_samples)

    # -------------------------------------------------------------
    # 10. Tracking Latency
    # -------------------------------------------------------------
    print("[10/30] Measuring Tracking Latency...")
    from pipeline.tracking.tracker import PersonTracker

    tracker = PersonTracker()
    track_samples = []
    dummy_detections = [{"bbox": [100, 50, 300, 450], "confidence": 0.95, "class": "person"}]
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = tracker.update(dummy_detections, frame_shape=(480, 640, 3))
        track_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["10_tracking_latency_ms"] = compute_distribution(track_samples)

    # -------------------------------------------------------------
    # 11. Silhouette Extraction Latency
    # -------------------------------------------------------------
    print("[11/30] Measuring Silhouette Extraction Latency...")
    from pipeline.silhouette.extractor import SilhouetteExtractor

    sil_extractor = SilhouetteExtractor()
    crop = sample_frame[50:450, 100:300]
    if crop.size == 0:
        crop = np.zeros((400, 200, 3), dtype=np.uint8)
    sil_samples = []
    sample_sil = None
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        sil = sil_extractor.extract_from_crop(crop)
        sil_samples.append((time.perf_counter() - t0) * 1000.0)
        if sil is not None:
            sample_sil = sil
    if sample_sil is None:
        sample_sil = np.zeros((128, 64), dtype=np.uint8)
    report["metrics"]["11_silhouette_extraction_latency_ms"] = compute_distribution(sil_samples)

    # -------------------------------------------------------------
    # 12. GEI Accumulation Latency
    # -------------------------------------------------------------
    print("[12/30] Measuring GEI Accumulation Latency...")
    from preprocessing.gei_builder import GEIBuilder

    gei_builder = GEIBuilder(size=(64, 128))
    accum_samples = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        gei_builder.add_frame(sample_sil)
        accum_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["12_gei_accumulation_latency_ms"] = compute_distribution(accum_samples)

    # -------------------------------------------------------------
    # 13. GEI Generation Latency
    # -------------------------------------------------------------
    print("[13/30] Measuring GEI Generation Latency...")
    for _ in range(20):
        gei_builder.add_frame(sample_sil)
    gen_samples = []
    sample_gei = None
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        g = gei_builder.build()
        gen_samples.append((time.perf_counter() - t0) * 1000.0)
        if g is not None:
            sample_gei = g
    if sample_gei is None:
        sample_gei = np.zeros((128, 64), dtype=np.uint8)
    report["metrics"]["13_gei_generation_latency_ms"] = compute_distribution(gen_samples)

    # -------------------------------------------------------------
    # 14. ByGaitLight Inference Latency
    # -------------------------------------------------------------
    print("[14/30] Measuring ByGaitLight Inference Latency...")
    from pipeline.steps.feature_extraction import FeatureExtractionStep

    fe_step = FeatureExtractionStep()
    norm_sil = (sample_gei.astype(np.float32) / 255.0)
    bygait_samples = []
    sample_gait_emb = None
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        emb = fe_step.backend.predict(norm_sil)
        bygait_samples.append((time.perf_counter() - t0) * 1000.0)
        sample_gait_emb = emb
    report["metrics"]["14_bygait_light_inference_latency_ms"] = compute_distribution(bygait_samples)

    # -------------------------------------------------------------
    # 15. Gait Embedding Generation Latency (complete step)
    # -------------------------------------------------------------
    print("[15/30] Measuring Gait Embedding Generation Latency...")
    gait_emb_samples = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = fe_step.backend.predict(norm_sil).flatten().astype(np.float32)
        gait_emb_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["15_gait_embedding_latency_ms"] = compute_distribution(gait_emb_samples)

    # -------------------------------------------------------------
    # 16. OSNet Inference Latency
    # -------------------------------------------------------------
    print("[16/30] Measuring OSNet Inference Latency...")
    from intelligence.appearance_embedding import AppearanceEmbeddingExtractor

    app_extractor = AppearanceEmbeddingExtractor()
    osnet_samples = []
    sample_app_emb = None
    if app_extractor.backbone is not None:
        for _ in range(min(25, num_iterations)):
            t0 = time.perf_counter()
            emb = app_extractor.backbone.extract(crop)
            osnet_samples.append((time.perf_counter() - t0) * 1000.0)
            sample_app_emb = emb
    report["metrics"]["16_osnet_inference_latency_ms"] = compute_distribution(osnet_samples)

    # -------------------------------------------------------------
    # 17. Appearance Embedding Generation Latency
    # -------------------------------------------------------------
    print("[17/30] Measuring Appearance Embedding Latency...")
    app_emb_samples = []
    for _ in range(min(25, num_iterations)):
        t0 = time.perf_counter()
        _ = app_extractor.extract(crop, track_id=None)
        app_emb_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["17_appearance_embedding_latency_ms"] = compute_distribution(app_emb_samples)

    # -------------------------------------------------------------
    # 18. Gait VectorStore Search Latency
    # -------------------------------------------------------------
    print("[18/30] Measuring Gait VectorStore Search Latency...")
    from pipeline.steps.matching_step import MatchingStep

    gait_matcher = MatchingStep()
    gait_query = sample_gait_emb.flatten() if sample_gait_emb is not None else np.zeros((256,), dtype=np.float32)
    # Warm up cache
    _ = gait_matcher.match(gait_query, gs.gallery_features, gs.gallery_labels, gs.metadata)
    gait_search_samples = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = gait_matcher.match(
            gait_query,
            gs.gallery_features,
            gs.gallery_labels,
            gs.metadata,
        )
        gait_search_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["18_gait_vector_search_latency_ms"] = compute_distribution(gait_search_samples)

    # -------------------------------------------------------------
    # 19. Appearance VectorStore Search Latency
    # -------------------------------------------------------------
    print("[19/30] Measuring Appearance VectorStore Search Latency...")
    from pipeline.steps.reid_matching_step import ReIDMatchingStep

    app_matcher = ReIDMatchingStep()
    app_query = sample_app_emb.flatten() if sample_app_emb is not None else np.zeros((512,), dtype=np.float32)
    # Warm up cache
    _ = app_matcher.match(app_query, gs.appearance_gallery_features, gs.appearance_gallery_labels, gs.appearance_metadata)
    app_search_samples = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = app_matcher.match(
            app_query,
            gs.appearance_gallery_features,
            gs.appearance_gallery_labels,
            gs.appearance_metadata,
        )
        app_search_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["19_appearance_vector_search_latency_ms"] = compute_distribution(app_search_samples)

    # -------------------------------------------------------------
    # 20. Fusion Latency
    # -------------------------------------------------------------
    print("[20/30] Measuring Fusion Latency...")
    from intelligence.dual_modal_fusion import DualModalFusion

    fusion = DualModalFusion(enabled=True)
    # Warm up
    _ = fusion.fuse(
        gait_score=0.88,
        reid_score=0.82,
        crop=crop,
        gei_frame_count=15,
        gei=sample_gei,
        confidence=0.88,
        gait_embedding=gait_query,
        gait_gallery_embedding=gait_query,
        reid_embedding=app_query,
        reid_gallery_embedding=app_query,
    )
    fusion_samples = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = fusion.fuse(
            gait_score=0.88,
            reid_score=0.82,
            crop=crop,
            gei_frame_count=15,
            gei=sample_gei,
            confidence=0.88,
            gait_embedding=gait_query,
            gait_gallery_embedding=gait_query,
            reid_embedding=app_query,
            reid_gallery_embedding=app_query,
        )
        fusion_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["20_fusion_latency_ms"] = compute_distribution(fusion_samples)

    # -------------------------------------------------------------
    # 21. Complete Frame-to-Identity Latency
    # -------------------------------------------------------------
    print("[21/30] Measuring Complete Frame-to-Identity Latency...")
    ok, buf = cv2.imencode(".jpg", sample_frame)
    jpeg_bytes = buf.tobytes()
    # Warm up pipeline
    _ = gs.process_image_bytes(jpeg_bytes, camera_id="benchmark-cam")
    frame_to_id_samples = []
    for _ in range(min(20, num_iterations)):
        t0 = time.perf_counter()
        _ = gs.process_image_bytes(jpeg_bytes, camera_id="benchmark-cam")
        frame_to_id_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["21_frame_to_identity_latency_ms"] = compute_distribution(frame_to_id_samples)

    # -------------------------------------------------------------
    # 22. End-to-End Camera-to-Identity Latency
    # -------------------------------------------------------------
    print("[22/30] Measuring End-to-End Camera-to-Identity Latency...")
    cap = cv2.VideoCapture(0, backend)
    e2e_samples = []
    if cap.isOpened():
        for _ in range(min(15, num_iterations)):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if ret and frame is not None:
                ok, b = cv2.imencode(".jpg", frame)
                if ok:
                    _ = gs.process_image_bytes(b.tobytes(), camera_id="benchmark-cam")
                    e2e_samples.append((time.perf_counter() - t0) * 1000.0)
        cap.release()
    report["metrics"]["22_e2e_camera_to_identity_latency_ms"] = compute_distribution(e2e_samples)

    # -------------------------------------------------------------
    # 23. CPU Utilization
    # -------------------------------------------------------------
    print("[23/30] Measuring CPU Utilization...")
    cpu_percent = psutil.cpu_percent(interval=0.5)
    report["metrics"]["23_cpu_utilization_percent"] = {"mean": cpu_percent, "unit": "%"}

    # -------------------------------------------------------------
    # 24. RAM Usage
    # -------------------------------------------------------------
    print("[24/30] Measuring RAM Usage...")
    proc = psutil.Process(os.getpid())
    ram_mb = proc.memory_info().rss / (1024 * 1024)
    report["metrics"]["24_ram_usage_mb"] = {"mean": round(ram_mb, 2), "unit": "MB"}

    # -------------------------------------------------------------
    # 25. GPU Utilization & 26. GPU VRAM Usage
    # -------------------------------------------------------------
    print("[25-26/30] Measuring GPU Utilization & VRAM...")
    gpu_util = 0.0
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    report["metrics"]["25_gpu_utilization_percent"] = {"mean": gpu_util, "unit": "%"}
    report["metrics"]["26_gpu_vram_usage_mb"] = {"mean": round(vram_mb, 2), "unit": "MB"}

    # -------------------------------------------------------------
    # 27. FPS & 28. Frames Processed Per Second
    # -------------------------------------------------------------
    print("[27-28/30] Measuring Throughput FPS...")
    mean_e2e_ms = report["metrics"]["21_frame_to_identity_latency_ms"]["mean"]
    calc_fps = round(1000.0 / mean_e2e_ms, 2) if mean_e2e_ms > 0 else 0.0
    report["metrics"]["27_theoretical_fps"] = {"mean": calc_fps, "unit": "FPS"}
    report["metrics"]["28_frames_processed_per_sec"] = {"mean": calc_fps, "unit": "FPS"}

    # -------------------------------------------------------------
    # 29. Queue / Wait Time
    # -------------------------------------------------------------
    print("[29/30] Measuring Queue / Wait Time...")
    from queue import Queue

    q = Queue(maxsize=10)
    q_samples = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        q.put(sample_frame, block=False)
        _ = q.get(block=False)
        q_samples.append((time.perf_counter() - t0) * 1000.0)
    report["metrics"]["29_queue_wait_time_ms"] = compute_distribution(q_samples)

    # -------------------------------------------------------------
    # 30. Firebase / Background Synchronization Latency
    # -------------------------------------------------------------
    print("[30/30] Measuring Firebase Background Sync Latency...")
    from storage.firebase_embedding_store import FirebaseEmbeddingStore

    fb_store = FirebaseEmbeddingStore()
    t0 = time.perf_counter()
    _, fb_diag = fb_store.check_connection_health()
    fb_latency_ms = (time.perf_counter() - t0) * 1000.0
    report["metrics"]["30_firebase_sync_latency_ms"] = {
        "mean": round(fb_latency_ms, 2),
        "status": fb_diag.get("status", "UNKNOWN"),
        "mode": fb_diag.get("mode", "UNKNOWN"),
    }

    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print(f"BENCHMARK COMPLETED — Results saved to {output_path}")
    print("=" * 80)
    return report


def print_comparison(baseline_path: str, optimized_path: str) -> None:
    if not os.path.exists(baseline_path) or not os.path.exists(optimized_path):
        print("Comparison files missing.")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        b_data = json.load(f)
    with open(optimized_path, "r", encoding="utf-8") as f:
        o_data = json.load(f)

    b_metrics = b_data.get("metrics", {})
    o_metrics = o_data.get("metrics", {})

    print("\n" + "=" * 95)
    print(f"{'METRIC':<45} | {'BEFORE (mean)':<15} | {'AFTER (mean)':<15} | {'SPEEDUP':<10}")
    print("=" * 95)

    speedups = []
    for k in sorted(b_metrics.keys()):
        b_val = b_metrics.get(k, {}).get("mean")
        o_val = o_metrics.get(k, {}).get("mean")
        if b_val is None or o_val is None:
            continue

        unit = b_metrics.get(k, {}).get("unit", "ms")
        if unit == "ms" and o_val > 0 and b_val > 0:
            speedup = b_val / o_val
            speedups.append((k, speedup))
            sp_str = f"{speedup:.2f}x"
            if speedup >= 10.0:
                sp_str += " (10x+)"
            elif speedup >= 5.0:
                sp_str += " (5x+)"
            elif speedup >= 2.0:
                sp_str += " (2x+)"
            print(f"{k:<45} | {b_val:>11.4f} ms | {o_val:>11.4f} ms | {sp_str:>10}")
        elif unit in ("FPS", "%", "MB"):
            print(f"{k:<45} | {b_val:>11.2f} {unit} | {o_val:>11.2f} {unit} | {'N/A':>10}")

    print("=" * 95)
    ten_x = [k for k, sp in speedups if sp >= 10.0]
    five_x = [k for k, sp in speedups if 5.0 <= sp < 10.0]
    two_x = [k for k, sp in speedups if 2.0 <= sp < 5.0]
    one_x = [k for k, sp in speedups if sp < 2.0]

    print(f"Summary of Latency Optimizations ({len(speedups)} latency metrics evaluated):")
    print(f"  - 10x+ Speedup achieved: {len(ten_x)} metrics ({', '.join([k.split('_')[1] for k in ten_x]) or 'None'})")
    print(f"  - 5x-9.9x Speedup:       {len(five_x)} metrics ({', '.join([k.split('_')[1] for k in five_x]) or 'None'})")
    print(f"  - 2x-4.9x Speedup:       {len(two_x)} metrics ({', '.join([k.split('_')[1] for k in two_x]) or 'None'})")
    print(f"  - <2x or Baseline-bound: {len(one_x)} metrics")
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGUS AI Benchmark Suite")
    parser.add_argument("--iterations", type=int, default=30, help="Iterations for microbenchmarks")
    parser.add_argument("--output", type=str, default="outputs/reports/baseline_performance.json")
    parser.add_argument("--compare", type=str, default=None, help="Path to baseline report to compare against")
    args = parser.parse_args()
    if args.compare:
        run_benchmark(num_iterations=args.iterations, output_path=args.output)
        print_comparison(args.compare, args.output)
    else:
        run_benchmark(num_iterations=args.iterations, output_path=args.output)
