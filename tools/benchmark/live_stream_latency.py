import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from services.camera_source_resolver import CameraSourceResolver
from services.gait_service import GaitService


def measure_isolated_stages() -> dict[str, Any]:
    print("=" * 70)
    print("STAGE-BY-STAGE ISOLATED LATENCY PROFILING")
    print("=" * 70)
    results: dict[str, Any] = {}


    resolver = CameraSourceResolver()
    t0 = time.perf_counter()
    probe_ok = resolver.probe_usb_webcam(0)
    t1 = time.perf_counter()
    results["probe_usb_0_duration_s"] = t1 - t0
    results["probe_usb_0_success"] = probe_ok
    print(f"  Stage D-E: probe_usb_webcam(0) [Open+Read+Close] : {t1 - t0:7.4f} s (success={probe_ok})")


    results["index_probe_breakdown"] = {}
    for idx in range(4):
        t_start = time.perf_counter()
        if sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)
        is_opened = cap.isOpened()
        ret, frame = cap.read() if is_opened else (False, None)
        cap.release()
        t_end = time.perf_counter()
        dur = t_end - t_start
        results["index_probe_breakdown"][f"index_{idx}"] = {
            "duration_s": dur,
            "opened": is_opened,
            "read_success": bool(ret),
        }
        print(f"    - Index {idx} DirectShow probe : {dur:7.4f} s (opened={is_opened}, read={ret})")


    t0 = time.perf_counter()
    res = resolver.resolve_source(camera_id="BENCH-PROBE", requested_source="auto")
    t1 = time.perf_counter()
    resolver.release_source_by_camera_id("BENCH-PROBE")
    results["resolver_full_auto_s"] = t1 - t0
    results["resolved_source"] = res.get("resolved_source")
    print(f"  Stage D: resolver.resolve_source('auto')          : {t1 - t0:7.4f} s (source={res.get('resolved_source')})")


    t0 = time.perf_counter()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
    t1 = time.perf_counter()
    is_opened = cap.isOpened()
    results["vcap_open_s"] = t1 - t0
    results["vcap_opened"] = is_opened
    print(f"  Stage E: VideoCapture(0, CAP_DSHOW) open         : {t1 - t0:7.4f} s (opened={is_opened})")


    t0 = time.perf_counter()
    ret, frame = cap.read() if is_opened else (False, None)
    t1 = time.perf_counter()
    results["first_read_s"] = t1 - t0
    results["first_read_ok"] = bool(ret)
    frame_shape = list(frame.shape) if frame is not None else None
    results["frame_shape"] = frame_shape
    print(f"  Stage F-G: VideoCapture.read() (first frame)     : {t1 - t0:7.4f} s (read={ret}, shape={frame_shape})")


    if frame is not None:
        t0 = time.perf_counter()
        resized = cv2.resize(frame, (640, 480))
        enc_ok, enc_buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        t1 = time.perf_counter()
        results["jpeg_encode_s"] = t1 - t0
        results["jpeg_size_bytes"] = len(enc_buf) if enc_ok and enc_buf is not None else 0
        print(f"  Stage H: cv2.resize + JPEG encode (quality=75)   : {t1 - t0:7.4f} s (bytes={results['jpeg_size_bytes']})")
    cap.release()

    return results


def run_single_end_to_end_start(run_index: int, service: GaitService, camera_id: str = "CCTV-BENCH") -> dict[str, float]:
    gc.collect()
    time.sleep(0.5)

    t_start_click = time.perf_counter()


    t_api_received = time.perf_counter()
    service.start_camera(
        camera_id=camera_id,
        source="auto",
        location="Forensic Benchmark Zone",
    )
    t_api_completed = time.perf_counter()

    worker = service.get_camera_worker(camera_id)
    if not worker:
        raise RuntimeError("Worker not created by start_camera")


    t_stream_request = time.perf_counter()
    first_jpeg = None
    stream_wait_deadline = time.perf_counter() + 10.0
    while time.perf_counter() < stream_wait_deadline:
        first_jpeg = worker.get_latest_jpeg()
        if first_jpeg is not None and len(first_jpeg) > 0:
            break
        time.sleep(0.005)

    t_first_frame_received = time.perf_counter()


    if first_jpeg is None or len(first_jpeg) == 0:
        raise RuntimeError("Failed to receive first valid frame from camera worker")


    nparr = np.frombuffer(first_jpeg, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError("First frame JPEG is corrupt or invalid")


    service.stop_camera(camera_id)
    time.sleep(0.2)

    total_e2e = t_first_frame_received - t_start_click
    api_duration = t_api_completed - t_api_received
    stream_to_frame = t_first_frame_received - t_stream_request

    print(
        f"  Run {run_index}: Total E2E = {total_e2e:7.4f} s | "
        f"start_camera() API = {api_duration:7.4f} s | "
        f"Stream-to-Frame = {stream_to_frame:7.4f} s | "
        f"Frame: {img.shape[1]}x{img.shape[0]} ({len(first_jpeg)} bytes)"
    )

    return {
        "run": run_index,
        "total_e2e_s": total_e2e,
        "api_duration_s": api_duration,
        "stream_to_frame_s": stream_to_frame,
        "jpeg_bytes": len(first_jpeg),
        "frame_width": img.shape[1],
        "frame_height": img.shape[0],
    }


def run_benchmark(num_runs: int = 5) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"BENCHMARK: {num_runs} REPEATED END-TO-END START STREAM RUNS")
    print("=" * 70)


    isolated = measure_isolated_stages()


    service = GaitService()
    runs_data = []

    for i in range(1, num_runs + 1):
        try:
            run_res = run_single_end_to_end_start(i, service, camera_id="CCTV-BENCH")
            runs_data.append(run_res)
        except (RuntimeError, ValueError, OSError, TimeoutError, cv2.error) as exc:
            print(f"  Run {i} FAILED: {exc}")

    if not runs_data:
        raise RuntimeError("All benchmark runs failed!")

    e2e_times = [r["total_e2e_s"] for r in runs_data]
    api_times = [r["api_duration_s"] for r in runs_data]

    stats = {
        "num_runs": len(runs_data),
        "min_s": min(e2e_times),
        "max_s": max(e2e_times),
        "mean_s": statistics.mean(e2e_times),
        "median_s": statistics.median(e2e_times),
        "stdev_s": statistics.stdev(e2e_times) if len(e2e_times) > 1 else 0.0,
        "api_mean_s": statistics.mean(api_times),
        "runs": runs_data,
        "isolated_stages": isolated,
    }

    print("-" * 70)
    print("BENCHMARK SUMMARY (START_STREAM -> FIRST_VALID_FRAME_VISIBLE):")
    print(f"  Minimum : {stats['min_s']:7.4f} s")
    print(f"  Maximum : {stats['max_s']:7.4f} s")
    print(f"  Mean    : {stats['mean_s']:7.4f} s")
    print(f"  Median  : {stats['median_s']:7.4f} s")
    print(f"  StdDev  : {stats['stdev_s']:7.4f} s")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    benchmark_stats = run_benchmark(num_runs=5)
    output_path = Path("baseline_stream_startup_benchmark.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_stats, f, indent=2)
    print(f"Saved baseline benchmark results to {output_path.resolve()}")
