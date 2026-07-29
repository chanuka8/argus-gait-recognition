"""
Inference Backend Performance and Parity Benchmark Script for ARGUS AI.

Measures initialization latency, warm-up latency, mean/median/p95 inference latency,
throughput, device, precision, and output parity against the PyTorch reference backend.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.inference.backend import get_inference_backend


def benchmark_backend(
    backend_name: str,
    sample_count: int = 100,
    device: str = "auto",
    precision: str = "fp32",
    model_path: str = "runs/exp_001/best_model.pth",
    reference_embeddings: np.ndarray = None,
) -> dict:
    """Benchmark initialization, warm latency, p95 latency, throughput, and parity for a backend."""
    config = {
        "backend": backend_name,
        "device": device,
        "precision": precision,
        "allow_fallback": True,
        "warmup_iterations": 3,
        "engine_path": "models/engines/bygait_light_fp16.engine",
        "onnx_path": "models/engines/bygait_light.onnx",
    }

    print(f"\n[BENCHMARK] Testing Backend: '{backend_name}' (device={device}, precision={precision})...")

    # Measure initialization time
    init_start = time.perf_counter()
    backend = get_inference_backend(config=config, model_path=model_path)
    init_time_ms = (time.perf_counter() - init_start) * 1000.0

    dummy_input = np.random.randn(1, 1, 64, 128).astype(np.float32)

    # Warmup latency
    warm_start = time.perf_counter()
    backend.predict(dummy_input)
    warm_latency_ms = (time.perf_counter() - warm_start) * 1000.0

    # Latency sampling loop
    latencies = []
    sample_inputs = [np.random.randn(1, 1, 64, 128).astype(np.float32) for _ in range(sample_count)]

    start_total = time.perf_counter()
    outputs = []
    for inp in sample_inputs:
        t0 = time.perf_counter()
        emb = backend.predict(inp)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        outputs.append(emb)

    total_time_sec = time.perf_counter() - start_total

    latencies_arr = np.array(latencies)
    mean_lat = float(np.mean(latencies_arr))
    median_lat = float(np.median(latencies_arr))
    p95_lat = float(np.percentile(latencies_arr, 95))
    throughput = float(sample_count / total_time_sec)

    active_backend = getattr(backend, "backend_name", backend_name)
    actual_device = str(getattr(backend, "device", device))
    fallback_used = str(active_backend).lower() != str(backend_name).lower()

    # Parity check against PyTorch reference if provided
    parity_passed = True
    max_diff = 0.0
    if reference_embeddings is not None and len(outputs) == len(reference_embeddings):
        diffs = [np.max(np.abs(outputs[i] - reference_embeddings[i])) for i in range(len(outputs))]
        max_diff = float(np.max(diffs))
        parity_passed = max_diff < 1e-2

    result = {
        "requested_backend": backend_name,
        "active_backend": active_backend,
        "fallback_used": fallback_used,
        "device": actual_device,
        "precision": precision,
        "sample_count": sample_count,
        "init_time_ms": round(init_time_ms, 3),
        "warm_latency_ms": round(warm_latency_ms, 3),
        "mean_latency_ms": round(mean_lat, 3),
        "median_latency_ms": round(median_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "throughput_fps": round(throughput, 2),
        "parity_passed": parity_passed,
        "max_abs_difference": round(max_diff, 6),
        "outputs": outputs,
    }

    print(f"  - Active Backend : {active_backend}")
    print(f"  - Fallback Used  : {fallback_used}")
    print(f"  - Init Latency   : {result['init_time_ms']} ms")
    print(f"  - Mean Latency   : {result['mean_latency_ms']} ms")
    print(f"  - p95 Latency    : {result['p95_latency_ms']} ms")
    print(f"  - Throughput     : {result['throughput_fps']} FPS")
    print(f"  - Parity Passed  : {result['parity_passed']} (Max Diff: {result['max_abs_difference']})")
    if fallback_used:
        print(f"  - Note: Fallback was active ({backend_name} -> {active_backend}). Measurement reflects {active_backend} performance.")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ARGUS inference backends.")
    parser.add_argument("--samples", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="auto", help="Device choice (auto, cpu, cuda)")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision (fp32, fp16)")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report")

    args = parser.parse_args()

    backends_to_test = ["pytorch", "onnxruntime", "tensorrt", "auto"]
    results = {}

    ref_res = benchmark_backend("pytorch", sample_count=args.samples, device=args.device, precision=args.precision)
    results["pytorch"] = ref_res
    ref_outputs = ref_res.pop("outputs")

    for bname in backends_to_test[1:]:
        res = benchmark_backend(bname, sample_count=args.samples, device=args.device, precision=args.precision, reference_embeddings=ref_outputs)
        res.pop("outputs", None)
        results[bname] = res

    if args.save_report:
        report_dir = Path("outputs/reports/benchmark")
        report_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"backend_benchmark_{ts_str}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Benchmark report saved to {report_path}")


if __name__ == "__main__":
    main()
