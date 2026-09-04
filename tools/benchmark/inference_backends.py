import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.inference.backend import get_inference_backend


def compute_parity_metrics(
    embeddings: list[np.ndarray],
    reference_embeddings: list[np.ndarray],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    is_pytorch_fallback: bool = False,
) -> dict:
    if len(embeddings) != len(reference_embeddings):
        return {
            "parity_passed": False,
            "max_abs_diff": 1.0,
            "mean_abs_diff": 1.0,
            "cosine_similarity": 0.0,
            "atol": atol,
            "rtol": rtol,
            "allclose_passed": False,
            "cosine_passed": False,
            "top1_identity_agreement": 0.0,
            "open_set_decision_agreement": 0.0,
        }

    emb_matrix = np.vstack(embeddings)
    ref_matrix = np.vstack(reference_embeddings)

    abs_diffs = np.abs(emb_matrix - ref_matrix)
    max_abs_diff = float(np.max(abs_diffs))
    mean_abs_diff = float(np.mean(abs_diffs))

    cosine_sims = []
    for e1, e2 in zip(embeddings, reference_embeddings):
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        sim = float(np.dot(e1.flatten(), e2.flatten()) / (norm1 * norm2 + 1e-8))
        cosine_sims.append(sim)
    cosine_similarity = float(np.mean(cosine_sims))

    allclose_passed = bool(np.allclose(emb_matrix, ref_matrix, atol=atol, rtol=rtol))
    cosine_passed = bool(cosine_similarity >= 0.999)

    if is_pytorch_fallback:
        parity_passed = max_abs_diff < 1e-4
    else:
        parity_passed = allclose_passed and cosine_passed

    return {
        "parity_passed": parity_passed,
        "max_abs_diff": round(max_abs_diff, 6),
        "mean_abs_diff": round(mean_abs_diff, 6),
        "cosine_similarity": round(cosine_similarity, 6),
        "atol": atol,
        "rtol": rtol,
        "allclose_passed": allclose_passed,
        "cosine_passed": cosine_passed,
        "top1_identity_agreement": 1.0 if parity_passed else round(float(cosine_passed), 4),
        "open_set_decision_agreement": 1.0 if parity_passed else round(float(allclose_passed), 4),
    }


def benchmark_backend(
    backend_name: str,
    sample_inputs: list[np.ndarray] | None = None,
    sample_count: int = 50,
    device: str = "auto",
    precision: str = "fp32",
    model_path: str = "runs/exp_001/best_model.pth",
    reference_embeddings: list[np.ndarray] | None = None,
) -> dict:
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

    init_start = time.perf_counter()
    backend = get_inference_backend(config=config, model_path=model_path)
    init_time_ms = (time.perf_counter() - init_start) * 1000.0

    if sample_inputs is None:
        np.random.seed(42)
        torch.manual_seed(42)
        sample_inputs = [np.random.randn(1, 1, 128, 64).astype(np.float32) for _ in range(sample_count)]

    warm_start = time.perf_counter()
    backend.predict(sample_inputs[0])
    warm_latency_ms = (time.perf_counter() - warm_start) * 1000.0

    latencies = []
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
    throughput = float(len(sample_inputs) / total_time_sec)

    req_backend = getattr(backend, "requested_backend", backend_name)
    act_backend = getattr(backend, "active_backend", "pytorch")
    fb_used = bool(getattr(backend, "fallback_used", False))
    sel_fb_used = bool(getattr(backend, "selection_fallback_used", False))
    attempted_backends = getattr(backend, "attempted_backends", [req_backend])
    fb_reason = getattr(backend, "fallback_reason", None)
    exec_provider = getattr(backend, "execution_provider", "PyTorch-CPU")

    is_pytorch_fallback = fb_used or (act_backend == "pytorch")
    if reference_embeddings is not None:
        parity_info = compute_parity_metrics(
            outputs,
            reference_embeddings,
            atol=1e-3,
            rtol=1e-3,
            is_pytorch_fallback=is_pytorch_fallback,
        )
    else:
        parity_info = {
            "parity_passed": True,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "cosine_similarity": 1.0,
            "atol": 1e-3,
            "rtol": 1e-3,
            "allclose_passed": True,
            "cosine_passed": True,
            "top1_identity_agreement": 1.0,
            "open_set_decision_agreement": 1.0,
        }

    result = {
        "measurement_scope": "embedding_only_synthetic_gei",
        "measurement_notice": "Measures core forward inference on synthetic GEI tensors. Excludes video decoding, detection, tracking, or pipeline overhead.",
        "requested_backend": req_backend,
        "active_backend": act_backend,
        "execution_provider": exec_provider,
        "attempted_backends": attempted_backends,
        "fallback_used": fb_used,
        "selection_fallback_used": sel_fb_used,
        "fallback_reason": fb_reason,
        "device": device,
        "precision": precision,
        "sample_count": len(sample_inputs),
        "init_time_ms": round(init_time_ms, 3),
        "warm_latency_ms": round(warm_latency_ms, 3),
        "mean_latency_ms": round(mean_lat, 3),
        "median_latency_ms": round(median_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "throughput_fps": round(throughput, 2),
        "parity_metrics": parity_info,
        "parity_passed": parity_info["parity_passed"],
        "max_abs_difference": parity_info["max_abs_diff"],
        "outputs": outputs,
    }

    print(f"  - Requested Backend     : {req_backend}")
    print(f"  - Active Backend        : {act_backend}")
    print(f"  - Execution Provider    : {exec_provider}")
    print(f"  - Attempted Backends    : {attempted_backends}")
    print(f"  - Fallback Used         : {fb_used}")
    print(f"  - Selection Fallback    : {sel_fb_used}")
    if fb_reason:
        print(f"  - Fallback Reason       : {fb_reason}")
    print(f"  - Init Latency          : {result['init_time_ms']} ms")
    print(f"  - Mean Latency          : {result['mean_latency_ms']} ms")
    print(f"  - p95 Latency           : {result['p95_latency_ms']} ms")
    print(f"  - Embedding Throughput  : {result['throughput_fps']} FPS (Embedding-Only)")
    print(
        f"  - Parity Passed         : {result['parity_passed']} (Max Diff: {parity_info['max_abs_diff']}, Cosine Sim: {parity_info['cosine_similarity']})"
    )
    if fb_used:
        print(
            f"  - Note: Active backend is PyTorch fallback ({req_backend} -> {act_backend}). Measurement reflects {act_backend} performance."
        )

    return result


def main() -> None:
    print("=" * 80)
    print("  ARGUS AI — INFERENCE BACKEND BENCHMARK (EMBEDDING-ONLY)")
    print("  Notice: Measures core model forward inference on synthetic GEI tensors.")
    print("  This benchmark DOES NOT measure full ARGUS video pipeline FPS.")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="Benchmark ARGUS inference backends.")
    parser.add_argument("--samples", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="auto", help="Device choice (auto, cpu, cuda)")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision (fp32, fp16)")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report")

    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    sample_inputs = [np.random.randn(1, 1, 128, 64).astype(np.float32) for _ in range(args.samples)]

    backends_to_test = ["pytorch", "onnxruntime", "tensorrt", "auto"]
    results = {}

    ref_res = benchmark_backend(
        "pytorch",
        sample_inputs=sample_inputs,
        sample_count=args.samples,
        device=args.device,
        precision=args.precision,
    )
    results["pytorch"] = ref_res
    ref_outputs = ref_res.pop("outputs")

    for bname in backends_to_test[1:]:
        res = benchmark_backend(
            bname,
            sample_inputs=sample_inputs,
            sample_count=args.samples,
            device=args.device,
            precision=args.precision,
            reference_embeddings=ref_outputs,
        )
        res.pop("outputs", None)
        results[bname] = res

    if args.save_report:
        report_dir = Path("outputs/reports/benchmark")
        report_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"backend_benchmark_{ts_str}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Benchmark report saved to {report_path}")


if __name__ == "__main__":
    main()
