"""
ARGUS AI — Complete GPU Validation Audit
=========================================
Verifies actual GPU device placement, inference correctness, latency,
VRAM usage, GPU utilization, and numerical consistency for both
ByGaitLight and OSNet production inference paths.

DOES NOT modify architecture. READ + BENCHMARK only.
"""

import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def section(title: str) -> None:
    print(f"\n{'=' * 76}")
    print(f"  {title}")
    print(f"{'=' * 76}")


def nvidia_smi_gpu_util() -> dict:
    """Query nvidia-smi for real GPU utilization. Returns dict or error string."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "gpu_utilization_percent": float(parts[0]),
                    "memory_utilization_percent": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                }
        return {"error": f"nvidia-smi returned code {result.returncode}: {result.stderr.strip()}"}
    except FileNotFoundError:
        return {"error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out"}


def sample_gpu_utilization_during(func, duration_hint_sec=2.0) -> dict:
    """Run func() in a thread while sampling nvidia-smi utilization."""
    samples = []
    stop_event = threading.Event()

    def sampler():
        while not stop_event.is_set():
            s = nvidia_smi_gpu_util()
            if "error" not in s:
                samples.append(s)
            time.sleep(0.3)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    func()
    stop_event.set()
    t.join(timeout=2)

    if not samples:
        return {"error": "GPU utilization measurement unavailable (no samples collected)"}

    gpu_utils = [s["gpu_utilization_percent"] for s in samples]
    return {
        "samples": len(gpu_utils),
        "mean_gpu_percent": round(np.mean(gpu_utils), 1),
        "max_gpu_percent": round(max(gpu_utils), 1),
        "min_gpu_percent": round(min(gpu_utils), 1),
    }


def compute_stats(samples: list[float]) -> dict:
    if not samples:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "n": 0}
    arr = np.array(samples)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "n": len(samples),
    }


def main() -> None:
    report = {}

    print("ARGUS AI — COMPLETE GPU VALIDATION AUDIT")
    print("=" * 76)

    # ==================================================================
    # A. ENVIRONMENT
    # ==================================================================
    section("A. ENVIRONMENT")
    env = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
        "python_version": sys.version.split()[0],
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = f"{torch.cuda.get_device_capability(0)}"
        env["gpu_memory_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1)
    else:
        env["gpu_name"] = "UNAVAILABLE"

    for k, v in env.items():
        print(f"  {k:30s} = {v}")
    report["A_environment"] = env

    if not torch.cuda.is_available():
        print("\n  *** CUDA NOT AVAILABLE — ABORTING AUDIT ***")
        report["verdict"] = "GPU READY: NO (CUDA not available)"
        _save_report(report)
        return

    # ==================================================================
    # B. MODEL DEVICE PLACEMENT — ByGaitLight
    # ==================================================================
    section("B. BYGAITLIGHT — DEVICE PLACEMENT")
    from automation.device_manager import DeviceManager
    dm = DeviceManager.get_instance()
    print(f"  DeviceManager.device        = {dm.device}")
    print(f"  DeviceManager.is_cuda       = {dm.is_cuda}")
    print(f"  resolve_component('auto')   = {dm.resolve_component_device('auto')}")

    from pipeline.steps.feature_extraction import FeatureExtractionStep
    fe_step = FeatureExtractionStep()
    bygait_backend = fe_step.backend

    print(f"  Backend class               = {type(bygait_backend).__name__}")
    print(f"  backend.device              = {bygait_backend.device}")
    print(f"  backend.device.type         = {bygait_backend.device.type}")
    print(f"  backend.execution_provider  = {bygait_backend.execution_provider}")

    bygait_model = bygait_backend.model
    param_device = next(bygait_model.parameters()).device
    is_eval = not bygait_model.training
    print(f"  next(params).device         = {param_device}")
    print(f"  model.training              = {bygait_model.training}")
    print(f"  model.eval()                = {is_eval}")

    bygait_on_cuda = param_device.type == "cuda"
    report["B_bygaitlight"] = {
        "backend_device": str(bygait_backend.device),
        "param_device": str(param_device),
        "on_cuda": bygait_on_cuda,
        "eval_mode": is_eval,
    }

    # ==================================================================
    # B2. MODEL DEVICE PLACEMENT — OSNet
    # ==================================================================
    section("B2. OSNET — DEVICE PLACEMENT")
    from models.reid.osnet_backbone import OSNetBackbone
    osnet = OSNetBackbone()
    print(f"  osnet.device                = {osnet.device}")
    print(f"  osnet.device.type           = {osnet.device.type}")

    osnet_model = osnet._ensure_model()
    osnet_param_device = next(osnet_model.parameters()).device
    osnet_eval = not osnet_model.training
    print(f"  next(params).device         = {osnet_param_device}")
    print(f"  model.training              = {osnet_model.training}")
    print(f"  model.eval()                = {osnet_eval}")
    print(f"  _mean.device                = {osnet._mean.device}")
    print(f"  _std.device                 = {osnet._std.device}")

    osnet_on_cuda = osnet_param_device.type == "cuda"
    report["B2_osnet"] = {
        "osnet_device": str(osnet.device),
        "param_device": str(osnet_param_device),
        "on_cuda": osnet_on_cuda,
        "eval_mode": osnet_eval,
        "mean_device": str(osnet._mean.device),
        "std_device": str(osnet._std.device),
    }

    # ==================================================================
    # C. TENSOR DEVICE PLACEMENT TRACE
    # ==================================================================
    section("C. TENSOR DEVICE TRACE — ByGaitLight")
    sample_gei = np.random.rand(128, 64).astype(np.float32)
    tensor = torch.from_numpy(sample_gei).float()
    print(f"  After from_numpy:            {tensor.device}")
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    print(f"  After unsqueeze:             {tensor.device}")
    tensor_dev = tensor.to(bygait_backend.device, non_blocking=True)
    print(f"  After .to(backend.device):   {tensor_dev.device}")

    with torch.inference_mode():
        output = bygait_model(tensor_dev)
        print(f"  Model output device:         {output.device}")
        print(f"  Model output shape:          {output.shape}")
        out_cpu = output.cpu().numpy()
        print(f"  After .cpu().numpy() shape:  {out_cpu.shape}")

    report["C_tensor_trace_bygait"] = {
        "input_device_initial": "cpu",
        "input_device_after_to": str(tensor_dev.device),
        "output_device": str(output.device),
        "output_shape": list(output.shape),
    }

    section("C2. TENSOR DEVICE TRACE — OSNet")
    sample_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(sample_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 256))
    t_osnet = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(osnet.device, non_blocking=True) / 255.0
    print(f"  After preprocess + .to():    {t_osnet.device}")
    t_osnet = (t_osnet - osnet._mean) / osnet._std
    print(f"  After normalization:         {t_osnet.device}")

    with torch.inference_mode():
        out_osnet = osnet_model(t_osnet)
        normed = F.normalize(out_osnet, p=2, dim=-1)
        print(f"  Model output device:         {out_osnet.device}")
        print(f"  Model output shape:          {out_osnet.shape}")
        out_osnet_np = normed.squeeze(0).cpu().numpy()
        print(f"  After normalize+cpu shape:   {out_osnet_np.shape}")

    report["C2_tensor_trace_osnet"] = {
        "input_device_after_to": str(t_osnet.device),
        "output_device": str(out_osnet.device),
        "output_shape": list(out_osnet.shape),
    }

    # ==================================================================
    # D. GPU SMOKE TEST
    # ==================================================================
    section("D. GPU SMOKE TEST")
    torch.cuda.reset_peak_memory_stats(0)
    vram_before = torch.cuda.memory_allocated(0) / 1024 / 1024

    # ByGaitLight smoke
    gei_input = torch.randn(1, 1, 128, 64, device=bygait_backend.device)
    with torch.inference_mode():
        smoke_out = bygait_model(gei_input)
    torch.cuda.synchronize()
    print(f"  ByGaitLight smoke: input={gei_input.device} output={smoke_out.device} shape={smoke_out.shape}")
    assert smoke_out.device.type == "cuda" if bygait_on_cuda else True, "ByGaitLight output not on expected device"

    # OSNet smoke
    osnet_input = torch.randn(1, 3, 256, 128, device=osnet.device)
    osnet_input = (osnet_input - osnet._mean) / osnet._std
    with torch.inference_mode():
        smoke_osnet = osnet_model(osnet_input)
    torch.cuda.synchronize()
    print(f"  OSNet smoke:       input={osnet_input.device} output={smoke_osnet.device} shape={smoke_osnet.shape}")

    vram_after = torch.cuda.memory_allocated(0) / 1024 / 1024
    peak_vram = torch.cuda.max_memory_allocated(0) / 1024 / 1024
    print(f"  VRAM before smoke: {vram_before:.2f} MB")
    print(f"  VRAM after smoke:  {vram_after:.2f} MB")
    print(f"  VRAM peak:         {peak_vram:.2f} MB")

    report["D_smoke_test"] = {
        "bygait_ok": smoke_out.shape[-1] > 0,
        "osnet_ok": smoke_osnet.shape[-1] > 0,
        "vram_before_mb": round(vram_before, 2),
        "vram_after_mb": round(vram_after, 2),
        "vram_peak_mb": round(peak_vram, 2),
    }

    # ==================================================================
    # E. CPU vs GPU LATENCY — ByGaitLight
    # ==================================================================
    section("E. LATENCY COMPARISON — ByGaitLight")
    WARMUP = 10
    MEASURE = 100

    # GPU timing
    gpu_input = torch.randn(1, 1, 128, 64, device=bygait_backend.device)
    for _ in range(WARMUP):
        with torch.inference_mode():
            _ = bygait_model(gpu_input)
    torch.cuda.synchronize()

    bygait_gpu_times = []
    for _ in range(MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = bygait_model(gpu_input)
        torch.cuda.synchronize()
        bygait_gpu_times.append((time.perf_counter() - t0) * 1000.0)

    # CPU timing
    bygait_model_cpu = type(bygait_model)(part_bins=bygait_model.embedding.in_features // 128 if hasattr(bygait_model, 'embedding') else 4)
    bygait_model_cpu.load_state_dict(bygait_model.state_dict())
    bygait_model_cpu.to("cpu")
    bygait_model_cpu.eval()

    cpu_input = torch.randn(1, 1, 128, 64, device="cpu")
    for _ in range(WARMUP):
        with torch.inference_mode():
            _ = bygait_model_cpu(cpu_input)

    bygait_cpu_times = []
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = bygait_model_cpu(cpu_input)
        bygait_cpu_times.append((time.perf_counter() - t0) * 1000.0)

    gpu_stats = compute_stats(bygait_gpu_times)
    cpu_stats = compute_stats(bygait_cpu_times)
    speedup = round(cpu_stats["mean"] / gpu_stats["mean"], 2) if gpu_stats["mean"] > 0 else 0

    print(f"  ByGaitLight GPU:  mean={gpu_stats['mean']:.4f}ms  p50={gpu_stats['median']:.4f}ms  p95={gpu_stats['p95']:.4f}ms")
    print(f"  ByGaitLight CPU:  mean={cpu_stats['mean']:.4f}ms  p50={cpu_stats['median']:.4f}ms  p95={cpu_stats['p95']:.4f}ms")
    print(f"  Speedup (CPU/GPU): {speedup}x")

    report["E_bygait_latency"] = {"gpu": gpu_stats, "cpu": cpu_stats, "speedup": speedup}

    del bygait_model_cpu
    torch.cuda.empty_cache()

    # ==================================================================
    # E2. CPU vs GPU LATENCY — OSNet
    # ==================================================================
    section("E2. LATENCY COMPARISON — OSNet")
    from models.reid.osnet_backbone import _build_osnet_x0_25

    # GPU timing
    osnet_gpu_input = torch.randn(1, 3, 256, 128, device=osnet.device)
    mean_dev = osnet._mean
    std_dev = osnet._std
    osnet_gpu_input_norm = (osnet_gpu_input - mean_dev) / std_dev

    for _ in range(WARMUP):
        with torch.inference_mode():
            _ = osnet_model(osnet_gpu_input_norm)
    torch.cuda.synchronize()

    osnet_gpu_times = []
    for _ in range(MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = osnet_model(osnet_gpu_input_norm)
        torch.cuda.synchronize()
        osnet_gpu_times.append((time.perf_counter() - t0) * 1000.0)

    # CPU timing
    osnet_cpu_model = _build_osnet_x0_25()
    osnet_cpu_model.load_state_dict(osnet_model.state_dict())
    osnet_cpu_model.to("cpu")
    osnet_cpu_model.eval()

    osnet_cpu_input = torch.randn(1, 3, 256, 128, device="cpu")
    mean_cpu = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_cpu = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    osnet_cpu_input_norm = (osnet_cpu_input - mean_cpu) / std_cpu

    for _ in range(WARMUP):
        with torch.inference_mode():
            _ = osnet_cpu_model(osnet_cpu_input_norm)

    osnet_cpu_times = []
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = osnet_cpu_model(osnet_cpu_input_norm)
        osnet_cpu_times.append((time.perf_counter() - t0) * 1000.0)

    osnet_gpu_stats = compute_stats(osnet_gpu_times)
    osnet_cpu_stats = compute_stats(osnet_cpu_times)
    osnet_speedup = round(osnet_cpu_stats["mean"] / osnet_gpu_stats["mean"], 2) if osnet_gpu_stats["mean"] > 0 else 0

    print(f"  OSNet GPU:  mean={osnet_gpu_stats['mean']:.4f}ms  p50={osnet_gpu_stats['median']:.4f}ms  p95={osnet_gpu_stats['p95']:.4f}ms")
    print(f"  OSNet CPU:  mean={osnet_cpu_stats['mean']:.4f}ms  p50={osnet_cpu_stats['median']:.4f}ms  p95={osnet_cpu_stats['p95']:.4f}ms")
    print(f"  Speedup (CPU/GPU): {osnet_speedup}x")

    report["E2_osnet_latency"] = {"gpu": osnet_gpu_stats, "cpu": osnet_cpu_stats, "speedup": osnet_speedup}

    del osnet_cpu_model
    torch.cuda.empty_cache()

    # ==================================================================
    # F. VRAM USAGE
    # ==================================================================
    section("F. VRAM USAGE")
    vram_allocated = torch.cuda.memory_allocated(0) / 1024 / 1024
    vram_reserved = torch.cuda.memory_reserved(0) / 1024 / 1024
    vram_peak = torch.cuda.max_memory_allocated(0) / 1024 / 1024
    print(f"  Allocated: {vram_allocated:.2f} MB")
    print(f"  Reserved:  {vram_reserved:.2f} MB")
    print(f"  Peak:      {vram_peak:.2f} MB")

    smi = nvidia_smi_gpu_util()
    if "error" not in smi:
        print(f"  nvidia-smi used: {smi['memory_used_mb']:.0f} MB / {smi['memory_total_mb']:.0f} MB")
    else:
        print(f"  nvidia-smi: {smi['error']}")

    report["F_vram"] = {
        "torch_allocated_mb": round(vram_allocated, 2),
        "torch_reserved_mb": round(vram_reserved, 2),
        "torch_peak_mb": round(vram_peak, 2),
        "nvidia_smi": smi,
    }

    # ==================================================================
    # G. GPU UTILIZATION DURING SUSTAINED INFERENCE
    # ==================================================================
    section("G. GPU UTILIZATION (SUSTAINED INFERENCE)")

    def sustained_inference():
        inp_b = torch.randn(1, 1, 128, 64, device=bygait_backend.device)
        inp_o = torch.randn(1, 3, 256, 128, device=osnet.device)
        inp_o_norm = (inp_o - mean_dev) / std_dev
        for _ in range(500):
            with torch.inference_mode():
                _ = bygait_model(inp_b)
                _ = osnet_model(inp_o_norm)
        torch.cuda.synchronize()

    util_result = sample_gpu_utilization_during(sustained_inference)
    if "error" in util_result:
        print(f"  {util_result['error']}")
    else:
        print(f"  Samples collected:  {util_result['samples']}")
        print(f"  Mean GPU util:      {util_result['mean_gpu_percent']}%")
        print(f"  Max GPU util:       {util_result['max_gpu_percent']}%")
        print(f"  Min GPU util:       {util_result['min_gpu_percent']}%")

    report["G_gpu_utilization"] = util_result

    # ==================================================================
    # H. NUMERICAL CONSISTENCY (CPU vs GPU)
    # ==================================================================
    section("H. NUMERICAL CONSISTENCY")

    # ByGaitLight: same input, compare CPU vs GPU output
    test_gei = np.random.rand(128, 64).astype(np.float32)
    test_tensor = torch.from_numpy(test_gei).float().unsqueeze(0).unsqueeze(0)

    # GPU inference
    with torch.inference_mode():
        gpu_emb = bygait_model(test_tensor.to(bygait_backend.device)).cpu().numpy().flatten()

    # CPU inference via fresh model
    bygait_cpu_check = type(bygait_model)(part_bins=bygait_model.embedding.in_features // 128 if hasattr(bygait_model, 'embedding') else 4)
    bygait_cpu_check.load_state_dict(bygait_model.state_dict())
    bygait_cpu_check.to("cpu")
    bygait_cpu_check.eval()
    with torch.inference_mode():
        cpu_emb = bygait_cpu_check(test_tensor).cpu().numpy().flatten()

    bygait_dim = gpu_emb.shape[0]
    bygait_max_diff = float(np.max(np.abs(gpu_emb - cpu_emb)))
    bygait_cosine_sim = float(np.dot(gpu_emb, cpu_emb) / (np.linalg.norm(gpu_emb) * np.linalg.norm(cpu_emb) + 1e-12))
    bygait_finite = bool(np.all(np.isfinite(gpu_emb)))

    print(f"  ByGaitLight embedding dim:    {bygait_dim} (expected: 256)")
    print(f"  Max abs diff (CPU vs GPU):    {bygait_max_diff:.8f}")
    print(f"  Cosine similarity:            {bygait_cosine_sim:.8f}")
    print(f"  All finite:                   {bygait_finite}")
    print(f"  Dim check (==256):            {'PASS' if bygait_dim == 256 else 'FAIL'}")

    del bygait_cpu_check

    # OSNet: same input, compare CPU vs GPU
    test_crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

    # GPU path via production extract()
    gpu_osnet_emb = osnet.extract(test_crop)

    # CPU path
    osnet_cpu_check = _build_osnet_x0_25()
    osnet_cpu_check.load_state_dict(osnet_model.state_dict())
    osnet_cpu_check.to("cpu")
    osnet_cpu_check.eval()

    rgb = cv2.cvtColor(test_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (128, 256))
    t_cpu = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t_cpu = (t_cpu - mean_cpu) / std_cpu
    with torch.inference_mode():
        cpu_osnet_raw = osnet_cpu_check(t_cpu)
        cpu_osnet_emb = F.normalize(cpu_osnet_raw, p=2, dim=-1).squeeze(0).numpy().astype(np.float32)

    osnet_dim = gpu_osnet_emb.shape[0]
    osnet_max_diff = float(np.max(np.abs(gpu_osnet_emb - cpu_osnet_emb)))
    osnet_cosine_sim = float(np.dot(gpu_osnet_emb, cpu_osnet_emb) / (np.linalg.norm(gpu_osnet_emb) * np.linalg.norm(cpu_osnet_emb) + 1e-12))
    osnet_finite = bool(np.all(np.isfinite(gpu_osnet_emb)))

    print(f"\n  OSNet embedding dim:          {osnet_dim} (expected: 512)")
    print(f"  Max abs diff (CPU vs GPU):    {osnet_max_diff:.8f}")
    print(f"  Cosine similarity:            {osnet_cosine_sim:.8f}")
    print(f"  All finite:                   {osnet_finite}")
    print(f"  Dim check (==512):            {'PASS' if osnet_dim == 512 else 'FAIL'}")

    del osnet_cpu_check

    report["H_numerical"] = {
        "bygait": {
            "dim": bygait_dim, "expected": 256, "dim_ok": bygait_dim == 256,
            "max_abs_diff": bygait_max_diff, "cosine_sim": bygait_cosine_sim,
            "all_finite": bygait_finite,
        },
        "osnet": {
            "dim": osnet_dim, "expected": 512, "dim_ok": osnet_dim == 512,
            "max_abs_diff": osnet_max_diff, "cosine_sim": osnet_cosine_sim,
            "all_finite": osnet_finite,
        },
    }

    # ==================================================================
    # I. PRODUCTION PATH VERIFICATION
    # ==================================================================
    section("I. PRODUCTION PATH VERIFICATION")
    print("  Testing via FeatureExtractionStep.extract_from_gei() [ByGaitLight]...")
    test_gei_prod = np.random.randint(0, 255, (128, 64), dtype=np.uint8)
    prod_emb = fe_step.extract_from_gei(test_gei_prod)
    print(f"    Output shape: {prod_emb.shape}")
    print(f"    Output dtype: {prod_emb.dtype}")
    print(f"    All finite:   {np.all(np.isfinite(prod_emb))}")
    print(f"    Dim:          {prod_emb.shape[-1]} (expected: 256)")

    print("\n  Testing via OSNetBackbone.extract() [OSNet]...")
    test_crop_prod = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    prod_osnet_emb = osnet.extract(test_crop_prod)
    print(f"    Output shape: {prod_osnet_emb.shape}")
    print(f"    Output dtype: {prod_osnet_emb.dtype}")
    print(f"    All finite:   {np.all(np.isfinite(prod_osnet_emb))}")
    print(f"    Dim:          {prod_osnet_emb.shape[-1]} (expected: 512)")

    # Test via AppearanceEmbeddingExtractor (production wrapper)
    print("\n  Testing via AppearanceEmbeddingExtractor.extract() [Production OSNet path]...")
    from intelligence.appearance_embedding import AppearanceEmbeddingExtractor
    app_ext = AppearanceEmbeddingExtractor()
    if app_ext.is_available():
        prod_app_emb = app_ext.extract(test_crop_prod, track_id=None)
        if prod_app_emb is not None:
            print(f"    Output shape: {prod_app_emb.shape}")
            print(f"    Output dtype: {prod_app_emb.dtype}")
            print(f"    All finite:   {np.all(np.isfinite(prod_app_emb))}")
            print(f"    Dim:          {prod_app_emb.shape[-1]} (expected: 512)")
        else:
            print("    *** Returned None — check logs ***")
    else:
        print("    *** AppearanceEmbeddingExtractor not available ***")

    # Test via GaitService.process_image_bytes (full production e2e)
    print("\n  Testing via GaitService.process_image_bytes() [Full production e2e]...")
    from services.gait_service import GaitService
    gs = GaitService()
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", test_frame)
    if ok:
        torch.cuda.synchronize()
        vram_before_e2e = torch.cuda.memory_allocated(0) / 1024 / 1024
        t0 = time.perf_counter()
        result = gs.process_image_bytes(buf.tobytes(), camera_id="gpu-audit-cam")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        vram_after_e2e = torch.cuda.memory_allocated(0) / 1024 / 1024
        print(f"    Latency:       {(t1-t0)*1000:.2f} ms")
        print(f"    VRAM delta:    {vram_after_e2e - vram_before_e2e:.2f} MB")
        if isinstance(result, dict):
            print(f"    Result keys:   {list(result.keys())[:10]}")
        else:
            print(f"    Result type:   {type(result).__name__}")

    report["I_production_path"] = {
        "bygait_prod_dim": int(prod_emb.shape[-1]),
        "osnet_prod_dim": int(prod_osnet_emb.shape[-1]),
        "appearance_extractor_available": app_ext.is_available(),
    }

    # ==================================================================
    # J. REMAINING BOTTLENECK ANALYSIS
    # ==================================================================
    section("J. BOTTLENECK ANALYSIS")

    # Measure full production path with real camera if available
    cam_backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, cam_backend)

    if cap.isOpened():
        print("  Physical camera detected — running real e2e benchmark (30 frames)...")

        # Warmup camera + pipeline
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    _ = gs.process_image_bytes(buf.tobytes(), camera_id="gpu-audit-cam")

        capture_times = []
        encode_times = []
        process_times = []
        total_times = []

        for i in range(30):
            t_total = time.perf_counter()

            t_cap = time.perf_counter()
            ret, frame = cap.read()
            capture_times.append((time.perf_counter() - t_cap) * 1000.0)

            if not ret or frame is None:
                continue

            t_enc = time.perf_counter()
            ok, buf = cv2.imencode(".jpg", frame)
            encode_times.append((time.perf_counter() - t_enc) * 1000.0)

            if not ok:
                continue

            torch.cuda.synchronize()
            t_proc = time.perf_counter()
            _ = gs.process_image_bytes(buf.tobytes(), camera_id="gpu-audit-cam")
            torch.cuda.synchronize()
            process_times.append((time.perf_counter() - t_proc) * 1000.0)

            total_times.append((time.perf_counter() - t_total) * 1000.0)

        cap.release()

        def print_stats(name, samples):
            s = compute_stats(samples)
            print(f"  {name:32s}: mean={s['mean']:8.2f}ms  p50={s['median']:8.2f}ms  p95={s['p95']:8.2f}ms  p99={s['p99']:8.2f}ms")
            return s

        print()
        cap_s = print_stats("Camera Capture", capture_times)
        enc_s = print_stats("JPEG Encode (overhead)", encode_times)
        proc_s = print_stats("Pipeline Processing", process_times)
        tot_s = print_stats("Total (cap+enc+proc)", total_times)

        actual_fps = round(1000.0 / tot_s["mean"], 2) if tot_s["mean"] > 0 else 0
        processing_fps = round(1000.0 / proc_s["mean"], 2) if proc_s["mean"] > 0 else 0

        print(f"\n  Actual Physical Camera FPS:     {actual_fps}")
        print(f"  Processing-only FPS:            {processing_fps}")

        report["J_bottleneck"] = {
            "camera_capture": cap_s,
            "jpeg_encode": enc_s,
            "pipeline_processing": proc_s,
            "total_e2e": tot_s,
            "actual_camera_fps": actual_fps,
            "processing_fps": processing_fps,
        }
    else:
        print("  *** Physical camera not available — skipping real e2e benchmark ***")
        report["J_bottleneck"] = {"camera": "not available"}

    # ==================================================================
    # FINAL VERDICT
    # ==================================================================
    section("FINAL VERDICT")

    bygait_gpu = bygait_on_cuda
    osnet_gpu = osnet_on_cuda
    production_gpu = bygait_gpu and osnet_gpu
    gpu_ready = torch.cuda.is_available() and bygait_gpu and osnet_gpu

    print(f"  GPU READY:                              {'YES' if gpu_ready else 'NO'}")
    print(f"  ByGaitLight actually running on GPU:     {'YES' if bygait_gpu else 'NO'}")
    print(f"  OSNet actually running on GPU:           {'YES' if osnet_gpu else 'NO'}")
    print(f"  Production inference path using GPU:     {'YES' if production_gpu else 'NO'}")

    if bygait_gpu:
        print(f"\n  ByGaitLight GPU speedup vs CPU:          {report['E_bygait_latency']['speedup']}x")
    if osnet_gpu:
        print(f"  OSNet GPU speedup vs CPU:                {report['E2_osnet_latency']['speedup']}x")

    report["verdict"] = {
        "gpu_ready": gpu_ready,
        "bygait_on_gpu": bygait_gpu,
        "osnet_on_gpu": osnet_gpu,
        "production_using_gpu": production_gpu,
    }

    _save_report(report)
    print("\n  Report saved to: outputs/reports/gpu_validation_audit.json")


def _save_report(report: dict) -> None:
    os.makedirs("outputs/reports", exist_ok=True)
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open("outputs/reports/gpu_validation_audit.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
