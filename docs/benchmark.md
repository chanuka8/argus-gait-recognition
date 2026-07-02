# Performance Benchmark

This document details the latency, throughput, and system resource requirements of the ARGUS gait recognition model.

---

## 1. Benchmarking Objectives

In surveillance environments, processing frames must complete in real-time to avoid video lagging or stream freezes. This benchmark measures:
- **Gallery Load Latency:** Speed to read biometric profiles from disk and build flat search arrays.
- **Pipeline Initialization Time:** Overhead to instantiate deep models and tracking components.
- **Single Inference Time:** Time required to pass a single Gait Energy Image (GEI) through the CNN and query the vector database.
- **Total Runtime Duration:** Full loop execution time for verification checks.

---

## 2. Verified Benchmark Results

The following metrics were compiled from system executions (results recorded in [outputs/reports/benchmark_report.json](file:///e:/ARGUS_AI/outputs/reports/benchmark_report.json)):

### Gallery Diagnostics
- **Gallery Status:** `OK`
- **Total Embeddings:** 13,544
- **Unique Identities (Subjects):** 124
- **Gallery Loading Time:** **0.0185 seconds** (18.47 milliseconds).

### Inference Latency
- **Pipeline Initialization Latency:** **0.0880 seconds** (88.04 milliseconds).
- **Single Image Inference (CNN + Database Lookup):** **0.1135 seconds** (113.50 milliseconds).
- **Average Inference Throughput:** ~8.8 Frames Per Second (FPS) on CPU.

### Summary
- **Total Benchmark Running Duration:** **0.2227 seconds**.

*Note:* Vector database matching scale remains sub-millisecond ($< 0.5$ ms) due to vectorized matrix multiplication in NumPy, making the CNN forward pass the main computation driver.

---

## 3. Benchmarking Command

To run the speed benchmark:
```bash
python cli.py --mode benchmark
```
*Alternative (Makefile):*
```bash
make benchmark
```

The script will:
1. Initialize the inference pipeline using [pipeline/inference_pipeline.py](file:///e:/ARGUS_AI/pipeline/inference_pipeline.py).
2. Measure gallery read speed and dimension scaling.
3. Pass a sample GEI image (`data/casia_processed/gei/034/034_nm-01_126.png`) through the model.
4. Verify classification outcomes and output log records to `outputs/reports/benchmark_report.json`.
