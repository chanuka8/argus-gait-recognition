import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.inference_pipeline import InferencePipeline
from storage.vector_store import VectorStore


def benchmark_gallery_load() -> dict:
    start = time.perf_counter()

    gallery = VectorStore().load()

    elapsed = time.perf_counter() - start

    if gallery is None:
        return {
            "status": "failed",
            "time_seconds": elapsed,
            "embeddings": 0,
            "people": 0,
        }

    features, labels, metadata = gallery

    return {
        "status": "ok",
        "time_seconds": elapsed,
        "embeddings": len(features),
        "labels": len(labels),
        "people": len(metadata),
    }


def benchmark_single_inference(
    image_path: str = "data/casia_processed/gei/034/034_nm-01_126.png",
) -> dict:
    start_init = time.perf_counter()

    pipeline = InferencePipeline(gallery_dir="models/gallery")

    init_time = time.perf_counter() - start_init

    start_predict = time.perf_counter()

    result = pipeline.predict(image_path)

    predict_time = time.perf_counter() - start_predict

    identity = result["identity"]
    score = result["score"]
    threshold = pipeline.matcher.threshold

    reason = None
    if identity == "UNKNOWN":
        if score == 0.0:
            active_count = 0
            if pipeline.metadata:
                for label, entry in pipeline.metadata.items():
                    status = entry.get("status", "DISABLED").upper()
                    enabled = entry.get("enabled", status == "ACTIVE")
                    if status == "ACTIVE" and enabled:
                        active_count += 1
            if len(pipeline.gallery_features) == 0:
                reason = "Gallery has no embeddings"
            elif active_count == 0:
                reason = "All gallery identities are filtered out (none have status ACTIVE and enabled=True)"
            else:
                reason = "No valid embeddings found or zero similarity"
        else:
            reason = f"Best similarity score {score:.4f} is below the recognition threshold of {threshold:.2f}"

    return {
        "image_path": image_path,
        "pipeline_init_seconds": init_time,
        "prediction_seconds": predict_time,
        "prediction_time_ms": round(predict_time * 1000.0, 4),
        "fps": round(1.0 / predict_time, 2) if predict_time > 0 else 0.0,
        "identity": identity,
        "score": score,
        "reason": reason,
    }


def save_report(report: dict) -> Path:
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "benchmark_report.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return output_file


def benchmark_inference_average(
    image_path: str = "data/casia_processed/gei/034/034_nm-01_126.png",
    num_iterations: int = 10,
) -> dict:
    pipeline = InferencePipeline(gallery_dir="models/gallery")

    inference_times = []

    for _ in range(num_iterations):
        t_start = time.perf_counter()
        pipeline.predict(image_path)
        t_end = time.perf_counter()
        inference_times.append(t_end - t_start)

    times_ms = [t * 1000.0 for t in inference_times]
    total_time = sum(inference_times)

    return {
        "num_iterations": num_iterations,
        "avg_inference_time_ms": round(sum(times_ms) / len(times_ms), 4),
        "min_inference_time_ms": round(min(times_ms), 4),
        "max_inference_time_ms": round(max(times_ms), 4),
        "fps": round(num_iterations / total_time, 2) if total_time > 0 else 0.0,
    }


def main() -> None:
    start_total = time.perf_counter()

    gallery_result = benchmark_gallery_load()
    inference_result = benchmark_single_inference()

    inference_avg = benchmark_inference_average()

    total_time = time.perf_counter() - start_total

    report = {
        "gallery_load": gallery_result,
        "single_inference": inference_result,
        "inference_average": inference_avg,
        "total_benchmark_seconds": total_time,
    }

    report_file = save_report(report)

    print("\n=== ARGUS BENCHMARK REPORT ===")

    print("\nGallery Load:")
    for key, value in gallery_result.items():
        print(f"  {key}: {value}")

    print("\nSingle Inference:")
    for key, value in inference_result.items():
        if key == "reason" and value is None:
            continue
        print(f"  {key}: {value}")

    print("\nInference Average:")
    for key, value in inference_avg.items():
        print(f"  {key}: {value}")

    print(f"\nTotal benchmark seconds: {total_time:.4f}")
    print(f"Report saved -> {report_file}")


if __name__ == "__main__":
    main()