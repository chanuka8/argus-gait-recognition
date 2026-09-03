import json
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

# Ensure project root is in sys.path when executed directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.steps.live_gei import LiveGEI
from pipeline.steps.silhouette_step import SilhouetteStep
from services.missing_person_processor import MissingPersonVideoProcessor


def create_benchmark_video(filepath: Path, num_frames: int = 90, width: int = 320, height: int = 240, fps: float = 30.0) -> Path:
    """Generates a standard 90-frame (3-second) benchmark video of walking human figure."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

    for idx in range(num_frames):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        cx = 100 + (idx % 20) * 3
        # Head
        cv2.circle(frame, (cx, 50), 16, (255, 255, 255), -1)
        # Torso
        cv2.rectangle(frame, (cx - 22, 68), (cx + 22, 145), (255, 255, 255), -1)
        # Arms
        arm_shift = int((idx % 10 - 5) * 3)
        cv2.line(frame, (cx - 22, 80), (cx - 30, 125 + arm_shift), (255, 255, 255), 6)
        cv2.line(frame, (cx + 22, 80), (cx + 30, 125 - arm_shift), (255, 255, 255), 6)
        # Legs
        leg_shift = int((idx % 8 - 4) * 4)
        cv2.line(frame, (cx - 12, 145), (cx - 16 + leg_shift, 215), (255, 255, 255), 8)
        cv2.line(frame, (cx + 12, 145), (cx + 16 - leg_shift, 215), (255, 255, 255), 8)

        writer.write(frame)

    writer.release()
    return filepath


def run_baseline_pipeline(video_path: Path, temp_work_dir: Path, mock_tracking: bool = True) -> tuple[float, int, int]:
    """Baseline approach representing legacy AutoEnrollmentService:
    - Normal torch execution (without inference_mode)
    - Sequential frame reading
    - Saves every GEI to disk as temporary .png file
    - Re-reads GEI from disk before feature extraction
    - No embedding deduplication
    """
    start_time = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    silhouette_step = SilhouetteStep()
    extractor = FeatureExtractionStep()

    live_gei = LiveGEI(max_frames=15, min_frames=10)
    gei_files: list[Path] = []
    frame_count = 0
    gei_idx = 0

    gei_dir = temp_work_dir / "baseline_geis"
    gei_dir.mkdir(parents=True, exist_ok=True)

    import supervision as sv
    mock_detections = sv.Detections(
        xyxy=np.array([[80, 30, 180, 230]], dtype=np.float32),
        confidence=np.array([0.95], dtype=np.float32),
        tracker_id=np.array([1], dtype=int),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Baseline: simulate detection & tracking pass
        xyxy = mock_detections.xyxy
        if xyxy is not None and len(xyxy) > 0:
            box = xyxy[0]
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            sil = silhouette_step.extract_from_crop(crop)
            if sil is not None:
                live_gei.add(sil)
                if live_gei.ready() and (frame_count % 15 == 0):
                    gei_arr = live_gei.build()
                    if gei_arr is not None:
                        gei_path = gei_dir / f"gei_{gei_idx}.png"
                        cv2.imwrite(str(gei_path), gei_arr)
                        gei_files.append(gei_path)
                        gei_idx += 1

    cap.release()

    # Re-read GEIs from disk and extract (legacy disk roundtrip)
    embeddings = []
    for g_path in gei_files:
        gei_img = cv2.imread(str(g_path), cv2.IMREAD_GRAYSCALE)
        if gei_img is not None:
            emb = extractor.extract_from_gei(gei_img)
            embeddings.append(emb)

    elapsed = time.perf_counter() - start_time
    return elapsed, frame_count, len(embeddings)


def run_optimized_pipeline(video_path: Path, temp_work_dir: Path, mock_tracking: bool = True) -> tuple[float, int, dict]:
    """Optimized approach (MissingPersonVideoProcessor):
    - torch.inference_mode()
    - Zero intermediate disk I/O (in-memory GEIs)
    - Dominant target track isolation
    - Strict validation & cosine deduplication
    """
    gait_gallery = temp_work_dir / "opt_gallery"
    db_dir = temp_work_dir / "opt_db"

    processor = MissingPersonVideoProcessor(
        gait_gallery_dir=str(gait_gallery),
        db_dir=str(db_dir),
    )

    if mock_tracking:
        from unittest.mock import MagicMock

        import supervision as sv
        mock_det = sv.Detections(
            xyxy=np.array([[80, 30, 180, 230]], dtype=np.float32),
            confidence=np.array([0.95], dtype=np.float32),
            tracker_id=np.array([1], dtype=int),
        )
        mock_tr = MagicMock()
        mock_tr.track.return_value = mock_det
        processor.tracker = mock_tr

    start_time = time.perf_counter()
    result = processor.process_reference_video(
        person_id="Bench_Subject",
        video_path=video_path,
    )
    elapsed = time.perf_counter() - start_time

    return elapsed, result.get("frames_processed", 0), result


def main():
    print("=" * 60)
    print("ARGUS AI - Reference Video Processing Benchmark")
    print("Comparing Baseline (AutoEnrollmentService Disk Flow)")
    print("vs Optimized (MissingPersonVideoProcessor In-Memory Flow)")
    print("=" * 60)

    work_dir = Path(tempfile.mkdtemp(prefix="argus_bench_"))
    try:
        video_file = work_dir / "benchmark_walk_3s.mp4"
        num_frames = 90
        fps = 30.0
        duration_sec = num_frames / fps
        print(f"[*] Generating benchmark video ({num_frames} frames, {duration_sec:.1f}s @ {fps} FPS)...")
        create_benchmark_video(video_file, num_frames=num_frames, fps=fps)

        # 1. Benchmark Baseline
        print("\n[*] Running Baseline Pipeline (Disk I/O + Autograd Tracking)...")
        tracemalloc.start()
        base_time, base_frames, base_embs = run_baseline_pipeline(video_file, work_dir)
        _, peak_ram_base = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        base_fps = base_frames / max(0.001, base_time)
        print(f"    Baseline Time:     {base_time:.3f} s")
        print(f"    Baseline Rate:     {base_fps:.1f} FPS")
        print(f"    Baseline Embeds:   {base_embs}")
        print(f"    Peak RAM:          {peak_ram_base / (1024 * 1024):.1f} MB")

        # 2. Benchmark Optimized
        print("\n[*] Running Optimized Pipeline (In-Memory GEI + torch.inference_mode + Dedup)...")
        tracemalloc.start()
        opt_time, opt_frames, opt_result = run_optimized_pipeline(video_file, work_dir)
        _, peak_ram_opt = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        opt_fps = opt_frames / max(0.001, opt_time)
        speedup = base_time / max(0.001, opt_time)

        print(f"    Optimized Time:    {opt_time:.3f} s")
        print(f"    Optimized Rate:    {opt_fps:.1f} FPS")
        print(f"    Valid Sequences:   {opt_result.get('valid_sequences', 0)}")
        print(f"    Generated Embeds:  {opt_result.get('embeddings_generated', 0)}")
        print(f"    Deduplicated:      {opt_result.get('embeddings_deduplicated', 0)}")
        print(f"    Committed Embeds:  {opt_result.get('embeddings_committed', 0)}")
        print(f"    Peak RAM:          {peak_ram_opt / (1024 * 1024):.1f} MB")
        print(f"    Measured Speedup:  {speedup:.2f}X")

        report = {
            "timestamp": time.time(),
            "video": {
                "frames": num_frames,
                "fps": fps,
                "duration_seconds": duration_sec,
                "resolution": "320x240",
            },
            "baseline": {
                "description": "AutoEnrollmentService disk-based GEI write/read + autograd tracking",
                "time_seconds": round(base_time, 4),
                "effective_fps": round(base_fps, 2),
                "embeddings_count": base_embs,
                "peak_ram_mb": round(peak_ram_base / (1024 * 1024), 2),
            },
            "optimized": {
                "description": "MissingPersonVideoProcessor in-memory GEI + inference_mode + dedup",
                "time_seconds": round(opt_time, 4),
                "effective_fps": round(opt_fps, 2),
                "valid_sequences": opt_result.get("valid_sequences", 0),
                "embeddings_generated": opt_result.get("embeddings_generated", 0),
                "embeddings_deduplicated": opt_result.get("embeddings_deduplicated", 0),
                "embeddings_committed": opt_result.get("embeddings_committed", 0),
                "peak_ram_mb": round(peak_ram_opt / (1024 * 1024), 2),
            },
            "comparison": {
                "speedup_ratio": round(speedup, 2),
                "ram_reduction_ratio": round(peak_ram_base / max(1, peak_ram_opt), 2),
                "target_speedup_target": "10X",
                "actual_measured_speedup": f"{speedup:.2f}X",
            },
            "hardware": {
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            }
        }

        out_path = Path("outputs/reports/benchmark/reference_processor_benchmark.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)

        print(f"\n[+] Benchmark report written to {out_path}")
        print("=" * 60)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
