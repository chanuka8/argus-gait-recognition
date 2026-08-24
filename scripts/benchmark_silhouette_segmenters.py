import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pipeline.steps.silhouette_step import LearnedSilhouetteSegmenter, OtsuSilhouetteExtractor
from training.silhouette_dataset import SilhouetteSegmentationDataset


def benchmark_silhouette_segmenters(
    zip_path: str = "data/casia_b_raw.zip",
    num_samples: int = 100,
) -> dict:
    if not Path(zip_path).exists():
        print(f"[!] CASIA-B raw archive not found at: {zip_path}")
        return {}

    print(f"[*] Benchmarking Silhouette Segmenters on {num_samples} CASIA-B validation samples...")
    val_ds = SilhouetteSegmentationDataset(
        zip_path=zip_path, subject_range=(63, 74), max_samples=num_samples, seed=101
    )

    learned = LearnedSilhouetteSegmenter(model_path="models/weights/silhouette_segmenter.onnx")
    otsu = OtsuSilhouetteExtractor()

    if not learned.is_available():
        print("[!] Learned silhouette segmenter model asset is not available!")
        return {}

    learned_dice_list, learned_iou_list, learned_prec_list, learned_rec_list = [], [], [], []
    otsu_dice_list, otsu_iou_list, otsu_prec_list, otsu_rec_list = [], [], [], []

    learned_times, otsu_times = [], []

    for idx in range(len(val_ds)):
        gt_mask = val_ds.masks[idx]
        if gt_mask is None or gt_mask.size == 0:
            continue

        rgb_crop = val_ds._render_synthetic_rgb_crop(gt_mask, idx)
        target_gt = (gt_mask > 128).astype(np.float32)

        t0 = time.perf_counter()
        learned_raw_mask = learned.segment(rgb_crop)
        learned_dt = (time.perf_counter() - t0) * 1000.0
        learned_times.append(learned_dt)

        if learned_raw_mask is not None:
            learned_pred = (learned_raw_mask > 128).astype(np.float32)
            tp = float(np.sum(learned_pred * target_gt))
            fp = float(np.sum(learned_pred * (1.0 - target_gt)))
            fn = float(np.sum((1.0 - learned_pred) * target_gt))

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

            learned_dice_list.append(dice)
            learned_iou_list.append(iou)
            learned_prec_list.append(prec)
            learned_rec_list.append(rec)

        t0 = time.perf_counter()
        otsu_raw_mask = otsu.extract_mask(rgb_crop)
        otsu_dt = (time.perf_counter() - t0) * 1000.0
        otsu_times.append(otsu_dt)

        if otsu_raw_mask is not None:
            otsu_pred = (otsu_raw_mask > 128).astype(np.float32)
            tp = float(np.sum(otsu_pred * target_gt))
            fp = float(np.sum(otsu_pred * (1.0 - target_gt)))
            fn = float(np.sum((1.0 - otsu_pred) * target_gt))

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

            otsu_dice_list.append(dice)
            otsu_iou_list.append(iou)
            otsu_prec_list.append(prec)
            otsu_rec_list.append(rec)

    results = {
        "learned": {
            "dice": float(np.mean(learned_dice_list)),
            "iou": float(np.mean(learned_iou_list)),
            "precision": float(np.mean(learned_prec_list)),
            "recall": float(np.mean(learned_rec_list)),
            "latency_ms": float(np.mean(learned_times)),
            "fps": float(1000.0 / np.mean(learned_times)),
        },
        "otsu": {
            "dice": float(np.mean(otsu_dice_list)),
            "iou": float(np.mean(otsu_iou_list)),
            "precision": float(np.mean(otsu_prec_list)),
            "recall": float(np.mean(otsu_rec_list)),
            "latency_ms": float(np.mean(otsu_times)),
            "fps": float(1000.0 / np.mean(otsu_times)),
        },
    }

    print("\n--- BENCHMARK RESULTS ---")
    print("Learned UNet ONNX Segmenter:")
    print(f"  Dice:      {results['learned']['dice']:.4f}")
    print(f"  IoU:       {results['learned']['iou']:.4f}")
    print(f"  Precision: {results['learned']['precision']:.4f}")
    print(f"  Recall:    {results['learned']['recall']:.4f}")
    print(f"  Latency:   {results['learned']['latency_ms']:.2f} ms ({results['learned']['fps']:.1f} FPS)")

    print("\nOtsu Thresholding Fallback:")
    print(f"  Dice:      {results['otsu']['dice']:.4f}")
    print(f"  IoU:       {results['otsu']['iou']:.4f}")
    print(f"  Precision: {results['otsu']['precision']:.4f}")
    print(f"  Recall:    {results['otsu']['recall']:.4f}")
    print(f"  Latency:   {results['otsu']['latency_ms']:.2f} ms ({results['otsu']['fps']:.1f} FPS)")

    return results


if __name__ == "__main__":
    benchmark_silhouette_segmenters()
