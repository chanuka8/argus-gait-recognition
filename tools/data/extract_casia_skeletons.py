import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extract_all_skeletons(
    zip_path: str = "data/casia_b_raw.zip",
    out_root: str = "data/casia_processed/skeletons",
    min_sub: int = 1,
    max_sub: int = 124,
):
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLOv8n-pose on device: {device} for subjects {min_sub:03d}..{max_sub:03d}...")
    model = YOLO("models/weights/yolov8n-pose.pt").to(device)

    print(f"Opening Zip archive: {zip_path}...")
    t0 = time.time()

    seq_map = defaultdict(list)

    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = z.namelist()

        for f in all_files:
            if not f.endswith(".png"):
                continue
            parts = f.split("/")
            if len(parts) >= 5 and parts[1].isdigit():
                sub_num = int(parts[1])
                sub = parts[1]
                cond = parts[2].lower()
                view = parts[3]
                if min_sub <= sub_num <= max_sub:
                    seq_map[(sub, cond, view)].append(f)

        print(f"Found {len(seq_map)} valid walking sequences across subjects {min_sub:03d}..{max_sub:03d}.")

        extracted_count = 0
        skipped_count = 0

        sorted_seq_keys = sorted(seq_map.keys())

        batch_imgs = []
        batch_size = 16

        for idx, (sub, cond, view) in enumerate(sorted_seq_keys):
            save_path = out_dir / sub / f"{sub}_{cond}_{view}.npy"
            if save_path.exists():
                skipped_count += 1
                continue

            save_path.parent.mkdir(parents=True, exist_ok=True)

            files = sorted(seq_map[(sub, cond, view)])
            frames_kpts = []

            for frame_file in files:
                img_bytes = z.read(frame_file)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                img_resized = cv2.resize(img, (320, 320))
                batch_imgs.append(img_resized)

                if len(batch_imgs) >= batch_size:
                    res = model.predict(batch_imgs, imgsz=320, conf=0.05, verbose=False, device=device)
                    for r in res:
                        if r.keypoints is not None and len(r.keypoints.xy) > 0:
                            xy = r.keypoints.xy[0].cpu().numpy()
                            conf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
                            xy_norm = xy / np.array([320.0, 320.0])
                            kpts = np.concatenate([xy_norm, conf[:, None]], axis=-1)
                        else:
                            kpts = np.zeros((17, 3), dtype=np.float32)
                        frames_kpts.append(kpts)
                    batch_imgs.clear()

            if batch_imgs:
                res = model.predict(batch_imgs, imgsz=320, conf=0.05, verbose=False, device=device)
                for r in res:
                    if r.keypoints is not None and len(r.keypoints.xy) > 0:
                        xy = r.keypoints.xy[0].cpu().numpy()
                        conf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
                        xy_norm = xy / np.array([320.0, 320.0])
                        kpts = np.concatenate([xy_norm, conf[:, None]], axis=-1)
                    else:
                        kpts = np.zeros((17, 3), dtype=np.float32)
                    frames_kpts.append(kpts)
                batch_imgs.clear()

            if frames_kpts:
                seq_arr = np.stack(frames_kpts, axis=0).astype(np.float32)
                np.save(save_path, seq_arr)
                extracted_count += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == len(sorted_seq_keys):
                elapsed = time.time() - t0
                print(
                    f"Processed {idx + 1}/{len(sorted_seq_keys)} sequences | Extracted: {extracted_count} | Skipped: {skipped_count} | Elapsed: {elapsed:.1f}s"
                )

    print(f"\n[SUCCESS] Extraction completed. Total sequences in {out_dir}: {extracted_count + skipped_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-sub", type=int, default=1)
    parser.add_argument("--max-sub", type=int, default=124)
    args = parser.parse_args()
    extract_all_skeletons(min_sub=args.min_sub, max_sub=args.max_sub)
