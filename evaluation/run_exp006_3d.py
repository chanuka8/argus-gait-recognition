import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from evaluation.dataset_split import load_or_create_subject_split
from evaluation.evaluate_exp004 import evaluate_checkpoint
from evaluation.gallery_probe_builder import build_gallery_and_probe_sets
from evaluation.metrics import compute_biometric_rates, compute_roc_auc_eer
from models.architectures.pose_gait_3d import PoseGait3DNet, PoseLifter3D


def extract_2d_keypoints_sequence(yolo_model: YOLO, img_paths: list[str]) -> np.ndarray:
    seq_kpts = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if img.ndim == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        results = yolo_model(img, conf=0.05, verbose=False)
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            xy = results[0].keypoints.xy[0].cpu().numpy()
            conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
            h, w = img.shape[:2]
            xy_norm = xy / np.array([max(w, 1), max(h, 1)])
            kpts = np.concatenate([xy_norm, conf[:, None]], axis=-1)
        else:
            kpts = np.zeros((17, 3), dtype=np.float32)
        seq_kpts.append(kpts)

    if not seq_kpts:
        seq_kpts = [np.zeros((17, 3), dtype=np.float32) for _ in range(15)]

    seq_arr = np.stack(seq_kpts, axis=0)
    if len(seq_arr) < 30:
        repeats = (30 // len(seq_arr)) + 1
        seq_arr = np.tile(seq_arr, (repeats, 1, 1))[:30]
    else:
        seq_arr = seq_arr[:30]

    return seq_arr.astype(np.float32)


def main():
    print("\n=======================================================")
    print("  RUNNING EXP-006: 3D Pose Gait Branch Controlled Experiment")
    print("=======================================================\n")

    output_dir = Path("runs/exp_006_3d_gait")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading YOLOv8n-pose and 3D Gait models...")
    yolo_pose = YOLO("models/weights/yolov8n-pose.pt")
    pose_lifter = PoseLifter3D().to(device).eval()
    gait3d_net = PoseGait3DNet(embedding_dim=256).to(device).eval()

    split_manifest = load_or_create_subject_split("configs/subject_split.json", "data/casia_processed/gei")
    test_subs = split_manifest["test_subjects"]
    gei_root = "data/casia_processed/gei"

    gallery_items, probe_items = build_gallery_and_probe_sets(subjects=test_subs, gei_root=gei_root)
    print(f"Loaded Gallery items: {len(gallery_items)} | Probe items: {len(probe_items)}")

    print("Extracting 3D Gait embeddings...")
    start_time = time.time()

    def get_3d_embedding(item_path: str) -> np.ndarray:
        p_dir = Path(item_path).parent
        frame_paths = sorted(p_dir.glob("*.png"))[:5]
        if not frame_paths:
            frame_paths = [item_path]

        seq_2d = extract_2d_keypoints_sequence(yolo_pose, [str(f) for f in frame_paths])
        tensor_2d = torch.from_numpy(seq_2d).float().unsqueeze(0).to(device)

        with torch.no_grad():
            joints_3d = pose_lifter(tensor_2d)
            emb_3d = gait3d_net(joints_3d).squeeze(0).cpu().numpy()

        return emb_3d

    gal_embs = np.stack([get_3d_embedding(item["path"]) for item in gallery_items], axis=0)
    gal_labels = np.array([item["subject_id"] for item in gallery_items])

    prb_embs = np.stack([get_3d_embedding(item["path"]) for item in probe_items], axis=0)
    prb_labels = np.array([item["subject_id"] for item in probe_items])
    prb_conds = np.array([item["condition"] for item in probe_items])

    total_time = time.time() - start_time
    fps = (len(gallery_items) + len(probe_items)) / max(total_time, 1e-5)
    latency_ms = (total_time / max(len(gallery_items) + len(probe_items), 1)) * 1000.0

    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    print(
        f"Extraction completed in {total_time:.2f}s | FPS: {fps:.1f} | Latency: {latency_ms:.2f}ms/item | VRAM: {vram_mb:.1f}MB"
    )

    print("\n--- Evaluating 3D Gait Closed-Set Identification ---")
    sim_matrix = np.dot(prb_embs, gal_embs.T)

    top1_hits = 0
    top5_hits = 0
    for i in range(len(prb_labels)):
        top_k_indices = np.argsort(sim_matrix[i])[::-1][:5]
        top1_hit = any(gal_labels[idx] == prb_labels[i] for idx in top_k_indices[:1])
        top5_hit = any(gal_labels[idx] == prb_labels[i] for idx in top_k_indices[:5])
        if top1_hit:
            top1_hits += 1
        if top5_hit:
            top5_hits += 1

    rank1_acc = top1_hits / max(len(prb_labels), 1)
    rank5_acc = top5_hits / max(len(prb_labels), 1)

    cond_accs = {}
    for c in ["NM", "BG", "CL"]:
        mask = prb_conds == c
        if np.sum(mask) > 0:
            c_hits = sum(
                any(gal_labels[idx] == prb_labels[i] for idx in np.argsort(sim_matrix[i])[::-1][:1])
                for i in np.where(mask)[0]
            )
            cond_accs[c] = c_hits / float(np.sum(mask))
        else:
            cond_accs[c] = 0.0

    print(f"Rank-1 Accuracy: {rank1_acc * 100:.2f}%")
    print(f"Rank-5 Accuracy: {rank5_acc * 100:.2f}%")
    print(f"NM Accuracy:     {cond_accs['NM'] * 100:.2f}%")
    print(f"BG Accuracy:     {cond_accs['BG'] * 100:.2f}%")
    print(f"CL Accuracy:     {cond_accs['CL'] * 100:.2f}%")

    known_subs = set(test_subs[: len(test_subs) // 2])
    scores = np.max(sim_matrix, axis=1)
    is_genuine = np.array(
        [
            prb_labels[i] in known_subs and gal_labels[np.argmax(sim_matrix[i])] == prb_labels[i]
            for i in range(len(prb_labels))
        ],
        dtype=bool,
    )

    roc_res = compute_roc_auc_eer(scores, is_genuine)
    eer_th = roc_res["eer_threshold"]
    op_metrics = compute_biometric_rates(scores, is_genuine, threshold=eer_th)

    exp004b_eval = evaluate_checkpoint(
        model_path="runs/exp_003e_hpp_arcface_triplet025/best_model.pth",
        output_dir=str(output_dir / "exp004b_baseline"),
        margin_threshold=0.05,
    )

    summary = {
        "experiment_id": "EXP-006",
        "description": "3D Pose Gait Branch (YOLOv8-pose 2D -> 3D PoseLifter -> PoseGait3DNet)",
        "3d_gait_metrics": {
            "Rank-1": round(rank1_acc, 4),
            "Rank-5": round(rank5_acc, 4),
            "NM": round(cond_accs["NM"], 4),
            "BG": round(cond_accs["BG"], 4),
            "CL": round(cond_accs["CL"], 4),
            "ROC_AUC": round(roc_res["roc_auc"], 4),
            "EER": round(roc_res["eer"], 4),
            "FAR": round(op_metrics["FAR"], 4),
            "FRR": round(op_metrics["FRR"], 4),
            "TAR": round(op_metrics["TAR"], 4),
        },
        "2d_exp004b_baseline": {
            "Rank-1": exp004b_eval["rank1"],
            "Rank-5": exp004b_eval["rank5"],
            "NM": exp004b_eval["NM"],
            "BG": exp004b_eval["BG"],
            "CL": exp004b_eval["CL"],
            "ROC_AUC": exp004b_eval["ROC_AUC"],
            "EER": exp004b_eval["EER"],
            "margin_aware_FAR": exp004b_eval["margin_aware_FAR"],
            "margin_aware_FRR": exp004b_eval["margin_aware_FRR"],
            "margin_aware_TAR": exp004b_eval["margin_aware_TAR"],
        },
        "performance": {
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 2),
            "vram_mb": round(vram_mb, 1),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_file = output_dir / "exp006_3d_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n=======================================================")
    print("  EXP-003E 2D VS EXP-006 3D POSE GAIT COMPARISON")
    print("=======================================================")
    print("Metric             | 2D Baseline (EXP-003E/004B) | 3D Pose Gait (EXP-006)")
    print("-------------------------------------------------------------------------")
    print(f"Rank-1 Accuracy    | {exp004b_eval['rank1'] * 100:.2f}%                      | {rank1_acc * 100:.2f}%")
    print(f"Rank-5 Accuracy    | {exp004b_eval['rank5'] * 100:.2f}%                      | {rank5_acc * 100:.2f}%")
    print(f"NM Accuracy        | {exp004b_eval['NM'] * 100:.2f}%                      | {cond_accs['NM'] * 100:.2f}%")
    print(f"BG Accuracy        | {exp004b_eval['BG'] * 100:.2f}%                      | {cond_accs['BG'] * 100:.2f}%")
    print(f"CL Accuracy        | {exp004b_eval['CL'] * 100:.2f}%                      | {cond_accs['CL'] * 100:.2f}%")
    print(f"ROC-AUC            | {exp004b_eval['ROC_AUC']:.4f}                      | {roc_res['roc_auc']:.4f}")
    print(f"EER                | {exp004b_eval['EER'] * 100:.2f}%                      | {roc_res['eer'] * 100:.2f}%")
    print(f"Throughput (FPS)   | ~45.0                      | {fps:.1f}")
    print(f"Latency (ms)       | ~22.0ms                    | {latency_ms:.2f}ms")
    print(f"Peak VRAM          | ~850MB                     | {vram_mb:.1f}MB")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
