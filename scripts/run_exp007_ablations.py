"""
EXP-007 Controlled Ablation Study and Optimization:
Ablates Encoders (TCN, ST-GCN, CTR-GCN), Sequence Lengths (30, 60, 90),
and Loss Parameters (ArcFace s=30/64, Triplet w=0.25/0.50) on VAL ONLY (063-074).

Selects the best performing candidate on VAL, evaluates ONCE on TEST (075-124),
and promotes candidate to models/candidates/gait_3d_exp007_best.pth.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset_split import load_or_create_subject_split
from evaluation.evaluator_3d import Evaluator3D
from scripts.evaluate_exp004 import evaluate_checkpoint
from training.gait_3d_trainer import Gait3DTrainer


def evaluate_on_val(model_path: str, sequence_length: int = 30) -> float:
    """Evaluates candidate checkpoint on VAL set (063-074) only."""
    evaluator = Evaluator3D(
        model_path=model_path,
        data_dir="data/casia_processed/skeletons",
        sequence_length=sequence_length,
    )
    lifter, gait_net = evaluator._load_model()
    split_manifest = load_or_create_subject_split("configs/subject_split.json", "data/casia_processed/gei")
    val_subs = split_manifest["val_subjects"]

    gal_items, prb_items = evaluator._build_gallery_and_probes(val_subs)
    if not gal_items or not prb_items:
        return 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_emb(path: str) -> np.ndarray:
        seq = np.load(path)
        if len(seq) < sequence_length:
            repeats = (sequence_length // len(seq)) + 1
            seq_fixed = np.tile(seq, (repeats, 1, 1))[:sequence_length]
        else:
            seq_fixed = seq[:sequence_length]
        tensor = torch.from_numpy(seq_fixed).float().unsqueeze(0).to(device)
        with torch.no_grad():
            j3d = lifter(tensor)
            emb = gait_net(j3d).squeeze(0).cpu().numpy()
        return emb

    gal_feats = np.stack([extract_emb(g["path"]) for g in gal_items], axis=0)
    gal_labels = np.array([g["subject_id"] for g in gal_items])

    prb_feats = np.stack([extract_emb(p["path"]) for p in prb_items], axis=0)
    prb_labels = np.array([p["subject_id"] for p in prb_items])

    sims = np.dot(prb_feats, gal_feats.T)
    top1_hits = 0
    for i in range(len(prb_labels)):
        best_idx = np.argmax(sims[i])
        if gal_labels[best_idx] == prb_labels[i]:
            top1_hits += 1

    return top1_hits / float(len(prb_labels))


def main():
    print("\n=======================================================")
    print("  ARGUS EXP-007 3D POSE GAIT CONTROLLED ABLATION STUDY")
    print("=======================================================\n")

    base_dir = Path("runs/exp_007_ablations")
    base_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {"name": "exp007_tcn_seq30", "encoder": "tcn", "seq_len": 30, "s": 30.0, "m": 0.50, "triplet_w": 0.25},
        {"name": "exp007_stgcn_seq30", "encoder": "stgcn", "seq_len": 30, "s": 30.0, "m": 0.50, "triplet_w": 0.25},
        {"name": "exp007_ctrgcn_seq30", "encoder": "ctrgcn", "seq_len": 30, "s": 30.0, "m": 0.50, "triplet_w": 0.25},
        {"name": "exp007_stgcn_seq60", "encoder": "stgcn", "seq_len": 60, "s": 30.0, "m": 0.50, "triplet_w": 0.25},
        {"name": "exp007_ctrgcn_seq60", "encoder": "ctrgcn", "seq_len": 60, "s": 30.0, "m": 0.50, "triplet_w": 0.25},
        {"name": "exp007_ctrgcn_s64_w05", "encoder": "ctrgcn", "seq_len": 30, "s": 64.0, "m": 0.50, "triplet_w": 0.50},
    ]

    results = []
    best_val_score = -1.0
    best_config = None
    best_ckpt_path = None

    for cfg in configs:
        run_dir = base_dir / cfg["name"]
        print(f"\n---> Training Candidate Ablation: {cfg['name']} (Encoder: {cfg['encoder'].upper()}, SeqLen: {cfg['seq_len']})")

        trainer = Gait3DTrainer(
            data_dir="data/casia_processed/skeletons",
            run_dir=str(run_dir),
            encoder_type=cfg["encoder"],
            epochs=15,
            batch_size=32,
            learning_rate=1e-3,
            arcface_scale=cfg["s"],
            arcface_margin=cfg["m"],
            triplet_weight=cfg["triplet_w"],
            sequence_length=cfg["seq_len"],
            seed=42,
        )

        train_res = trainer.train()
        ckpt_path = str(run_dir / "best_model.pth")

        val_rank1 = evaluate_on_val(ckpt_path, sequence_length=cfg["seq_len"])
        print(f"[{cfg['name']}] VAL Rank-1 Accuracy: {val_rank1*100:.2f}% (Val Training Acc: {train_res['best_val_acc']*100:.2f}%)")

        res_record = {
            "config": cfg,
            "val_acc_training": train_res["best_val_acc"],
            "val_rank1_eval": round(val_rank1, 4),
            "ckpt_path": ckpt_path,
        }
        results.append(res_record)

        if val_rank1 > best_val_score:
            best_val_score = val_rank1
            best_config = cfg
            best_ckpt_path = ckpt_path

    print("\n=======================================================")
    print(f"  WINNING CANDIDATE ON VAL: {best_config['name']}")
    print(f"  VAL Rank-1 Accuracy: {best_val_score*100:.2f}%")
    print("=======================================================\n")

    cand_dir = Path("models/candidates")
    cand_dir.mkdir(parents=True, exist_ok=True)
    promoted_path = cand_dir / "gait_3d_exp007_best.pth"

    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    torch.save(best_ckpt, promoted_path)
    print(f"Promoted winning checkpoint to: {promoted_path}")

    print("\nStarting Final TEST Evaluation of Promoted Candidate (075-124)...")
    evaluator = Evaluator3D(
        model_path=str(promoted_path),
        data_dir="data/casia_processed/skeletons",
        sequence_length=best_config["seq_len"],
        margin_threshold=0.05,
    )
    test_eval_3d = evaluator.evaluate(output_dir=str(base_dir / "best_test_eval"))

    test_eval_2d = evaluate_checkpoint(
        model_path="runs/exp_003e_hpp_arcface_triplet025/best_model.pth",
        output_dir=str(base_dir / "eval_2d_baseline"),
        margin_threshold=0.05,
    )

    summary = {
        "experiment": "EXP-007 3D Pose Gait Ablation and Optimization Study",
        "best_config": best_config,
        "val_ablation_results": results,
        "test_eval_winning_3d": test_eval_3d,
        "test_eval_2d_baseline": test_eval_2d,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(base_dir / "exp007_ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    m3d = test_eval_3d["metrics"]
    p3d = test_eval_3d["performance"]

    print("\n=========================================================================")
    print("  EXP-006 VS EXP-007 ABLATIONS VS 2D BASELINE FINAL COMPARISON")
    print("=========================================================================")
    print("Metric                 | 2D Baseline (EXP-003E) | EXP-006 (Baseline 3D) | EXP-007 (Best 3D)")
    print("---------------------------------------------------------------------------------")
    print(f"Rank-1 Accuracy        | 72.63%                 | 19.45%                | {m3d['rank1']*100:.2f}%")
    print(f"Rank-5 Accuracy        | 82.76%                 | 49.35%                | {m3d['rank5']*100:.2f}%")
    print(f"NM Accuracy            | 97.00%                 | 26.82%                | {m3d['NM']*100:.2f}%")
    print(f"BG Accuracy            | 78.26%                 | 16.29%                | {m3d['BG']*100:.2f}%")
    print(f"CL Accuracy            | 42.64%                 | 8.24%                 | {m3d['CL']*100:.2f}%")
    print(f"ROC-AUC                | 0.8776                 | 0.6552                | {m3d['ROC_AUC']:.4f}")
    print(f"EER                    | 20.46%                 | 39.90%                | {m3d['EER']*100:.2f}%")
    print(f"Open-Set FAR (M=0.05)  | 18.05%                 | 0.00%                 | {m3d['margin_aware_FAR']*100:.2f}%")
    print(f"Open-Set TAR (M=0.05)  | 33.18%                 | 0.00%                 | {m3d['margin_aware_TAR']*100:.2f}%")
    print(f"Throughput (FPS)       | ~45.0                  | 84.9                  | {p3d['fps']:.1f}")
    print(f"Latency (ms)           | ~22.0ms                | 11.79ms               | {p3d['latency_ms']:.2f}ms")
    print(f"Peak VRAM              | ~850MB                 | 18.1MB                | {p3d['vram_mb']:.1f}MB")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
