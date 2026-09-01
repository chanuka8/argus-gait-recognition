import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator_3d import Evaluator3D
from scripts.evaluate_exp004 import evaluate_checkpoint
from training.gait_3d_trainer import Gait3DTrainer


def main():
    print("\n=======================================================")
    print("  ARGUS EXP-006 3D POSE GAIT FULL TRAINING & EVALUATION")
    print("=======================================================\n")

    run_dir = Path("runs/exp_006_3d")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Starting 3D Gait Model Training (15 Epochs, ArcFace + Triplet 0.25)...")
    trainer = Gait3DTrainer(
        data_dir="data/casia_processed/skeletons",
        run_dir=str(run_dir),
        epochs=15,
        batch_size=32,
        learning_rate=1e-3,
        arcface_scale=30.0,
        arcface_margin=0.50,
        triplet_weight=0.25,
        sequence_length=30,
        seed=42,
    )
    train_results = trainer.train()
    print(f"Training Complete! Best Val Accuracy: {train_results['best_val_acc'] * 100:.2f}%\n")

    print("Starting Strict Subject-Disjoint 3D Gait Evaluation on Test Set (075-124)...")
    evaluator = Evaluator3D(
        model_path=str(run_dir / "best_model.pth"),
        data_dir="data/casia_processed/skeletons",
        margin_threshold=0.05,
    )
    eval_3d = evaluator.evaluate(output_dir=str(run_dir / "evaluation"))
    print("3D Evaluation Complete!\n")

    print("Evaluating 2D Baseline (EXP-003E + EXP-004B) for Comparison...")
    eval_2d = evaluate_checkpoint(
        model_path="runs/exp_003e_hpp_arcface_triplet025/best_model.pth",
        output_dir=str(run_dir / "eval_2d_baseline"),
        margin_threshold=0.05,
    )

    comparison = {
        "experiment": "EXP-006 3D Gait Branch vs EXP-003E/004B 2D Baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "3d_gait": eval_3d,
        "2d_baseline": eval_2d,
    }
    with open(run_dir / "exp006_full_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)

    m3d = eval_3d["metrics"]
    p3d = eval_3d["performance"]

    print("\n=======================================================")
    print("  EXP-003E 2D VS EXP-006 3D POSE GAIT FINAL COMPARISON")
    print("=======================================================")
    print("Metric             | 2D Baseline (EXP-003E/004B) | EXP-006 3D Pose Gait")
    print("-------------------------------------------------------------------------")
    print(f"Rank-1 Accuracy    | {eval_2d['rank1'] * 100:.2f}%                      | {m3d['rank1'] * 100:.2f}%")
    print(f"Rank-5 Accuracy    | {eval_2d['rank5'] * 100:.2f}%                      | {m3d['rank5'] * 100:.2f}%")
    print(f"NM Accuracy        | {eval_2d['NM'] * 100:.2f}%                      | {m3d['NM'] * 100:.2f}%")
    print(f"BG Accuracy        | {eval_2d['BG'] * 100:.2f}%                      | {m3d['BG'] * 100:.2f}%")
    print(f"CL Accuracy        | {eval_2d['CL'] * 100:.2f}%                      | {m3d['CL'] * 100:.2f}%")
    print(f"ROC-AUC            | {eval_2d['ROC_AUC']:.4f}                      | {m3d['ROC_AUC']:.4f}")
    print(f"EER                | {eval_2d['EER'] * 100:.2f}%                      | {m3d['EER'] * 100:.2f}%")
    print(
        f"Open-Set FAR (M)   | {eval_2d['margin_aware_FAR'] * 100:.2f}%                      | {m3d['margin_aware_FAR'] * 100:.2f}%"
    )
    print(
        f"Open-Set FRR (M)   | {eval_2d['margin_aware_FRR'] * 100:.2f}%                      | {m3d['margin_aware_FRR'] * 100:.2f}%"
    )
    print(f"Throughput (FPS)   | ~45.0                      | {p3d['fps']:.1f}")
    print(f"Latency (ms)       | ~22.0ms                    | {p3d['latency_ms']:.2f}ms")
    print(f"Peak VRAM          | ~850MB                     | {p3d['vram_mb']:.1f}MB")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
