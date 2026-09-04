import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_subject_disjoint import main as run_evaluation
from training.trainer import Trainer

EXPERIMENTS = [
    {
        "id": "EXP-003A",
        "name": "exp_003a_global_ce_triplet",
        "part_bins": 1,
        "loss_mode": "ce",
        "triplet_weight": 0.5,
        "arcface_scale": 30.0,
        "arcface_margin": 0.50,
        "description": "Original global pooling + CE + Triplet baseline reproduction",
    },
    {
        "id": "EXP-003B",
        "name": "exp_003b_hpp_ce_triplet",
        "part_bins": 4,
        "loss_mode": "ce",
        "triplet_weight": 0.5,
        "arcface_scale": 30.0,
        "arcface_margin": 0.50,
        "description": "HPP part_bins=4 + CE + Triplet (HPP effect alone)",
    },
    {
        "id": "EXP-003C",
        "name": "exp_003c_global_arcface_triplet",
        "part_bins": 1,
        "loss_mode": "ce_arcface",
        "triplet_weight": 0.5,
        "arcface_scale": 30.0,
        "arcface_margin": 0.50,
        "description": "Original global pooling + ArcFace + Triplet (ArcFace effect alone)",
    },
    {
        "id": "EXP-003D",
        "name": "exp_003d_hpp_arcface_only",
        "part_bins": 4,
        "loss_mode": "ce_arcface",
        "triplet_weight": 0.0,
        "arcface_scale": 30.0,
        "arcface_margin": 0.50,
        "description": "HPP part_bins=4 + ArcFace only (no triplet interaction)",
    },
    {
        "id": "EXP-003E",
        "name": "exp_003e_hpp_arcface_triplet025",
        "part_bins": 4,
        "loss_mode": "ce_arcface",
        "triplet_weight": 0.25,
        "arcface_scale": 30.0,
        "arcface_margin": 0.50,
        "description": "HPP part_bins=4 + ArcFace + Triplet weight 0.25",
    },
]


def run_experiment(exp: dict, epochs: int = 25, batch_size: int = 16, lr: float = 0.0001) -> dict:
    run_dir = Path("runs") / exp["name"]
    eval_dir = run_dir / "evaluation_subject_disjoint"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f" STARTING EXPERIMENT: {exp['id']} ({exp['name']})")
    print(f" Description: {exp['description']}")
    print(
        f" Config: part_bins={exp['part_bins']}, loss_mode={exp['loss_mode']}, triplet_weight={exp['triplet_weight']}"
    )
    print("=======================================================\n")

    trainer = Trainer(
        data_dir="data/casia_processed/gei",
        run_dir=str(run_dir),
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        part_bins=exp["part_bins"],
        loss_mode=exp["loss_mode"],
        triplet_weight=exp["triplet_weight"],
        arcface_scale=exp["arcface_scale"],
        arcface_margin=exp["arcface_margin"],
        split_config_path="configs/subject_split.json",
    )
    trainer.train()

    best_ckpt = run_dir / "best_model.pth"
    orig_argv = sys.argv
    sys.argv = [
        "evaluate_subject_disjoint.py",
        "--model-path",
        str(best_ckpt),
        "--gei-root",
        "data/casia_processed/gei",
        "--split-config",
        "configs/subject_split.json",
        "--output-dir",
        str(eval_dir),
    ]
    try:
        run_evaluation()
    finally:
        sys.argv = orig_argv

    with open(eval_dir / "closed_set_eval_report.json", "r", encoding="utf-8") as f:
        closed_res = json.load(f)
    with open(eval_dir / "open_set_report.json", "r", encoding="utf-8") as f:
        open_res = json.load(f)
    with open(eval_dir / "threshold_calibration.json", "r", encoding="utf-8") as f:
        calib_res = json.load(f)

    summary = {
        "exp_id": exp["id"],
        "name": exp["name"],
        "part_bins": exp["part_bins"],
        "loss_mode": exp["loss_mode"],
        "triplet_weight": exp["triplet_weight"],
        "rank1": closed_res["rank1_accuracy"],
        "rank5": closed_res["rank5_accuracy"],
        "nm": closed_res["condition_wise_accuracy"]["NM"]["rank1_accuracy"],
        "bg": closed_res["condition_wise_accuracy"]["BG"]["rank1_accuracy"],
        "cl": closed_res["condition_wise_accuracy"]["CL"]["rank1_accuracy"],
        "roc_auc": open_res["ROC_AUC"],
        "eer": open_res["EER"],
        "far": open_res["operating_metrics"]["FAR"],
        "frr": open_res["operating_metrics"]["FRR"],
        "threshold": calib_res["selected_threshold"],
        "calib_min_score": calib_res["calibration_score_min"],
        "calib_max_score": calib_res["calibration_score_max"],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Full ARGUS Gait Ablation Study (EXP-003A..E)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()

    results = []
    for exp in EXPERIMENTS:
        res = run_experiment(exp, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        results.append(res)

    out_file = Path("runs/ablation_study_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n=======================================================")
    print("         ABLATION STUDY COMPLETE - SUMMARY TABLE       ")
    print("=======================================================\n")
    header = f"{'Exp ID':<10} | {'Pooling':<8} | {'Loss':<12} | {'TripW':<6} | {'Rank-1':<7} | {'Rank-5':<7} | {'NM':<7} | {'BG':<7} | {'CL':<7} | {'ROC-AUC':<7} | {'FAR':<7} | {'Thresh':<7}"
    print(header)
    print("-" * len(header))
    for r in results:
        pool_str = "HPP (4)" if r["part_bins"] == 4 else "Global(1)"
        print(
            f"{r['exp_id']:<10} | {pool_str:<8} | {r['loss_mode']:<12} | {r['triplet_weight']:<6.2f} | {r['rank1'] * 100:<6.2f}% | {r['rank5'] * 100:<6.2f}% | {r['nm'] * 100:<6.2f}% | {r['bg'] * 100:<6.2f}% | {r['cl'] * 100:<6.2f}% | {r['roc_auc']:<7.4f} | {r['far'] * 100:<6.2f}% | {r['threshold']:<7.4f}"
        )

    print(f"\nSaved full ablation summary to {out_file}\n")


if __name__ == "__main__":
    main()
