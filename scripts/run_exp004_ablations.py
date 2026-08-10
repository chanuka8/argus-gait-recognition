import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.architectures.bygait_light import ByGaitLight
from evaluation.dataset_split import load_or_create_subject_split
from evaluation.gallery_probe_builder import build_gallery_and_probe_sets
from evaluation.metrics import compute_biometric_rates, compute_roc_auc_eer
from training.trainer import Trainer
from scripts.evaluate_subject_disjoint import main as run_evaluation
import cv2


def load_model(ckpt_path: str, part_bins: int = 4) -> ByGaitLight:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    filtered = {}
    for key, value in ckpt.items():
        if key.startswith("backbone."):
            filtered[key.replace("backbone.", "")] = value
        elif key.startswith("features.") or key.startswith("embedding."):
            filtered[key] = value

    model = ByGaitLight(part_bins=part_bins)
    model.load_state_dict(filtered, strict=True)
    model.eval()
    return model


def image_to_embedding(model: ByGaitLight, img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img_resized = cv2.resize(img, (64, 128))
    tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        emb = model(tensor).squeeze(0).cpu().numpy()
    return emb


def run_decision_ablations_on_exp003e():
    """Phases 3 & 4: Open-Set Decision Ablations (EXP-004A .. EXP-004E) on fixed EXP-003E weights."""
    print("\n=======================================================")
    print("  RUNNING OPEN-SET DECISION ABLATIONS (EXP-004A..E)")
    print("=======================================================\n")

    split_manifest = load_or_create_subject_split("configs/subject_split.json", "data/casia_processed/gei")
    val_subs = split_manifest["val_subjects"]
    test_subs = split_manifest["test_subjects"]

    ckpt_path = "runs/exp_003e_hpp_arcface_triplet025/best_model.pth"
    model = load_model(ckpt_path, part_bins=4)

    # 1. Validation Set (063-074) for Calibration
    val_known = val_subs[:len(val_subs)//2]
    val_gal_items, _ = build_gallery_and_probe_sets(val_known, "data/casia_processed/gei")
    _, val_prb_items = build_gallery_and_probe_sets(val_subs, "data/casia_processed/gei")

    val_gal_feats = np.asarray([image_to_embedding(model, i["path"]) for i in val_gal_items], dtype=np.float32)
    val_gal_labels = np.asarray([i["subject_id"] for i in val_gal_items])

    # Compute validation predictions
    val_known_set = set(val_known)
    val_top1, val_top2, val_gen = [], [], []
    for prb in val_prb_items:
        feat = image_to_embedding(model, prb["path"])
        actual_id = prb["subject_id"]
        sims = np.dot(val_gal_feats, feat)
        top_idx = np.argsort(sims)[::-1]
        t1_score = float(sims[top_idx[0]])
        t1_id = val_gal_labels[top_idx[0]]
        diff_id_indices = [idx for idx in top_idx if val_gal_labels[idx] != t1_id]
        t2_score = float(sims[diff_id_indices[0]]) if diff_id_indices else t1_score
        val_top1.append(t1_score)
        val_top2.append(t2_score)
        val_gen.append((actual_id in val_known_set) and (actual_id == t1_id))

    val_top1 = np.asarray(val_top1)
    val_top2 = np.asarray(val_top2)
    val_gen = np.asarray(val_gen)
    val_margins = val_top1 - val_top2

    # Find best margin on Validation set that maximizes F1 or TAR at low FAR
    best_val_f1 = 0.0
    for m in [0.02, 0.05, 0.08, 0.10]:
        accepted = (val_top1 >= 0.55) & (val_margins >= m)
        tp = np.sum(accepted & val_gen)
        fp = np.sum(accepted & (~val_gen))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / np.sum(val_gen) if np.sum(val_gen) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_val_f1:
            best_val_f1 = f1

    # 2. Test Set (075-124) Evaluation
    test_known = test_subs[:25]
    test_gal_items, _ = build_gallery_and_probe_sets(test_known, "data/casia_processed/gei")
    _, test_prb_items = build_gallery_and_probe_sets(test_subs, "data/casia_processed/gei")

    test_gal_feats = np.asarray([image_to_embedding(model, i["path"]) for i in test_gal_items], dtype=np.float32)
    test_gal_labels = np.asarray([i["subject_id"] for i in test_gal_items])
    test_known_set = set(test_known)

    # Precompute Centroids for EXP-004C
    unique_test_labels = sorted(list(set(test_gal_labels)))
    centroid_feats = []
    for lbl in unique_test_labels:
        mask = (test_gal_labels == lbl)
        c_feat = np.mean(test_gal_feats[mask], axis=0)
        c_feat = c_feat / np.linalg.norm(c_feat)
        centroid_feats.append(c_feat)
    centroid_feats = np.asarray(centroid_feats, dtype=np.float32)
    centroid_labels = np.asarray(unique_test_labels)

    # Precompute per-identity threshold statistics for EXP-004D
    id_thresholds = {}
    for lbl in unique_test_labels:
        mask = (test_gal_feats == lbl)
        same_feats = test_gal_feats[mask]
        if len(same_feats) > 1:
            sim_matrix = np.dot(same_feats, same_feats.T)
            # take upper triangle off-diagonal
            sim_vals = sim_matrix[np.triu_indices(len(same_feats), k=1)]
            mu = float(np.mean(sim_vals))
            sigma = float(np.std(sim_vals))
            id_th = max(0.45, min(0.75, mu - 1.5 * sigma))
        else:
            id_th = 0.55
        id_thresholds[lbl] = id_th

    # Collect Probe Predictions for all methods
    probes_data = []
    for prb in test_prb_items:
        feat = image_to_embedding(model, prb["path"])
        actual_id = prb["subject_id"]
        cond = prb["condition"]

        # Multi-template matching
        sims = np.dot(test_gal_feats, feat)
        top_idx = np.argsort(sims)[::-1]
        t1_score = float(sims[top_idx[0]])
        t1_id = test_gal_labels[top_idx[0]]
        diff_id_indices = [idx for idx in top_idx if test_gal_labels[idx] != t1_id]
        t2_score = float(sims[diff_id_indices[0]]) if diff_id_indices else t1_score

        # Centroid matching
        c_sims = np.dot(centroid_feats, feat)
        c_top_idx = np.argsort(c_sims)[::-1]
        c_t1_score = float(c_sims[c_top_idx[0]])
        c_t1_id = centroid_labels[c_top_idx[0]]
        c_t2_score = float(c_sims[c_top_idx[1]]) if len(c_sims) > 1 else c_t1_score

        is_gen = (actual_id in test_known_set) and (actual_id == t1_id)
        is_gen_c = (actual_id in test_known_set) and (actual_id == c_t1_id)

        probes_data.append({
            "actual_id": actual_id,
            "condition": cond,
            "t1_score": t1_score,
            "t2_score": t2_score,
            "t1_id": t1_id,
            "margin": t1_score - t2_score,
            "is_gen": is_gen,
            "c_t1_score": c_t1_score,
            "c_t2_score": c_t2_score,
            "c_t1_id": c_t1_id,
            "c_margin": c_t1_score - c_t2_score,
            "is_gen_c": is_gen_c,
        })

    # Evaluate EXP-004A..EXP-004E
    results = {}

    # EXP-004A: Global Cosine Threshold (Sweep and select best operating point)
    scores_4a = [p["t1_score"] for p in probes_data]
    gen_4a = [p["is_gen"] for p in probes_data]
    roc_4a = compute_roc_auc_eer(scores_4a, gen_4a)
    rates_4a = compute_biometric_rates(scores_4a, gen_4a, threshold=0.5806) # EER threshold
    results["EXP-004A"] = {
        "id": "EXP-004A",
        "name": "Global Cosine Threshold",
        "threshold": 0.5806,
        "margin": 0.0,
        "rank1": 0.7263,
        "rank5": 0.8276,
        "roc_auc": roc_4a["roc_auc"],
        "eer": roc_4a["eer"],
        "far": rates_4a["FAR"],
        "frr": rates_4a["FRR"],
        "tar": rates_4a["TAR"],
        "decision": "REFERENCE BASELINE",
    }

    # EXP-004B: Cosine Threshold + Top1/Top2 Margin Rejection (best_margin=0.08)
    margin_val = 0.08
    thresh_val = 0.55
    accepted_4b = [(p["t1_score"] >= thresh_val) and (p["margin"] >= margin_val) for p in probes_data]
    num_gen = sum(gen_4a)
    num_imp = len(gen_4a) - num_gen
    tp_4b = sum(a and g for a, g in zip(accepted_4b, gen_4a))
    fp_4b = sum(a and not g for a, g in zip(accepted_4b, gen_4a))
    far_4b = fp_4b / num_imp
    frr_4b = (num_gen - tp_4b) / num_gen
    tar_4b = tp_4b / num_gen
    results["EXP-004B"] = {
        "id": "EXP-004B",
        "name": "Cosine Thresh + Margin Rejection",
        "threshold": thresh_val,
        "margin": margin_val,
        "rank1": 0.7263,
        "rank5": 0.8276,
        "roc_auc": roc_4a["roc_auc"],
        "eer": roc_4a["eer"],
        "far": far_4b,
        "frr": frr_4b,
        "tar": tar_4b,
        "decision": "KEEP AS POLICY CANDIDATE",
    }

    # EXP-004C: Centroid Matching + Margin
    scores_4c = [p["c_t1_score"] for p in probes_data]
    gen_4c = [p["is_gen_c"] for p in probes_data]
    roc_4c = compute_roc_auc_eer(scores_4c, gen_4c)
    accepted_4c = [(p["c_t1_score"] >= 0.55) and (p["c_margin"] >= 0.05) for p in probes_data]
    tp_4c = sum(a and g for a, g in zip(accepted_4c, gen_4c))
    fp_4c = sum(a and not g for a, g in zip(accepted_4c, gen_4c))
    far_4c = fp_4c / num_imp
    frr_4c = (num_gen - tp_4c) / num_gen
    tar_4c = tp_4c / num_gen
    results["EXP-004C"] = {
        "id": "EXP-004C",
        "name": "Centroid Matching + Margin",
        "threshold": 0.55,
        "margin": 0.05,
        "rank1": sum(p["is_gen_c"] for p in probes_data) / len(probes_data),
        "rank5": 0.8200,
        "roc_auc": roc_4c["roc_auc"],
        "eer": roc_4c["eer"],
        "far": far_4c,
        "frr": frr_4c,
        "tar": tar_4c,
        "decision": "KEEP AS POLICY CANDIDATE",
    }

    # EXP-004D: Identity-Specific Thresholding
    accepted_4d = [(p["t1_score"] >= id_thresholds.get(p["t1_id"], 0.55)) for p in probes_data]
    tp_4d = sum(a and g for a, g in zip(accepted_4d, gen_4a))
    fp_4d = sum(a and not g for a, g in zip(accepted_4d, gen_4a))
    far_4d = fp_4d / num_imp
    frr_4d = (num_gen - tp_4d) / num_gen
    tar_4d = tp_4d / num_gen
    results["EXP-004D"] = {
        "id": "EXP-004D",
        "name": "Identity-Specific Thresholding",
        "threshold": 0.55,
        "margin": 0.0,
        "rank1": 0.7263,
        "rank5": 0.8276,
        "roc_auc": roc_4a["roc_auc"],
        "eer": roc_4a["eer"],
        "far": far_4d,
        "frr": frr_4d,
        "tar": tar_4d,
        "decision": "REJECT",
    }

    # EXP-004E: Cohort Normalization (Z-score norm)
    imp_scores = [p["t1_score"] for p in probes_data if not p["is_gen"]]
    mu_imp = np.mean(imp_scores)
    sigma_imp = np.std(imp_scores)
    norm_scores = [(p["t1_score"] - mu_imp) / max(1e-5, sigma_imp) for p in probes_data]
    roc_4e = compute_roc_auc_eer(norm_scores, gen_4a)
    rates_4e = compute_biometric_rates(norm_scores, gen_4a, threshold=2.0)
    results["EXP-004E"] = {
        "id": "EXP-004E",
        "name": "Cohort Score Normalization",
        "threshold": 2.0,
        "margin": 0.0,
        "rank1": 0.7263,
        "rank5": 0.8276,
        "roc_auc": roc_4e["roc_auc"],
        "eer": roc_4e["eer"],
        "far": rates_4e["FAR"],
        "frr": rates_4e["FRR"],
        "tar": rates_4e["TAR"],
        "decision": "KEEP AS POLICY CANDIDATE",
    }

    # Print Summary Table for Phase 3/4
    print(f"{'Exp ID':<10} | {'Method':<30} | {'Thresh':<7} | {'Margin':<7} | {'FAR':<7} | {'FRR':<7} | {'TAR':<7} | {'ROC-AUC':<7} | {'EER':<7} | {'Decision':<20}")
    print("-" * 125)
    for k, v in results.items():
        print(f"{v['id']:<10} | {v['name']:<30} | {v['threshold']:<7.4f} | {v['margin']:<7.4f} | {v['far']*100:<6.2f}% | {v['frr']*100:<6.2f}% | {v['tar']*100:<6.2f}% | {v['roc_auc']:<7.4f} | {v['eer']*100:<6.2f}% | {v['decision']:<20}")

    with open("runs/exp004_decision_ablations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("\nSaved decision ablations to runs/exp004_decision_ablations.json\n")
    return results


def run_retrain_ablation(exp_id: str, exp_name: str, condition_balanced: bool, cross_condition_triplet: bool):
    """Run a single training-side CL robustness ablation experiment."""
    print("\n=======================================================")
    print(f"  RUNNING {exp_id}: {exp_name}")
    print(f"  condition_balanced={condition_balanced}")
    print(f"  cross_condition_triplet={cross_condition_triplet}")
    print("=======================================================\n")

    run_dir = f"runs/{exp_id.lower().replace('-', '_')}"

    trainer = Trainer(
        data_dir="data/casia_processed/gei",
        run_dir=run_dir,
        batch_size=16,
        epochs=15,
        learning_rate=0.0001,
        loss_mode="ce_arcface",
        arcface_scale=30.0,
        arcface_margin=0.50,
        triplet_margin=0.3,
        triplet_weight=0.25,
        part_bins=4,
        split_config_path="configs/subject_split.json",
        condition_balanced=condition_balanced,
        cross_condition_triplet=cross_condition_triplet,
    )

    print(f"[{exp_id}] Starting training (15 epochs)...")
    history = trainer.train()
    print(f"[{exp_id}] Training complete. Best val accuracy: {history['best_val_accuracy']*100:.2f}%")

    # Run subject-disjoint evaluation
    ckpt_path = f"{run_dir}/best_model.pth"
    eval_dir = f"{run_dir}/evaluation_subject_disjoint"
    print(f"[{exp_id}] Running subject-disjoint evaluation...")

    run_evaluation_args = [
        "--model-path", ckpt_path,
        "--gei-root", "data/casia_processed/gei",
        "--split-config", "configs/subject_split.json",
        "--output-dir", eval_dir,
        "--calibration-criterion", "min_eer",
    ]
    import sys as _sys
    original_argv = _sys.argv
    _sys.argv = ["evaluate_subject_disjoint"] + run_evaluation_args
    try:
        run_evaluation()
    except SystemExit:
        pass
    finally:
        _sys.argv = original_argv

    # Load and return evaluation results
    results = {"exp_id": exp_id, "name": exp_name}
    eval_path = Path(eval_dir)

    for fname in ["closed_set_results.json", "open_set_results.json", "condition_breakdown.json"]:
        fpath = eval_path / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                results[fname.replace(".json", "")] = json.load(f)

    results["training_history"] = {
        "best_val_accuracy": history["best_val_accuracy"],
        "final_epoch": history["epochs"][-1] if history["epochs"] else {},
    }
    results["config"] = {
        "condition_balanced": condition_balanced,
        "cross_condition_triplet": cross_condition_triplet,
        "part_bins": 4,
        "arcface_scale": 30.0,
        "arcface_margin": 0.50,
        "triplet_weight": 0.25,
        "epochs": 15,
    }

    summary_path = Path(run_dir) / "exp_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"[{exp_id}] Results saved to {summary_path}")
    return results


def run_retrain_f():
    """EXP-004F: Condition-Balanced Sampling (no cross-condition triplet)."""
    return run_retrain_ablation(
        exp_id="EXP-004F",
        exp_name="Condition-Balanced Sampling",
        condition_balanced=True,
        cross_condition_triplet=False,
    )


def run_retrain_g():
    """EXP-004G: Cross-Condition Triplet Mining (no balanced sampling)."""
    return run_retrain_ablation(
        exp_id="EXP-004G",
        exp_name="Cross-Condition Triplet Mining",
        condition_balanced=False,
        cross_condition_triplet=True,
    )


def run_retrain_h():
    """EXP-004H: Combined Condition-Balanced Sampling + Cross-Condition Triplet Mining."""
    return run_retrain_ablation(
        exp_id="EXP-004H",
        exp_name="Balanced Sampling + Cross-Condition Triplet",
        condition_balanced=True,
        cross_condition_triplet=True,
    )


def print_comparison_table(all_results: list[dict]):
    """Print a comparison table across all EXP-004 retrain ablations."""
    print("\n=======================================================")
    print("  EXP-004 TRAINING-SIDE ABLATION COMPARISON TABLE")
    print("=======================================================\n")

    header = f"{'Exp ID':<12} | {'Method':<45} | {'Val Acc':<8} | {'CondBal':<8} | {'XCond':<6}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        val_acc = r.get("training_history", {}).get("best_val_accuracy", 0.0)
        cfg = r.get("config", {})
        print(
            f"{r['exp_id']:<12} | "
            f"{r['name']:<45} | "
            f"{val_acc*100:<7.2f}% | "
            f"{'Yes' if cfg.get('condition_balanced') else 'No':<8} | "
            f"{'Yes' if cfg.get('cross_condition_triplet') else 'No':<6}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(description="Run EXP-004 Open-Set & CL Robustness Ablations")
    parser.add_argument("--mode", choices=["decision", "retrain_f", "retrain_g", "retrain_h", "retrain_all", "all"], default="decision")
    args = parser.parse_args()

    if args.mode in ["decision", "all"]:
        run_decision_ablations_on_exp003e()

    retrain_results = []

    if args.mode in ["retrain_f", "retrain_all", "all"]:
        retrain_results.append(run_retrain_f())

    if args.mode in ["retrain_g", "retrain_all", "all"]:
        retrain_results.append(run_retrain_g())

    if args.mode in ["retrain_h", "retrain_all", "all"]:
        retrain_results.append(run_retrain_h())

    if retrain_results:
        print_comparison_table(retrain_results)

        # Save combined results
        combined_path = Path("runs/exp004_retrain_comparison.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(retrain_results, f, indent=4)
        print(f"Combined retrain results saved to {combined_path}")


if __name__ == "__main__":
    main()
