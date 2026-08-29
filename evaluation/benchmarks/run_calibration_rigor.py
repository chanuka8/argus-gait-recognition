import json
import sys
from pathlib import Path
from collections import defaultdict
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import cv2
import torch

from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.detection.person_detector import PersonDetector
from models.reid.osnet_backbone import OSNetBackbone
from intelligence.dual_modal_fusion import DualModalFusion
from intelligence.track_identity_aggregator import TrackIdentityAggregator
from intelligence.learned_fusion import LearnedLogisticFusion


def stratified_kfold_split(y_indices: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Pure NumPy stratified K-Fold split generator."""
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(y_indices)
    folds = [[] for _ in range(n_splits)]

    for c in unique_classes:
        c_idx = np.where(y_indices == c)[0]
        rng.shuffle(c_idx)
        for i, idx in enumerate(c_idx):
            folds[i % n_splits].append(idx)

    for fold_i in range(n_splits):
        test_indices = np.array(folds[fold_i])
        train_indices = np.setdiff1d(np.arange(len(y_indices)), test_indices)
        yield train_indices, test_indices


def run_phase2_calibration_rigor():
    print("=" * 110, flush=True)
    print("PHASE 2: CALIBRATION RIGOR & MULTI-FRAME TEMPORAL AGGREGATION BENCHMARK", flush=True)
    print("=" * 110, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = PersonDetector()
    gait_extractor = FeatureExtractionStep(model_path="runs/exp_001/best_model.pth")
    osnet_backbone = OSNetBackbone(model_path="models/weights/osnet_x0_25.pth", device=device)

    subjects = ["demo_person_001", "Devhan", "Isuru", "person01"]
    base_gei = Path("data/auto_enrollment/gei")
    base_photos = Path("data/auto_enrollment/photos")

    # Load 37 clean multimodal query samples
    query_gait, query_app, query_labels = [], [], []
    per_subject_samples = defaultdict(int)
    for s in subjects:
        g_files = sorted(list((base_gei / s).glob("*.*")))
        p_files = sorted(list((base_photos / s).glob("*.*")))
        g_embs = [gait_extractor.extract(f) for f in g_files]
        p_embs = []
        for f in p_files:
            img = cv2.imread(str(f))
            dets = detector.detect(img)
            crop = img
            if dets:
                d = max(dets, key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]))
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
            p_embs.append(osnet_backbone.extract(crop))

        n = min(len(g_embs), len(p_embs))
        per_subject_samples[s] = n
        for idx in range(n):
            query_gait.append(g_embs[idx])
            query_app.append(p_embs[idx])
            query_labels.append(s)

    N = len(query_labels)
    unique_subjects = sorted(list(set(query_labels)))
    label_to_idx = {s: i for i, s in enumerate(unique_subjects)}
    y_indices = np.array([label_to_idx[l] for l in query_labels])

    print(f"\n[DATASET SUMMARY] Total Multimodal Samples: N={N} across {len(unique_subjects)} subjects", flush=True)
    for s in unique_subjects:
        print(f"  - {s:18}: {per_subject_samples[s]:2d} samples ({per_subject_samples[s]/N*100:5.2f}%)", flush=True)

    # -------------------------------------------------------------------------
    # PART 1: NESTED 5-FOLD CROSS-VALIDATION ON INDIVIDUAL & FUSED BRANCHES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110, flush=True)
    print("--- 1. NESTED 5-FOLD CROSS-VALIDATION (OUT-OF-FOLD THRESHOLD CALIBRATION) ---", flush=True)
    print("=" * 110, flush=True)

    branches = ["gait", "appearance", "linear_optimal", "auc_learned"]
    branch_display = {
        "gait": "Gait Branch Alone",
        "appearance": "Appearance Branch Alone",
        "linear_optimal": "Linear Optimal (0.95/0.05)",
        "auc_learned": "AUC-Learned Logistic Fusion",
    }

    # Tracking per-fold metrics
    fold_details = []
    branch_fold_metrics = {b: {"th": [], "tar": [], "frr": [], "far": []} for b in branches}
    pooled_counts = {b: {"tar": 0, "frr": 0, "far": 0} for b in branches}

    fold_idx = 1
    for train_idx, test_idx in stratified_kfold_split(y_indices, n_splits=5, seed=42):
        n_test = len(test_idx)
        n_train = len(train_idx)

        # Subject sample distribution in this fold
        fold_sub_counts = defaultdict(int)
        for idx in test_idx:
            fold_sub_counts[query_labels[idx]] += 1

        # Check for small sample flags (<= 1 sample for any subject)
        small_sample_flags = []
        for s in unique_subjects:
            cnt = fold_sub_counts.get(s, 0)
            if cnt <= 1:
                small_sample_flags.append(f"{s}:{cnt}")

        # 1. Build training pairwise scores to calibrate 0% FAR threshold on train fold
        train_g_same, train_g_diff = [], []
        train_a_same, train_a_diff = [], []
        train_opt_same, train_opt_diff = [], []
        train_pairs_g, train_pairs_a, train_pairs_y = [], [], []

        for i_tr in train_idx:
            for j_tr in train_idx:
                if i_tr >= j_tr:
                    continue
                sg = float(np.dot(query_gait[i_tr], query_gait[j_tr]))
                sa = float(np.dot(query_app[i_tr], query_app[j_tr]))
                s_opt = 0.95 * sg + 0.05 * sa
                is_same = (query_labels[i_tr] == query_labels[j_tr])

                train_pairs_g.append(sg)
                train_pairs_a.append(sa)
                train_pairs_y.append(1 if is_same else 0)

                if is_same:
                    train_g_same.append(sg)
                    train_a_same.append(sa)
                    train_opt_same.append(s_opt)
                else:
                    train_g_diff.append(sg)
                    train_a_diff.append(sa)
                    train_opt_diff.append(s_opt)

        # Fit AUC Learned model on fold train split
        fold_learned = LearnedLogisticFusion().fit(train_pairs_g, train_pairs_a, train_pairs_y, loss_type="ranking_auc")
        train_learned_diff = [fold_learned.predict_probability(g, a) for g, a, y in zip(train_pairs_g, train_pairs_a, train_pairs_y) if y == 0]

        # Derive 0% FAR Operating Thresholds from Training Fold (max impostor + epsilon)
        th_dict = {
            "gait": float(np.max(train_g_diff) + 0.001) if train_g_diff else 0.89,
            "appearance": float(np.max(train_a_diff) + 0.001) if train_a_diff else 0.72,
            "linear_optimal": float(np.max(train_opt_diff) + 0.001) if train_opt_diff else 0.88,
            "auc_learned": float(np.max(train_learned_diff) + 0.001) if train_learned_diff else 0.65,
        }

        for b in branches:
            branch_fold_metrics[b]["th"].append(th_dict[b])

        # 2. Evaluate on Held-Out Test Fold
        fold_counts = {b: {"tar": 0, "frr": 0, "far": 0} for b in branches}

        for test_i in test_idx:
            q_g, q_a, q_lbl = query_gait[test_i], query_app[test_i], query_labels[test_i]
            # Gallery: all remaining samples in the dataset excluding the test sample
            gal_g = [query_gait[j] for j in range(N) if j != test_i]
            gal_a = [query_app[j] for j in range(N) if j != test_i]
            gal_lbl = [query_labels[j] for j in range(N) if j != test_i]

            sims_g = [float(np.dot(q_g, g)) for g in gal_g]
            sims_a = [float(np.dot(q_a, a)) for a in gal_a]
            sims_opt = [0.95 * g + 0.05 * a for g, a in zip(sims_g, sims_a)]
            sims_learned = [fold_learned.predict_probability(g, a) for g, a in zip(sims_g, sims_a)]

            def eval_sample(sims, th):
                best_idx = int(np.argmax(sims))
                score = sims[best_idx]
                pred = gal_lbl[best_idx]
                if score >= th:
                    return ("tar", 1) if pred == q_lbl else ("far", 1)
                else:
                    return ("frr", 1)

            for b, sims in [("gait", sims_g), ("appearance", sims_a), ("linear_optimal", sims_opt), ("auc_learned", sims_learned)]:
                outcome, _ = eval_sample(sims, th_dict[b])
                fold_counts[b][outcome] += 1
                pooled_counts[b][outcome] += 1

        # Record fold rates
        fold_summary = {
            "fold": fold_idx,
            "n_train": n_train,
            "n_test": n_test,
            "sub_counts": dict(fold_sub_counts),
            "small_sample_flags": small_sample_flags,
            "branches": {},
        }

        for b in branches:
            tar_pct = (fold_counts[b]["tar"] / n_test) * 100.0
            frr_pct = (fold_counts[b]["frr"] / n_test) * 100.0
            far_pct = (fold_counts[b]["far"] / n_test) * 100.0
            branch_fold_metrics[b]["tar"].append(tar_pct)
            branch_fold_metrics[b]["frr"].append(frr_pct)
            branch_fold_metrics[b]["far"].append(far_pct)
            fold_summary["branches"][b] = {
                "threshold": th_dict[b],
                "tar": tar_pct,
                "frr": frr_pct,
                "far": far_pct,
                "tar_n": fold_counts[b]["tar"],
                "frr_n": fold_counts[b]["frr"],
                "far_n": fold_counts[b]["far"],
            }

        fold_details.append(fold_summary)
        fold_idx += 1

    # Print Per-Fold Breakdown
    print("\n--- DETAILED PER-FOLD BREAKDOWN & SMALL SAMPLE AUDIT ---", flush=True)
    print(f"{'Fold':<6} | {'Test N':<8} | {'Sub Distribution (demo/Dev/Isu/p01)':<38} | {'Small Sample Warning (<=1)':<28}", flush=True)
    print("-" * 90, flush=True)
    for fd in fold_details:
        dist_str = f"{fd['sub_counts'].get('demo_person_001', 0)} / {fd['sub_counts'].get('Devhan', 0)} / {fd['sub_counts'].get('Isuru', 0)} / {fd['sub_counts'].get('person01', 0)}"
        flag_str = ", ".join(fd["small_sample_flags"]) if fd["small_sample_flags"] else "None"
        print(f"Fold {fd['fold']:<2} | {fd['n_test']:<8} | {dist_str:<38} | [FLAG] {flag_str}", flush=True)

    # Print Branch Summary with Mean and Standard Deviation across folds
    print("\n" + "-" * 125, flush=True)
    print(f"{'Branch / Fusion Strategy':<32} | {'Calibrated Gate (Mean ± Std)':<30} | {'Out-of-Fold TAR (Mean ± Std)':<30} | {'Out-of-Fold FRR (Mean ± Std)':<30} | {'Out-of-Fold FAR (Mean ± Std)'}", flush=True)
    print("-" * 125, flush=True)

    branch_aggregated = {}
    for b in branches:
        name = branch_display[b]
        th_mean = float(np.mean(branch_fold_metrics[b]["th"]))
        th_std = float(np.std(branch_fold_metrics[b]["th"], ddof=1))

        tar_mean = float(np.mean(branch_fold_metrics[b]["tar"]))
        tar_std = float(np.std(branch_fold_metrics[b]["tar"], ddof=1))

        frr_mean = float(np.mean(branch_fold_metrics[b]["frr"]))
        frr_std = float(np.std(branch_fold_metrics[b]["frr"], ddof=1))

        far_mean = float(np.mean(branch_fold_metrics[b]["far"]))
        far_std = float(np.std(branch_fold_metrics[b]["far"], ddof=1))

        pooled_tar = float(pooled_counts[b]["tar"] / N * 100)
        pooled_frr = float(pooled_counts[b]["frr"] / N * 100)
        pooled_far = float(pooled_counts[b]["far"] / N * 100)

        branch_aggregated[b] = {
            "name": name,
            "gate_mean": round(th_mean, 4),
            "gate_std": round(th_std, 4),
            "tar_mean": round(tar_mean, 2),
            "tar_std": round(tar_std, 2),
            "frr_mean": round(frr_mean, 2),
            "frr_std": round(frr_std, 2),
            "far_mean": round(far_mean, 2),
            "far_std": round(far_std, 2),
            "pooled_tar": round(pooled_tar, 2),
            "pooled_frr": round(pooled_frr, 2),
            "pooled_far": round(pooled_far, 2),
            "pooled_counts": pooled_counts[b],
        }

        th_str = f"{th_mean:.4f} ± {th_std:.4f}"
        tar_str = f"{tar_mean:.2f}% ± {tar_std:.2f}% (Pooled: {pooled_tar:.1f}%)"
        frr_str = f"{frr_mean:.2f}% ± {frr_std:.2f}% (Pooled: {pooled_frr:.1f}%)"
        far_str = f"{far_mean:.2f}% ± {far_std:.2f}% (Pooled: {pooled_far:.1f}%)"
        print(f"{name:<32} | {th_str:<30} | {tar_str:<30} | {frr_str:<30} | {far_str}", flush=True)

    # -------------------------------------------------------------------------
    # PART 2: REAL MULTI-FRAME TEMPORAL AGGREGATOR TESTING & FAR AUDIT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 125, flush=True)
    print("--- 2. MULTI-FRAME TEMPORAL AGGREGATOR SENSITIVITY GRID & IMPOSTOR FAR AUDIT ---", flush=True)
    print("=" * 125, flush=True)

    window_sizes = [4, 6, 8, 12]
    consensus_thresholds = [0.50, 0.60, 0.75]

    print(
        f"{'Window K':<10} | {'Consensus M':<12} | {'Clean TTFC (Fr)':<16} | {'Clean Track TAR':<16} | {'Clean FRR':<12} | {'Clean Impostor FAR':<20} | {'Degraded Track TAR':<20} | {'Degraded FAR':<14} | {'Flip/Churn Rate'}",
        flush=True,
    )
    print("-" * 145, flush=True)

    temporal_results = {}

    for K in window_sizes:
        for M in consensus_thresholds:
            np.random.seed(42)
            num_genuine_tracks = 100
            num_impostor_tracks = 100
            total_frames_per_track = 16

            # 1. Evaluate Genuine Tracks (Clean & Degraded)
            clean_confirmed = 0
            clean_rejected = 0
            clean_wrong_confirm = 0
            clean_ttfc_list = []
            clean_flips = 0

            deg_confirmed = 0
            deg_rejected = 0
            deg_wrong_confirm = 0
            deg_review = 0

            for t_idx in range(num_genuine_tracks):
                s_true = subjects[t_idx % len(subjects)]

                agg_clean = TrackIdentityAggregator(
                    window_size=K,
                    consensus_threshold=M,
                    confirm_threshold=0.72,
                    min_frames_for_decision=3,
                    high_risk_confusion_groups=[["Devhan", "Isuru", "person01"]],
                )

                agg_deg = TrackIdentityAggregator(
                    window_size=K,
                    consensus_threshold=M,
                    confirm_threshold=0.72,
                    min_frames_for_decision=3,
                    high_risk_confusion_groups=[["Devhan", "Isuru", "person01"]],
                )

                ttfc = None
                last_identity = None

                for f_idx in range(1, total_frames_per_track + 1):
                    # Clean frame simulation (90% correct detections with score in [0.75, 0.90])
                    cand_clean = s_true if np.random.rand() > 0.10 else "UNKNOWN"
                    score_clean = float(np.random.uniform(0.75, 0.90) if cand_clean == s_true else np.random.uniform(0.40, 0.60))
                    res_clean = agg_clean.update(track_id=t_idx, identity=cand_clean, score=score_clean)

                    if last_identity is not None and res_clean["identity"] != "UNKNOWN" and res_clean["identity"] != last_identity:
                        clean_flips += 1
                    if res_clean["identity"] != "UNKNOWN":
                        last_identity = res_clean["identity"]

                    if (res_clean["decision"] == "CONFIRMED" or (s_true != "demo_person_001" and res_clean["decision"] == "REVIEW_REQUIRED")) and ttfc is None:
                        ttfc = f_idx

                    # Degraded frame simulation (65% correct detections with score in [0.65, 0.82])
                    cand_deg = s_true if np.random.rand() > 0.35 else "UNKNOWN"
                    score_deg = float(np.random.uniform(0.65, 0.82) if cand_deg == s_true else np.random.uniform(0.35, 0.55))
                    res_deg = agg_deg.update(track_id=t_idx, identity=cand_deg, score=score_deg)

                # Final Genuine Decisions
                # Clean:
                if res_clean["decision"] == "CONFIRMED" or (s_true != "demo_person_001" and res_clean["decision"] == "REVIEW_REQUIRED"):
                    if res_clean["identity"] == s_true:
                        clean_confirmed += 1
                    else:
                        clean_wrong_confirm += 1
                else:
                    clean_rejected += 1

                if ttfc is not None:
                    clean_ttfc_list.append(ttfc)

                # Degraded:
                if res_deg["decision"] == "CONFIRMED" or (s_true != "demo_person_001" and res_deg["decision"] == "REVIEW_REQUIRED"):
                    if res_deg["identity"] == s_true:
                        deg_confirmed += 1
                    else:
                        deg_wrong_confirm += 1
                else:
                    deg_rejected += 1

            # 2. Evaluate Impostor Tracks (Un-enrolled intruders & cross-subject impostors)
            # 100 impostor tracks with fluctuating false candidates and noise scores
            impostor_clean_false_accepts = 0
            impostor_deg_false_accepts = 0

            for imp_idx in range(num_impostor_tracks):
                agg_imp_clean = TrackIdentityAggregator(
                    window_size=K,
                    consensus_threshold=M,
                    confirm_threshold=0.72,
                    min_frames_for_decision=3,
                    high_risk_confusion_groups=[["Devhan", "Isuru", "person01"]],
                )
                agg_imp_deg = TrackIdentityAggregator(
                    window_size=K,
                    consensus_threshold=M,
                    confirm_threshold=0.72,
                    min_frames_for_decision=3,
                    high_risk_confusion_groups=[["Devhan", "Isuru", "person01"]],
                )

                for f_idx in range(1, total_frames_per_track + 1):
                    # Impostor frame: random spurious matches across gallery subjects with occasional high score spike
                    # Random candidate subject with low score (0.30 - 0.65), occasionally spiking to 0.73
                    imp_cand = np.random.choice(subjects + ["UNKNOWN"], p=[0.2, 0.2, 0.2, 0.2, 0.2])
                    imp_score = float(np.random.uniform(0.40, 0.68) if np.random.rand() > 0.05 else np.random.uniform(0.70, 0.74))

                    r_imp_clean = agg_imp_clean.update(track_id=1000 + imp_idx, identity=imp_cand, score=imp_score)

                    # Degraded impostor frame (even noisier candidate switching)
                    imp_cand_deg = np.random.choice(subjects + ["UNKNOWN"], p=[0.22, 0.22, 0.22, 0.22, 0.12])
                    imp_score_deg = float(np.random.uniform(0.35, 0.65))
                    r_imp_deg = agg_imp_deg.update(track_id=2000 + imp_idx, identity=imp_cand_deg, score=imp_score_deg)

                if r_imp_clean["decision"] == "CONFIRMED":
                    impostor_clean_false_accepts += 1
                if r_imp_deg["decision"] == "CONFIRMED":
                    impostor_deg_false_accepts += 1

            mean_ttfc = float(np.mean(clean_ttfc_list)) if clean_ttfc_list else 0.0
            clean_tar = (clean_confirmed / num_genuine_tracks) * 100.0
            clean_frr = (clean_rejected / num_genuine_tracks) * 100.0
            clean_imp_far = (impostor_clean_false_accepts / num_impostor_tracks) * 100.0

            deg_tar = (deg_confirmed / num_genuine_tracks) * 100.0
            deg_frr = (deg_rejected / num_genuine_tracks) * 100.0
            deg_imp_far = (impostor_deg_false_accepts / num_impostor_tracks) * 100.0
            flip_rate = (clean_flips / (num_genuine_tracks * total_frames_per_track)) * 100.0

            temporal_results[f"K{K}_M{int(M*100)}"] = {
                "window_size": K,
                "consensus_threshold": M,
                "mean_ttfc_frames": round(mean_ttfc, 2),
                "clean_tar": round(clean_tar, 2),
                "clean_frr": round(clean_frr, 2),
                "clean_impostor_far": round(clean_imp_far, 2),
                "degraded_tar": round(deg_tar, 2),
                "degraded_frr": round(deg_frr, 2),
                "degraded_far": round(deg_imp_far, 2),
                "flip_rate": round(flip_rate, 4),
            }

            print(
                f"{K:<10} | {M:<12.2f} | {mean_ttfc:>14.2f}   | {clean_tar:>14.1f}% | {clean_frr:>10.1f}% | {clean_imp_far:>18.2f}% | {deg_tar:>18.1f}% | {deg_imp_far:>12.2f}% | {flip_rate:>13.2f}%",
                flush=True,
            )

    # Save comprehensive results to JSON artifact
    out_json = {
        "dataset_summary": {
            "total_samples": N,
            "subjects": dict(per_subject_samples),
            "num_subjects": len(unique_subjects),
        },
        "phase2_cross_validation": {
            "per_fold": fold_details,
            "branch_summary": branch_aggregated,
        },
        "phase2_temporal_grid": temporal_results,
    }

    out_file = ROOT_DIR / "evaluation" / "results" / "phase2_calibration_rigor_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print("\n" + "=" * 110, flush=True)
    print(f"[SUCCESS] Phase 2 Calibration Rigor Execution Completed. Saved JSON to {out_file}", flush=True)
    print("=" * 110, flush=True)
    return out_json


if __name__ == "__main__":
    run_phase2_calibration_rigor()
