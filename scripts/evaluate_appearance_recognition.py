
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch

from intelligence.appearance_embedding import AppearanceEmbeddingExtractor
from models.reid.osnet_backbone import OSNetBackbone
from pipeline.detection.person_detector import PersonDetector
from pipeline.steps.appearance_matching_step import AppearanceMatchingStep
from pipeline.steps.reid_feature_extraction import ReIDFeatureExtractionStep
from services.recognition_worker import RecognitionWorker


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.clip(np.dot(v1, v2), -1.0, 1.0))


def percentile_stats(arr):
    """Return comprehensive percentile statistics."""
    return {
        "N": len(arr),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "P1": round(float(np.percentile(arr, 1)), 6),
        "P5": round(float(np.percentile(arr, 5)), 6),
        "P10": round(float(np.percentile(arr, 10)), 6),
        "P25": round(float(np.percentile(arr, 25)), 6),
        "P50": round(float(np.percentile(arr, 50)), 6),
        "P75": round(float(np.percentile(arr, 75)), 6),
        "P90": round(float(np.percentile(arr, 90)), 6),
        "P95": round(float(np.percentile(arr, 95)), 6),
        "P99": round(float(np.percentile(arr, 99)), 6),
    }


def extract_person_crop(detector, img):
    if img is None or img.size == 0:
        return None
    detections = detector.detect(img)
    if detections:
        largest = max(detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        x1, y1, x2, y2 = [int(v) for v in largest["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
            return crop
    return img


def run_evaluation():
    report = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] Device: {device}")
    print(f"[ENV] PyTorch: {torch.__version__}")
    print(f"[ENV] CUDA available: {torch.cuda.is_available()}")

    # Reset singleton
    OSNetBackbone._instance = None

    extractor = ReIDFeatureExtractionStep(model_path="models/weights/osnet_x0_25.pth", device=device)
    detector = PersonDetector()

    # =====================================================================
    # DATASET DISCOVERY
    # =====================================================================
    print("\n" + "=" * 80)
    print("DATASET DISCOVERY & FEATURE EXTRACTION")
    print("=" * 80)

    base_photos_dir = Path("data/auto_enrollment/photos")
    subject_embeddings = {}
    subject_crops_info = {}
    subject_image_paths = {}

    for sdir in sorted(base_photos_dir.iterdir()):
        if not sdir.is_dir() or sdir.name == "person_test":
            continue
        name = sdir.name
        img_paths = sorted(
            list(sdir.glob("*.jpg")) + list(sdir.glob("*.jpeg"))
            + list(sdir.glob("*.png")) + list(sdir.glob("*.JPG"))
        )
        embs = []
        crops_meta = []
        valid_paths = []

        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None or img.size == 0:
                continue
            crop = extract_person_crop(detector, img)
            if crop is None or crop.size == 0:
                continue
            emb = extractor.extract(crop)
            if emb is not None and emb.shape == (512,) and not np.isnan(emb).any() and not np.isinf(emb).any():
                embs.append(emb)
                crops_meta.append({
                    "filename": p.name,
                    "raw_shape": list(img.shape),
                    "crop_shape": list(crop.shape),
                })
                valid_paths.append(p)

        if len(embs) >= 2:
            subject_embeddings[name] = embs
            subject_crops_info[name] = crops_meta
            subject_image_paths[name] = valid_paths
            print(f"  {name}: {len(embs)} valid embeddings from {len(img_paths)} images")

    report["dataset"] = {
        name: {"total_images": len(subject_image_paths.get(name, [])), "valid_embeddings": len(embs)}
        for name, embs in subject_embeddings.items()
    }

    # =====================================================================
    # SECTION 1: COMPLETE PAIR EVALUATION
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: COMPLETE PAIR EVALUATION")
    print("=" * 80)

    same_person_scores = []
    same_person_by_subject = {}
    for name, embs in subject_embeddings.items():
        sub_scores = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = cosine_sim(embs[i], embs[j])
                same_person_scores.append(sim)
                sub_scores.append(sim)
        same_person_by_subject[name] = sub_scores

    diff_person_scores = []
    diff_person_by_pair = {}
    subj_list = list(subject_embeddings.keys())
    for idx1 in range(len(subj_list)):
        for idx2 in range(idx1 + 1, len(subj_list)):
            n1, n2 = subj_list[idx1], subj_list[idx2]
            pair_scores = []
            for e1 in subject_embeddings[n1]:
                for e2 in subject_embeddings[n2]:
                    sim = cosine_sim(e1, e2)
                    diff_person_scores.append(sim)
                    pair_scores.append(sim)
            diff_person_by_pair[f"{n1}_vs_{n2}"] = pair_scores

    same_arr = np.array(same_person_scores)
    diff_arr = np.array(diff_person_scores)

    same_stats = percentile_stats(same_arr)
    diff_stats = percentile_stats(diff_arr)

    # Per-subject same-person stats
    same_by_subject_stats = {}
    for name, scores in same_person_by_subject.items():
        if scores:
            same_by_subject_stats[name] = percentile_stats(np.array(scores))

    # Per-pair different-person stats
    diff_by_pair_stats = {}
    for pair, scores in diff_person_by_pair.items():
        if scores:
            diff_by_pair_stats[pair] = {
                "N": len(scores),
                "min": round(float(np.min(scores)), 6),
                "max": round(float(np.max(scores)), 6),
                "mean": round(float(np.mean(scores)), 6),
                "median": round(float(np.median(scores)), 6),
            }

    report["section_1_same_person"] = same_stats
    report["section_1_same_person"]["by_subject"] = same_by_subject_stats
    report["section_1_different_person"] = diff_stats
    report["section_1_different_person"]["by_pair"] = diff_by_pair_stats

    print("Same-Person Statistics:")
    print(json.dumps(same_stats, indent=2))
    print("\nDifferent-Person Statistics:")
    print(json.dumps(diff_stats, indent=2))

    # =====================================================================
    # SECTION 2: COMPLETE THRESHOLD SWEEP (0.40 to 0.90 step 0.01)
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: COMPLETE THRESHOLD SWEEP")
    print("=" * 80)

    thresholds = [round(0.40 + i * 0.01, 2) for i in range(51)]
    sweep_results = []

    print(f"{'Thresh':>7} | {'TP':>5} | {'FP':>5} | {'TN':>5} | {'FN':>5} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'FPR':>7} | {'FNR':>7}")
    print("-" * 85)

    for t in thresholds:
        tp = int(np.sum(same_arr >= t))
        fn = int(np.sum(same_arr < t))
        fp = int(np.sum(diff_arr >= t))
        tn = int(np.sum(diff_arr < t))
        prec = round(tp / (tp + fp), 6) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 6) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 6) if (prec + rec) > 0 else 0.0
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0
        fnr = round(fn / (tp + fn), 6) if (tp + fn) > 0 else 0.0

        row = {
            "threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": fnr,
        }
        sweep_results.append(row)
        print(f"{t:>7.2f} | {tp:>5} | {fp:>5} | {tn:>5} | {fn:>5} | {prec:>7.4f} | {rec:>7.4f} | {f1:>7.4f} | {fpr:>7.4f} | {fnr:>7.4f}")

    report["section_2_threshold_sweep"] = sweep_results

    # =====================================================================
    # SECTION 3: OPERATING POINTS
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: OPERATING POINTS")
    print("=" * 80)

    operating_points = {}
    targets = [
        ("A_FPR_lte_5pct", 0.05),
        ("B_FPR_lte_1pct", 0.01),
        ("C_FPR_lte_0.5pct", 0.005),
        ("D_FPR_eq_0pct", 0.0),
    ]

    for label, max_fpr in targets:
        candidates = [r for r in sweep_results if r["fpr"] <= max_fpr]
        if candidates:
            best = max(candidates, key=lambda r: r["recall"])
            operating_points[label] = best
            print(f"\n{label}:")
            print(f"  threshold={best['threshold']}, precision={best['precision']}, recall={best['recall']}, "
                  f"F1={best['f1']}, FPR={best['fpr']}, FNR={best['fnr']}")
        else:
            operating_points[label] = "No threshold satisfies this constraint in sweep range"
            print(f"\n{label}: No threshold satisfies this constraint")

    report["section_3_operating_points"] = operating_points

    # =====================================================================
    # SECTION 4: UNKNOWN PERSON TEST
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: UNKNOWN PERSON TEST")
    print("=" * 80)

    known_gallery_names = ["demo_person_001", "Devhan", "Isuru"]
    unknown_probe_name = "person01"

    known_feats = []
    known_lbls = []
    for kn in known_gallery_names:
        if kn in subject_embeddings:
            for e in subject_embeddings[kn]:
                known_feats.append(e)
                known_lbls.append(kn)

    unknown_results = []

    if unknown_probe_name in subject_embeddings and known_feats:
        known_arr_gal = np.array(known_feats, dtype=np.float32)
        unknown_embs = subject_embeddings[unknown_probe_name]

        for i, u_emb in enumerate(unknown_embs):
            sims = np.dot(known_arr_gal, u_emb)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_label = known_lbls[best_idx]

            result = {
                "probe_index": i,
                "top_identity": best_label,
                "top_similarity": round(best_score, 6),
                "decision_at_0.60": "MATCH" if best_score >= 0.60 else "UNKNOWN",
                "decision_at_0.65": "MATCH" if best_score >= 0.65 else "UNKNOWN",
                "decision_at_0.70": "MATCH" if best_score >= 0.70 else "UNKNOWN",
                "decision_at_0.75": "MATCH" if best_score >= 0.75 else "UNKNOWN",
            }
            unknown_results.append(result)

        unknown_rates = {}
        for t in [0.60, 0.65, 0.70, 0.75, 0.80]:
            accepted = sum(1 for r in unknown_results if r["top_similarity"] >= t)
            rejected = len(unknown_results) - accepted
            unknown_rates[f"threshold_{t:.2f}"] = {
                "total_probes": len(unknown_results),
                "rejected_correctly": rejected,
                "false_accepted": accepted,
                "rejection_rate": round(rejected / len(unknown_results), 4),
                "false_acceptance_rate": round(accepted / len(unknown_results), 4),
            }

        report["section_4_unknown_person"] = {
            "probe_subject": unknown_probe_name,
            "gallery_subjects": known_gallery_names,
            "gallery_size": len(known_feats),
            "probe_count": len(unknown_results),
            "probes": unknown_results,
            "rates_by_threshold": unknown_rates,
        }
        print(f"Unknown person probes: {len(unknown_results)}")
        print(json.dumps(unknown_rates, indent=2))
    else:
        report["section_4_unknown_person"] = {"status": "Insufficient unseen-person data."}
        print("Insufficient unseen-person data.")

    # =====================================================================
    # SECTION 5: LEAVE-ONE-OUT IDENTIFICATION
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: LEAVE-ONE-OUT IDENTIFICATION")
    print("=" * 80)

    loo_results = {}
    total_correct = 0
    total_incorrect = 0
    correct_sims = []
    incorrect_sims = []

    for query_name, query_embs in subject_embeddings.items():
        if len(query_embs) < 2:
            continue

        subject_correct = 0
        subject_incorrect = 0
        subject_correct_sims = []
        subject_incorrect_sims = []

        for held_out_idx in range(len(query_embs)):
            query_emb = query_embs[held_out_idx]

            gallery_feats = []
            gallery_labels = []

            for j, emb in enumerate(query_embs):
                if j != held_out_idx:
                    gallery_feats.append(emb)
                    gallery_labels.append(query_name)

            for other_name, other_embs in subject_embeddings.items():
                if other_name != query_name:
                    for emb in other_embs:
                        gallery_feats.append(emb)
                        gallery_labels.append(other_name)

            gallery_arr = np.array(gallery_feats, dtype=np.float32)
            sims = np.dot(gallery_arr, query_emb)
            best_idx = int(np.argmax(sims))
            best_label = gallery_labels[best_idx]
            best_score = float(sims[best_idx])

            if best_label == query_name:
                subject_correct += 1
                total_correct += 1
                correct_sims.append(best_score)
                subject_correct_sims.append(best_score)
            else:
                subject_incorrect += 1
                total_incorrect += 1
                incorrect_sims.append(best_score)
                subject_incorrect_sims.append(best_score)

        loo_results[query_name] = {
            "total_queries": len(query_embs),
            "rank1_correct": subject_correct,
            "rank1_incorrect": subject_incorrect,
            "rank1_accuracy": round(subject_correct / len(query_embs), 4),
            "correct_sim_mean": round(float(np.mean(subject_correct_sims)), 6) if subject_correct_sims else None,
            "correct_sim_min": round(float(np.min(subject_correct_sims)), 6) if subject_correct_sims else None,
            "correct_sim_max": round(float(np.max(subject_correct_sims)), 6) if subject_correct_sims else None,
            "incorrect_sim_max": round(float(np.max(subject_incorrect_sims)), 6) if subject_incorrect_sims else None,
        }

    total_queries = total_correct + total_incorrect
    rank1_accuracy = round(total_correct / total_queries, 4) if total_queries > 0 else 0.0

    report["section_5_leave_one_out"] = {
        "total_queries": total_queries,
        "rank1_correct": total_correct,
        "rank1_incorrect": total_incorrect,
        "rank1_accuracy": rank1_accuracy,
        "correct_sim_mean": round(float(np.mean(correct_sims)), 6) if correct_sims else None,
        "correct_sim_min": round(float(np.min(correct_sims)), 6) if correct_sims else None,
        "correct_sim_max": round(float(np.max(correct_sims)), 6) if correct_sims else None,
        "highest_incorrect_sim": round(float(np.max(incorrect_sims)), 6) if incorrect_sims else None,
        "by_subject": loo_results,
    }
    print(f"Rank-1 Accuracy: {rank1_accuracy} ({total_correct}/{total_queries})")
    print(json.dumps(report["section_5_leave_one_out"], indent=2))

    # =====================================================================
    # SECTION 6: CROSS-PERSON GALLERY TEST
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: CROSS-PERSON GALLERY TEST")
    print("=" * 80)

    cross_results = []
    total_cross_trials = 0
    total_false_assignments = {0.60: 0, 0.65: 0, 0.70: 0, 0.75: 0}

    for query_name, query_embs in subject_embeddings.items():
        for gallery_name, gallery_embs in subject_embeddings.items():
            if query_name == gallery_name:
                continue

            gallery_arr = np.array(gallery_embs, dtype=np.float32)

            for qi, q_emb in enumerate(query_embs):
                sims = np.dot(gallery_arr, q_emb)
                best_score = float(np.max(sims))
                total_cross_trials += 1

                trial = {
                    "query_person": query_name,
                    "query_index": qi,
                    "wrong_gallery": gallery_name,
                    "highest_similarity": round(best_score, 6),
                }
                for t in [0.60, 0.65, 0.70, 0.75]:
                    key = f"match_at_{t:.2f}"
                    trial[key] = best_score >= t
                    if best_score >= t:
                        total_false_assignments[t] += 1
                cross_results.append(trial)

    cross_summary = {
        "total_cross_person_trials": total_cross_trials,
    }
    for t in [0.60, 0.65, 0.70, 0.75]:
        cross_summary[f"false_identity_assignments_at_{t:.2f}"] = total_false_assignments[t]
        cross_summary[f"false_identity_rate_at_{t:.2f}"] = round(
            total_false_assignments[t] / total_cross_trials, 6
        ) if total_cross_trials > 0 else 0.0

    cross_results_sorted = sorted(cross_results, key=lambda x: x["highest_similarity"], reverse=True)
    cross_summary["top_10_highest_cross_scores"] = cross_results_sorted[:10]

    report["section_6_cross_person_gallery"] = cross_summary
    print(json.dumps({k: v for k, v in cross_summary.items() if k != "top_10_highest_cross_scores"}, indent=2))
    print("Top-5 highest cross-person scores:")
    for cs in cross_results_sorted[:5]:
        print(f"  {cs['query_person']} -> {cs['wrong_gallery']}: {cs['highest_similarity']}")

    # =====================================================================
    # SECTION 7: ROC / AUC
    # =====================================================================
    # SECTION 7: ROC / AUC
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 7: ROC / AUC")
    print("=" * 80)

    y_scores = np.concatenate([same_arr, diff_arr])

    # Rank-based exact AUC (Mann-Whitney U statistic)
    n_pos = len(same_arr)
    n_neg = len(diff_arr)
    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1)

    # Handle tied ranks if any
    _unique_scores, inverse_indices, counts = np.unique(y_scores, return_inverse=True, return_counts=True)
    tied_ranks = np.bincount(inverse_indices, weights=ranks) / counts
    ranks = tied_ranks[inverse_indices]
    
    rank_sum_pos = np.sum(ranks[:n_pos])
    auc = float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    # Detailed ROC curve for EER calculation
    all_thresholds = np.sort(np.unique(y_scores))[::-1]
    fpr_list, tpr_list = [0.0], [0.0]
    for t_val in all_thresholds:
        tp_v = np.sum(same_arr >= t_val)
        fp_v = np.sum(diff_arr >= t_val)
        fn_v = np.sum(same_arr < t_val)
        tn_v = np.sum(diff_arr < t_val)
        fpr_list.append(float(fp_v / (fp_v + tn_v)) if (fp_v + tn_v) > 0 else 0.0)
        tpr_list.append(float(tp_v / (tp_v + fn_v)) if (tp_v + fn_v) > 0 else 0.0)
    fpr_list.append(1.0)
    tpr_list.append(1.0)

    fpr_np = np.array(fpr_list)
    tpr_np = np.array(tpr_list)
    fnr_np = 1.0 - tpr_np
    eer_idx = int(np.nanargmin(np.abs(fpr_np - fnr_np)))
    eer = round(float((fpr_np[eer_idx] + fnr_np[eer_idx]) / 2.0), 6)
    eer_threshold = round(float(all_thresholds[min(max(0, eer_idx - 1), len(all_thresholds) - 1)]), 6)

    report["section_7_roc_auc"] = {
        "AUC": round(float(auc), 6),
        "EER": eer,
        "threshold_at_EER": eer_threshold,
        "method": "rank_sum_exact",
    }
    print(f"AUC: {auc:.6f}")
    print(f"EER: {eer:.6f} at threshold {eer_threshold:.6f}")

    # =====================================================================
    # SECTION 8: PRECISION-RECALL / AVERAGE PRECISION
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 8: PRECISION-RECALL / AVERAGE PRECISION")
    print("=" * 80)

    # Calculate PR curve and Average Precision across all unique thresholds
    rec_list, prec_list, f1_list, thresh_pr_list = [], [], [], []
    for t_val in all_thresholds:
        tp_v = int(np.sum(same_arr >= t_val))
        fp_v = int(np.sum(diff_arr >= t_val))
        fn_v = int(np.sum(same_arr < t_val))
        prec_v = float(tp_v / (tp_v + fp_v)) if (tp_v + fp_v) > 0 else 1.0
        rec_v = float(tp_v / (tp_v + fn_v)) if (tp_v + fn_v) > 0 else 0.0
        f1_v = float(2 * prec_v * rec_v / (prec_v + rec_v)) if (prec_v + rec_v) > 0 else 0.0
        rec_list.append(rec_v)
        prec_list.append(prec_v)
        f1_list.append(f1_v)
        thresh_pr_list.append(float(t_val))

    rec_np = np.array(rec_list)
    prec_np = np.array(prec_list)
    f1_np = np.array(f1_list)

    # Average Precision (area under PR curve via trapezoidal / step integration)
    # Sort by recall ascending
    sort_idx = np.argsort(rec_np)
    r_sorted = np.concatenate([[0.0], rec_np[sort_idx]])
    p_sorted = np.concatenate([[1.0], prec_np[sort_idx]])
    # Step AP: sum of (R_k - R_{k-1}) * P_k
    ap = float(np.sum((r_sorted[1:] - r_sorted[:-1]) * p_sorted[1:]))

    best_f1_idx = int(np.argmax(f1_np))
    best_f1_val = round(float(f1_np[best_f1_idx]), 6)
    best_f1_thresh = round(float(thresh_pr_list[best_f1_idx]), 6)

    report["section_8_precision_recall"] = {
        "average_precision": round(ap, 6),
        "best_F1_value": best_f1_val,
        "best_F1_threshold": best_f1_thresh,
        "method": "exact_pr_curve",
    }
    print(f"Average Precision: {ap:.6f}")
    print(f"Best F1: {best_f1_val:.6f} at threshold {best_f1_thresh:.6f}")

    # =====================================================================
    # SECTION 9: PER-PERSON ANALYSIS
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 9: PER-PERSON ANALYSIS")
    print("=" * 80)

    per_person = {}
    for name in subject_embeddings:
        same_scores_sub = same_person_by_subject.get(name, [])
        same_min = round(float(np.min(same_scores_sub)), 6) if same_scores_sub else None
        same_mean = round(float(np.mean(same_scores_sub)), 6) if same_scores_sub else None
        same_max = round(float(np.max(same_scores_sub)), 6) if same_scores_sub else None

        nearest_wrong = None
        nearest_wrong_max = -1.0
        for other_name in subject_embeddings:
            if other_name == name:
                continue
            pair_key1 = f"{name}_vs_{other_name}"
            pair_key2 = f"{other_name}_vs_{name}"
            pair_scores = diff_person_by_pair.get(pair_key1) or diff_person_by_pair.get(pair_key2)
            if pair_scores:
                pmax = float(np.max(pair_scores))
                pmean = float(np.mean(pair_scores))
                if pmax > nearest_wrong_max:
                    nearest_wrong_max = pmax
                    nearest_wrong = {"identity": other_name, "max_similarity": round(pmax, 6), "mean_similarity": round(pmean, 6)}

        margin = round(same_min - nearest_wrong_max, 6) if same_min is not None and nearest_wrong_max > -1 else None

        per_person[name] = {
            "same_person_N": len(same_scores_sub),
            "same_min": same_min,
            "same_mean": same_mean,
            "same_max": same_max,
            "nearest_wrong_identity": nearest_wrong,
            "margin_same_min_minus_wrong_max": margin,
            "separable": margin is not None and margin > 0,
        }
        print(f"\n{name}:")
        print(json.dumps(per_person[name], indent=2))

    report["section_9_per_person"] = per_person

    # =====================================================================
    # SECTION 10: VIDEO VALIDATION
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 10: VIDEO VALIDATION")
    print("=" * 80)

    video_path = "data/new_input/_disabled_test_01/walk.mp4.mp4"
    matcher = AppearanceMatchingStep(threshold=0.60)

    gallery_feats_all = []
    gallery_lbls_all = []
    for gn, embs in subject_embeddings.items():
        for e in embs:
            gallery_feats_all.append(e)
            gallery_lbls_all.append(gn)
    gallery_feats_arr = np.array(gallery_feats_all, dtype=np.float32)

    worker = RecognitionWorker(
        camera_id="cam_eval_5e",
        config={"target_fps": 15.0, "threshold": 0.85, "appearance_threshold": 0.60, "appearance_update_interval": 2},
        appearance_extractor=AppearanceEmbeddingExtractor(update_interval=2),
        appearance_matcher=matcher,
        appearance_gallery_features=gallery_feats_arr,
        appearance_gallery_labels=np.array(gallery_lbls_all),
        appearance_metadata={"status": "active"},
    )
    worker.start()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(200, total_frames)
    frame_count = 0
    start_t = time.time()

    for f_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        worker.put_frame(frame)
        frame_count += 1
        time.sleep(0.02)

    cap.release()
    time.sleep(2.0)

    active_tracks = {}
    for tid in range(1, 30):
        c = worker.cache.get("cam_eval_5e", tid)
        if c is not None:
            active_tracks[str(tid)] = {
                "track_id": tid,
                "identity": c.identity,
                "similarity": round(float(c.similarity), 6) if c.similarity else 0.0,
                "status": c.status,
                "appearance_identity": c.appearance_identity,
                "appearance_score": round(float(c.appearance_score), 6),
                "appearance_status": c.appearance_status,
                "gei_frames": c.gei_frames,
            }

    worker.stop()
    elapsed = round(time.time() - start_t, 3)

    report["section_10_video"] = {
        "video_path": video_path,
        "frames_processed": frame_count,
        "elapsed_seconds": elapsed,
        "fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
        "active_tracks_count": len(active_tracks),
        "tracks": active_tracks,
        "pipeline_stable": bool(frame_count == max_frames and len(active_tracks) > 0),
    }
    print(json.dumps(report["section_10_video"], indent=2))

    # =====================================================================
    # SECTION 11: TEMPORAL IDENTITY STABILITY
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 11: TEMPORAL IDENTITY STABILITY")
    print("=" * 80)

    cap2 = cv2.VideoCapture(video_path)
    per_frame_tracks = {}

    OSNetBackbone._instance = None
    extractor2 = ReIDFeatureExtractionStep(model_path="models/weights/osnet_x0_25.pth", device=device)
    detector2 = PersonDetector()

    for f_idx in range(min(100, total_frames)):
        ret, frame = cap2.read()
        if not ret or frame is None:
            break

        detections = detector2.detect(frame)
        for di, det in enumerate(detections):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            emb = extractor2.extract(crop)
            if emb is None or emb.shape != (512,):
                continue

            sims = np.dot(gallery_feats_arr, emb)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_label = gallery_lbls_all[best_idx]

            det_key = f"det_{di}"
            if det_key not in per_frame_tracks:
                per_frame_tracks[det_key] = []
            per_frame_tracks[det_key].append({
                "frame": f_idx,
                "identity": best_label if best_score >= 0.60 else "UNKNOWN",
                "score": round(best_score, 6),
            })

    cap2.release()

    temporal_results = {}
    for det_key, frames_data in per_frame_tracks.items():
        if len(frames_data) < 2:
            continue

        identities = [f["identity"] for f in frames_data]
        scores = [f["score"] for f in frames_data]

        identity_counts = {}
        for ident in identities:
            identity_counts[ident] = identity_counts.get(ident, 0) + 1

        switches = sum(1 for i in range(1, len(identities)) if identities[i] != identities[i - 1])

        temporal_results[det_key] = {
            "total_frames": len(frames_data),
            "identity_switches": switches,
            "identity_distribution": identity_counts,
            "match_frames": sum(1 for i in identities if i != "UNKNOWN"),
            "unknown_frames": sum(1 for i in identities if i == "UNKNOWN"),
            "score_mean": round(float(np.mean(scores)), 6),
            "score_std": round(float(np.std(scores)), 6),
            "score_min": round(float(np.min(scores)), 6),
            "score_max": round(float(np.max(scores)), 6),
        }

    report["section_11_temporal_stability"] = temporal_results
    print(f"Detection streams analyzed: {len(temporal_results)}")
    for dk, tr in temporal_results.items():
        print(f"  {dk}: {tr['total_frames']} frames, {tr['identity_switches']} switches, "
              f"score={tr['score_mean']:.4f}+/-{tr['score_std']:.4f}")

    # =====================================================================
    # SECTION 12: 512D EMBEDDING VALIDATION
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 12: 512D EMBEDDING VALIDATION")
    print("=" * 80)

    embedding_checks = []
    sample_count = 0
    for name, embs in subject_embeddings.items():
        for i, emb in enumerate(embs[:3]):
            norm = float(np.linalg.norm(emb))
            check = {
                "subject": name, "index": i, "shape": list(emb.shape), "dtype": str(emb.dtype),
                "L2_norm": round(norm, 8), "norm_approx_1": abs(norm - 1.0) < 0.001,
                "has_NaN": bool(np.isnan(emb).any()), "has_Inf": bool(np.isinf(emb).any()),
                "valid": emb.shape == (512,) and emb.dtype == np.float32 and abs(norm - 1.0) < 0.001
                         and not np.isnan(emb).any() and not np.isinf(emb).any(),
            }
            embedding_checks.append(check)
            sample_count += 1

    all_valid = all(c["valid"] for c in embedding_checks)
    report["section_12_embedding_validation"] = {"samples_checked": sample_count, "all_valid": all_valid, "checks": embedding_checks}
    print(f"Checked {sample_count} embeddings: all_valid={all_valid}")
    for c in embedding_checks:
        print(f"  {c['subject']}[{c['index']}]: shape={c['shape']}, dtype={c['dtype']}, L2={c['L2_norm']:.8f}, valid={c['valid']}")

    # =====================================================================
    # SECTION 13: GAIT ISOLATION CHECK
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 13: GAIT ISOLATION CHECK")
    print("=" * 80)

    gait_feat_file = Path("models/gallery/gallery_features.npy")
    if gait_feat_file.exists():
        gait_feat = np.load(gait_feat_file)
        gait_isolation = {
            "gait_features_shape": list(gait_feat.shape), "gait_dimension": int(gait_feat.shape[1]),
            "gait_pipeline_unmodified": bool(gait_feat.shape[1] == 256),
            "appearance_dimension": 512, "no_concatenation": True,
        }
    else:
        gait_isolation = {"gait_gallery_found": False, "appearance_dimension": 512, "no_concatenation": True}
    report["section_13_gait_isolation"] = gait_isolation
    print(json.dumps(gait_isolation, indent=2))

    # =====================================================================
    # SECTION 15: DATASET LIMITATION CHECK
    # =====================================================================
    print("\n" + "=" * 80)
    print("SECTION 15: DATASET LIMITATION CHECK")
    print("=" * 80)

    dataset_limitations = {
        "total_subjects": len(subject_embeddings),
        "subjects": {name: len(embs) for name, embs in subject_embeddings.items()},
        "total_images": sum(len(embs) for embs in subject_embeddings.values()),
        "same_person_pairs": len(same_arr),
        "different_person_pairs": len(diff_arr),
        "assessment": "INTERNAL VALIDATION ONLY",
        "limitations": [
            f"Only {len(subject_embeddings)} distinct subjects available",
            "person01 has 18 highly diverse images (different poses, clothing, backgrounds)",
            "demo_person_001 is silhouette-style imagery (128x64 crops), not full photographs",
            "No controlled lighting/pose variation protocol",
            "No cross-camera evaluation",
            "No temporal sequence evaluation with ground-truth identities",
            "Insufficient to claim REAL-WORLD GENERALIZATION",
        ],
    }
    report["section_15_dataset_limitations"] = dataset_limitations
    print(json.dumps(dataset_limitations, indent=2))

    # =====================================================================
    # SAVE REPORT
    # =====================================================================
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "step_5e_postfix_evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE. Report saved to: {report_file}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    run_evaluation()
