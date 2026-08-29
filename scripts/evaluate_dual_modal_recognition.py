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
from intelligence.dual_modal_fusion import DualModalFusion
from models.reid.osnet_backbone import OSNetBackbone
from pipeline.detection.person_detector import PersonDetector
from pipeline.steps.appearance_matching_step import AppearanceMatchingStep
from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.steps.reid_feature_extraction import ReIDFeatureExtractionStep
from services.recognition_worker import RecognitionWorker


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def percentile_stats(arr):
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


def compute_roc_pr_metrics(same_scores, diff_scores):
    same_arr = np.array(same_scores)
    diff_arr = np.array(diff_scores)
    y_scores = np.concatenate([same_arr, diff_arr])
    n_pos = len(same_arr)
    n_neg = len(diff_arr)

    # Rank-based exact AUC
    order = np.argsort(y_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1)
    _unique_scores, inverse_indices, counts = np.unique(y_scores, return_inverse=True, return_counts=True)
    tied_ranks = np.bincount(inverse_indices, weights=ranks) / counts
    ranks = tied_ranks[inverse_indices]
    rank_sum_pos = np.sum(ranks[:n_pos])
    auc = float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    # ROC thresholds for EER
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
    eer_thresh = round(float(all_thresholds[min(max(0, eer_idx - 1), len(all_thresholds) - 1)]), 6)

    # Average Precision
    rec_list, prec_list = [], []
    for t_val in all_thresholds:
        tp_v = int(np.sum(same_arr >= t_val))
        fp_v = int(np.sum(diff_arr >= t_val))
        prec_v = float(tp_v / (tp_v + fp_v)) if (tp_v + fp_v) > 0 else 1.0
        rec_v = float(tp_v / n_pos) if n_pos > 0 else 0.0
        rec_list.append(rec_v)
        prec_list.append(prec_v)
    rec_np = np.array(rec_list)
    prec_np = np.array(prec_list)
    sort_idx = np.argsort(rec_np)
    r_sorted = np.concatenate([[0.0], rec_np[sort_idx]])
    p_sorted = np.concatenate([[1.0], prec_np[sort_idx]])
    ap = float(np.sum((r_sorted[1:] - r_sorted[:-1]) * p_sorted[1:]))

    return {
        "AUC": round(auc, 6),
        "EER": eer,
        "threshold_at_EER": eer_thresh,
        "AP": round(ap, 6),
    }


def extract_person_crop(detector, img):
    if img is None or img.size == 0:
        return None
    dets = detector.detect(img)
    if dets:
        largest = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        x1, y1, x2, y2 = [int(v) for v in largest["bbox"]]
        crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
            return crop
    return img


def run_dual_modal_evaluation():
    print("=" * 80)
    print("ARGUS AI - STEP 5F: DUAL-MODAL RECOGNITION VALIDATION")
    print("=" * 80)

    report = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] Device: {device}")

    # Initialize feature extraction modules
    gait_extractor = FeatureExtractionStep()
    OSNetBackbone._instance = None
    reid_extractor = ReIDFeatureExtractionStep(model_path="models/weights/osnet_x0_25.pth", device=device)
    detector = PersonDetector()
    fusion_engine = DualModalFusion(default_gait_weight=0.70, default_reid_weight=0.30, enabled=True)

    # 1. Discover and load multimodal features for all 4 subjects
    base_gei = Path("data/auto_enrollment/gei")
    base_photos = Path("data/auto_enrollment/photos")
    subjects = ["demo_person_001", "Devhan", "Isuru", "person01"]

    subject_data = {}
    for name in subjects:
        g_files = sorted((base_gei / name).glob("*.*"))
        p_files = sorted((base_photos / name).glob("*.*"))

        # Extract gait (256D)
        g_embs = [gait_extractor.extract(f) for f in g_files]

        # Extract appearance (512D)
        p_embs = []
        for f in p_files:
            img = cv2.imread(str(f))
            crop = extract_person_crop(detector, img)
            emb = reid_extractor.extract(crop)
            p_embs.append(emb)

        # Pair instances: each sample i has (gait_i, appearance_i)
        n_samples = min(len(g_embs), len(p_embs))
        subject_data[name] = {
            "gait": g_embs[:n_samples],
            "appearance": p_embs[:n_samples],
            "n_samples": n_samples,
        }
        print(f"Loaded {name}: {n_samples} multimodal samples (256D gait + 512D appearance)")

    # 2. Pairwise Evaluations: Same-person and Different-person
    same_gait_scores = []
    same_app_scores = []
    same_fused_scores = []

    diff_gait_scores = []
    diff_app_scores = []
    diff_fused_scores = []

    confusion_pair_stats = {}

    for name, data in subject_data.items():
        n = data["n_samples"]
        for i in range(n):
            for j in range(i + 1, n):
                s_g = cosine_sim(data["gait"][i], data["gait"][j])
                s_a = cosine_sim(data["appearance"][i], data["appearance"][j])
                f_res = fusion_engine.fuse(gait_score=s_g, reid_score=s_a)
                s_f = f_res["final_score"]

                same_gait_scores.append(s_g)
                same_app_scores.append(s_a)
                same_fused_scores.append(s_f)

    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            n1, n2 = subjects[i], subjects[j]
            pair_key = f"{n1}_vs_{n2}"
            pair_gait = []
            pair_app = []
            pair_fused = []

            for idx1 in range(subject_data[n1]["n_samples"]):
                for idx2 in range(subject_data[n2]["n_samples"]):
                    s_g = cosine_sim(subject_data[n1]["gait"][idx1], subject_data[n2]["gait"][idx2])
                    s_a = cosine_sim(subject_data[n1]["appearance"][idx1], subject_data[n2]["appearance"][idx2])
                    f_res = fusion_engine.fuse(gait_score=s_g, reid_score=s_a)
                    s_f = f_res["final_score"]

                    diff_gait_scores.append(s_g)
                    diff_app_scores.append(s_a)
                    diff_fused_scores.append(s_f)

                    pair_gait.append(s_g)
                    pair_app.append(s_a)
                    pair_fused.append(s_f)

            confusion_pair_stats[pair_key] = {
                "N": len(pair_fused),
                "gait": {"min": round(float(np.min(pair_gait)), 6), "max": round(float(np.max(pair_gait)), 6), "mean": round(float(np.mean(pair_gait)), 6)},
                "appearance": {"min": round(float(np.min(pair_app)), 6), "max": round(float(np.max(pair_app)), 6), "mean": round(float(np.mean(pair_app)), 6)},
                "fused": {"min": round(float(np.min(pair_fused)), 6), "max": round(float(np.max(pair_fused)), 6), "mean": round(float(np.mean(pair_fused)), 6)},
            }

    report["same_person"] = {
        "gait": percentile_stats(same_gait_scores),
        "appearance": percentile_stats(same_app_scores),
        "fused": percentile_stats(same_fused_scores),
    }

    report["different_person"] = {
        "gait": percentile_stats(diff_gait_scores),
        "appearance": percentile_stats(diff_app_scores),
        "fused": percentile_stats(diff_fused_scores),
    }

    report["confusion_pairs"] = confusion_pair_stats

    # 3. Full Threshold Sweep for Fused Score (0.40 to 0.90 step 0.01)
    thresholds = [round(0.40 + i * 0.01, 2) for i in range(51)]
    fused_same_arr = np.array(same_fused_scores)
    fused_diff_arr = np.array(diff_fused_scores)
    sweep_table = []

    for t in thresholds:
        tp = int(np.sum(fused_same_arr >= t))
        fn = int(np.sum(fused_same_arr < t))
        fp = int(np.sum(fused_diff_arr >= t))
        tn = int(np.sum(fused_diff_arr < t))
        prec = round(tp / (tp + fp), 6) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 6) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 6) if (prec + rec) > 0 else 0.0
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0
        fnr = round(fn / (tp + fn), 6) if (tp + fn) > 0 else 0.0
        sweep_table.append({
            "threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": fnr,
        })

    report["threshold_sweep_fused"] = sweep_table

    # Operating Points for Fused Score
    op_points = {}
    for label, max_fpr in [("FPR_lte_5pct", 0.05), ("FPR_lte_1pct", 0.01), ("FPR_lte_0.5pct", 0.005), ("FPR_eq_0pct", 0.0)]:
        cand = [r for r in sweep_table if r["fpr"] <= max_fpr]
        if cand:
            op_points[label] = max(cand, key=lambda r: r["recall"])
    report["operating_points_fused"] = op_points

    # 4. Unknown Person Rejection Test (person01 as unknown against {demo_person_001, Devhan, Isuru})
    known_subjs = ["demo_person_001", "Devhan", "Isuru"]
    unknown_subj = "person01"

    known_gait_feats = []
    known_app_feats = []
    known_lbls = []
    for kn in known_subjs:
        for idx in range(subject_data[kn]["n_samples"]):
            known_gait_feats.append(subject_data[kn]["gait"][idx])
            known_app_feats.append(subject_data[kn]["appearance"][idx])
            known_lbls.append(kn)
    known_gait_arr = np.array(known_gait_feats, dtype=np.float32)
    known_app_arr = np.array(known_app_feats, dtype=np.float32)

    unknown_test_results = {"app_only": {}, "gait_only": {}, "fused": {}}
    for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        unknown_test_results["app_only"][f"{t:.2f}"] = {"accepted": 0, "rejected": 0}
        unknown_test_results["gait_only"][f"{t:.2f}"] = {"accepted": 0, "rejected": 0}
        unknown_test_results["fused"][f"{t:.2f}"] = {"accepted": 0, "rejected": 0}

    n_unknown = subject_data[unknown_subj]["n_samples"]
    for idx in range(n_unknown):
        u_g = subject_data[unknown_subj]["gait"][idx]
        u_a = subject_data[unknown_subj]["appearance"][idx]

        g_sims = [cosine_sim(u_g, k_g) for k_g in known_gait_arr]
        a_sims = [cosine_sim(u_a, k_a) for k_a in known_app_arr]

        best_g_idx = int(np.argmax(g_sims))
        best_a_idx = int(np.argmax(a_sims))

        best_g_score = g_sims[best_g_idx]
        best_a_score = a_sims[best_a_idx]

        # Fused decision
        best_fused_score = 0.70 * best_g_score + 0.30 * best_a_score

        for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            # App
            if best_a_score >= t:
                unknown_test_results["app_only"][f"{t:.2f}"]["accepted"] += 1
            else:
                unknown_test_results["app_only"][f"{t:.2f}"]["rejected"] += 1
            # Gait
            if best_g_score >= t:
                unknown_test_results["gait_only"][f"{t:.2f}"]["accepted"] += 1
            else:
                unknown_test_results["gait_only"][f"{t:.2f}"]["rejected"] += 1
            # Fused
            if best_fused_score >= t:
                unknown_test_results["fused"][f"{t:.2f}"]["accepted"] += 1
            else:
                unknown_test_results["fused"][f"{t:.2f}"]["rejected"] += 1

    report["unknown_person_rejection"] = {
        "n_probes": n_unknown,
        "results": unknown_test_results,
    }

    # 5. Leave-One-Out (LOO) Rank-1 Accuracy
    loo_summary = {"gait": {"correct": 0, "total": 0}, "appearance": {"correct": 0, "total": 0}, "fused": {"correct": 0, "total": 0}}
    per_subject_loo = {}

    for query_name in subjects:
        n_q = subject_data[query_name]["n_samples"]
        per_subject_loo[query_name] = {"gait": 0, "appearance": 0, "fused": 0, "total": n_q}

        for held_out_idx in range(n_q):
            q_g = subject_data[query_name]["gait"][held_out_idx]
            q_a = subject_data[query_name]["appearance"][held_out_idx]

            gal_g = []
            gal_a = []
            gal_lbl = []

            for other_name in subjects:
                n_o = subject_data[other_name]["n_samples"]
                for j in range(n_o):
                    if other_name == query_name and j == held_out_idx:
                        continue
                    gal_g.append(subject_data[other_name]["gait"][j])
                    gal_a.append(subject_data[other_name]["appearance"][j])
                    gal_lbl.append(other_name)

            gal_g_arr = np.array(gal_g, dtype=np.float32)
            gal_a_arr = np.array(gal_a, dtype=np.float32)

            g_sims = np.array([cosine_sim(q_g, g) for g in gal_g_arr])
            a_sims = np.array([cosine_sim(q_a, a) for a in gal_a_arr])
            f_sims = 0.70 * g_sims + 0.30 * a_sims

            best_g_lbl = gal_lbl[int(np.argmax(g_sims))]
            best_a_lbl = gal_lbl[int(np.argmax(a_sims))]
            best_f_lbl = gal_lbl[int(np.argmax(f_sims))]

            if best_g_lbl == query_name:
                loo_summary["gait"]["correct"] += 1
                per_subject_loo[query_name]["gait"] += 1
            if best_a_lbl == query_name:
                loo_summary["appearance"]["correct"] += 1
                per_subject_loo[query_name]["appearance"] += 1
            if best_f_lbl == query_name:
                loo_summary["fused"]["correct"] += 1
                per_subject_loo[query_name]["fused"] += 1

            loo_summary["gait"]["total"] += 1
            loo_summary["appearance"]["total"] += 1
            loo_summary["fused"]["total"] += 1

    report["loo_rank1_accuracy"] = {
        "gait_rank1": round(loo_summary["gait"]["correct"] / loo_summary["gait"]["total"], 4),
        "appearance_rank1": round(loo_summary["appearance"]["correct"] / loo_summary["appearance"]["total"], 4),
        "fused_rank1": round(loo_summary["fused"]["correct"] / loo_summary["fused"]["total"], 4),
        "counts": loo_summary,
        "by_subject": per_subject_loo,
    }

    # 6. ROC AUC, EER, and AP
    roc_gait = compute_roc_pr_metrics(same_gait_scores, diff_gait_scores)
    roc_app = compute_roc_pr_metrics(same_app_scores, diff_app_scores)
    roc_fused = compute_roc_pr_metrics(same_fused_scores, diff_fused_scores)

    report["roc_pr_comparison"] = {
        "gait_only": roc_gait,
        "appearance_only": roc_app,
        "fused": roc_fused,
    }

    # 7. Video Stream Evaluation (walk.mp4.mp4)
    video_path = "data/new_input/_disabled_test_01/walk.mp4.mp4"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(150, total_frames)

    # Build galleries
    gal_lbls_all = []
    gal_g_all = []
    gal_a_all = []
    for name in subjects:
        for idx in range(subject_data[name]["n_samples"]):
            gal_g_all.append(subject_data[name]["gait"][idx])
            gal_a_all.append(subject_data[name]["appearance"][idx])
            gal_lbls_all.append(name)
    gal_g_arr = np.array(gal_g_all, dtype=np.float32)
    gal_a_arr = np.array(gal_a_all, dtype=np.float32)

    matcher = AppearanceMatchingStep(threshold=0.60)
    worker = RecognitionWorker(
        camera_id="cam_dual_modal_eval",
        config={
            "target_fps": 15.0,
            "threshold": 0.85,
            "appearance_threshold": 0.60,
            "appearance_update_interval": 2,
            "dual_modal_fusion": {"enabled": True, "gait_weight": 0.70, "appearance_weight": 0.30},
        },
        appearance_extractor=AppearanceEmbeddingExtractor(update_interval=2),
        appearance_matcher=matcher,
        appearance_gallery_features=gal_a_arr,
        appearance_gallery_labels=np.array(gal_lbls_all),
        appearance_metadata={"status": "active"},
    )
    worker.start()

    frame_count = 0
    for _ in range(max_frames):
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
        c = worker.cache.get("cam_dual_modal_eval", tid)
        if c is not None:
            active_tracks[str(tid)] = {
                "track_id": tid,
                "gait_identity": c.identity,
                "gait_score": round(float(c.similarity), 4),
                "appearance_identity": c.appearance_identity,
                "appearance_score": round(float(c.appearance_score), 4),
                "gei_frames": c.gei_frames,
            }
    worker.stop()

    report["video_stream_results"] = {
        "frames_processed": frame_count,
        "tracks": active_tracks,
    }

    # Save artifact report
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "step_5f_dual_modal_evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"DUAL-MODAL EVALUATION COMPLETE. Report saved to: {report_file}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    run_dual_modal_evaluation()
