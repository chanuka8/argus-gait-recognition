import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch

from pipeline.steps.feature_extraction import FeatureExtractionStep
from pipeline.detection.person_detector import PersonDetector
from models.reid.osnet_backbone import OSNetBackbone
from intelligence.learned_fusion import LearnedLogisticFusion
from intelligence.score_calibrator import PlattScoreCalibrator
from evaluation.benchmarks.execute_phase_b_master import compute_cmc, compute_map_minp, compute_roc_eer_tar_at_far


def evaluate_all_fusion_strategies(output_dir: str = "configs/fusion_profiles"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = PersonDetector()
    gait_extractor = FeatureExtractionStep(model_path="runs/exp_001/best_model.pth")
    osnet_backbone = OSNetBackbone(model_path="models/weights/osnet_x0_25.pth", device=device)

    subjects = ["demo_person_001", "Devhan", "Isuru", "person01"]
    base_gei = Path("data/auto_enrollment/gei")
    base_photos = Path("data/auto_enrollment/photos")

    # Load multimodal query samples
    query_gait, query_app, query_labels = [], [], []
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
        for idx in range(n):
            query_gait.append(g_embs[idx])
            query_app.append(p_embs[idx])
            query_labels.append(s)

    N = len(query_labels)

    # 1. Build pairwise genuine & impostor score pools
    pairwise_gait = []
    pairwise_app = []
    pairwise_labels = []

    for i in range(N):
        for j in range(i + 1, N):
            s_g = float(np.dot(query_gait[i], query_gait[j]))
            s_a = float(np.dot(query_app[i], query_app[j]))
            pairwise_gait.append(s_g)
            pairwise_app.append(s_a)
            pairwise_labels.append(1 if query_labels[i] == query_labels[j] else 0)

    # 2. Build Leave-One-Out similarity matrices
    sim_matrix_base = np.zeros((N, N - 1), dtype=np.float32)
    sim_matrix_opt_linear = np.zeros((N, N - 1), dtype=np.float32)
    sim_matrix_calib = np.zeros((N, N - 1), dtype=np.float32)
    sim_matrix_learned_auc = np.zeros((N, N - 1), dtype=np.float32)

    loo_gallery_labels = []

    for i in range(N):
        q_g, q_a, q_lbl = query_gait[i], query_app[i], query_labels[i]
        gal_g, gal_a, gal_lbl = [], [], []
        for j in range(N):
            if i == j:
                continue
            gal_g.append(query_gait[j])
            gal_a.append(query_app[j])
            gal_lbl.append(query_labels[j])

        if i == 0:
            loo_gallery_labels = gal_lbl

        # Train pair dataset strictly disjoint from query i
        train_pairs_g, train_pairs_a, train_pairs_y = [], [], []
        for r1 in range(N):
            if r1 == i:
                continue
            for r2 in range(r1 + 1, N):
                if r2 == i:
                    continue
                s_g = float(np.dot(query_gait[r1], query_gait[r2]))
                s_a = float(np.dot(query_app[r1], query_app[r2]))
                train_pairs_g.append(s_g)
                train_pairs_a.append(s_a)
                train_pairs_y.append(1 if query_labels[r1] == query_labels[r2] else 0)

        # Fit Platt Calibrators & Learned AUC Fusion
        gait_calib = PlattScoreCalibrator().fit(train_pairs_g, train_pairs_y)
        app_calib = PlattScoreCalibrator().fit(train_pairs_a, train_pairs_y)
        learned_auc = LearnedLogisticFusion().fit(train_pairs_g, train_pairs_a, train_pairs_y, loss_type="ranking_auc")

        for gal_idx in range(N - 1):
            g_sim = float(np.dot(q_g, gal_g[gal_idx]))
            a_sim = float(np.dot(q_a, gal_a[gal_idx]))

            sim_matrix_base[i, gal_idx] = 0.30 * g_sim + 0.70 * a_sim
            sim_matrix_opt_linear[i, gal_idx] = 0.95 * g_sim + 0.05 * a_sim

            cg = gait_calib.calibrate(g_sim)
            ca = app_calib.calibrate(a_sim)
            sim_matrix_calib[i, gal_idx] = 0.30 * cg + 0.70 * ca

            sim_matrix_learned_auc[i, gal_idx] = learned_auc.predict_probability(g_sim, a_sim)

    # Ranking metrics
    _, r_base = compute_cmc(sim_matrix_base, query_labels, loo_gallery_labels)
    _, r_opt_linear = compute_cmc(sim_matrix_opt_linear, query_labels, loo_gallery_labels)
    _, r_calib = compute_cmc(sim_matrix_calib, query_labels, loo_gallery_labels)
    _, r_learned_auc = compute_cmc(sim_matrix_learned_auc, query_labels, loo_gallery_labels)

    map_base, minp_base = compute_map_minp(sim_matrix_base, query_labels, loo_gallery_labels)
    map_opt_linear, minp_opt_linear = compute_map_minp(sim_matrix_opt_linear, query_labels, loo_gallery_labels)
    map_calib, minp_calib = compute_map_minp(sim_matrix_calib, query_labels, loo_gallery_labels)
    map_learned_auc, minp_learned_auc = compute_map_minp(sim_matrix_learned_auc, query_labels, loo_gallery_labels)

    # Pairwise verification ROC-AUC & EER
    global_learned_auc = LearnedLogisticFusion().fit(pairwise_gait, pairwise_app, pairwise_labels, loss_type="ranking_auc")
    global_calib_g = PlattScoreCalibrator().fit(pairwise_gait, pairwise_labels)
    global_calib_a = PlattScoreCalibrator().fit(pairwise_app, pairwise_labels)

    same_base, diff_base = [], []
    same_opt, diff_opt = [], []
    same_calib, diff_calib = [], []
    same_learned_auc, diff_learned_auc = [], []

    for i in range(N):
        for j in range(i + 1, N):
            g_sim = float(np.dot(query_gait[i], query_gait[j]))
            a_sim = float(np.dot(query_app[i], query_app[j]))

            sb = 0.30 * g_sim + 0.70 * a_sim
            so = 0.95 * g_sim + 0.05 * a_sim
            sc = 0.30 * global_calib_g.calibrate(g_sim) + 0.70 * global_calib_a.calibrate(a_sim)
            sl = global_learned_auc.predict_probability(g_sim, a_sim)

            if query_labels[i] == query_labels[j]:
                same_base.append(sb)
                same_opt.append(so)
                same_calib.append(sc)
                same_learned_auc.append(sl)
            else:
                diff_base.append(sb)
                diff_opt.append(so)
                diff_calib.append(sc)
                diff_learned_auc.append(sl)

    roc_base = compute_roc_eer_tar_at_far(same_base, diff_base)
    roc_opt = compute_roc_eer_tar_at_far(same_opt, diff_opt)
    roc_calib = compute_roc_eer_tar_at_far(same_calib, diff_calib)
    roc_learned_auc = compute_roc_eer_tar_at_far(same_learned_auc, diff_learned_auc)

    def compute_gated_metrics(sim_matrix, same_scores, diff_scores):
        th = float(np.max(diff_scores) + 0.001)
        correct = sum(1 for i in range(N) if sim_matrix[i, int(np.argmax(sim_matrix[i]))] >= th and loo_gallery_labels[int(np.argmax(sim_matrix[i]))] == query_labels[i])
        unknown = sum(1 for i in range(N) if sim_matrix[i, int(np.argmax(sim_matrix[i]))] < th)
        return (correct / N * 100), (unknown / N * 100), th

    tar_base, frr_base, th_base = compute_gated_metrics(sim_matrix_base, same_base, diff_base)
    tar_opt, frr_opt, th_opt = compute_gated_metrics(sim_matrix_opt_linear, same_opt, diff_opt)
    tar_calib, frr_calib, th_calib = compute_gated_metrics(sim_matrix_calib, same_calib, diff_calib)
    tar_learned_auc, frr_learned_auc, th_learned_auc = compute_gated_metrics(sim_matrix_learned_auc, same_learned_auc, diff_learned_auc)

    return {
        "baseline": {"rank1": r_base[1], "rank5": r_base[5], "map": map_base, "auc": roc_base["auc"], "eer": roc_base["eer"], "tar": tar_base, "frr": frr_base},
        "linear_optimal": {"rank1": r_opt_linear[1], "rank5": r_opt_linear[5], "map": map_opt_linear, "auc": roc_opt["auc"], "eer": roc_opt["eer"], "tar": tar_opt, "frr": frr_opt},
        "calibrated": {"rank1": r_calib[1], "rank5": r_calib[5], "map": map_calib, "auc": roc_calib["auc"], "eer": roc_calib["eer"], "tar": tar_calib, "frr": frr_calib},
        "auc_learned": {"rank1": r_learned_auc[1], "rank5": r_learned_auc[5], "map": map_learned_auc, "auc": roc_learned_auc["auc"], "eer": roc_learned_auc["eer"], "tar": tar_learned_auc, "frr": frr_learned_auc},
    }

if __name__ == "__main__":
    results = evaluate_all_fusion_strategies()
    print("Fusion evaluation completed successfully.")
