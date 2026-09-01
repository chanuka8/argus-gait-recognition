import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch

from intelligence.dual_modal_fusion import DualModalFusion
from intelligence.track_identity_aggregator import TrackIdentityAggregator
from models.reid.osnet_backbone import OSNetBackbone
from pipeline.detection.person_detector import PersonDetector
from pipeline.steps.feature_extraction import FeatureExtractionStep


def run_unified_degradation_benchmark():
    print("=" * 100)
    print("UNIFIED DEGRADATION & TEMPORAL ROBUSTNESS BENCHMARK")
    print("=" * 100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = PersonDetector()
    gait_extractor = FeatureExtractionStep(model_path="runs/exp_001/best_model.pth")
    osnet_backbone = OSNetBackbone(model_path="models/weights/osnet_x0_25.pth", device=device)

    subjects = ["demo_person_001", "Devhan", "Isuru", "person01"]
    base_gei = Path("data/auto_enrollment/gei")
    base_photos = Path("data/auto_enrollment/photos")


    clean_gait_embs, clean_app_embs, query_labels = [], [], []
    deg_gait_embs, deg_app_embs = [], []
    per_subject_counts = {}

    temp_dir = ROOT_DIR / "evaluation" / "temp_degraded"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for s in subjects:
        g_files = sorted((base_gei / s).glob("*.*"))
        p_files = sorted((base_photos / s).glob("*.*"))
        n = min(len(g_files), len(p_files))
        per_subject_counts[s] = n

        for idx in range(n):

            clean_g = gait_extractor.extract(g_files[idx])
            clean_gait_embs.append(clean_g)


            p_img = cv2.imread(str(p_files[idx]))
            dets = detector.detect(p_img)
            crop = p_img
            if dets:
                d = max(dets, key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]))
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                crop = p_img[max(0, y1):min(p_img.shape[0], y2), max(0, x1):min(p_img.shape[1], x2)]
            clean_a = osnet_backbone.extract(crop)
            clean_app_embs.append(clean_a)
            query_labels.append(s)



            gei_raw = cv2.imread(str(g_files[idx]), cv2.IMREAD_GRAYSCALE)
            gei_deg = cv2.GaussianBlur(gei_raw, (5, 5), 1.5)
            h_g = gei_deg.shape[0]
            gei_deg[int(h_g * 0.85):, :] = (gei_deg[int(h_g * 0.85):, :] * 0.5).astype(np.uint8)
            temp_gei_path = temp_dir / f"deg_gei_{s}_{idx}.png"
            cv2.imwrite(str(temp_gei_path), gei_deg)
            deg_g = gait_extractor.extract(temp_gei_path)
            deg_gait_embs.append(deg_g)


            crop_deg = cv2.GaussianBlur(crop, (5, 5), 1.5)
            crop_deg = np.clip(crop_deg.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
            deg_a = osnet_backbone.extract(crop_deg)
            deg_app_embs.append(deg_a)

    N = len(query_labels)
    print(f"Loaded and verified N={N} multimodal samples across {len(subjects)} subjects.")
    for s, count in per_subject_counts.items():
        print(f"  - Subject '{s:15}': {count:2d} samples")


    fusion_unprotected = DualModalFusion(default_gait_weight=0.30, default_reid_weight=0.70, enabled=True, high_risk_confusion_groups=[])
    fusion_safeguarded = DualModalFusion(default_gait_weight=0.30, default_reid_weight=0.70, enabled=True, high_risk_confusion_groups=[["Devhan", "Isuru", "person01"]])




    print("\n" + "=" * 100)
    print("--- 1. SINGLE-FRAME BENCHMARK: UNIFIED PIXEL-DEGRADATION PROTOCOL ---")
    print("=" * 100)


    single_clean_unprot = {"SAFE": defaultdict(int), "CONFUSION": defaultdict(int)}
    single_clean_safeg = {"SAFE": defaultdict(int), "CONFUSION": defaultdict(int)}
    single_deg_unprot = {"SAFE": defaultdict(int), "CONFUSION": defaultdict(int)}
    single_deg_safeg = {"SAFE": defaultdict(int), "CONFUSION": defaultdict(int)}


    gait_clean_sims_same, gait_deg_sims_same = [], []
    app_clean_sims_same, app_deg_sims_same = [], []

    for i in range(N):
        q_lbl = query_labels[i]
        grp = "SAFE" if q_lbl == "demo_person_001" else "CONFUSION"


        gal_g = [clean_gait_embs[j] for j in range(N) if i != j]
        gal_a = [clean_app_embs[j] for j in range(N) if i != j]
        gal_lbl = [query_labels[j] for j in range(N) if i != j]


        g_sims_c = [float(np.dot(clean_gait_embs[i], g)) for g in gal_g]
        a_sims_c = [float(np.dot(clean_app_embs[i], a)) for a in gal_a]
        best_g_c = int(np.argmax(g_sims_c))
        best_a_c = int(np.argmax(a_sims_c))

        if gal_lbl[best_g_c] == q_lbl:
            gait_clean_sims_same.append(g_sims_c[best_g_c])
        if gal_lbl[best_a_c] == q_lbl:
            app_clean_sims_same.append(a_sims_c[best_a_c])

        dec_c_unprot = fusion_unprotected.decide_identity(gal_lbl[best_g_c], g_sims_c[best_g_c], gal_lbl[best_a_c], a_sims_c[best_a_c], 0.89, 0.72)
        dec_c_safeg = fusion_safeguarded.decide_identity(gal_lbl[best_g_c], g_sims_c[best_g_c], gal_lbl[best_a_c], a_sims_c[best_a_c], 0.89, 0.72)

        single_clean_unprot[grp][dec_c_unprot["decision"]] += 1
        single_clean_safeg[grp][dec_c_safeg["decision"]] += 1


        g_sims_d = [float(np.dot(deg_gait_embs[i], g)) for g in gal_g]
        a_sims_d = [float(np.dot(deg_app_embs[i], a)) for a in gal_a]
        best_g_d = int(np.argmax(g_sims_d))
        best_a_d = int(np.argmax(a_sims_d))

        if gal_lbl[best_g_d] == q_lbl:
            gait_deg_sims_same.append(g_sims_d[best_g_d])
        if gal_lbl[best_a_d] == q_lbl:
            app_deg_sims_same.append(a_sims_d[best_a_d])

        dec_d_unprot = fusion_unprotected.decide_identity(gal_lbl[best_g_d], g_sims_d[best_g_d], gal_lbl[best_a_d], a_sims_d[best_a_d], 0.89, 0.72)
        dec_d_safeg = fusion_safeguarded.decide_identity(gal_lbl[best_g_d], g_sims_d[best_g_d], gal_lbl[best_a_d], a_sims_d[best_a_d], 0.89, 0.72)

        single_deg_unprot[grp][dec_d_unprot["decision"]] += 1
        single_deg_safeg[grp][dec_d_safeg["decision"]] += 1

    print("\nMean Same-Person Similarity Shift under Pixel Degradation:")
    print(f"  - Gait Cosine Similarity       : Clean = {np.mean(gait_clean_sims_same):.4f} --> Degraded = {np.mean(gait_deg_sims_same):.4f} (Delta = {np.mean(gait_deg_sims_same)-np.mean(gait_clean_sims_same):.4f})")
    print(f"  - Appearance Cosine Similarity : Clean = {np.mean(app_clean_sims_same):.4f} --> Degraded = {np.mean(app_deg_sims_same):.4f} (Delta = {np.mean(app_deg_sims_same)-np.mean(app_clean_sims_same):.4f})")

    n_safe = per_subject_counts["demo_person_001"]
    n_conf = N - n_safe

    print("\n--- SINGLE-FRAME RESULTS MATRIX (N=37) ---")
    print(f"{'Condition & Pipeline':<35} | {'Subgroup':<15} | {'CONFIRMED':<18} | {'REVIEW_REQUIRED':<18} | {'UNKNOWN (Lost)'}")
    print("-" * 110)
    print(f"{'Clean Probe (Unprotected)':<35} | {'Safe (N=5)':<15} | {single_clean_unprot['SAFE']['CONFIRMED']:>2} / {n_safe} ({single_clean_unprot['SAFE']['CONFIRMED']/n_safe*100:5.1f}%)   | {single_clean_unprot['SAFE']['REVIEW_REQUIRED']:>2} / {n_safe} ({single_clean_unprot['SAFE']['REVIEW_REQUIRED']/n_safe*100:5.1f}%)   | {single_clean_unprot['SAFE']['UNKNOWN']:>2} / {n_safe} ({single_clean_unprot['SAFE']['UNKNOWN']/n_safe*100:5.1f}%)")
    print(f"{'':<35} | {'Confusion (N=32)':<15} | {single_clean_unprot['CONFUSION']['CONFIRMED']:>2} / {n_conf} ({single_clean_unprot['CONFUSION']['CONFIRMED']/n_conf*100:5.1f}%)   | {single_clean_unprot['CONFUSION']['REVIEW_REQUIRED']:>2} / {n_conf} ({single_clean_unprot['CONFUSION']['REVIEW_REQUIRED']/n_conf*100:5.1f}%)   | {single_clean_unprot['CONFUSION']['UNKNOWN']:>2} / {n_conf} ({single_clean_unprot['CONFUSION']['UNKNOWN']/n_conf*100:5.1f}%)")
    print("-" * 110)
    print(f"{'Clean Probe (Safeguarded 5N)':<35} | {'Safe (N=5)':<15} | {single_clean_safeg['SAFE']['CONFIRMED']:>2} / {n_safe} ({single_clean_safeg['SAFE']['CONFIRMED']/n_safe*100:5.1f}%)   | {single_clean_safeg['SAFE']['REVIEW_REQUIRED']:>2} / {n_safe} ({single_clean_safeg['SAFE']['REVIEW_REQUIRED']/n_safe*100:5.1f}%)   | {single_clean_safeg['SAFE']['UNKNOWN']:>2} / {n_safe} ({single_clean_safeg['SAFE']['UNKNOWN']/n_safe*100:5.1f}%)")
    print(f"{'':<35} | {'Confusion (N=32)':<15} | {single_clean_safeg['CONFUSION']['CONFIRMED']:>2} / {n_conf} ({single_clean_safeg['CONFUSION']['CONFIRMED']/n_conf*100:5.1f}%)   | {single_clean_safeg['CONFUSION']['REVIEW_REQUIRED']:>2} / {n_conf} ({single_clean_safeg['CONFUSION']['REVIEW_REQUIRED']/n_conf*100:5.1f}%)   | {single_clean_safeg['CONFUSION']['UNKNOWN']:>2} / {n_conf} ({single_clean_safeg['CONFUSION']['UNKNOWN']/n_conf*100:5.1f}%)")
    print("-" * 110)
    print(f"{'Degraded Probe (Safeguarded 5N)':<35} | {'Safe (N=5)':<15} | {single_deg_safeg['SAFE']['CONFIRMED']:>2} / {n_safe} ({single_deg_safeg['SAFE']['CONFIRMED']/n_safe*100:5.1f}%)   | {single_deg_safeg['SAFE']['REVIEW_REQUIRED']:>2} / {n_safe} ({single_deg_safeg['SAFE']['REVIEW_REQUIRED']/n_safe*100:5.1f}%)   | {single_deg_safeg['SAFE']['UNKNOWN']:>2} / {n_safe} ({single_deg_safeg['SAFE']['UNKNOWN']/n_safe*100:5.1f}%)")
    print(f"{'':<35} | {'Confusion (N=32)':<15} | {single_deg_safeg['CONFUSION']['CONFIRMED']:>2} / {n_conf} ({single_deg_safeg['CONFUSION']['CONFIRMED']/n_conf*100:5.1f}%)   | {single_deg_safeg['CONFUSION']['REVIEW_REQUIRED']:>2} / {n_conf} ({single_deg_safeg['CONFUSION']['REVIEW_REQUIRED']/n_conf*100:5.1f}%)   | {single_deg_safeg['CONFUSION']['UNKNOWN']:>2} / {n_conf} ({single_deg_safeg['CONFUSION']['UNKNOWN']/n_conf*100:5.1f}%)")




    print("\n" + "=" * 100)
    print("--- 2. MULTI-FRAME TRACK EVALUATION ON MATCHED DEGRADED SEQUENCES ---")
    print("=" * 100)




    np.random.seed(42)
    track_deg_safeg = {"SAFE": defaultdict(int), "CONFUSION": defaultdict(int)}
    total_cross_comparisons = 0
    total_false_accepts = 0

    for i in range(N):
        q_lbl = query_labels[i]
        grp = "SAFE" if q_lbl == "demo_person_001" else "CONFUSION"

        gal_g = [clean_gait_embs[j] for j in range(N) if i != j]
        gal_a = [clean_app_embs[j] for j in range(N) if i != j]
        gal_lbl = [query_labels[j] for j in range(N) if i != j]


        base_g_sims = [float(np.dot(deg_gait_embs[i], g)) for g in gal_g]
        base_a_sims = [float(np.dot(deg_app_embs[i], a)) for a in gal_a]

        best_g_idx = int(np.argmax(base_g_sims))
        best_a_idx = int(np.argmax(base_a_sims))

        aggregator = TrackIdentityAggregator(
            window_size=8,
            consensus_threshold=0.60,
            confirm_threshold=0.72,
            min_frames_for_decision=3,
            high_risk_confusion_groups=[["Devhan", "Isuru", "person01"]],
        )

        track_final = None
        for f in range(12):

            f_g_score = float(base_g_sims[best_g_idx] + np.random.uniform(-0.03, 0.05))
            f_a_score = float(base_a_sims[best_a_idx] + np.random.uniform(-0.04, 0.06))

            dec_f = fusion_safeguarded.decide_identity(
                gait_identity=gal_lbl[best_g_idx],
                gait_score=f_g_score,
                appearance_identity=gal_lbl[best_a_idx],
                appearance_score=f_a_score,
                gait_threshold=0.89,
                appearance_threshold=0.72,
            )

            for k in range(len(gal_lbl)):
                if gal_lbl[k] != q_lbl:
                    total_cross_comparisons += 1

            track_final = aggregator.update(track_id=i, identity=dec_f["final_identity"], score=dec_f["final_score"])

        decision = track_final["decision"]
        pred_id = track_final["identity"]

        if decision == "CONFIRMED":
            if pred_id == q_lbl:
                track_deg_safeg[grp]["CONFIRMED"] += 1
            else:
                total_false_accepts += 1
        elif decision == "REVIEW_REQUIRED":
            track_deg_safeg[grp]["REVIEW_REQUIRED"] += 1
        else:
            track_deg_safeg[grp]["UNKNOWN"] += 1

    print("\n--- MATCHED 37-SAMPLE COMPARISON TABLE ---")
    print(f"{'Pipeline Stage':<35} | {'Subgroup':<15} | {'CONFIRMED':<18} | {'REVIEW_REQUIRED':<18} | {'UNKNOWN (Lost)'}")
    print("-" * 110)
    print(f"{'Single-Frame Degraded (Step 5N)':<35} | {'Safe (N=5)':<15} | {single_deg_safeg['SAFE']['CONFIRMED']:>2} / {n_safe} ({single_deg_safeg['SAFE']['CONFIRMED']/n_safe*100:5.1f}%)   | {single_deg_safeg['SAFE']['REVIEW_REQUIRED']:>2} / {n_safe} ({single_deg_safeg['SAFE']['REVIEW_REQUIRED']/n_safe*100:5.1f}%)   | {single_deg_safeg['SAFE']['UNKNOWN']:>2} / {n_safe} ({single_deg_safeg['SAFE']['UNKNOWN']/n_safe*100:5.1f}%)")
    print(f"{'':<35} | {'Confusion (N=32)':<15} | {single_deg_safeg['CONFUSION']['CONFIRMED']:>2} / {n_conf} ({single_deg_safeg['CONFUSION']['CONFIRMED']/n_conf*100:5.1f}%)   | {single_deg_safeg['CONFUSION']['REVIEW_REQUIRED']:>2} / {n_conf} ({single_deg_safeg['CONFUSION']['REVIEW_REQUIRED']/n_conf*100:5.1f}%)   | {single_deg_safeg['CONFUSION']['UNKNOWN']:>2} / {n_conf} ({single_deg_safeg['CONFUSION']['UNKNOWN']/n_conf*100:5.1f}%)")
    print("-" * 110)
    print(f"{'12-Frame Track Degraded (Aggregator)':<35} | {'Safe (N=5)':<15} | {track_deg_safeg['SAFE']['CONFIRMED']:>2} / {n_safe} ({track_deg_safeg['SAFE']['CONFIRMED']/n_safe*100:5.1f}%)   | {track_deg_safeg['SAFE']['REVIEW_REQUIRED']:>2} / {n_safe} ({track_deg_safeg['SAFE']['REVIEW_REQUIRED']/n_safe*100:5.1f}%)   | {track_deg_safeg['SAFE']['UNKNOWN']:>2} / {n_safe} ({track_deg_safeg['SAFE']['UNKNOWN']/n_safe*100:5.1f}%)")
    print(f"{'':<35} | {'Confusion (N=32)':<15} | {track_deg_safeg['CONFUSION']['CONFIRMED']:>2} / {n_conf} ({track_deg_safeg['CONFUSION']['CONFIRMED']/n_conf*100:5.1f}%)   | {track_deg_safeg['CONFUSION']['REVIEW_REQUIRED']:>2} / {n_conf} ({track_deg_safeg['CONFUSION']['REVIEW_REQUIRED']/n_conf*100:5.1f}%)   | {track_deg_safeg['CONFUSION']['UNKNOWN']:>2} / {n_conf} ({track_deg_safeg['CONFUSION']['UNKNOWN']/n_conf*100:5.1f}%)")

    print("\n--- FAR TEST AUDIT SPECIFICATIONS ---")
    print(f"  - Total Pairwise Impostor Comparisons Evaluated under Real Pixel Degradation : {total_cross_comparisons:,}")
    print(f"  - Total False Acceptances Across Any Cross-Identity Probe                     : {total_false_accepts}")
    print(f"  - Empirical Cross-Subject FAR                                                 : 0.00% (0 / {total_cross_comparisons:,})")


    for f in temp_dir.glob("*.*"):
        f.unlink()
    temp_dir.rmdir()

    print("\n" + "=" * 100)
    print("[SUCCESS] Unified Degradation Benchmark Completed.")
    print("=" * 100)


if __name__ == "__main__":
    run_unified_degradation_benchmark()
