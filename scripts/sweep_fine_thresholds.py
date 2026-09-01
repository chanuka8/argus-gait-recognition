import json
import sys
from pathlib import Path


_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import cv2
import numpy as np
import torch

from models.architectures.bygait_light import ByGaitLight


def run_fine_sweep():
    bygait_path = "runs/exp_001/best_model.pth"
    casia_gei_dir = Path("data/casia_processed/gei")
    eval_subjects = ["101", "102", "103", "104", "105", "106", "107", "108", "109", "110"]

    bygait_model = ByGaitLight(embedding_dim=256, part_bins=1)
    state = torch.load(bygait_path, map_location="cpu", weights_only=True)
    clean = {k.replace("backbone.", ""): v for k, v in state.items() if k.replace("backbone.", "") in bygait_model.state_dict()}
    bygait_model.load_state_dict(clean, strict=False)
    bygait_model.eval()

    def extract_bygait_emb(gei_arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            img = np.asarray(gei_arr, dtype=np.float32)
            if img.ndim == 2:
                img = img[np.newaxis, np.newaxis, :, :]
            elif img.ndim == 3:
                img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
            t = torch.from_numpy(img)
            emb = bygait_model(t).cpu().numpy().flatten()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-6 else emb

    gallery_embs = {}
    probe_list = []

    for sid in eval_subjects:
        s_dir = casia_gei_dir / sid
        g_files = list(s_dir.glob(f"{sid}_nm-0[1-4]_*.png")) + list(s_dir.glob(f"{sid}_nm-0[1-4]_*.jpg"))
        p_files = list(s_dir.glob(f"{sid}_nm-0[5-6]_*.png")) + list(s_dir.glob(f"{sid}_cl-*.png")) + list(s_dir.glob(f"{sid}_bg-*.png"))

        if g_files:
            g_imgs = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in g_files[:4] if cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) is not None]
            if g_imgs:
                g_avg = np.mean(g_imgs, axis=0).astype(np.uint8)
                gallery_embs[sid] = extract_bygait_emb(g_avg)

        for pf in p_files[:6]:
            p_img = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
            if p_img is not None:
                probe_list.append((sid, pf.name, p_img))

    genuine_scores = []
    impostor_scores = []

    for p_sid, pf_name, p_img in probe_list:
        p_emb = extract_bygait_emb(p_img)
        for g_sid, g_emb in gallery_embs.items():
            sim = float(np.dot(p_emb, g_emb))
            if p_sid == g_sid:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)

    gen_arr = np.array(genuine_scores)
    imp_arr = np.array(impostor_scores)
    N_gen = len(gen_arr)
    N_imp = len(imp_arr)

    thresholds = [round(t, 3) for t in np.arange(0.900, 1.000, 0.001)]
    results = []

    for th in thresholds:
        tp = int(np.sum(gen_arr >= th))
        fn = int(N_gen - tp)
        fp = int(np.sum(imp_arr >= th))
        tn = int(N_imp - fp)

        prec = round(tp / max(tp + fp, 1) * 100.0, 2) if (tp + fp) > 0 else 0.0
        rec = round(tp / N_gen * 100.0, 2)
        f1 = round(2 * (prec * rec) / max(prec + rec, 1e-8), 2) if (prec + rec) > 0 else 0.0
        tar = rec
        far = round(fp / N_imp * 100.0, 2)
        frr = round(fn / N_gen * 100.0, 2)
        balanced_acc = round((tar + (100.0 - far)) / 2.0, 2)
        youden_j = round(tar - far, 2)

        results.append({
            "threshold": th,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tar": tar,
            "far": far,
            "frr": frr,
            "balanced_acc": balanced_acc,
            "youden_j": youden_j,
        })


    max_f1_pt = max(results, key=lambda x: (x["f1"], x["precision"]))
    best_bal_pt = max(results, key=lambda x: (x["balanced_acc"], x["youden_j"]))
    eer_pt = min(results, key=lambda x: abs(x["far"] - x["frr"]))


    far_10_pts = [r for r in results if r["far"] <= 10.0]
    far_10_pt = max(far_10_pts, key=lambda x: x["tar"]) if far_10_pts else None

    far_5_pts = [r for r in results if r["far"] <= 5.0]
    far_5_pt = max(far_5_pts, key=lambda x: x["tar"]) if far_5_pts else None

    far_1_pts = [r for r in results if r["far"] <= 1.0]
    far_1_pt = max(far_1_pts, key=lambda x: x["tar"]) if far_1_pts else None

    output_data = {
        "sweep": results,
        "max_f1_pt": max_f1_pt,
        "best_bal_pt": best_bal_pt,
        "eer_pt": eer_pt,
        "far_10_pt": far_10_pt,
        "far_5_pt": far_5_pt,
        "far_1_pt": far_1_pt,
    }

    with open("outputs/fine_threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("SWEEP COMPLETED")
    print("MAX F1:", max_f1_pt)
    print("BEST BALANCED:", best_bal_pt)
    print("EER:", eer_pt)
    print("FAR <= 10%:", far_10_pt)
    print("FAR <= 5%:", far_5_pt)
    print("FAR <= 1%:", far_1_pt)

if __name__ == "__main__":
    run_fine_sweep()
