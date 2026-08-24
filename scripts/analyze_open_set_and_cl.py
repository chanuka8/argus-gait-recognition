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


def main():
    split_manifest = load_or_create_subject_split("configs/subject_split.json", "data/casia_processed/gei")
    val_subs = split_manifest["val_subjects"]
    test_subs = split_manifest["test_subjects"]

    ckpt_path = "runs/exp_003e_hpp_arcface_triplet025/best_model.pth"
    model = load_model(ckpt_path, part_bins=4)

    print("--- Extracting Validation Set Features (063-074) ---")
    val_known = val_subs[:len(val_subs)//2]

    val_gal_items, _ = build_gallery_and_probe_sets(val_known, "data/casia_processed/gei")
    _, _ = build_gallery_and_probe_sets(val_subs, "data/casia_processed/gei")

    print("--- Extracting Test Set Features (075-124) ---")
    test_known = test_subs[:25]

    test_gal_items, _ = build_gallery_and_probe_sets(test_known, "data/casia_processed/gei")
    _, test_prb_items = build_gallery_and_probe_sets(test_subs, "data/casia_processed/gei")

    test_gal_feats = np.asarray([image_to_embedding(model, i["path"]) for i in test_gal_items], dtype=np.float32)
    test_gal_labels = np.asarray([i["subject_id"] for i in test_gal_items])

    test_scores_top1 = []
    test_scores_top2 = []
    test_margins = []
    test_is_genuine = []
    test_conditions = []
    test_actual_ids = []

    test_known_set = set(test_known)

    for prb in test_prb_items:
        prb_feat = image_to_embedding(model, prb["path"])
        actual_id = prb["subject_id"]
        cond = prb["condition"]

        sims = np.dot(test_gal_feats, prb_feat)
        top_indices = np.argsort(sims)[::-1]

        top1_idx = top_indices[0]
        top1_score = float(sims[top1_idx])
        top1_id = test_gal_labels[top1_idx]

        different_id_indices = [idx for idx in top_indices if test_gal_labels[idx] != top1_id]
        top2_score = float(sims[different_id_indices[0]]) if different_id_indices else top1_score

        margin = top1_score - top2_score
        is_gen = (actual_id in test_known_set) and (actual_id == top1_id)

        test_scores_top1.append(top1_score)
        test_scores_top2.append(top2_score)
        test_margins.append(margin)
        test_is_genuine.append(is_gen)
        test_conditions.append(cond)
        test_actual_ids.append(actual_id)

    test_scores_top1 = np.asarray(test_scores_top1)
    test_margins = np.asarray(test_margins)
    test_is_genuine = np.asarray(test_is_genuine)

    print("\n=======================================================")
    print("      EXP-003E OPEN-SET SIMILARITY & MARGIN STATS      ")
    print("=======================================================")
    gen_scores = test_scores_top1[test_is_genuine]
    imp_scores = test_scores_top1[~test_is_genuine]
    gen_margins = test_margins[test_is_genuine]
    imp_margins = test_margins[~test_is_genuine]

    print(f"Genuine Scores: mean={np.mean(gen_scores):.4f}, std={np.std(gen_scores):.4f}, min={np.min(gen_scores):.4f}, max={np.max(gen_scores):.4f}")
    print(f"Impostor Scores: mean={np.mean(imp_scores):.4f}, std={np.std(imp_scores):.4f}, min={np.min(imp_scores):.4f}, max={np.max(imp_scores):.4f}")
    print(f"Genuine Margins: mean={np.mean(gen_margins):.4f}, std={np.std(gen_margins):.4f}")
    print(f"Impostor Margins: mean={np.mean(imp_margins):.4f}, std={np.std(imp_margins):.4f}")

    print("\n=======================================================")
    print("      EXP-003E CONDITION-WISE INTRA/INTER SIMILARITY   ")
    print("=======================================================")
    nm_feats, bg_feats, cl_feats = [], [], []

    for prb in test_prb_items:
        prb_feat = image_to_embedding(model, prb["path"])
        cond = prb["condition"]
        if cond == "NM":
            nm_feats.append(prb_feat)
        elif cond == "BG":
            bg_feats.append(prb_feat)
        elif cond == "CL":
            cl_feats.append(prb_feat)

    nm_feats = np.asarray(nm_feats)
    bg_feats = np.asarray(bg_feats)
    cl_feats = np.asarray(cl_feats)

    sub_to_cond_feats = {}
    for prb in test_prb_items:
        s = prb["subject_id"]
        c = prb["condition"]
        if s not in sub_to_cond_feats:
            sub_to_cond_feats[s] = {"NM": [], "BG": [], "CL": []}
        sub_to_cond_feats[s][c].append(image_to_embedding(model, prb["path"]))

    nm_nm_sims, nm_bg_sims, nm_cl_sims = [], [], []
    for s, conds in sub_to_cond_feats.items():
        nm_list = conds["NM"]
        bg_list = conds["BG"]
        cl_list = conds["CL"]

        if len(nm_list) > 1:
            for i in range(len(nm_list)):
                for j in range(i+1, len(nm_list)):
                    nm_nm_sims.append(np.dot(nm_list[i], nm_list[j]))

        for nm in nm_list:
            for bg in bg_list:
                nm_bg_sims.append(np.dot(nm, bg))
            for cl in cl_list:
                nm_cl_sims.append(np.dot(nm, cl))

    print(f"Intra-Subject NM-NM Similarity: mean={np.mean(nm_nm_sims):.4f}, std={np.std(nm_nm_sims):.4f}")
    print(f"Intra-Subject NM-BG Similarity: mean={np.mean(nm_bg_sims):.4f}, std={np.std(nm_bg_sims):.4f}")
    print(f"Intra-Subject NM-CL Similarity: mean={np.mean(nm_cl_sims):.4f}, std={np.std(nm_cl_sims):.4f}")


if __name__ == "__main__":
    main()
