"""
EXP-004B CL Root Cause Analysis: HPP Part-Level Similarity Investigation.
Compares per-part-bin cosine similarity across NM, BG, CL conditions
to identify which body regions cause clothing-change degradation.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from evaluation.dataset_split import load_or_create_subject_split
from models.architectures.bygait_light import ByGaitLight


def load_model():
    ckpt_path = "runs/exp_003e_hpp_arcface_triplet025/best_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    filtered = {}
    for k, v in ckpt.items():
        if k.startswith("backbone."):
            filtered[k.replace("backbone.", "")] = v
        elif k.startswith(("features.", "embedding.")):
            filtered[k] = v
    model = ByGaitLight(part_bins=4)
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model()
    model.to(device)

    split_manifest = load_or_create_subject_split("configs/subject_split.json", "data/casia_processed/gei")
    test_subs = split_manifest["test_subjects"]

    def extract_features(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (64, 128))
        tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        with torch.no_grad():
            feat = model.features(tensor)
            part_pooled = model.pool(feat)
            bins = part_pooled.squeeze(-1).squeeze(0).T
            bins = F.normalize(bins, p=2, dim=1)

            flat = torch.flatten(part_pooled, 1)
            emb = model.embedding(flat)
            emb = F.normalize(emb, p=2, dim=1).squeeze(0)
        return emb.numpy(), bins.numpy()

    def get_condition(filename):
        stem = filename.lower()
        if "_nm-" in stem:
            return "NM"
        if "_bg-" in stem:
            return "BG"
        if "_cl-" in stem:
            return "CL"
        return None

    gei_root = Path("data/casia_processed/gei")
    sub_data = {}
    for sub in test_subs[:20]:
        sub_dir = gei_root / sub
        if not sub_dir.exists():
            continue
        sub_data[sub] = {"NM": [], "BG": [], "CL": []}
        for p in sorted(sub_dir.glob("*.png")):
            cond = get_condition(p.name)
            if cond is None:
                continue
            emb, parts = extract_features(str(p))
            sub_data[sub][cond].append({"emb": emb, "parts": parts})

    print(f"Subjects loaded: {len(sub_data)}")
    sample_sub = next(iter(sub_data))
    sample_counts = sub_data[sample_sub]
    print(
        f"Sample {sample_sub}: "
        f"NM={len(sample_counts['NM'])}, "
        f"BG={len(sample_counts['BG'])}, "
        f"CL={len(sample_counts['CL'])}"
    )

    nm_nm_sims, nm_bg_sims, nm_cl_sims = [], [], []
    part_sims = {
        "NM-NM": [[] for _ in range(4)],
        "NM-BG": [[] for _ in range(4)],
        "NM-CL": [[] for _ in range(4)],
    }

    for sub, conds in sub_data.items():
        nms = conds["NM"]
        bgs = conds["BG"]
        cls = conds["CL"]

        for i in range(len(nms)):
            for j in range(i + 1, len(nms)):
                nm_nm_sims.append(float(np.dot(nms[i]["emb"], nms[j]["emb"])))
                for b in range(4):
                    part_sims["NM-NM"][b].append(float(np.dot(nms[i]["parts"][b], nms[j]["parts"][b])))

        for nm_item in nms:
            for bg_item in bgs:
                nm_bg_sims.append(float(np.dot(nm_item["emb"], bg_item["emb"])))
                for b in range(4):
                    part_sims["NM-BG"][b].append(float(np.dot(nm_item["parts"][b], bg_item["parts"][b])))

        for nm_item in nms:
            for cl_item in cls:
                nm_cl_sims.append(float(np.dot(nm_item["emb"], cl_item["emb"])))
                for b in range(4):
                    part_sims["NM-CL"][b].append(float(np.dot(nm_item["parts"][b], cl_item["parts"][b])))

    print(f"\nPairs: NM-NM={len(nm_nm_sims)}, NM-BG={len(nm_bg_sims)}, NM-CL={len(nm_cl_sims)}")

    print("\n=== FULL EMBEDDING INTRA-SUBJECT SIMILARITIES ===")
    print(f"NM-NM: mean={np.mean(nm_nm_sims):.4f} +/- {np.std(nm_nm_sims):.4f}")
    print(f"NM-BG: mean={np.mean(nm_bg_sims):.4f} +/- {np.std(nm_bg_sims):.4f}")
    print(f"NM-CL: mean={np.mean(nm_cl_sims):.4f} +/- {np.std(nm_cl_sims):.4f}")
    gap = np.mean(nm_nm_sims) - np.mean(nm_cl_sims)
    print(f"Gap NM-NM vs NM-CL: {gap:.4f}")

    print("\n=== HPP PART BIN INTRA-SUBJECT SIMILARITIES ===")
    part_names = [
        "Bin 0 (Head/Shoulders)",
        "Bin 1 (Upper Torso)",
        "Bin 2 (Lower Torso/Hips)",
        "Bin 3 (Legs/Feet)",
    ]
    for b in range(4):
        nm_nm_p = np.mean(part_sims["NM-NM"][b])
        nm_bg_p = np.mean(part_sims["NM-BG"][b])
        nm_cl_p = np.mean(part_sims["NM-CL"][b])
        drop_pct = (nm_nm_p - nm_cl_p) / max(nm_nm_p, 1e-8) * 100
        print(f"{part_names[b]}:")
        print(f"  NM-NM={nm_nm_p:.4f} | NM-BG={nm_bg_p:.4f} | NM-CL={nm_cl_p:.4f} | Drop={drop_pct:.1f}%")

    inter_sims = []
    subs_list = list(sub_data.keys())
    for i in range(min(10, len(subs_list))):
        for j in range(i + 1, min(10, len(subs_list))):
            s1_nms = sub_data[subs_list[i]]["NM"]
            s2_nms = sub_data[subs_list[j]]["NM"]
            for a in s1_nms[:2]:
                for b_item in s2_nms[:2]:
                    inter_sims.append(float(np.dot(a["emb"], b_item["emb"])))
    if inter_sims:
        print("\n=== INTER-SUBJECT SIMILARITY (NM-NM, different subjects) ===")
        print(f"mean={np.mean(inter_sims):.4f} +/- {np.std(inter_sims):.4f}")


if __name__ == "__main__":
    main()
