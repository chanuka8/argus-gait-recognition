import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset_split import load_or_create_subject_split
from evaluation.metrics import compute_biometric_rates, compute_roc_auc_eer
from models.architectures.pose_gait_3d import PoseLifter3D


class Evaluator3D:
    def __init__(
        self,
        model_path: str = "runs/exp_006_3d/best_model.pth",
        data_dir: str = "data/casia_processed/skeletons",
        split_config_path: str = "configs/subject_split.json",
        margin_threshold: float = 0.05,
        sequence_length: int = 30,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.split_config_path = split_config_path
        self.margin_threshold = margin_threshold
        self.sequence_length = sequence_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> tuple[PoseLifter3D, torch.nn.Module]:
        if not self.model_path.exists():
            raise FileNotFoundError(f"3D Gait model checkpoint not found: {self.model_path}")

        ckpt = torch.load(self.model_path, map_location=self.device)
        lifter = PoseLifter3D().to(self.device)

        encoder_type = ckpt.get("encoder_type", "tcn")
        from training.gait_3d_trainer import get_gait3d_model

        gait_net = get_gait3d_model(encoder_type=encoder_type, embedding_dim=256).to(self.device)

        lifter.load_state_dict(ckpt["lifter"])
        gait_net.load_state_dict(ckpt["gait_net"])

        lifter.eval()
        gait_net.eval()

        return lifter, gait_net

    def _build_gallery_and_probes(self, test_subjects: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        gallery_items = []
        probe_items = []
        test_set = set(test_subjects)

        for sub_dir in sorted(self.data_dir.glob("*")):
            if not sub_dir.is_dir() or sub_dir.name not in test_set:
                continue

            for npy_path in sorted(sub_dir.glob("*.npy")):
                stem = npy_path.stem
                parts = stem.split("_")
                if len(parts) >= 3:
                    sub_id = parts[0]
                    cond_raw = parts[1].upper()
                    view = parts[2]

                    item = {
                        "path": str(npy_path),
                        "subject_id": sub_id,
                        "condition_raw": parts[1],
                        "view": view,
                    }

                    if "NM-01" in cond_raw or "NM-02" in cond_raw:
                        gallery_items.append(item)
                    else:
                        probe_items.append(item)

        gal_paths = {g["path"] for g in gallery_items}
        prb_paths = {p["path"] for p in probe_items}
        overlap = gal_paths.intersection(prb_paths)
        if len(overlap) > 0:
            raise RuntimeError(f"CRITICAL LEAKAGE DETECTED: {len(overlap)} paths exist in both Gallery and Probe sets!")

        return gallery_items, probe_items

    def evaluate(self, output_dir: str = "runs/exp_006_3d/evaluation") -> dict[str, Any]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        lifter, gait_net = self._load_model()
        split_manifest = load_or_create_subject_split(self.split_config_path, "data/casia_processed/gei")
        test_subs = split_manifest["test_subjects"]

        gallery_items, probe_items = self._build_gallery_and_probes(test_subs)
        if not gallery_items or not probe_items:
            raise RuntimeError(f"Gallery ({len(gallery_items)}) or Probe ({len(probe_items)}) set empty!")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()

        def extract_emb(npy_path: str) -> np.ndarray:
            seq = np.load(npy_path)
            T_curr = len(seq)
            if T_curr == 0:
                seq_fixed = np.zeros((self.sequence_length, 17, 3), dtype=np.float32)
            elif T_curr < self.sequence_length:
                repeats = (self.sequence_length // T_curr) + 1
                seq_fixed = np.tile(seq, (repeats, 1, 1))[: self.sequence_length]
            else:
                seq_fixed = seq[: self.sequence_length]

            tensor = torch.from_numpy(seq_fixed).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                j3d = lifter(tensor)
                emb = gait_net(j3d).squeeze(0).cpu().numpy()
            return emb

        gal_features = np.stack([extract_emb(g["path"]) for g in gallery_items], axis=0)
        gal_labels = np.array([g["subject_id"] for g in gallery_items])

        prb_features = np.stack([extract_emb(p["path"]) for p in probe_items], axis=0)
        prb_labels = np.array([p["subject_id"] for p in probe_items])
        prb_conds = np.array([p["condition_raw"].upper() for p in probe_items])
        prb_views = np.array([p["view"] for p in probe_items])

        t_total = time.time() - t0
        total_queries = len(gallery_items) + len(probe_items)
        fps = total_queries / max(t_total, 1e-5)
        latency_ms = (t_total / max(total_queries, 1)) * 1000.0

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        sim_matrix = np.dot(prb_features, gal_features.T)

        rank1_hits = 0
        rank5_hits = 0

        probe_details = []

        for i in range(len(probe_items)):
            sims = sim_matrix[i]
            top_k_idx = np.argsort(sims)[::-1][:5]
            best_idx = top_k_idx[0]
            best_sim = float(sims[best_idx])
            predicted_id = gal_labels[best_idx]
            actual_id = prb_labels[i]

            margin = 0.0
            if len(top_k_idx) > 1:
                diff_ids = [idx for idx in top_k_idx if gal_labels[idx] != predicted_id]
                second_sim = float(sims[diff_ids[0]]) if diff_ids else float(sims[top_k_idx[1]])
                margin = best_sim - second_sim

            is_rank1 = actual_id == predicted_id
            is_rank5 = any(gal_labels[idx] == actual_id for idx in top_k_idx)

            if is_rank1:
                rank1_hits += 1
            if is_rank5:
                rank5_hits += 1

            probe_details.append(
                {
                    "path": probe_items[i]["path"],
                    "actual_id": actual_id,
                    "predicted_id": predicted_id,
                    "score": best_sim,
                    "margin": round(margin, 4),
                    "is_genuine": is_rank1,
                    "condition": prb_conds[i],
                    "view": prb_views[i],
                }
            )

        rank1_acc = rank1_hits / max(len(probe_items), 1)
        rank5_acc = rank5_hits / max(len(probe_items), 1)

        cond_accs = {}
        for c in ["NM", "BG", "CL"]:
            mask = np.array([c in cond for cond in prb_conds])
            if np.sum(mask) > 0:
                c_hits = sum(1 for i in np.where(mask)[0] if probe_details[i]["is_genuine"])
                cond_accs[c] = round(c_hits / float(np.sum(mask)), 4)
            else:
                cond_accs[c] = 0.0

        view_accs = {}
        unique_views = sorted(set(prb_views))
        for v in unique_views:
            mask = prb_views == v
            if np.sum(mask) > 0:
                v_hits = sum(1 for i in np.where(mask)[0] if probe_details[i]["is_genuine"])
                view_accs[v] = round(v_hits / float(np.sum(mask)), 4)

        known_test_subs = set(test_subs[: len(test_subs) // 2])
        scores_arr = np.array([p["score"] for p in probe_details], dtype=np.float32)
        margins_arr = np.array([p["margin"] for p in probe_details], dtype=np.float32)
        is_genuine_arr = np.array(
            [p["actual_id"] in known_test_subs and p["is_genuine"] for p in probe_details], dtype=bool
        )

        roc_res = compute_roc_auc_eer(scores_arr, is_genuine_arr)
        operating_rates = compute_biometric_rates(scores_arr, is_genuine_arr, threshold=roc_res["eer_threshold"])

        accepted = (scores_arr >= roc_res["eer_threshold"]) & (margins_arr >= self.margin_threshold)
        tp = int(np.sum(accepted & is_genuine_arr))
        fp = int(np.sum(accepted & (~is_genuine_arr)))
        n_gen = int(np.sum(is_genuine_arr))
        n_imp = len(is_genuine_arr) - n_gen

        margin_far = fp / n_imp if n_imp > 0 else 0.0
        margin_frr = (n_gen - tp) / n_gen if n_gen > 0 else 0.0
        margin_tar = tp / n_gen if n_gen > 0 else 0.0

        summary = {
            "evaluation_type": "3D Pose Gait Strict Subject-Disjoint Evaluation",
            "model_path": str(self.model_path),
            "gallery_count": len(gallery_items),
            "probe_count": len(probe_items),
            "disjoint_check": "VERIFIED (0 path overlap)",
            "metrics": {
                "rank1": round(rank1_acc, 4),
                "rank5": round(rank5_acc, 4),
                "NM": cond_accs.get("NM", 0.0),
                "BG": cond_accs.get("BG", 0.0),
                "CL": cond_accs.get("CL", 0.0),
                "ROC_AUC": round(roc_res["roc_auc"], 4),
                "EER": round(roc_res["eer"], 4),
                "threshold": round(roc_res["eer_threshold"], 4),
                "score_only_FAR": round(operating_rates["FAR"], 4),
                "score_only_FRR": round(operating_rates["FRR"], 4),
                "margin_aware_FAR": round(margin_far, 4),
                "margin_aware_FRR": round(margin_frr, 4),
                "margin_aware_TAR": round(margin_tar, 4),
            },
            "condition_wise": cond_accs,
            "view_wise": view_accs,
            "performance": {
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 2),
                "vram_mb": round(vram_mb, 1),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(out_dir / "eval_summary_3d.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return summary
