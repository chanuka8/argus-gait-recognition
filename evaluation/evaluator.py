import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dataset_split import load_or_create_subject_split
from evaluation.gallery_probe_builder import build_gallery_and_probe_sets
from evaluation.leakage_validator import (
    assert_subject_disjointness,
    assert_gallery_probe_disjointness,
)
from evaluation.metrics import (
    compute_rank_k_accuracies,
    compute_cmc_curve,
    compute_biometric_rates,
)
from models.architectures.bygait_light import ByGaitLight
from pipeline.steps.matching_step import MatchingStep


class SubjectDisjointEvaluator:
    """
    Evaluates gait recognition model on held-out test subjects with strict sequence separation.
    Zero data leakage guaranteed.
    """

    def __init__(
        self,
        gei_root: str = "data/casia_processed/gei",
        model_path: str = "runs/exp_001/best_model.pth",
        split_config_path: str = "configs/subject_split.json",
        threshold: float = 0.85,
        report_dir: str = "runs/exp_001/evaluation_subject_disjoint",
    ) -> None:
        self.gei_root = Path(gei_root)
        self.model_path = Path(model_path)
        self.split_manifest = load_or_create_subject_split(config_path=split_config_path, data_dir=gei_root)
        self.threshold = threshold

        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.matcher = MatchingStep(threshold=threshold)

        # Validate disjointness upfront
        assert_subject_disjointness(
            self.split_manifest["train_subjects"],
            self.split_manifest["val_subjects"],
            self.split_manifest["test_subjects"],
        )

        self.model = self._load_model()
        self.feature_cache = {}

    def _load_model(self) -> ByGaitLight:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        model = ByGaitLight()
        checkpoint = torch.load(self.model_path, map_location="cpu")

        filtered = {}
        for key, value in checkpoint.items():
            if key.startswith("backbone."):
                filtered[key.replace("backbone.", "")] = value
            elif key in model.state_dict():
                filtered[key] = value

        model.load_state_dict(filtered, strict=True)
        model.eval()
        return model

    def image_to_embedding(self, image_path: Path) -> np.ndarray:
        p_str = str(image_path.resolve())
        if p_str in self.feature_cache:
            return self.feature_cache[p_str]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        # Resize to (64, 128) - matching training GEIDataset
        image = cv2.resize(image, (64, 128))
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().flatten().astype(np.float32)

        # L2 Normalize
        norm = np.linalg.norm(embedding)
        embedding = embedding / (norm + 1e-8)

        self.feature_cache[p_str] = embedding
        return embedding

    def evaluate(self, gallery_view_filter: str | None = None) -> dict:
        test_subjects = self.split_manifest["test_subjects"]

        # Build gallery & probe items
        gallery_items, probe_items = build_gallery_and_probe_sets(
            subjects=test_subjects,
            gei_root=str(self.gei_root),
            gallery_view_filter=gallery_view_filter,
        )

        # Assert no path overlap and no train subjects
        assert_gallery_probe_disjointness(
            gallery_paths=[i["path"] for i in gallery_items],
            probe_paths=[i["path"] for i in probe_items],
            train_subjects=self.split_manifest["train_subjects"],
        )

        print(f"Extracting features for {len(gallery_items)} gallery and {len(probe_items)} probe items...")
        gallery_features = np.asarray([self.image_to_embedding(Path(i["path"])) for i in gallery_items], dtype=np.float32)
        gallery_labels = np.asarray([i["subject_id"] for i in gallery_items])

        metadata = {
            sub: {"embeddings": int(np.sum(gallery_labels == sub)), "status": "ACTIVE", "enabled": True}
            for sub in set(gallery_labels)
        }

        top_k_lists = []
        true_labels = []
        inference_times = []
        condition_results = {"NM": {"correct": 0, "total": 0}, "BG": {"correct": 0, "total": 0}, "CL": {"correct": 0, "total": 0}}

        scores_list = []
        is_genuine_list = []

        for prb in probe_items:
            t0 = time.perf_counter()
            prb_feat = self.image_to_embedding(Path(prb["path"]))
            actual_id = prb["subject_id"]
            cond = prb["condition"]

            matches = self.matcher.top_k_matches(
                query_feature=prb_feat,
                gallery_features=gallery_features,
                gallery_labels=gallery_labels,
                metadata=metadata,
                k=20,
            )
            t1 = time.perf_counter()
            inference_times.append(t1 - t0)

            ranked_ids = [m[0] for m in matches]
            top_k_lists.append(ranked_ids)
            true_labels.append(actual_id)

            best_id, best_score = matches[0] if matches else ("UNKNOWN", 0.0)
            scores_list.append(best_score)
            is_genuine_list.append(actual_id == best_id)

            cond_key = cond if cond in condition_results else "NM"
            condition_results[cond_key]["total"] += 1
            if matches and matches[0][0] == actual_id:
                condition_results[cond_key]["correct"] += 1

        rank_accs = compute_rank_k_accuracies(top_k_lists, true_labels, ks=(1, 5, 10))
        cmc = compute_cmc_curve(top_k_lists, true_labels, max_k=20)
        rates = compute_biometric_rates(scores_list, is_genuine_list, threshold=self.threshold)

        avg_lat_ms = (sum(inference_times) / len(inference_times)) * 1000.0 if inference_times else 0.0
        fps = len(inference_times) / sum(inference_times) if sum(inference_times) > 0 else 0.0

        cond_accs = {
            cond: {
                "correct": res["correct"],
                "total": res["total"],
                "rank1_accuracy": res["correct"] / res["total"] if res["total"] > 0 else 0.0,
            }
            for cond, res in condition_results.items()
        }

        report = {
            "evaluation_type": "Subject-Disjoint Closed-Set Identification",
            "checkpoint": str(self.model_path),
            "split_manifest_path": "configs/subject_split.json",
            "threshold": self.threshold,
            "test_subjects_count": len(test_subjects),
            "gallery_sample_count": len(gallery_items),
            "probe_sample_count": len(probe_items),
            "rank1_accuracy": rank_accs[1],
            "rank5_accuracy": rank_accs[5],
            "rank10_accuracy": rank_accs[10],
            "cmc_curve": cmc,
            "biometric_rates": rates,
            "condition_wise_accuracy": cond_accs,
            "avg_inference_latency_ms": round(avg_lat_ms, 4),
            "inference_fps": round(fps, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.report_dir / "closed_set_eval_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return report