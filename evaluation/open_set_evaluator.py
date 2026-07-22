import csv
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import SubjectDisjointEvaluator
from evaluation.gallery_probe_builder import build_gallery_and_probe_sets
from evaluation.leakage_validator import assert_gallery_probe_disjointness
from evaluation.metrics import compute_biometric_rates, compute_roc_auc_eer


class SubjectDisjointOpenSetEvaluator(SubjectDisjointEvaluator):
    """
    Evaluates open-set identification and unknown subject rejection performance
    using held-out test subjects partitioned into Known Enrolled vs Unknown Open-Set identities.
    """

    def __init__(
        self,
        gei_root: str = "data/casia_processed/gei",
        model_path: str = "runs/exp_001/best_model.pth",
        split_config_path: str = "configs/subject_split.json",
        threshold: float = 0.85,
        known_ratio: float = 0.5,
        report_dir: str = "runs/exp_001/evaluation_subject_disjoint",
    ) -> None:
        super().__init__(
            gei_root=gei_root,
            model_path=model_path,
            split_config_path=split_config_path,
            threshold=threshold,
            report_dir=report_dir,
        )
        self.known_ratio = known_ratio

    def evaluate_open_set_protocol(self) -> dict:
        test_subjects = sorted(self.split_manifest["test_subjects"])
        num_known = max(1, int(len(test_subjects) * self.known_ratio))

        known_test_subjects = test_subjects[:num_known]       # e.g., 075-099
        unknown_test_subjects = test_subjects[num_known:]     # e.g., 100-124

        # Build gallery for KNOWN test subjects ONLY
        gallery_items, _ = build_gallery_and_probe_sets(
            subjects=known_test_subjects,
            gei_root=str(self.gei_root),
        )

        # Build probes for ALL test subjects (both known and unknown)
        _, probe_items = build_gallery_and_probe_sets(
            subjects=test_subjects,
            gei_root=str(self.gei_root),
        )

        # Disjointness assertion: ensure unknown test subjects DO NOT exist in gallery
        assert_gallery_probe_disjointness(
            gallery_paths=[i["path"] for i in gallery_items],
            probe_paths=[i["path"] for i in probe_items],
            train_subjects=self.split_manifest["train_subjects"],
            unknown_subjects=unknown_test_subjects,
            gallery_subjects=[i["subject_id"] for i in gallery_items],
        )

        print(f"Extracting features for Open-Set Evaluation ({len(gallery_items)} gallery, {len(probe_items)} probes)...")
        gal_features = np.asarray([self.image_to_embedding(Path(i["path"])) for i in gallery_items], dtype=np.float32)
        gal_labels = np.asarray([i["subject_id"] for i in gallery_items])

        metadata = {sub: {"status": "ACTIVE", "enabled": True} for sub in set(gal_labels)}

        scores = []
        is_genuine = []
        probe_details = []

        known_set = set(known_test_subjects)

        for prb in probe_items:
            prb_feat = self.image_to_embedding(Path(prb["path"]))
            actual_id = prb["subject_id"]

            matches = self.matcher.top_k_matches(
                query_feature=prb_feat,
                gallery_features=gal_features,
                gallery_labels=gal_labels,
                metadata=metadata,
                k=1,
            )

            best_id, best_score = matches[0] if matches else ("UNKNOWN", 0.0)
            scores.append(best_score)

            is_gen = actual_id in known_set and actual_id == best_id
            is_genuine.append(is_gen)

            probe_details.append({
                "probe_path": prb["path"],
                "actual_id": actual_id,
                "is_known_subject": actual_id in known_set,
                "predicted_id": best_id,
                "score": best_score,
                "is_genuine_match": is_gen,
            })

        scores_arr = np.asarray(scores, dtype=np.float32)
        is_genuine_arr = np.asarray(is_genuine, dtype=bool)

        # Compute ROC and EER across score range
        roc_results = compute_roc_auc_eer(scores_arr, is_genuine_arr, num_thresholds=200)

        # Compute biometric rates at operating threshold
        operating_rates = compute_biometric_rates(scores_arr, is_genuine_arr, threshold=self.threshold)

        # Verify that FAR/FRR metrics change across thresholds
        far_values = roc_results["far_list"]
        if len(set(far_values)) <= 1:
            raise ValueError("CRITICAL METRIC FAILURE: FAR metrics do not change across threshold sweep! Score distribution is degenerate.")

        report = {
            "evaluation_type": "Subject-Disjoint Open-Set Identification",
            "checkpoint": str(self.model_path),
            "split_manifest_path": "configs/subject_split.json",
            "operating_threshold": self.threshold,
            "known_test_subjects": known_test_subjects,
            "unknown_test_subjects": unknown_test_subjects,
            "gallery_samples_count": len(gallery_items),
            "total_probe_count": len(probe_items),
            "known_probe_count": sum(1 for p in probe_details if p["is_known_subject"]),
            "unknown_probe_count": sum(1 for p in probe_details if not p["is_known_subject"]),
            "ROC_AUC": round(roc_results["roc_auc"], 4),
            "EER": round(roc_results["eer"], 4),
            "EER_threshold": round(roc_results["eer_threshold"], 4),
            "operating_metrics": operating_rates,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save report JSON
        json_path = self.report_dir / "open_set_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        # Save scores JSON
        scores_path = self.report_dir / "open_set_scores.json"
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(probe_details, f, indent=2)

        # Save CSV
        csv_path = self.report_dir / "open_set_report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["ROC_AUC", report["ROC_AUC"]])
            writer.writerow(["EER", report["EER"]])
            writer.writerow(["EER_threshold", report["EER_threshold"]])
            writer.writerow(["FAR", operating_rates["FAR"]])
            writer.writerow(["FRR", operating_rates["FRR"]])
            writer.writerow(["TAR", operating_rates["TAR"]])
            writer.writerow(["TNR", operating_rates["TNR"]])
            writer.writerow(["Precision", operating_rates["precision"]])
            writer.writerow(["Recall", operating_rates["recall"]])
            writer.writerow(["F1_Score", operating_rates["f1_score"]])

        return report
