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
from intelligence.open_set_recognizer import OpenSetRecognizer


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
        margin_threshold: float = 0.05,
    ) -> None:
        super().__init__(
            gei_root=gei_root,
            model_path=model_path,
            split_config_path=split_config_path,
            threshold=threshold,
            report_dir=report_dir,
        )
        self.known_ratio = known_ratio
        self.margin_threshold = margin_threshold
        unk_th = max(0.05, threshold - 0.10)
        self.open_set_recognizer = OpenSetRecognizer(
            known_threshold=threshold,
            unknown_threshold=unk_th,
            margin_threshold=margin_threshold,
        )

    def evaluate_open_set_protocol(self) -> dict:
        test_subjects = sorted(self.split_manifest["test_subjects"])
        num_known = max(1, int(len(test_subjects) * self.known_ratio))

        known_test_subjects = test_subjects[:num_known]
        unknown_test_subjects = test_subjects[num_known:]

        gallery_items, _ = build_gallery_and_probe_sets(
            subjects=known_test_subjects,
            gei_root=str(self.gei_root),
        )

        _, probe_items = build_gallery_and_probe_sets(
            subjects=test_subjects,
            gei_root=str(self.gei_root),
        )

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
                k=5,
            )

            open_set_decision = self.open_set_recognizer.evaluate_open_set_decision(top_matches=matches)
            open_set_state = open_set_decision.state.value

            best_id, best_score = matches[0] if matches else ("UNKNOWN", 0.0)
            scores.append(best_score)

            margin = 0.0
            if len(matches) >= 2:
                second_diff = [m for m in matches[1:] if m[0] != best_id]
                if second_diff:
                    margin = best_score - float(second_diff[0][1])
                else:
                    margin = best_score - float(matches[1][1])

            is_gen = actual_id in known_set and actual_id == best_id
            is_genuine.append(is_gen)

            probe_details.append({
                "probe_path": prb["path"],
                "actual_id": actual_id,
                "is_known_subject": actual_id in known_set,
                "predicted_id": best_id,
                "score": best_score,
                "margin": round(margin, 4),
                "open_set_state": open_set_state,
                "is_genuine_match": is_gen,
            })

        scores_arr = np.asarray(scores, dtype=np.float32)
        is_genuine_arr = np.asarray(is_genuine, dtype=bool)

        roc_results = compute_roc_auc_eer(scores_arr, is_genuine_arr, num_thresholds=200)

        operating_rates = compute_biometric_rates(scores_arr, is_genuine_arr, threshold=self.threshold)

        margins_arr = np.asarray([p["margin"] for p in probe_details], dtype=np.float32)
        margin_accepted = (scores_arr >= self.threshold) & (margins_arr >= self.margin_threshold)
        margin_tp = int(np.sum(margin_accepted & is_genuine_arr))
        margin_fp = int(np.sum(margin_accepted & (~is_genuine_arr)))
        total_genuine = int(np.sum(is_genuine_arr))
        total_impostor = len(is_genuine_arr) - total_genuine
        margin_far = margin_fp / total_impostor if total_impostor > 0 else 0.0
        margin_frr = (total_genuine - margin_tp) / total_genuine if total_genuine > 0 else 0.0
        margin_tar = margin_tp / total_genuine if total_genuine > 0 else 0.0
        margin_tnr = 1.0 - margin_far
        margin_prec = margin_tp / (margin_tp + margin_fp) if (margin_tp + margin_fp) > 0 else 0.0
        margin_f1 = 2 * margin_prec * margin_tar / (margin_prec + margin_tar) if (margin_prec + margin_tar) > 0 else 0.0

        margin_aware_rates = {
            "threshold": self.threshold,
            "margin_threshold": self.margin_threshold,
            "FAR": round(margin_far, 4),
            "FRR": round(margin_frr, 4),
            "TAR": round(margin_tar, 4),
            "TNR": round(margin_tnr, 4),
            "precision": round(margin_prec, 4),
            "f1_score": round(margin_f1, 4),
            "tp": margin_tp,
            "fp": margin_fp,
        }

        far_values = roc_results["far_list"]
        if len(set(far_values)) <= 1:
            raise ValueError("CRITICAL METRIC FAILURE: FAR metrics do not change across threshold sweep! Score distribution is degenerate.")

        report = {
            "evaluation_type": "Subject-Disjoint Open-Set Identification",
            "checkpoint": str(self.model_path),
            "split_manifest_path": "configs/subject_split.json",
            "operating_threshold": self.threshold,
            "margin_threshold": self.margin_threshold,
            "known_test_subjects": known_test_subjects,
            "unknown_test_subjects": unknown_test_subjects,
            "gallery_samples_count": len(gallery_items),
            "total_probe_count": len(probe_items),
            "known_probe_count": sum(1 for p in probe_details if p["is_known_subject"]),
            "unknown_probe_count": sum(1 for p in probe_details if not p["is_known_subject"]),
            "open_set_state_counts": {
                "KNOWN": sum(1 for p in probe_details if p["open_set_state"] == "KNOWN"),
                "UNKNOWN": sum(1 for p in probe_details if p["open_set_state"] == "UNKNOWN"),
                "UNCERTAIN": sum(1 for p in probe_details if p["open_set_state"] == "UNCERTAIN"),
            },
            "ROC_AUC": round(roc_results["roc_auc"], 4),
            "EER": round(roc_results["eer"], 4),
            "EER_threshold": round(roc_results["eer_threshold"], 4),
            "operating_metrics": operating_rates,
            "margin_aware_metrics": margin_aware_rates,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


        json_path = self.report_dir / "open_set_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        scores_path = self.report_dir / "open_set_scores.json"
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(probe_details, f, indent=2)

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
