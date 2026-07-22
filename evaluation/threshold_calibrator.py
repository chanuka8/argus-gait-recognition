import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.gallery_probe_builder import build_gallery_and_probe_sets


class ThresholdCalibrator:
    """
    Calibrates recognition operating threshold using ONLY validation subjects.
    Never tunes or optimizes thresholds on the test set.
    """

    def __init__(
        self,
        val_subjects: list[str],
        feature_extractor_fn,  # Callable[[Path], np.ndarray]
        known_ratio: float = 0.5,
    ) -> None:
        self.val_subjects = sorted(val_subjects)
        self.feature_extractor = feature_extractor_fn

        num_known = max(1, int(len(self.val_subjects) * known_ratio))
        self.known_val_subjects = set(self.val_subjects[:num_known])
        self.unknown_val_subjects = set(self.val_subjects[num_known:])

    def calibrate(
        self,
        criterion: str = "min_eer",
        target_far: float = 0.01,
        gei_root: str = "data/casia_processed/gei",
        output_dir: str = "runs/exp_001/evaluation_subject_disjoint",
    ) -> dict:
        # Build gallery for KNOWN validation subjects only
        gallery_items, _ = build_gallery_and_probe_sets(
            subjects=sorted(list(self.known_val_subjects)),
            gei_root=gei_root,
        )

        # Build probes for ALL validation subjects (both known and unknown)
        _, probe_items = build_gallery_and_probe_sets(
            subjects=self.val_subjects,
            gei_root=gei_root,
        )

        if not gallery_items or not probe_items:
            raise RuntimeError("Validation gallery or probe items empty. Cannot calibrate threshold.")

        # Extract embeddings
        gal_features = np.asarray([self.feature_extractor(Path(item["path"])) for item in gallery_items], dtype=np.float32)
        gal_labels = np.asarray([item["subject_id"] for item in gallery_items])

        # L2 normalize gallery
        gal_norms = np.linalg.norm(gal_features, axis=1, keepdims=True)
        gal_features_norm = gal_features / (gal_norms + 1e-8)

        # Evaluate probes
        scores = []
        is_genuine = []
        true_labels = []

        for prb in probe_items:
            prb_path = Path(prb["path"])
            sub_id = prb["subject_id"]
            feat = self.feature_extractor(prb_path)
            feat_norm = feat / (np.linalg.norm(feat) + 1e-8)

            # Cosine similarity matrix product
            sims = np.dot(gal_features_norm, feat_norm)
            best_idx = np.argmax(sims)
            best_sim = float(sims[best_idx])
            best_match_id = str(gal_labels[best_idx])

            scores.append(best_sim)
            true_labels.append(sub_id)
            is_genuine.append(sub_id in self.known_val_subjects and sub_id == best_match_id)

        scores = np.asarray(scores, dtype=np.float32)
        is_genuine = np.asarray(is_genuine, dtype=bool)

        # Sweep thresholds
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        thresholds = np.linspace(min_score - 0.05, max_score + 0.05, 201)

        sweep_records = []
        best_threshold = 0.85

        total_genuine_queries = int(np.sum([sub in self.known_val_subjects for sub in true_labels]))
        total_impostor_queries = len(probe_items) - total_genuine_queries

        for th in thresholds:
            th = float(th)
            # Accept if score >= th
            accepted = scores >= th

            # Genuine query accepted with correct ID => True Accept
            # Genuine query rejected (score < th) => False Reject
            # Impostor query (or genuine with wrong match) accepted (score >= th) => False Accept
            # Impostor query rejected (score < th) => True Reject
            
            tp = int(np.sum(accepted & is_genuine))
            fn = total_genuine_queries - tp
            fp = int(np.sum(accepted & (~is_genuine)))
            tn = total_impostor_queries - fp

            far = fp / total_impostor_queries if total_impostor_queries > 0 else 0.0
            frr = fn / total_genuine_queries if total_genuine_queries > 0 else 0.0
            tar = tp / total_genuine_queries if total_genuine_queries > 0 else 0.0
            tnr = tn / total_impostor_queries if total_impostor_queries > 0 else 0.0

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tar
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            sweep_records.append({
                "threshold": th,
                "FAR": far,
                "FRR": frr,
                "TAR": tar,
                "TNR": tnr,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "EER_gap": abs(far - frr),
            })

        # Find best threshold according to criterion
        if criterion == "min_eer":
            sweep_records_sorted = sorted(sweep_records, key=lambda x: x["EER_gap"])
            best_threshold = sweep_records_sorted[0]["threshold"]
        elif criterion == "max_f1":
            sweep_records_sorted = sorted(sweep_records, key=lambda x: x["f1_score"], reverse=True)
            best_threshold = sweep_records_sorted[0]["threshold"]
        elif criterion == "target_far":
            eligible = [r for r in sweep_records if r["FAR"] <= target_far]
            if eligible:
                # Pick one with highest TAR among eligible
                eligible_sorted = sorted(eligible, key=lambda x: x["TAR"], reverse=True)
                best_threshold = eligible_sorted[0]["threshold"]
            else:
                best_threshold = float(np.max(scores))

        calibration_result = {
            "protocol": "Validation Threshold Calibration (Validation Subjects Only)",
            "val_subjects": self.val_subjects,
            "known_val_subjects": sorted(list(self.known_val_subjects)),
            "unknown_val_subjects": sorted(list(self.unknown_val_subjects)),
            "criterion_used": criterion,
            "selected_threshold": round(float(best_threshold), 4),
            "calibration_score_min": round(min_score, 4),
            "calibration_score_max": round(max_score, 4),
            "total_val_probes": len(probe_items),
            "sweep_sample_count": len(sweep_records),
        }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "threshold_calibration.json", "w", encoding="utf-8") as f:
            json.dump(calibration_result, f, indent=4)

        return calibration_result
