import csv
import json
from pathlib import Path
import numpy as np

from evaluation.evaluator import SplitEvaluator
from pipeline.steps.centroid_matching_step import CentroidMatchingStep


class OpenSetEvaluator(SplitEvaluator):
    def __init__(
        self,
        gei_root: str = "data/casia_processed/gei",
        model_path: str = "runs/exp_001/best_model.pth",
        gallery_ratio: float = 0.5,
        threshold: float = 0.85,
        known_ratio: float = 0.6,
        report_dir: str = "outputs/eval_reports",
    ) -> None:
        self.known_ratio = known_ratio
        super().__init__(
            gei_root=gei_root,
            model_path=model_path,
            gallery_ratio=gallery_ratio,
            threshold=threshold,
            report_dir=report_dir,
        )
        self.centroid_matcher = CentroidMatchingStep(
            threshold=threshold,
            margin=0.05,
            top_k=5,
        )

    def _build_open_set_data(self):
        all_subjects = sorted([d.name for d in self.gei_root.iterdir() if d.is_dir()])
        num_subjects = len(all_subjects)

        num_known = int(num_subjects * self.known_ratio)
        num_known = max(1, min(num_known, num_subjects - 1))

        known_ids = set(all_subjects[:num_known])
        unknown_ids = set(all_subjects[num_known:])

        gallery_features = []
        gallery_labels = []
        test_items = []
        metadata = {}

        for person_dir in sorted(self.gei_root.iterdir()):
            if not person_dir.is_dir():
                continue

            person_id = person_dir.name
            image_paths = list(person_dir.glob("*.png"))
            if len(image_paths) < 2:
                continue

            if person_id in known_ids:
                gallery_images, test_images = self._split_person_images(image_paths)
                for img_path in gallery_images:
                    gallery_features.append(self._image_to_embedding(img_path))
                    gallery_labels.append(person_id)

                metadata[person_id] = {
                    "embeddings": len(gallery_images),
                    "status": "ACTIVE",
                    "enabled": True,
                    "source": "EVALUATION",
                }

                for img_path in test_images:
                    test_items.append((img_path, person_id, True))
            else:
                for img_path in image_paths:
                    test_items.append((img_path, person_id, False))

        return (
            np.asarray(gallery_features, dtype=np.float32),
            np.asarray(gallery_labels),
            metadata,
            test_items,
            known_ids,
            unknown_ids,
        )

    def evaluate_open_set(self, max_test_images: int | None = None, matching_mode: str = "flat") -> dict:
        (
            gallery_features,
            gallery_labels,
            metadata,
            test_items,
            known_ids,
            unknown_ids,
        ) = self._build_open_set_data()

        if max_test_images is not None:
            import random
            rng = random.Random(42)
            rng.shuffle(test_items)
            test_items = test_items[:max_test_images]

        tp = 0  # True Positive
        fp = 0  # False Positive
        tn = 0  # True Negative
        fn = 0  # False Negative

        correct_known = 0
        total_known = 0
        total_unknown = 0

        self.centroid_matcher.threshold = self.threshold

        for image_path, actual_id, is_known in test_items:
            query_feature = self._image_to_embedding(image_path)

            predicted_id, score = self.centroid_matcher.match(
                query_feature=query_feature,
                gallery_features=gallery_features,
                gallery_labels=gallery_labels,
                metadata=metadata,
                mode=matching_mode,
            )

            if is_known:
                total_known += 1
                if predicted_id == actual_id:
                    tp += 1
                    correct_known += 1
                elif predicted_id == "UNKNOWN":
                    fn += 1
                else:
                    fp += 1
            else:
                total_unknown += 1
                if predicted_id == "UNKNOWN":
                    tn += 1
                else:
                    fp += 1

        known_accuracy = correct_known / total_known if total_known else 0.0
        unknown_rejection_rate = tn / total_unknown if total_unknown else 0.0
        false_accept_rate = (total_unknown - tn) / total_unknown if total_unknown else 0.0
        false_reject_rate = fn / total_known if total_known else 0.0

        results = {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "known_accuracy": known_accuracy,
            "unknown_rejection_rate": unknown_rejection_rate,
            "false_accept_rate": false_accept_rate,
            "false_reject_rate": false_reject_rate,
            "total_known": total_known,
            "total_unknown": total_unknown,
            "total_tested": len(test_items),
            "threshold": self.threshold,
            "known_ratio": self.known_ratio,
            "matching_mode": matching_mode,
        }

        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = self.report_dir / "open_set_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        report_csv_path = self.report_dir / "open_set_report.csv"
        with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(results.keys())
            writer.writerow(results.values())

        return results
