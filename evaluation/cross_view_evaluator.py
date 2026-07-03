import csv
import json
from pathlib import Path

from evaluation.evaluator import SplitEvaluator


class CrossViewEvaluator(SplitEvaluator):
    def parse_view_angle(self, image_path: Path) -> str:
        stem = image_path.stem
        parts = stem.split("_")
        if len(parts) >= 3:
            angle = parts[-1]
            if angle.isdigit() and len(angle) == 3:
                return angle
        return "Fallback-View"

    def evaluate_cross_view(self, max_test_images: int | None = None) -> dict:
        (
            gallery_features,
            gallery_labels,
            metadata,
            test_items,
        ) = self._build_split_data()

        if max_test_images is not None:
            import random
            rng = random.Random(42)
            rng.shuffle(test_items)
            test_items = test_items[:max_test_images]


        view_stats = {}

        for image_path, actual_id in test_items:
            view_angle = self.parse_view_angle(image_path)

            if view_angle not in view_stats:
                view_stats[view_angle] = {"correct": 0, "total": 0}

            query_feature = self._image_to_embedding(image_path)

            top10 = self.matcher.top_k_matches(
                query_feature=query_feature,
                gallery_features=gallery_features,
                gallery_labels=gallery_labels,
                metadata=metadata,
                k=1,
            )

            if not top10:
                predicted_id = "UNKNOWN"
            else:
                best_identity, best_score = top10[0]
                if best_score >= self.threshold:
                    predicted_id = best_identity
                else:
                    predicted_id = "UNKNOWN"

            view_stats[view_angle]["total"] += 1
            if predicted_id == actual_id:
                view_stats[view_angle]["correct"] += 1

        per_view_results = {}
        for angle, stats in sorted(view_stats.items()):
            correct = stats["correct"]
            total = stats["total"]
            accuracy = correct / total if total > 0 else 0.0
            per_view_results[angle] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }

        fallback_used = "Fallback-View" in per_view_results

        report_data = {
            "per_view_metrics": per_view_results,
            "fallback_used": fallback_used,
            "threshold": self.threshold,
            "gallery_ratio": self.gallery_ratio,
        }

        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = self.report_dir / "cross_view_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)

        report_csv_path = self.report_dir / "cross_view_report.csv"
        with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["view_angle", "correct", "total", "accuracy"])
            for angle, metrics in per_view_results.items():
                writer.writerow([
                    angle,
                    metrics["correct"],
                    metrics["total"],
                    metrics["accuracy"]
                ])

        return report_data
