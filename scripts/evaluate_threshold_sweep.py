import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import SplitEvaluator
from evaluation.metrics import EvaluationMetrics
from evaluation.roc import ROCAnalyzer
from storage.vector_store import VectorStore


class SweepEvaluator(SplitEvaluator):
    def _build_split_data(self):
        store = VectorStore(gallery_dir="models/gallery")
        gallery = store.load()
        if gallery is None:
            raise RuntimeError("Gallery models/gallery not found. Please build it first.")
        features, labels, metadata = gallery

        person_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            person_to_indices[str(label)].append(idx)

        gallery_features = []
        gallery_labels = []
        test_items = []
        split_metadata = {}

        for person_id, indices in sorted(person_to_indices.items()):
            split_idx = int(len(indices) * self.gallery_ratio)
            split_idx = max(1, min(split_idx, len(indices) - 1))

            gallery_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            for idx in gallery_indices:
                gallery_features.append(features[idx])
                gallery_labels.append(person_id)

            for idx in test_indices:
                test_items.append((features[idx], person_id))

            split_metadata[person_id] = {
                "embeddings": len(gallery_indices),
                "status": "ACTIVE",
                "enabled": True,
                "source": "EVALUATION",
            }

        return (
            np.asarray(gallery_features, dtype=np.float32),
            np.asarray(gallery_labels),
            split_metadata,
            test_items,
        )

    def _image_to_embedding(self, image_path_or_feature):
        if isinstance(image_path_or_feature, np.ndarray):
            return image_path_or_feature
        return super()._image_to_embedding(image_path_or_feature)

    def evaluate(self, max_test_images: int | None = None) -> dict:
        (
            gallery_features,
            gallery_labels,
            metadata,
            test_items,
        ) = self._build_split_data()

        if max_test_images is not None:
            test_items = test_items[:max_test_images]

        tested = len(test_items)
        if tested == 0:
            return {}

        # 1. Prepare gallery once
        gallery_features_act, gallery_labels_act = self.matcher._prepare_gallery(
            gallery_features,
            gallery_labels,
            metadata,
        )

        if gallery_features_act is None or gallery_labels_act is None:
            return {}

        # 2. Extract and L2-normalize all queries in batch
        query_features = np.asarray(
            [self._image_to_embedding(item[0]) for item in test_items],
            dtype=np.float32,
        )
        norms = np.linalg.norm(query_features, axis=1, keepdims=True)
        query_features = query_features / (norms + 1e-8)

        # Normalize active gallery features
        gallery_norms = np.linalg.norm(gallery_features_act, axis=1, keepdims=True)
        gallery_features_act_norm = gallery_features_act / (gallery_norms + 1e-8)

        # 3. Batch matrix dot product
        scores = np.dot(query_features, gallery_features_act_norm.T)

        # 4. Compute metrics
        rank1 = 0
        rank5 = 0
        rank10 = 0
        false_match = 0
        false_non_match = 0

        self.metrics = EvaluationMetrics()
        self.roc = ROCAnalyzer()

        inference_times = []

        for idx, (_, actual_id) in enumerate(test_items):
            t_start = time.perf_counter()

            query_scores = scores[idx]
            # Get indices sorted by score descending
            top10_idx = np.argsort(query_scores)[::-1][:10]

            top10 = [
                (str(gallery_labels_act[i]), float(query_scores[i]))
                for i in top10_idx
            ]

            if not top10:
                predicted_id = "UNKNOWN"
                score = 0.0
            else:
                best_identity, best_score = top10[0]
                score = best_score

                if best_score >= self.threshold:
                    predicted_id = best_identity
                else:
                    predicted_id = "UNKNOWN"

            self.metrics.update(actual_id, predicted_id)

            top_ids = [identity for identity, _ in top10]

            if len(top_ids) > 0 and actual_id == top_ids[0]:
                rank1 += 1

            if actual_id in top_ids[:5]:
                rank5 += 1

            if actual_id in top_ids[:10]:
                rank10 += 1

            if predicted_id == "UNKNOWN":
                false_non_match += 1

            if predicted_id not in ("UNKNOWN", actual_id):
                false_match += 1

            self.roc.add(
                score=score,
                is_match=actual_id in top_ids[:1],
            )

            t_end = time.perf_counter()
            inference_times.append(t_end - t_start)

        summary = self.metrics.summary()

        rank1_accuracy = rank1 / tested if tested else 0.0
        rank5_accuracy = rank5 / tested if tested else 0.0
        rank10_accuracy = rank10 / tested if tested else 0.0

        fmr = false_match / tested if tested else 0.0
        fnmr = false_non_match / tested if tested else 0.0

        roc_summary = self.roc.compute(
            output_path=self.report_dir / f"roc_curve_{self.threshold:.2f}.png",
        )

        total_inference_time = sum(inference_times) if inference_times else 0.0
        avg_inference_time_ms = (
            (total_inference_time / len(inference_times)) * 1000.0
            if inference_times
            else 0.0
        )
        eval_fps = (
            len(inference_times) / total_inference_time
            if total_inference_time > 0
            else 0.0
        )

        summary.update(
            {
                "total": tested,
                "correct": rank1,
                "incorrect": tested - rank1,
                "rank1_accuracy": rank1_accuracy,
                "rank5_accuracy": rank5_accuracy,
                "rank10_accuracy": rank10_accuracy,
                "false_match_count": false_match,
                "false_non_match_count": false_non_match,
                "fmr": fmr,
                "fnmr": fnmr,
                "eer": roc_summary["eer"],
                "roc_auc": roc_summary["roc_auc"],
                "avg_inference_time_ms": round(avg_inference_time_ms, 4),
                "fps": round(eval_fps, 2),
                "gallery_size": int(len(gallery_labels)),
                "test_size_available": int(len(test_items)),
                "gallery_ratio": self.gallery_ratio,
                "threshold": self.threshold,
                "roc_curve": str(self.report_dir / f"roc_curve_{self.threshold:.2f}.png"),
            }
        )

        self._save_json(
            f"split_eval_report_{self.threshold:.2f}.json",
            summary,
        )

        self._save_json(
            f"confusion_matrix_{self.threshold:.2f}.json",
            summary["confusion_matrix"],
        )

        return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ARGUS thresholds via sweep evaluation"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max test images to process (None for all)",
    )
    parser.add_argument(
        "--gallery-ratio",
        type=float,
        default=0.5,
        help="Ratio of features to keep in gallery",
    )
    args = parser.parse_args()

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    sweep_results = []

    print(f"Starting optimized threshold sweep evaluation over: {thresholds}")

    for t in thresholds:
        print(f"Evaluating threshold: {t:.2f}...")
        evaluator = SweepEvaluator(
            gallery_ratio=args.gallery_ratio,
            threshold=t,
        )
        results = evaluator.evaluate(max_test_images=args.max_images)

        sweep_results.append(
            {
                "threshold": t,
                "rank1_accuracy": results["rank1_accuracy"],
                "rank5_accuracy": results["rank5_accuracy"],
                "rank10_accuracy": results["rank10_accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1_score": results["f1_score"],
                "false_match_count": results["false_match_count"],
                "false_non_match_count": results["false_non_match_count"],
                "fmr": results["fmr"],
                "fnmr": results["fnmr"],
                "eer": results["eer"],
                "roc_auc": results["roc_auc"],
                "avg_inference_time_ms": results["avg_inference_time_ms"],
                "fps": results["fps"],
            }
        )

    sorted_results = sorted(
        sweep_results,
        key=lambda x: x["rank1_accuracy"],
        reverse=True,
    )

    output_dir = Path("outputs/eval_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "threshold_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=4)

    csv_path = output_dir / "threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "rank1_accuracy",
                "rank5_accuracy",
                "rank10_accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_match_count",
                "false_non_match_count",
                "fmr",
                "fnmr",
                "eer",
                "roc_auc",
                "avg_inference_time_ms",
                "fps",
            ],
        )
        writer.writeheader()
        for row in sweep_results:
            writer.writerow(row)

    print("\n=== THRESHOLD SWEEP SUMMARY TABLE (Sorted by Rank-1 Accuracy) ===")
    print(
        f"{'Threshold':<10} | {'Rank-1 Acc':<12} | {'Rank-5 Acc':<12} | {'Rank-10 Acc':<12} | "
        f"{'Precision':<10} | {'Recall':<10} | {'F1':<10} | "
        f"{'FMR':<8} | {'FNMR':<8} | {'EER':<8} | {'ROC AUC':<8} | "
        f"{'Avg ms':<10} | {'FPS':<8}"
    )
    print("-" * 165)
    for res in sorted_results:
        print(
            f"{res['threshold']:<10.2f} | "
            f"{res['rank1_accuracy']:<12.4f} | "
            f"{res['rank5_accuracy']:<12.4f} | "
            f"{res['rank10_accuracy']:<12.4f} | "
            f"{res['precision']:<10.4f} | "
            f"{res['recall']:<10.4f} | "
            f"{res['f1_score']:<10.4f} | "
            f"{res['fmr']:<8.4f} | "
            f"{res['fnmr']:<8.4f} | "
            f"{res['eer']:<8.4f} | "
            f"{res['roc_auc']:<8.4f} | "
            f"{res['avg_inference_time_ms']:<10.4f} | "
            f"{res['fps']:<8.2f}"
        )

    print(f"\nSaved JSON report -> {json_path}")
    print(f"Saved CSV report -> {csv_path}")


if __name__ == "__main__":
    main()
