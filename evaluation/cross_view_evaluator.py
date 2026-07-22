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
from evaluation.metrics import compute_rank_k_accuracies

ALL_ANGLES = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]


class SubjectDisjointCrossViewEvaluator(SubjectDisjointEvaluator):
    """
    Evaluates Rank-1, Rank-5, Rank-10 cross-view recognition performance across all
    11 x 11 gallery-angle x probe-angle pair combinations on subject-disjoint test set.
    """

    def evaluate_cross_view_matrices(self) -> dict:
        test_subjects = self.split_manifest["test_subjects"]

        # Build all gallery & probe items for test subjects
        gallery_items, probe_items = build_gallery_and_probe_sets(
            subjects=test_subjects,
            gei_root=str(self.gei_root),
        )

        print(f"Building feature embeddings for Cross-View Matrix ({len(gallery_items)} gallery, {len(probe_items)} probes)...")
        # Pre-extract all embeddings into cache
        for item in gallery_items:
            self.image_to_embedding(Path(item["path"]))
        for item in probe_items:
            self.image_to_embedding(Path(item["path"]))

        # Group gallery items by (angle, subject_id)
        gal_by_angle = {angle: [i for i in gallery_items if i["angle"] == angle] for angle in ALL_ANGLES}
        prb_by_angle_cond = {
            cond: {angle: [i for i in probe_items if i["angle"] == angle and i["condition"] == cond] for angle in ALL_ANGLES}
            for cond in ["NM", "BG", "CL"]
        }
        prb_by_angle_all = {angle: [i for i in probe_items if i["angle"] == angle] for angle in ALL_ANGLES}

        # Structure to hold matrices: matrix[gal_angle][prb_angle] = rank1_acc
        matrix_rank1 = {g_ang: {} for g_ang in ALL_ANGLES}
        matrix_rank5 = {g_ang: {} for g_ang in ALL_ANGLES}
        matrix_rank10 = {g_ang: {} for g_ang in ALL_ANGLES}

        matrix_by_cond = {
            cond: {g_ang: {} for g_ang in ALL_ANGLES}
            for cond in ["NM", "BG", "CL"]
        }

        cross_view_rank1_values = []
        same_view_rank1_values = []
        all_rank1_values = []

        for g_ang in ALL_ANGLES:
            gal_subset = gal_by_angle[g_ang]
            if not gal_subset:
                continue

            gal_features = np.asarray([self.image_to_embedding(Path(i["path"])) for i in gal_subset], dtype=np.float32)
            gal_labels = np.asarray([i["subject_id"] for i in gal_subset])
            metadata = {sub: {"status": "ACTIVE", "enabled": True} for sub in set(gal_labels)}

            for p_ang in ALL_ANGLES:
                prb_subset = prb_by_angle_all[p_ang]
                if not prb_subset:
                    matrix_rank1[g_ang][p_ang] = 0.0
                    matrix_rank5[g_ang][p_ang] = 0.0
                    matrix_rank10[g_ang][p_ang] = 0.0
                    continue

                top_k_lists = []
                true_labels = []

                for prb in prb_subset:
                    prb_feat = self.image_to_embedding(Path(prb["path"]))
                    actual_id = prb["subject_id"]

                    matches = self.matcher.top_k_matches(
                        query_feature=prb_feat,
                        gallery_features=gal_features,
                        gallery_labels=gal_labels,
                        metadata=metadata,
                        k=10,
                    )

                    ranked_ids = [m[0] for m in matches]
                    top_k_lists.append(ranked_ids)
                    true_labels.append(actual_id)

                ranks = compute_rank_k_accuracies(top_k_lists, true_labels, ks=(1, 5, 10))
                matrix_rank1[g_ang][p_ang] = round(ranks[1], 4)
                matrix_rank5[g_ang][p_ang] = round(ranks[5], 4)
                matrix_rank10[g_ang][p_ang] = round(ranks[10], 4)

                all_rank1_values.append(ranks[1])
                if g_ang == p_ang:
                    same_view_rank1_values.append(ranks[1])
                else:
                    cross_view_rank1_values.append(ranks[1])

                # Per-condition matrix
                for cond in ["NM", "BG", "CL"]:
                    cond_prbs = prb_by_angle_cond[cond][p_ang]
                    if not cond_prbs:
                        matrix_by_cond[cond][g_ang][p_ang] = 0.0
                        continue

                    c_top_k = []
                    c_true = []
                    for prb in cond_prbs:
                        prb_feat = self.image_to_embedding(Path(prb["path"]))
                        actual_id = prb["subject_id"]
                        matches = self.matcher.top_k_matches(
                            query_feature=prb_feat,
                            gallery_features=gal_features,
                            gallery_labels=gal_labels,
                            metadata=metadata,
                            k=1,
                        )
                        c_top_k.append([m[0] for m in matches])
                        c_true.append(actual_id)

                    c_ranks = compute_rank_k_accuracies(c_top_k, c_true, ks=(1,))
                    matrix_by_cond[cond][g_ang][p_ang] = round(c_ranks[1], 4)

        avg_cross_view_rank1 = sum(cross_view_rank1_values) / len(cross_view_rank1_values) if cross_view_rank1_values else 0.0
        avg_same_view_rank1 = sum(same_view_rank1_values) / len(same_view_rank1_values) if same_view_rank1_values else 0.0
        avg_overall_rank1 = sum(all_rank1_values) / len(all_rank1_values) if all_rank1_values else 0.0

        report = {
            "evaluation_type": "Subject-Disjoint Cross-View Matrix Evaluation",
            "checkpoint": str(self.model_path),
            "split_manifest_path": "configs/subject_split.json",
            "test_subjects_count": len(test_subjects),
            "angles_evaluated": ALL_ANGLES,
            "overall_average_rank1": round(avg_overall_rank1, 4),
            "same_view_average_rank1": round(avg_same_view_rank1, 4),
            "cross_view_average_rank1": round(avg_cross_view_rank1, 4),
            "matrix_rank1": matrix_rank1,
            "matrix_rank5": matrix_rank5,
            "matrix_rank10": matrix_rank10,
            "matrix_per_condition_rank1": matrix_by_cond,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save JSON
        json_path = self.report_dir / "cross_view_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        # Save CSV
        csv_path = self.report_dir / "cross_view_matrix.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Gallery_Angle"] + ALL_ANGLES)
            for g_ang in ALL_ANGLES:
                row = [g_ang] + [matrix_rank1[g_ang].get(p_ang, 0.0) for p_ang in ALL_ANGLES]
                writer.writerow(row)

        # Save Markdown Report
        md_path = self.report_dir / "cross_view_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# ARGUS Cross-View Rank-1 Accuracy Matrix\n\n")
            f.write(f"- **Checkpoint:** `{self.model_path}`\n")
            f.write(f"- **Test Subjects:** {len(test_subjects)}\n")
            f.write(f"- **Cross-View Avg (Excl Same View):** `{avg_cross_view_rank1 * 100:.2f}%`\n")
            f.write(f"- **Same-View Avg:** `{avg_same_view_rank1 * 100:.2f}%`\n")
            f.write(f"- **Overall Avg:** `{avg_overall_rank1 * 100:.2f}%`\n\n")
            f.write("### Rank-1 Accuracy Matrix (Gallery View vs Probe View)\n\n")
            header = "| Gal \\ Probe | " + " | ".join(ALL_ANGLES) + " |\n"
            sep = "| --- | " + " | ".join(["---"] * len(ALL_ANGLES)) + " |\n"
            f.write(header + sep)
            for g_ang in ALL_ANGLES:
                row_str = f"| **{g_ang}°** | " + " | ".join([f"{matrix_rank1[g_ang].get(p_ang, 0.0)*100:.1f}%" for p_ang in ALL_ANGLES]) + " |\n"
                f.write(row_str)

        return report
