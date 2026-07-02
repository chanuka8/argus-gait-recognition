import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from evaluation.metrics import EvaluationMetrics
from evaluation.roc import ROCAnalyzer
from models.architectures.bygait_light import ByGaitLight
from pipeline.steps.matching_step import MatchingStep


class SplitEvaluator:
    def __init__(
        self,
        gei_root: str = "data/casia_processed/gei",
        model_path: str = "runs/exp_001/best_model.pth",
        gallery_ratio: float = 0.5,
        threshold: float = 0.75,
        report_dir: str = "outputs/eval_reports",
    ) -> None:
        self.gei_root = Path(
            gei_root,
        )

        self.model_path = model_path
        self.gallery_ratio = gallery_ratio
        self.threshold = threshold

        self.matcher = MatchingStep(
            threshold=threshold,
        )

        self.metrics = EvaluationMetrics()
        self.roc = ROCAnalyzer()

        self.report_dir = Path(
            report_dir,
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = self._load_model()

    def _load_model(
        self,
    ) -> ByGaitLight:
        model = ByGaitLight()

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
        )

        filtered = {}

        for key, value in checkpoint.items():
            if key.startswith(
                "backbone.",
            ):
                filtered[
                    key.replace(
                        "backbone.",
                        "",
                    )
                ] = value

        model.load_state_dict(
            filtered,
            strict=True,
        )

        model.eval()

        return model

    def _image_to_embedding(
        self,
        image_path: Path,
    ) -> np.ndarray:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to read image: {image_path}"
            )

        image = image.astype(
            np.float32,
        ) / 255.0

        tensor = torch.from_numpy(
            image,
        ).unsqueeze(
            0,
        ).unsqueeze(
            0,
        )

        with torch.no_grad():
            embedding = (
                self.model(
                    tensor,
                )
                .cpu()
                .numpy()
                .flatten()
                .astype(np.float32)
            )

        return embedding

    def _split_person_images(
        self,
        image_paths: list[Path],
    ) -> tuple[list[Path], list[Path]]:
        image_paths = sorted(
            image_paths,
        )

        split_index = int(
            len(image_paths) * self.gallery_ratio,
        )

        split_index = max(
            1,
            min(
                split_index,
                len(image_paths) - 1,
            ),
        )

        return (
            image_paths[:split_index],
            image_paths[split_index:],
        )

    def _build_split_data(
        self,
    ):
        gallery_features = []
        gallery_labels = []
        test_items = []
        metadata = {}

        for person_dir in sorted(
            self.gei_root.iterdir(),
        ):
            if not person_dir.is_dir():
                continue

            person_id = person_dir.name

            image_paths = list(
                person_dir.glob(
                    "*.png",
                )
            )

            if len(
                image_paths,
            ) < 2:
                continue

            gallery_images, test_images = self._split_person_images(
                image_paths,
            )

            for image_path in gallery_images:
                gallery_features.append(
                    self._image_to_embedding(
                        image_path,
                    )
                )

                gallery_labels.append(
                    person_id,
                )

            metadata[person_id] = {
                "embeddings": len(
                    gallery_images,
                ),
                "status": "ACTIVE",
                "enabled": True,
                "source": "EVALUATION",
            }

            for image_path in test_images:
                test_items.append(
                    (
                        image_path,
                        person_id,
                    )
                )

        return (
            np.asarray(
                gallery_features,
                dtype=np.float32,
            ),
            np.asarray(
                gallery_labels,
            ),
            metadata,
            test_items,
        )

    def _save_json(
        self,
        filename: str,
        data: dict,
    ) -> Path:
        output_path = self.report_dir / filename

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
            )

        return output_path

    def evaluate(
        self,
        max_test_images: int | None = None,
    ) -> dict:
        (
            gallery_features,
            gallery_labels,
            metadata,
            test_items,
        ) = self._build_split_data()

        rank1 = 0
        rank5 = 0
        rank10 = 0
        false_match = 0
        false_non_match = 0
        tested = 0

        # Array to track cumulative matches at ranks 1 to 10
        cmc_ranks = [0] * 10

        for image_path, actual_id in tqdm(
            test_items,
            desc="Split Evaluation",
        ):
            query_feature = self._image_to_embedding(
                image_path,
            )

            top10 = self.matcher.top_k_matches(
                query_feature=query_feature,
                gallery_features=gallery_features,
                gallery_labels=gallery_labels,
                metadata=metadata,
                k=10,
            )

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

            self.metrics.update(
                actual_id,
                predicted_id,
            )

            top_ids = [
                identity
                for identity, _
                in top10
            ]

            if len(top_ids) > 0 and actual_id == top_ids[0]:
                rank1 += 1

            if actual_id in top_ids[:5]:
                rank5 += 1

            if actual_id in top_ids[:10]:
                rank10 += 1

            # Update CMC ranks
            for r in range(1, 11):
                if actual_id in top_ids[:r]:
                    cmc_ranks[r - 1] += 1

            if predicted_id == "UNKNOWN":
                false_non_match += 1

            if predicted_id not in (
                "UNKNOWN",
                actual_id,
            ):
                false_match += 1

            self.roc.add(
                score=score,
                is_match=actual_id in top_ids[:1],
            )

            tested += 1

            if max_test_images is not None and tested >= max_test_images:
                break

        summary = self.metrics.summary()

        rank1_accuracy = rank1 / tested if tested else 0.0
        rank5_accuracy = rank5 / tested if tested else 0.0
        rank10_accuracy = rank10 / tested if tested else 0.0

        fmr = false_match / tested if tested else 0.0
        fnmr = false_non_match / tested if tested else 0.0

        roc_summary = self.roc.compute(
            output_path=self.report_dir / "roc_curve.png",
        )

        # Plot CMC curve
        cmc_accuracies = [count / tested if tested else 0.0 for count in cmc_ranks]
        import matplotlib.pyplot as plt
        ranks = list(range(1, 11))
        plt.figure()
        plt.plot(ranks, cmc_accuracies, marker='o', color='b', label='CMC Curve')
        plt.xlabel('Rank')
        plt.ylabel('Identification Rate')
        plt.title('Cumulative Match Characteristic (CMC) Curve')
        plt.xticks(ranks)
        plt.grid(True)
        plt.ylim([0.0, 1.05])
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.report_dir / "cmc_curve.png")
        plt.close()

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
                "gallery_size": int(
                    len(
                        gallery_labels,
                    )
                ),
                "test_size_available": int(
                    len(
                        test_items,
                    )
                ),
                "gallery_ratio": self.gallery_ratio,
                "threshold": self.threshold,
                "roc_curve": str(
                    self.report_dir / "roc_curve.png",
                ),
                "cmc_curve": str(
                    self.report_dir / "cmc_curve.png",
                ),
            }
        )

        self._save_json(
            "split_eval_report.json",
            summary,
        )

        self._save_json(
            "confusion_matrix.json",
            summary[
                "confusion_matrix"
            ],
        )

        return summary