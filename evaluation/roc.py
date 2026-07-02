from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ROCAnalyzer:
    def __init__(
        self,
    ) -> None:
        self.scores: list[float] = []
        self.labels: list[int] = []

    def add(
        self,
        score: float,
        is_match: bool,
    ) -> None:
        self.scores.append(
            float(score),
        )

        self.labels.append(
            1 if is_match else 0,
        )

    def _roc_points(
        self,
    ):
        if not self.scores:
            return (
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, 1.0]),
                np.asarray([0.0]),
            )

        scores = np.asarray(
            self.scores,
            dtype=np.float32,
        )

        labels = np.asarray(
            self.labels,
            dtype=np.int32,
        )

        thresholds = np.asarray(
            sorted(
                set(
                    scores.tolist(),
                ),
                reverse=True,
            ),
            dtype=np.float32,
        )

        if len(thresholds) == 0:
            thresholds = np.asarray(
                [0.0],
                dtype=np.float32,
            )

        fpr_values = []
        tpr_values = []

        positives = max(
            int(
                np.sum(
                    labels == 1,
                )
            ),
            1,
        )

        negatives = max(
            int(
                np.sum(
                    labels == 0,
                )
            ),
            1,
        )

        for threshold in thresholds:
            predicted_positive = scores >= threshold

            tp = int(
                np.sum(
                    predicted_positive & (labels == 1),
                )
            )

            fp = int(
                np.sum(
                    predicted_positive & (labels == 0),
                )
            )

            tpr = tp / positives
            fpr = fp / negatives

            tpr_values.append(
                tpr,
            )

            fpr_values.append(
                fpr,
            )

        fpr_array = np.asarray(
            fpr_values,
            dtype=np.float32,
        )

        tpr_array = np.asarray(
            tpr_values,
            dtype=np.float32,
        )

        order = np.argsort(
            fpr_array,
        )

        return (
            fpr_array[order],
            tpr_array[order],
            thresholds,
        )

    def compute(
        self,
        output_path: str | Path,
    ) -> dict:
        output_path = Path(
            output_path,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fpr, tpr, thresholds = self._roc_points()

        if len(fpr) > 1:
            roc_auc = float(
                np.trapezoid(
                    tpr,
                    fpr,
                )
            )
        else:
            roc_auc = 0.0

        fnr = 1.0 - tpr

        if len(fpr) > 0:
            index = int(
                np.nanargmin(
                    np.abs(
                        fnr - fpr,
                    )
                )
            )

            eer = float(
                (
                    fpr[index]
                    + fnr[index]
                )
                / 2.0
            )
        else:
            eer = 0.0

        plt.figure()
        plt.plot(
            fpr,
            tpr,
            label=f"ROC AUC = {roc_auc:.4f}",
        )
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random",
        )
        plt.xlabel(
            "False Positive Rate",
        )
        plt.ylabel(
            "True Positive Rate",
        )
        plt.title(
            "ARGUS Gait Recognition ROC Curve",
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_path,
        )
        plt.close()

        return {
            "roc_auc": roc_auc,
            "eer": eer,
            "threshold_count": int(
                len(
                    thresholds,
                )
            ),
        }