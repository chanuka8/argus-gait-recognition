import numpy as np


class ReIDMatchingStep:
    def __init__(
        self,
        threshold: float = 0.6,
    ) -> None:
        self.threshold = threshold

    def _is_active(
        self,
        label: str,
        metadata: dict | None,
    ) -> bool:
        if metadata is None:
            return True

        if not isinstance(metadata, dict):
            return True

        entry = metadata.get(
            str(label),
        )

        if entry is None:
            return True

        if isinstance(
            entry,
            dict,
        ):
            status = str(
                entry.get(
                    "status",
                    "ACTIVE" if entry.get("enabled", True) else "DISABLED",
                )
            ).upper()

            enabled = bool(
                entry.get(
                    "enabled",
                    status == "ACTIVE",
                )
            )

            return status == "ACTIVE" and enabled

        return True

    def match(
        self,
        query_feature,
        gallery_features,
        gallery_labels,
        metadata: dict | None = None,
        unknown_label: str = "UNKNOWN_PERSON",
    ):
        if gallery_features is None or gallery_labels is None:
            return unknown_label, 0.0

        gallery_features = np.asarray(
            gallery_features,
            dtype=np.float32,
        )

        gallery_labels = np.asarray(
            gallery_labels,
        )

        if len(gallery_features) == 0:
            return unknown_label, 0.0

        active_mask = np.asarray(
            [
                self._is_active(
                    str(label),
                    metadata,
                )
                for label in gallery_labels
            ],
            dtype=bool,
        )

        if not np.any(
            active_mask,
        ):
            return unknown_label, 0.0

        gallery_features = gallery_features[active_mask]

        gallery_labels = gallery_labels[active_mask]

        query_feature = np.asarray(
            query_feature,
            dtype=np.float32,
        )

        query_norm = np.linalg.norm(
            query_feature,
        )

        if query_norm == 0:
            return unknown_label, 0.0

        query_feature = query_feature / (query_norm + 1e-8)

        gallery_norms = np.linalg.norm(
            gallery_features,
            axis=1,
            keepdims=True,
        )

        gallery_features = gallery_features / (gallery_norms + 1e-8)

        scores = np.dot(
            gallery_features,
            query_feature,
        )

        best_index = int(
            np.argmax(
                scores,
            )
        )

        best_score = float(
            scores[best_index],
        )

        if best_score < self.threshold:
            return unknown_label, best_score

        return str(
            gallery_labels[best_index],
        ), best_score
