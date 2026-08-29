import numpy as np


class AppearanceMatchingStep:
    """
    Appearance Re-Identification matching using Cosine Similarity against
    512-dimensional appearance gallery vectors.
    """

    def __init__(
        self,
        threshold: float = 0.60,
    ) -> None:
        self.threshold = float(threshold)

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

    def _prepare_gallery(
        self,
        gallery_features,
        gallery_labels,
        metadata: dict | None,
    ):
        if gallery_features is None or gallery_labels is None:
            return None, None

        gallery_features = np.asarray(
            gallery_features,
            dtype=np.float32,
        )

        gallery_labels = np.asarray(
            gallery_labels,
        )

        if len(gallery_features) == 0:
            return None, None

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
            return None, None

        gallery_features = gallery_features[active_mask]
        gallery_labels = gallery_labels[active_mask]

        gallery_norms = np.linalg.norm(
            gallery_features,
            axis=1,
            keepdims=True,
        )

        gallery_features = gallery_features / (gallery_norms + 1e-8)

        return gallery_features, gallery_labels

    def _prepare_query(
        self,
        query_feature,
    ) -> np.ndarray | None:
        if query_feature is None:
            return None

        query_feature = np.asarray(
            query_feature,
            dtype=np.float32,
        ).ravel()

        if query_feature.size != 512:
            return None

        query_norm = float(np.linalg.norm(query_feature))
        if query_norm == 0.0:
            return None

        return (query_feature / (query_norm + 1e-8)).astype(np.float32)

    def match(
        self,
        query_feature,
        gallery_features,
        gallery_labels,
        metadata: dict | None = None,
        unknown_label: str = "UNKNOWN_PERSON",
    ) -> tuple[str, float]:
        """
        Match 512D query appearance embedding against gallery using cosine similarity.

        Returns:
            (identity, similarity_score)
        """
        query_vec = self._prepare_query(query_feature)
        if query_vec is None:
            return unknown_label, 0.0

        g_feats, g_lbls = self._prepare_gallery(
            gallery_features,
            gallery_labels,
            metadata,
        )

        if g_feats is None or g_lbls is None or len(g_feats) == 0:
            return unknown_label, 0.0

        scores = np.dot(
            g_feats,
            query_vec,
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
            g_lbls[best_index],
        ), best_score

    def top_k_matches(
        self,
        query_feature,
        gallery_features,
        gallery_labels,
        metadata: dict | None = None,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return top-K candidate matches ranked by cosine similarity."""
        query_vec = self._prepare_query(query_feature)
        if query_vec is None:
            return []

        g_feats, g_lbls = self._prepare_gallery(
            gallery_features,
            gallery_labels,
            metadata,
        )

        if g_feats is None or g_lbls is None or len(g_feats) == 0:
            return []

        scores = np.dot(
            g_feats,
            query_vec,
        )

        k = max(
            1,
            min(
                int(k),
                len(scores),
            ),
        )

        indices = np.argsort(
            scores,
        )[::-1][:k]

        return [
            (
                str(
                    g_lbls[idx],
                ),
                float(
                    scores[idx],
                ),
            )
            for idx in indices
        ]
