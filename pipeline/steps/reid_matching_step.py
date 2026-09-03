import numpy as np


class ReIDMatchingStep:
    def __init__(
        self,
        threshold: float = 0.6,
    ) -> None:
        self.threshold = threshold
        self._cache_key: tuple | None = None
        self._cached_active_features: np.ndarray | None = None
        self._cached_active_labels: np.ndarray | None = None

    def _is_active(
        self,
        label: str,
        metadata: dict | None,
    ) -> bool:
        if metadata is None or not isinstance(metadata, dict):
            return True

        entry = metadata.get(str(label))
        if entry is None:
            return True

        if isinstance(entry, dict):
            status = str(entry.get("status", "ACTIVE" if entry.get("enabled", True) else "DISABLED")).upper()
            enabled = bool(entry.get("enabled", status == "ACTIVE"))
            return status == "ACTIVE" and enabled

        return True

    def _prepare_gallery(
        self,
        gallery_features,
        gallery_labels,
        metadata: dict | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if gallery_features is None or gallery_labels is None:
            return None, None
        if len(gallery_features) == 0 or len(gallery_labels) == 0:
            return None, None

        current_key = (id(gallery_features), len(gallery_features), id(metadata))
        if self._cache_key == current_key and self._cached_active_features is not None:
            return self._cached_active_features, self._cached_active_labels

        features = np.asarray(gallery_features, dtype=np.float32)
        labels = np.asarray(gallery_labels)

        if len(features) == 0:
            return None, None

        if metadata:
            active_mask = np.fromiter(
                (self._is_active(str(lbl), metadata) for lbl in labels),
                dtype=bool,
                count=len(labels),
            )
            if not np.any(active_mask):
                return None, None
            features = features[active_mask]
            labels = labels[active_mask]

        # Pre-normalize and ensure C-contiguous for optimal BLAS gemv
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = np.ascontiguousarray(features / (norms + 1e-8), dtype=np.float32)

        self._cache_key = current_key
        self._cached_active_features = features
        self._cached_active_labels = tuple(str(x) for x in labels)
        return features, self._cached_active_labels

    def match(
        self,
        query_feature,
        gallery_features,
        gallery_labels,
        metadata: dict | None = None,
        unknown_label: str = "UNKNOWN_PERSON",
    ) -> tuple[str, float]:
        if query_feature is None or gallery_features is None or gallery_labels is None:
            return unknown_label, 0.0

        feats, lbls = self._prepare_gallery(gallery_features, gallery_labels, metadata)
        if feats is None or lbls is None or len(feats) == 0:
            return unknown_label, 0.0

        if isinstance(query_feature, np.ndarray) and query_feature.dtype == np.float32 and query_feature.ndim == 1:
            q = query_feature
        else:
            q = np.asarray(query_feature, dtype=np.float32).ravel()
            q_len_sq = float(np.dot(q, q))
            if q_len_sq == 0.0:
                return unknown_label, 0.0
            if abs(q_len_sq - 1.0) > 1e-4:
                q = q / np.sqrt(q_len_sq)

        scores = np.dot(feats, q)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        if best_score < self.threshold:
            return unknown_label, best_score

        return lbls[best_index], best_score
