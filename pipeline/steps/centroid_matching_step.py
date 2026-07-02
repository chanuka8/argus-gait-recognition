import numpy as np
from collections import Counter

from pipeline.steps.matching_step import MatchingStep


class CentroidMatchingStep:
    def __init__(
        self,
        threshold: float = 0.85,
        margin: float = 0.05,
        top_k: int = 5,
    ) -> None:
        self.threshold = threshold
        self.margin = margin
        self.top_k = top_k
        self.flat_matcher = MatchingStep(threshold=threshold)

    def _prepare_query(self, query_feature):
        return self.flat_matcher._prepare_query(query_feature)

    def _prepare_gallery(self, gallery_features, gallery_labels, metadata):
        return self.flat_matcher._prepare_gallery(gallery_features, gallery_labels, metadata)

    def _build_centroids(self, gallery_features, gallery_labels, metadata):
        active_features, active_labels = self._prepare_gallery(
            gallery_features, gallery_labels, metadata
        )
        if active_features is None or len(active_features) == 0:
            return None, None

        unique_labels = sorted(list(set(active_labels)))
        centroid_features = []

        for label in unique_labels:
            indices = np.where(active_labels == label)[0]
            feats = active_features[indices]
            centroid = np.mean(feats, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / (norm + 1e-8)
            centroid_features.append(centroid)

        return np.asarray(centroid_features, dtype=np.float32), np.asarray(unique_labels)

    def match(
        self,
        query_feature,
        gallery_features,
        gallery_labels,
        metadata: dict | None = None,
        mode: str = "flat",
    ) -> tuple[str, float]:
        """
        Supports mode:
        - 'flat': Standard matching over flat gallery features
        - 'centroid': Centroid matching
        - 'centroid_margin': Centroid matching with margin rule
        - 'centroid_margin_topk': Centroid matching + margin rule + topk consensus
        """
        # Flat mode fallback
        if mode == "flat":
            return self.flat_matcher.match(
                query_feature, gallery_features, gallery_labels, metadata
            )

        # Prepare Query
        q_feat = self._prepare_query(query_feature)
        if q_feat is None:
            return "UNKNOWN", 0.0

        # Build Centroids
        centroids, centroid_labels = self._build_centroids(
            gallery_features, gallery_labels, metadata
        )
        if centroids is None or len(centroids) == 0:
            return "UNKNOWN", 0.0

        # Compute centroid scores
        scores = np.dot(centroids, q_feat)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_identity = str(centroid_labels[best_idx])

        # Apply Rejection/Rules
        if mode == "centroid":
            if best_score < self.threshold:
                return "UNKNOWN", best_score
            return best_identity, best_score

        elif mode == "centroid_margin":
            # Check margin rule (best - second_best >= margin)
            if len(scores) > 1:
                sorted_scores = sorted(scores, reverse=True)
                if sorted_scores[0] - sorted_scores[1] < self.margin:
                    return "UNKNOWN", best_score
            if best_score < self.threshold:
                return "UNKNOWN", best_score
            return best_identity, best_score

        elif mode == "centroid_margin_topk":
            # Check margin rule first
            if len(scores) > 1:
                sorted_scores = sorted(scores, reverse=True)
                if sorted_scores[0] - sorted_scores[1] < self.margin:
                    return "UNKNOWN", best_score

            # Check threshold
            if best_score < self.threshold:
                return "UNKNOWN", best_score

            # Check top-K consensus in flat features
            active_features, active_labels = self._prepare_gallery(
                gallery_features, gallery_labels, metadata
            )
            if active_features is None or len(active_features) == 0:
                return "UNKNOWN", best_score

            flat_scores = np.dot(active_features, q_feat)
            top_k_indices = np.argsort(flat_scores)[::-1][:self.top_k]
            top_k_labels = [str(active_labels[i]) for i in top_k_indices]

            # Vote count for best_identity
            votes = Counter(top_k_labels)
            majority_thresh = len(top_k_labels) / 2.0
            if votes[best_identity] > majority_thresh:
                return best_identity, best_score
            else:
                return "UNKNOWN", best_score

        else:
            raise ValueError(f"Unknown matching mode: {mode}")
