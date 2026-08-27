"""
Appearance ReID Embedding Extraction and Caching Module.

Extracts 512D L2-normalized feature vectors using OSNet pretrained backbone.
Maintains per-track caching and performance gating to minimize inference overhead.
"""

from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


class AppearanceEmbeddingExtractor:
    """
    Appearance ReID embedding extractor and per-track cache manager.

    Wraps OSNet backbone to generate 512D L2-normalized feature embeddings.
    Caches features per active track ID to avoid repeated CNN inference.
    """

    def __init__(
        self,
        model_path: str = "models/weights/osnet_x0_25.pth",
        device: str = "auto",
        update_interval: int = 8,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.update_interval = max(1, int(update_interval))
        self._logger = get_logger("appearance_embedding")

        self.backbone = None
        self._cache: dict[int, dict[str, Any]] = {}
        self._init_model()

    def _init_model(self) -> None:
        """Initialize OSNet backbone with exception handling fallback."""
        try:
            from pipeline.steps.reid_feature_extraction import ReIDFeatureExtractionStep

            self.backbone = ReIDFeatureExtractionStep(
                model_path=self.model_path,
                device=self.device,
            )
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            self._logger.warning(f"[APPEARANCE] Backbone initialization failed: {exc}. Falling back to gait-only mode.")
            self.backbone = None

    def is_available(self) -> bool:
        """Check if appearance model is initialized and available."""
        return self.backbone is not None

    def extract(
        self,
        crop: np.ndarray | None,
        track_id: int | None = None,
        frame_index: int = 0,
        track_reliable: bool = True,
        recognition_deferred: bool = False,
    ) -> np.ndarray | None:
        """
        Extract 512D L2-normalized appearance embedding for a track crop.

        Uses caching and gating to prevent repeated inference:
        - If track is unreliable or recognition deferred, returns cached embedding if available.
        - If crop is invalid or track was updated recently (< update_interval), returns cached.
        - Otherwise extracts embedding via CNN, normalizes, updates cache, and returns vector.
        """
        if track_id is None:
            return self._extract_raw(crop)

        cached_entry = self._cache.get(track_id)
        cached_embedding = cached_entry["embedding"] if cached_entry else None

        if not track_reliable or recognition_deferred:
            return cached_embedding

        if crop is None or crop.size == 0 or len(crop.shape) != 3:
            return cached_embedding

        if cached_entry is not None:
            last_frame = cached_entry.get("last_updated_frame", 0)
            if frame_index - last_frame < self.update_interval:
                return cached_embedding

        new_embedding = self._extract_raw(crop)
        if new_embedding is not None:
            self._cache[track_id] = {
                "embedding": new_embedding,
                "last_updated_frame": frame_index,
            }
            return new_embedding

        return cached_embedding

    def _extract_raw(self, crop: np.ndarray | None) -> np.ndarray | None:
        """Extract and normalize 512D embedding vector from a BGR crop."""
        if not self.is_available() or crop is None or crop.size == 0:
            return None

        try:
            embedding = self.backbone.extract(crop)
            if embedding is None:
                return None

            vec = np.asarray(embedding, dtype=np.float32).ravel()
            if vec.size == 0:
                return None

            if vec.size != 512:
                if vec.size < 512:
                    padded = np.zeros((512,), dtype=np.float32)
                    padded[: vec.size] = vec
                    vec = padded
                else:
                    vec = vec[:512]

            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            else:
                return None

            return vec.astype(np.float32)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            self._logger.debug(f"[APPEARANCE] Feature extraction error: {exc}")
            return None

    def get_cached(self, track_id: int) -> np.ndarray | None:
        """Retrieve cached embedding for a track ID."""
        entry = self._cache.get(track_id)
        return entry["embedding"] if entry else None

    def clear_track(self, track_id: int) -> None:
        """Evict track entry from cache."""
        self._cache.pop(track_id, None)

    def clear_all(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
