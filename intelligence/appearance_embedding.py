from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


class AppearanceEmbeddingExtractor:
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
        return self.backbone is not None

    def extract(
        self,
        crop: np.ndarray | None,
        track_id: int | None = None,
        frame_index: int = 0,
        track_reliable: bool = True,
        recognition_deferred: bool = False,
    ) -> np.ndarray | None:
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
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
            self._logger.debug(f"[APPEARANCE] Feature extraction error: {exc}")
            return None

    def extract_batch(
        self,
        crops: list[np.ndarray | None],
        track_ids: list[int] | None = None,
        frame_index: int = 0,
    ) -> list[np.ndarray | None]:
        if not self.is_available() or not crops:
            return [None] * len(crops)

        valid_crops = []
        valid_indices = []
        for i, crop in enumerate(crops):
            if crop is not None and getattr(crop, "size", 0) > 0 and len(crop.shape) == 3:
                valid_crops.append(crop)
                valid_indices.append(i)

        results: list[np.ndarray | None] = [None] * len(crops)
        if not valid_crops:
            return results

        try:
            if hasattr(self.backbone, "extract_batch"):
                raw_embeddings = self.backbone.extract_batch(valid_crops)
            else:
                raw_embeddings = [self.backbone.extract(c) for c in valid_crops]

            for idx, raw_emb in zip(valid_indices, raw_embeddings):
                if raw_emb is None:
                    continue
                vec = np.asarray(raw_emb, dtype=np.float32).ravel()
                if vec.size == 0:
                    continue
                if vec.size != 512:
                    if vec.size < 512:
                        padded = np.zeros((512,), dtype=np.float32)
                        padded[: vec.size] = vec
                        vec = padded
                    else:
                        vec = vec[:512]
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8:
                    normalized = (vec / norm).astype(np.float32)
                    results[idx] = normalized
                    if track_ids is not None and idx < len(track_ids):
                        tid = track_ids[idx]
                        self._cache[tid] = {
                            "embedding": normalized,
                            "last_updated_frame": frame_index,
                        }
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
            self._logger.debug(f"[APPEARANCE] Batch feature extraction error: {exc}")

        return results

    def get_cached(self, track_id: int) -> np.ndarray | None:
        entry = self._cache.get(track_id)
        return entry["embedding"] if entry else None

    def clear_track(self, track_id: int) -> None:
        self._cache.pop(track_id, None)

    def clear_all(self) -> None:
        self._cache.clear()
