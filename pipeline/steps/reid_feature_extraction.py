import numpy as np

from models.reid.osnet_backbone import OSNetBackbone


class ReIDFeatureExtractionStep:
    """
    Person re-identification feature extraction
    using OSNet-x0.25 backbone.

    Secondary biometric module.
    Does not replace gait feature extraction.
    """

    def __init__(
        self,
        model_path: str = "models/weights/osnet_x0_25.pth",
        device: str = "auto",
    ) -> None:
        self.backbone = OSNetBackbone(
            model_path=model_path,
            device=device,
        )

    def extract(
        self,
        crop: np.ndarray,
    ) -> np.ndarray | None:
        """
        Extract normalized ReID embedding
        from a BGR person crop.

        Returns None if crop is invalid.
        """

        if crop is None:
            return None

        if crop.size == 0:
            return None

        if len(crop.shape) != 3:
            return None

        return self.backbone.extract(crop)

    def extract_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[np.ndarray | None]:
        """
        Extract ReID embeddings from
        a batch of BGR person crops.
        """

        valid = []
        valid_indices = []

        for i, crop in enumerate(crops):
            if (
                crop is not None
                and crop.size > 0
                and len(crop.shape) == 3
            ):
                valid.append(crop)
                valid_indices.append(i)

        if not valid:
            return [None] * len(crops)

        embeddings = self.backbone.extract_batch(
            valid,
        )

        results: list[np.ndarray | None] = (
            [None] * len(crops)
        )

        for idx, emb in zip(
            valid_indices,
            embeddings,
        ):
            results[idx] = emb

        return results
