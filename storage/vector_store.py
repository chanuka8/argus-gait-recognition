import json
import time
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(
        self,
        gallery_dir: str = "models/gallery",
    ) -> None:
        self.gallery_dir = Path(gallery_dir)

        self.gallery_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.features_file = self.gallery_dir / "gallery_features.npy"
        self.labels_file = self.gallery_dir / "gallery_labels.npy"
        self.metadata_file = self.gallery_dir / "gallery_metadata.json"

    def _normalize_metadata_entry(
        self,
        value,
    ) -> dict:
        if isinstance(value, dict):
            status = str(
                value.get(
                    "status",
                    "ACTIVE" if value.get("enabled", True) else "DISABLED",
                )
            ).upper()

            return {
                "embeddings": int(value.get("embeddings", 0)),
                "status": status,
                "enabled": status == "ACTIVE",
                "updated_at": float(value.get("updated_at", time.time())),
            }

        if isinstance(value, int):
            return {
                "embeddings": value,
                "status": "ACTIVE",
                "enabled": True,
                "updated_at": time.time(),
            }

        return {
            "embeddings": 0,
            "status": "ACTIVE",
            "enabled": True,
            "updated_at": time.time(),
        }

    def _normalize_metadata(
        self,
        metadata: dict,
    ) -> dict:
        return {
            str(person_id): self._normalize_metadata_entry(value)
            for person_id, value in metadata.items()
        }

    def save(
        self,
        features,
        labels,
        metadata,
    ) -> None:
        metadata = self._normalize_metadata(
            metadata or {},
        )

        np.save(
            self.features_file,
            np.asarray(features, dtype=np.float32),
        )

        np.save(
            self.labels_file,
            np.asarray(labels),
        )

        with open(
            self.metadata_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
            )

    def load(self):
        if not self.features_file.exists():
            return None

        features = np.load(
            self.features_file,
            allow_pickle=True,
        )

        labels = np.load(
            self.labels_file,
            allow_pickle=True,
        )

        if self.metadata_file.exists():
            with open(
                self.metadata_file,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)
        else:
            metadata = {}

        metadata = self._normalize_metadata(
            metadata,
        )

        return (
            features,
            labels,
            metadata,
        )