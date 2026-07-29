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
        if not self.features_file.exists() or not self.labels_file.exists():
            return None

        try:
            features = np.load(
                self.features_file,
                allow_pickle=False,
            )
        except Exception as err:
            err_msg = str(err)
            if "pickle" in err_msg.lower() or "object arrays" in err_msg.lower():
                raise ValueError(
                    f"Gallery features file '{self.features_file}' requires pickle deserialization, which is prohibited."
                ) from err
            raise ValueError(f"Failed to load gallery features file '{self.features_file}': {err}") from err

        try:
            labels = np.load(
                self.labels_file,
                allow_pickle=False,
            )
        except Exception as err:
            err_msg = str(err)
            if "pickle" in err_msg.lower() or "object arrays" in err_msg.lower():
                raise ValueError(
                    f"Gallery labels file '{self.labels_file}' requires pickle deserialization, which is prohibited."
                ) from err
            raise ValueError(f"Failed to load gallery labels file '{self.labels_file}': {err}") from err

        # Security & Validation Checks
        if features.dtype == object or features.dtype.kind == "O":
            raise ValueError(
                f"Gallery features array in '{self.features_file}' has invalid object dtype ({features.dtype}). Only numeric dtypes are allowed."
            )

        if labels.dtype == object or labels.dtype.kind == "O":
            raise ValueError(
                f"Gallery labels array in '{self.labels_file}' has invalid object dtype ({labels.dtype}). Only string or numeric dtypes are allowed."
            )

        if not np.issubdtype(features.dtype, np.number):
            raise ValueError(f"Gallery features array must be numeric, got {features.dtype}.")

        if features.ndim != 2:
            raise ValueError(
                f"Gallery features array must be 2-dimensional (N, D), got shape {features.shape}."
            )

        if labels.ndim != 1:
            raise ValueError(
                f"Gallery labels array must be 1-dimensional (N,), got shape {labels.shape}."
            )

        if len(features) != len(labels):
            raise ValueError(
                f"Mismatch between features length ({len(features)}) and labels length ({len(labels)})."
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
