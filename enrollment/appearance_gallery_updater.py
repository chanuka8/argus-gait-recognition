import time

import numpy as np

from storage.vector_store import VectorStore


class AppearanceGalleryUpdater:
    def __init__(
        self,
        gallery_dir: str = "models/appearance_gallery",
    ) -> None:
        self.gallery_dir = gallery_dir
        self.store = VectorStore(
            gallery_dir=gallery_dir,
        )

    def _metadata_entry(
        self,
        current_value,
        embeddings_added: int,
    ) -> dict:
        if isinstance(
            current_value,
            dict,
        ):
            previous = int(
                current_value.get(
                    "embeddings",
                    0,
                )
            )

            status = str(
                current_value.get(
                    "status",
                    "ACTIVE",
                )
            ).upper()

            return {
                "embeddings": previous + embeddings_added,
                "status": status,
                "enabled": status == "ACTIVE",
                "source": "PHOTO",
                "updated_at": time.time(),
            }

        return {
            "embeddings": embeddings_added,
            "status": "ACTIVE",
            "enabled": True,
            "source": "PHOTO",
            "updated_at": time.time(),
        }

    def add_person(
        self,
        person_id: str,
        embeddings: list,
    ) -> None:
        if not person_id or not str(person_id).strip():
            raise ValueError("person_id cannot be empty")

        if not embeddings or len(embeddings) == 0:
            raise ValueError(f"No embeddings provided for person {person_id}")

        validated_embeddings = []
        for i, emb in enumerate(embeddings):
            vec = np.asarray(emb, dtype=np.float32).ravel()
            if vec.size != 512:
                raise ValueError(
                    f"Appearance embeddings must be 512-dimensional (got shape {vec.shape}, size {vec.size} at index {i}). "
                    f"Gait vectors (256D) cannot be inserted into the appearance gallery."
                )

            norm = float(np.linalg.norm(vec))
            if norm == 0.0 or not np.isfinite(norm):
                raise ValueError(f"Invalid zero or non-finite norm vector at index {i}")

            vec = (vec / norm).astype(np.float32)
            validated_embeddings.append(vec)

        current = self.store.load()

        if current is None:
            features = []
            labels = []
            metadata = {}
        else:
            features, labels, metadata = current
            features = features.tolist()
            labels = labels.tolist()

        for embedding in validated_embeddings:
            features.append(
                embedding.tolist(),
            )
            labels.append(
                str(person_id),
            )

        metadata[str(person_id)] = self._metadata_entry(
            metadata.get(
                str(person_id),
            ),
            len(validated_embeddings),
        )

        self.store.save(
            features,
            labels,
            metadata,
        )

        print(f"Added appearance identity {person_id} ({len(validated_embeddings)} embeddings)")
