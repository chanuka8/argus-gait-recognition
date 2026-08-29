import time

from storage.vector_store import VectorStore


class GalleryUpdater:
    def __init__(
        self,
        gallery_dir: str = "models/live_gallery",
    ):
        self.store = VectorStore(
            gallery_dir=gallery_dir,
        )

    def _metadata_entry(
        self,
        current_value,
        embeddings_added: int,
    ) -> dict:
        if isinstance(current_value, dict):
            previous_embeddings = int(
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

            enabled = bool(
                current_value.get(
                    "enabled",
                    status == "ACTIVE",
                )
            )

            if status != "ACTIVE":
                enabled = False

            return {
                "embeddings": previous_embeddings + embeddings_added,
                "status": status,
                "enabled": enabled,
                "updated_at": time.time(),
            }

        if isinstance(current_value, int):
            return {
                "embeddings": current_value + embeddings_added,
                "status": "ACTIVE",
                "enabled": True,
                "updated_at": time.time(),
            }

        return {
            "embeddings": embeddings_added,
            "status": "ACTIVE",
            "enabled": True,
            "updated_at": time.time(),
        }

    def add_person(
        self,
        person_id: str,
        embeddings: list,
    ) -> None:
        """
        Add one or more 256D gait embeddings for a subject ID to the gait gallery.

        Raises:
            ValueError: If embeddings list is empty or any embedding is not 256-dimensional.
        """
        if not person_id or not str(person_id).strip():
            raise ValueError("person_id cannot be empty")

        if not embeddings or len(embeddings) == 0:
            raise ValueError(f"No embeddings provided for person {person_id}")

        import numpy as np

        validated_embeddings = []
        for i, emb in enumerate(embeddings):
            vec = np.asarray(emb, dtype=np.float32).ravel()
            if vec.size != 256:
                raise ValueError(
                    f"Gait embeddings must be 256-dimensional (got shape {vec.shape}, size {vec.size} at index {i}). "
                    f"Appearance vectors (512D) cannot be inserted into the gait gallery."
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
            metadata.get(str(person_id)),
            len(validated_embeddings),
        )

        self.store.save(
            features,
            labels,
            metadata,
        )

        entry = metadata[str(person_id)]

        print(
            f"Added {person_id} "
            f"({len(validated_embeddings)} embeddings) | "
            f"status={entry.get('status')} | "
            f"enabled={entry.get('enabled')}"
        )
