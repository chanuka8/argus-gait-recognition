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
        current = self.store.load()

        if current is None:
            features = []
            labels = []
            metadata = {}

        else:
            features, labels, metadata = current

            features = features.tolist()
            labels = labels.tolist()

        for embedding in embeddings:
            features.append(
                embedding,
            )

            labels.append(
                person_id,
            )

        metadata[person_id] = self._metadata_entry(
            metadata.get(person_id),
            len(embeddings),
        )

        self.store.save(
            features,
            labels,
            metadata,
        )

        entry = metadata[person_id]

        print(
            f"Added {person_id} "
            f"({len(embeddings)} embeddings) | "
            f"status={entry.get('status')} | "
            f"enabled={entry.get('enabled')}"
        )