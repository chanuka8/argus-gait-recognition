import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from storage.vector_store import VectorStore


def clean_gallery(
    remove_ids: list[str],
    gallery_dir: str,
) -> None:
    store = VectorStore(
        gallery_dir=gallery_dir,
    )

    current = store.load()

    if current is None:
        print(
            f"Gallery not found: {gallery_dir}"
        )
        return

    features, labels, metadata = current

    features = np.asarray(
        features,
    )

    labels = np.asarray(
        labels,
    )

    remove_set = {
        str(person_id)
        for person_id in remove_ids
    }

    keep_mask = np.asarray(
        [
            str(label) not in remove_set
            for label in labels
        ],
        dtype=bool,
    )

    removed_count = int(
        np.sum(
            ~keep_mask,
        )
    )

    for person_id in remove_set:
        metadata.pop(
            person_id,
            None,
        )

    store.save(
        features[keep_mask],
        labels[keep_mask],
        metadata,
    )

    print(
        f"Gallery cleaned: {gallery_dir}"
    )
    print(
        f"Removed identities: {sorted(remove_set)}"
    )
    print(
        f"Embeddings removed: {removed_count}"
    )
    print(
        f"Remaining embeddings: {int(np.sum(keep_mask))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean contaminated identities from ARGUS live gallery"
    )

    parser.add_argument(
        "--gallery-dir",
        default="models/live_gallery",
    )

    parser.add_argument(
        "--person-id",
        action="append",
        required=True,
        help="Identity to remove. Can be used multiple times.",
    )

    args = parser.parse_args()

    clean_gallery(
        remove_ids=args.person_id,
        gallery_dir=args.gallery_dir,
    )


if __name__ == "__main__":
    main()