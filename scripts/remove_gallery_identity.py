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


def remove_identity(
    person_id: str,
    gallery_dir: str,
    verbose: bool = True,
) -> bool:
    store = VectorStore(
        gallery_dir=gallery_dir,
    )

    current = store.load()

    if current is None:
        if verbose:
            print(f"Gallery not found: {gallery_dir}")
        return False

    features, labels, metadata = current

    features = np.asarray(
        features,
    )

    labels = np.asarray(
        labels,
    )

    keep_mask = labels != person_id

    removed = int(
        np.sum(~keep_mask),
    )

    if removed == 0:
        return False

    new_features = features[keep_mask]
    new_labels = labels[keep_mask]

    metadata.pop(
        person_id,
        None,
    )

    store.save(
        new_features,
        new_labels,
        metadata,
    )

    print(f"Removed identity: {person_id} from {gallery_dir}")
    print(f"Embeddings removed: {removed}")
    print(f"Remaining embeddings: {len(new_labels)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove an identity from ARGUS gallery"
    )

    parser.add_argument(
        "--person-id",
        required=True,
    )

    parser.add_argument(
        "--gallery-dir",
        default="models/live_gallery",
    )

    args = parser.parse_args()

    if args.gallery_dir == "models/live_gallery":
        removed_live = remove_identity(
            person_id=args.person_id,
            gallery_dir="models/live_gallery",
            verbose=False,
        )
        removed_app = remove_identity(
            person_id=args.person_id,
            gallery_dir="models/appearance_gallery",
            verbose=False,
        )
        if not removed_live and not removed_app:
            print(f"No gallery entries found for {args.person_id} in live or appearance galleries.")
    else:
        remove_identity(
            person_id=args.person_id,
            gallery_dir=args.gallery_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()