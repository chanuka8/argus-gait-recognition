import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from storage.vector_store import VectorStore


def is_numeric_identity(label: str) -> bool:
    return re.fullmatch(r"\d+", str(label)) is not None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove numeric CASIA-B identities from ARGUS gallery"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without saving changes",
    )

    args = parser.parse_args()

    store = VectorStore()
    current = store.load()

    if current is None:
        print("Gallery not found.")
        return

    features, labels, metadata = current

    features = np.asarray(features)
    labels = np.asarray(labels)

    remove_mask = np.array(
        [
            is_numeric_identity(label)
            for label in labels
        ],
        dtype=bool,
    )

    keep_mask = ~remove_mask

    removed_labels = sorted(
        set(
            str(label)
            for label in labels[remove_mask]
        )
    )

    removed_count = int(
        np.sum(remove_mask)
    )

    print(f"Numeric identities found: {len(removed_labels)}")
    print(f"Embeddings to remove: {removed_count}")

    if removed_labels:
        print("Labels:")
        print(", ".join(removed_labels[:50]))

        if len(removed_labels) > 50:
            print("...")

    if args.dry_run:
        print("Dry run only. Gallery not modified.")
        return

    new_features = features[keep_mask]
    new_labels = labels[keep_mask]

    for label in removed_labels:
        metadata.pop(
            label,
            None,
        )

    store.save(
        new_features,
        new_labels,
        metadata,
    )

    print("Numeric CASIA-B identities removed from gallery.")
    print(f"Remaining embeddings: {len(new_labels)}")


if __name__ == "__main__":
    main()