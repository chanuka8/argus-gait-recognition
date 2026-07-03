import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from storage.vector_store import VectorStore


VALID_STATUSES = {
    "ACTIVE",
    "DISABLED",
    "ARCHIVED",
}


def set_status(
    person_id: str,
    status: str,
    gallery_dir: str,
    verbose: bool = True,
) -> bool:
    status = status.upper()

    if status not in VALID_STATUSES:
        if verbose:
            print(
                f"Invalid status: {status}"
            )
            print(
                "Valid statuses: ACTIVE, DISABLED, ARCHIVED"
            )
        return False

    store = VectorStore(
        gallery_dir=gallery_dir,
    )

    current = store.load()

    if current is None:
        if verbose:
            print(
                f"Gallery not found: {gallery_dir}"
            )
        return False

    features, labels, metadata = current

    if person_id not in metadata:
        return False

    entry = metadata[person_id]

    if not isinstance(entry, dict):
        entry = {
            "embeddings": int(entry),
        }

    entry["status"] = status
    entry["enabled"] = status == "ACTIVE"
    entry["updated_at"] = time.time()

    metadata[person_id] = entry

    store.save(
        features,
        labels,
        metadata,
    )

    print(
        f"Identity status updated: {person_id} -> {status} ({gallery_dir})"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set ARGUS gallery identity status"
    )

    parser.add_argument(
        "--person-id",
        required=True,
    )

    parser.add_argument(
        "--status",
        required=True,
        choices=[
            "ACTIVE",
            "DISABLED",
            "ARCHIVED",
            "active",
            "disabled",
            "archived",
        ],
    )

    parser.add_argument(
        "--gallery-dir",
        default="models/live_gallery",
    )

    args = parser.parse_args()

    if args.gallery_dir == "models/live_gallery":
        updated_live = set_status(
            person_id=args.person_id,
            status=args.status,
            gallery_dir="models/live_gallery",
            verbose=False,
        )
        updated_app = set_status(
            person_id=args.person_id,
            status=args.status,
            gallery_dir="models/appearance_gallery",
            verbose=False,
        )
        if not updated_live and not updated_app:
            print(
                f"Identity not found in live or appearance galleries: {args.person_id}"
            )
    else:
        set_status(
            person_id=args.person_id,
            status=args.status,
            gallery_dir=args.gallery_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()