import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.folder_recognition import FolderRecognitionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARGUS folder-based GEI recognition"
    )

    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing GEI images",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum cosine similarity score required for identity acceptance",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path",
    )

    args = parser.parse_args()

    recognizer = FolderRecognitionPipeline()

    recognizer.run(
        folder_path=args.folder,
        threshold=args.threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()