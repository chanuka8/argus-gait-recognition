import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.dataset_builder import CasiaGEIDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GEI images from CASIA-B ZIP dataset"
    )

    parser.add_argument(
        "--zip",
        default="data/casia_b_raw.zip",
        help="Path to CASIA-B ZIP file",
    )

    parser.add_argument(
        "--output",
        default="data/casia_processed/gei",
        help="Output directory for generated GEI images",
    )

    parser.add_argument(
        "--min-frames",
        type=int,
        default=15,
        help="Minimum frames required to build a GEI",
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Limit number of sequences for testing",
    )

    args = parser.parse_args()

    builder = CasiaGEIDatasetBuilder(
        zip_path=args.zip,
        output_dir=args.output,
        min_frames=args.min_frames,
        max_sequences=args.max_sequences,
    )

    summary = builder.build()

    print("\nCASIA GEI preprocessing completed.")
    print(summary)


if __name__ == "__main__":
    main()