import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from enrollment.auto_enrollment_service import AutoEnrollmentService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARGUS auto enrollment service"
    )

    parser.add_argument(
        "--input",
        default="data/new_input",
        help="Folder containing person folders for auto enrollment",
    )

    parser.add_argument(
        "--processed",
        default="data/auto_enrollment/gei",
        help="Folder used to store generated enrollment GEI images",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch for new enrollment data",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-enrollment even if marker files already exist",
    )

    parser.add_argument(
        "--scan-interval",
        type=int,
        default=5,
        help="Seconds between folder scans in watch mode",
    )

    parser.add_argument(
        "--gei-frames",
        type=int,
        default=15,
        help="Number of silhouette frames required to build one GEI",
    )

    parser.add_argument(
        "--video-stride",
        type=int,
        default=10,
        help="Frame interval used when saving GEIs from video",
    )

    args = parser.parse_args()

    service = AutoEnrollmentService(
        input_dir=args.input,
        processed_dir=args.processed,
        gei_frames=args.gei_frames,
        video_stride=args.video_stride,
        scan_interval=args.scan_interval,
    )

    if args.watch:
        if args.force:
            service.enroll_pending(
                force=True,
            )

        service.watch()
        return

    results = service.enroll_pending(
        force=args.force,
    )

    print("\n=== AUTO ENROLLMENT SUMMARY ===")

    if not results:
        print("No new enrollment data found.")
        return

    for result in results:
        print(
            f"  {result.get('person_id')} | "
            f"success={result.get('success')} | "
            f"gait_embeddings={result.get('gait_embeddings_added', 0)} | "
            f"{result.get('message')}"
        )


if __name__ == "__main__":
    main()