import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.video_recognition import VideoRecognitionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARGUS video-file gait recognition"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to walking video file",
    )

    parser.add_argument(
        "--model",
        default="runs/exp_001/best_model.pth",
        help="Path to trained ByGaitLight model checkpoint",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum cosine similarity score required for identity acceptance",
    )

    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.80,
        help="Confidence threshold for alert manager",
    )

    parser.add_argument(
        "--security-threshold",
        type=float,
        default=0.80,
        help="Confidence threshold for security decision engine",
    )

    parser.add_argument(
        "--gei-frames",
        type=int,
        default=15,
        help="Number of silhouettes required to build one live GEI",
    )

    parser.add_argument(
        "--recognition-interval",
        type=int,
        default=10,
        help="Frame interval between repeated recognition attempts per track",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output report path",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated video while processing",
    )

    args = parser.parse_args()

    recognizer = VideoRecognitionPipeline(
        model_path=args.model,
        threshold=args.threshold,
        alert_threshold=args.alert_threshold,
        security_threshold=args.security_threshold,
        gei_frames=args.gei_frames,
        recognition_interval=args.recognition_interval,
    )

    recognizer.run(
        video_path=args.video,
        show=args.show,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()