import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from pipeline.live_recognition import LiveRecognitionPipeline


def main() -> None:
    pipeline = LiveRecognitionPipeline(
        threshold=0.85,
        alert_threshold=0.90,
        security_threshold=0.90,
    )

    pipeline.run()


if __name__ == "__main__":
    main()