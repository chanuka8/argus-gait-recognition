import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from pipeline.steps.detection import DetectionStep
from streaming.stream_engine import StreamEngine


def main() -> None:
    stream = StreamEngine(source=0)
    detector = DetectionStep()

    if not stream.is_opened():
        print("Camera not found.")
        return

    print("Webcam detection started.")
    print("Press Q to quit.")

    while True:
        ret, frame = stream.read()

        if not ret:
            print("Failed to read frame.")
            break

        results = detector.detect(frame)

        annotated = results[0].plot()

        cv2.imshow("ARGUS Webcam Person Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()