import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from pipeline.steps.tracking import TrackingStep
from streaming.stream_engine import StreamEngine


def main() -> None:
    stream = StreamEngine(source=0)
    tracker = TrackingStep()

    if not stream.is_opened():
        print("Camera not found.")
        return

    print("Tracking started")
    print("Press Q to quit")

    while True:
        ret, frame = stream.read()

        if not ret:
            print("Failed to read frame.")
            break

        detections = tracker.track(frame)

        xyxy = detections.xyxy
        tracker_ids = detections.tracker_id

        if tracker_ids is not None:
            for box, track_id in zip(xyxy, tracker_ids):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"ID {int(track_id)}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("ARGUS Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()