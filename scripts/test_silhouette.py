import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from pipeline.steps.silhouette_step import SilhouetteStep
from pipeline.steps.tracking import TrackingStep
from streaming.stream_engine import StreamEngine


def crop_person(frame, box):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def main() -> None:
    stream = StreamEngine(source=0)
    tracker = TrackingStep()
    silhouette_step = SilhouetteStep()

    if not stream.is_opened():
        print("Camera not found.")
        return

    print("Silhouette test started")
    print("Press Q to quit")

    while True:
        ret, frame = stream.read()

        if not ret:
            break

        detections = tracker.track(frame)

        xyxy = detections.xyxy
        tracker_ids = detections.tracker_id

        if tracker_ids is not None:
            for box, track_id in zip(xyxy, tracker_ids):
                crop = crop_person(frame, box)

                if crop is None:
                    continue

                silhouette = silhouette_step.extract_from_crop(crop)

                if silhouette is None:
                    continue

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

                cv2.imshow(
                    "ARGUS Person Crop",
                    crop,
                )

                cv2.imshow(
                    "ARGUS Silhouette",
                    silhouette,
                )

        cv2.imshow(
            "ARGUS Tracking + Silhouette",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()