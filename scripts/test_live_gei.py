import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from streaming.stream_engine import StreamEngine
from pipeline.steps.tracking import TrackingStep
from pipeline.steps.silhouette_step import SilhouetteStep
from pipeline.steps.live_gei import LiveGEI


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


def main():

    stream = StreamEngine()

    tracker = TrackingStep()

    silhouette_step = SilhouetteStep()

    gei_buffer = LiveGEI(
        max_frames=15
    )

    print("Live GEI started")
    print("Press Q to quit")

    while True:

        ret, frame = stream.read()

        if not ret:
            break

        detections = tracker.track(
            frame
        )

        xyxy = detections.xyxy
        ids = detections.tracker_id

        if ids is not None:

            for box, track_id in zip(
                xyxy,
                ids,
            ):

                crop = crop_person(
                    frame,
                    box,
                )

                if crop is None:
                    continue

                silhouette = (
                    silhouette_step.extract_from_crop(
                        crop
                    )
                )

                if silhouette is None:
                    continue

                gei_buffer.add(
                    silhouette
                )

                count = (
                    gei_buffer.count()
                )

                x1, y1, x2, y2 = map(
                    int,
                    box,
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"ID {int(track_id)} | GEI:{count}/15",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(
                    "Silhouette",
                    silhouette,
                )

        if gei_buffer.ready():

            gei = gei_buffer.build()

            cv2.imshow(
                "Live GEI",
                gei,
            )

        cv2.imshow(
            "ARGUS Live GEI",
            frame,
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):
            break

    stream.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()