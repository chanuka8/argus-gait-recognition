import threading
import numpy as np


def compute_iou(box1, box2) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Bounding box format: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = float((x_right - x_left) * (y_bottom - y_top))

    area1 = float((x2_1 - x1_1) * (y2_1 - y1_1))
    area2 = float((x2_2 - x1_2) * (y2_2 - y1_2))

    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


class BoxStabilizer:
    """
    Stabilizes bounding boxes across frames using:
    - EMA (Exponential Moving Average) smoothing.
    - IoU verification to reject sudden tracking glitches/jumps.
    - Speed/dimension jump checks to detect rapid tracking shifts.
    - Temporary box extrapolation for missed detection frames.
    - Thread-safe state tracking.
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.alpha = config.get("ema_alpha", 0.35)
        self.min_confidence = config.get("min_detection_confidence", 0.35)
        self.min_iou = config.get("min_iou_keep", 0.25)
        self.max_missed = config.get("max_missed_frames", 8)
        self.max_jump = config.get("max_jump_ratio", 0.35)
        self.prune_after = config.get("prune_after_frames", 30)

        self.tracks = {}  # track_id -> dict
        self.current_frame = 0
        self._lock = threading.Lock()

    def update(self, raw_detections: list, frame_shape: tuple) -> dict:
        """
        raw_detections: list of tuples (track_id, box_xyxy, confidence)
        frame_shape: tuple of (height, width)

        Returns:
            dict: track_id -> (stable_box_xyxy, is_valid, is_predicted)
        """
        if not self.enabled:
            return {
                tid: (np.array(box, dtype=np.float32), True, False)
                for tid, box, _ in raw_detections
            }

        with self._lock:
            self.current_frame += 1
            h, w = frame_shape[:2]
            frame_area = h * w

            detected_track_ids = set()

            # Step 1: Process current raw detections
            for track_id, box, conf in raw_detections:
                if conf < self.min_confidence:
                    continue

                track_id = int(track_id)
                raw_box = np.array(box, dtype=np.float32)

                box_w = raw_box[2] - raw_box[0]
                box_h = raw_box[3] - raw_box[1]
                box_area = box_w * box_h
                area_ratio = box_area / frame_area if frame_area > 0 else 0

                # Reject extremely small boxes or invalid coordinates
                if box_h < 10 or box_w < 10:
                    continue

                detected_track_ids.add(track_id)

                if track_id not in self.tracks:
                    # New track initialization
                    self.tracks[track_id] = {
                        "stable_box": raw_box,
                        "missed_frames": 0,
                        "last_seen_frame": self.current_frame,
                        "is_predicted": False,
                    }
                else:
                    # Existing track: compute updates with jump verification
                    prev_state = self.tracks[track_id]
                    prev_box = prev_state["stable_box"]

                    iou = compute_iou(raw_box, prev_box)

                    # Compute centroids and dimensions
                    prev_cx = (prev_box[0] + prev_box[2]) / 2.0
                    prev_cy = (prev_box[1] + prev_box[3]) / 2.0
                    prev_w = prev_box[2] - prev_box[0]
                    prev_h = prev_box[3] - prev_box[1]

                    curr_cx = (raw_box[0] + raw_box[2]) / 2.0
                    curr_cy = (raw_box[1] + raw_box[3]) / 2.0
                    curr_w = raw_box[2] - raw_box[0]
                    curr_h = raw_box[3] - raw_box[1]

                    cx_jump = abs(curr_cx - prev_cx) / prev_w if prev_w > 0 else 0.0
                    cy_jump = abs(curr_cy - prev_cy) / prev_h if prev_h > 0 else 0.0
                    w_jump = abs(curr_w - prev_w) / prev_w if prev_w > 0 else 0.0
                    h_jump = abs(curr_h - prev_h) / prev_h if prev_h > 0 else 0.0

                    is_jump = (
                        iou < self.min_iou
                        or cx_jump > self.max_jump
                        or cy_jump > self.max_jump
                        or w_jump > self.max_jump
                        or h_jump > self.max_jump
                    )

                    if is_jump:
                        # Sudden jump: retain last stable box coordinates and mark predicted
                        prev_state["missed_frames"] += 1
                        prev_state["is_predicted"] = True
                    else:
                        # Apply Exponential Moving Average (EMA) smoothing
                        prev_state["stable_box"] = (
                            self.alpha * raw_box + (1.0 - self.alpha) * prev_box
                        )
                        prev_state["missed_frames"] = 0
                        prev_state["last_seen_frame"] = self.current_frame
                        prev_state["is_predicted"] = False

            # Step 2: Handle temporarily missed tracks (not in current detections)
            for track_id, state in self.tracks.items():
                if track_id not in detected_track_ids:
                    state["missed_frames"] += 1
                    state["is_predicted"] = True

            # Step 3: Prune stale tracks to release memory
            to_prune = []
            for track_id, state in self.tracks.items():
                if self.current_frame - state["last_seen_frame"] > self.prune_after:
                    to_prune.append(track_id)

            for track_id in to_prune:
                self.tracks.pop(track_id, None)

            # Step 4: Map final results
            results = {}
            for track_id, state in self.tracks.items():
                is_valid = state["missed_frames"] <= self.max_missed
                results[track_id] = (
                    state["stable_box"],
                    is_valid,
                    state["is_predicted"],
                )

            return results
