from typing import Any

import numpy as np

from intelligence.track_identity_aggregator import TrackIdentityAggregator


class TemporalTrackEvaluator:
    """
    Evaluator for temporal identity aggregation across simulated and real video tracks.
    """

    def __init__(
        self,
        window_size: int = 8,
        consensus_threshold: float = 0.60,
        confirm_threshold: float = 0.72,
        high_risk_confusion_groups: list[list[str]] | None = None,
    ) -> None:
        self.window_size = window_size
        self.consensus_threshold = consensus_threshold
        self.confirm_threshold = confirm_threshold
        self.high_risk_confusion_groups = high_risk_confusion_groups or [["Devhan", "Isuru", "person01"]]

    def evaluate_synthetic_tracks(
        self,
        known_subjects: list[str],
        num_tracks: int = 200,
        track_length: int = 12,
        noise_level: float = 0.08,
        seed: int = 42,
    ) -> dict[str, Any]:
        np.random.seed(seed)
        confirmed_correct = 0
        confirmed_wrong = 0
        review_required = 0
        unknown_tracks = 0

        for t_idx in range(num_tracks):
            s_true = known_subjects[t_idx % len(known_subjects)]
            agg = TrackIdentityAggregator(
                window_size=self.window_size,
                consensus_threshold=self.consensus_threshold,
                confirm_threshold=self.confirm_threshold,
                high_risk_confusion_groups=self.high_risk_confusion_groups,
            )

            final_res = None
            for _ in range(track_length):
                is_correct = np.random.rand() > noise_level
                cand = s_true if is_correct else "UNKNOWN"
                score = np.random.uniform(0.74, 0.90) if is_correct else np.random.uniform(0.35, 0.60)
                final_res = agg.update(track_id=t_idx, identity=cand, score=score)

            decision = final_res["decision"]
            pred_id = final_res["identity"]

            if decision == "CONFIRMED":
                if pred_id == s_true:
                    confirmed_correct += 1
                else:
                    confirmed_wrong += 1
            elif decision == "REVIEW_REQUIRED":
                review_required += 1
            else:
                unknown_tracks += 1

        return {
            "num_tracks": num_tracks,
            "confirmed_correct": confirmed_correct,
            "confirmed_wrong": confirmed_wrong,
            "review_required": review_required,
            "unknown_tracks": unknown_tracks,
            "track_tar": round(confirmed_correct / num_tracks * 100, 2),
            "track_far": round(confirmed_wrong / num_tracks * 100, 2),
            "review_rate": round(review_required / num_tracks * 100, 2),
        }
