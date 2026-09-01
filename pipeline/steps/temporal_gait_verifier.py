import threading
from collections import Counter
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


class TemporalGaitVerifier:
    def __init__(
        self,
        window_size: int = 3,
    ) -> None:
        self.window_size = window_size
        self.logger = get_logger("temporal_gait_verifier")
        self._lock = threading.Lock()

        self.buffers: dict[int, list[np.ndarray]] = {}

        self.last_identities: dict[int, str] = {}

    def add_embedding(
        self,
        track_id: int,
        embedding: np.ndarray,
    ) -> None:
        if embedding is None:
            return

        with self._lock:
            if track_id not in self.buffers:
                self.buffers[track_id] = []

            buf = self.buffers[track_id]
            buf.append(embedding)

            if len(buf) > self.window_size:
                buf.pop(0)

    def get_buffer(
        self,
        track_id: int,
    ) -> list[np.ndarray]:
        with self._lock:
            return list(self.buffers.get(track_id, []))

    def verify_identity(
        self,
        track_id: int,
        matcher_func: Any,
        gallery_features: Any,
        gallery_labels: Any,
        metadata: dict | None = None,
    ) -> tuple[str, float, str]:
        embeddings = self.get_buffer(track_id)

        if not embeddings:
            return "UNKNOWN", 0.0, "UNCERTAIN"

        match_results: list[tuple[str, float]] = []

        for emb in embeddings:
            matches = matcher_func(
                emb,
                gallery_features,
                gallery_labels,
                metadata,
            )
            if matches and len(matches) > 0:
                top_id, top_score = matches[0][0], float(matches[0][1])
                match_results.append((top_id, top_score))
            else:
                match_results.append(("UNKNOWN", 0.0))

        if not match_results:
            return "UNKNOWN", 0.0, "UNCERTAIN"

        identities = [res[0] for res in match_results]
        scores = [res[1] for res in match_results]

        counts = Counter(identities)
        most_common_id, most_common_count = counts.most_common(1)[0]

        mean_score = float(np.mean(scores))

        required_votes = (len(embeddings) // 2) + 1

        with self._lock:
            prev_identity = self.last_identities.get(track_id, "UNKNOWN")

            if most_common_count >= required_votes and most_common_id != "UNKNOWN":
                final_identity = most_common_id
                decision = "MAJORITY_VOTE" if len(embeddings) > 1 else "SINGLE_MATCH"
                self.last_identities[track_id] = final_identity
            elif prev_identity != "UNKNOWN":
                final_identity = prev_identity
                decision = "PREVIOUS_IDENTITY"
                self.logger.info(
                    f"Track {track_id}: No majority in temporal buffer {identities}. Keeping previous identity {prev_identity}"
                )
            else:
                final_identity = "UNKNOWN"
                decision = "UNCERTAIN"
                self.logger.info(f"Track {track_id}: No majority in temporal buffer {identities}. Marked UNCERTAIN.")

        return final_identity, mean_score, decision

    def get_open_set_state(
        self,
        final_identity: str,
        score: float,
        decision_type: str,
        unknown_ceiling: float = 0.70,
    ) -> str:
        if final_identity != "UNKNOWN" and decision_type in ("MAJORITY_VOTE", "SINGLE_MATCH"):
            return "KNOWN"
        elif final_identity == "UNKNOWN" and score < unknown_ceiling:
            return "UNKNOWN"
        else:
            return "UNCERTAIN"

    def clear_track(
        self,
        track_id: int,
    ) -> None:
        with self._lock:
            self.buffers.pop(track_id, None)
            self.last_identities.pop(track_id, None)
            self.logger.debug(f"Cleared temporal buffer for track {track_id}")

    def clear_all(self) -> None:
        with self._lock:
            self.buffers.clear()
            self.last_identities.clear()
