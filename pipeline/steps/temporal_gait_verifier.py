"""
Temporal Gait Verification Module.

Maintains a rolling buffer of valid GEI embeddings per track.
Performs matching across all buffered valid embeddings and applies majority voting
for robust identity verification.
"""

from collections import Counter
import threading
from typing import Any, Dict, List, Tuple

import numpy as np

from monitoring.logging_config import get_logger


class TemporalGaitVerifier:
    """
    Temporal Gait Verifier.

    Buffers valid GEI embeddings (default window size: 3) per track ID,
    matches each against the gallery, and determines identity via majority voting.
    """

    def __init__(
        self,
        window_size: int = 3,
    ) -> None:
        self.window_size = window_size
        self.logger = get_logger("temporal_gait_verifier")
        self._lock = threading.Lock()

        # track_id -> List[np.ndarray] (up to window_size embeddings)
        self.buffers: Dict[int, List[np.ndarray]] = {}

        # track_id -> str (last verified identity)
        self.last_identities: Dict[int, str] = {}

    def add_embedding(
        self,
        track_id: int,
        embedding: np.ndarray,
    ) -> None:
        """Add a valid GEI embedding to the track's rolling buffer."""
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
    ) -> List[np.ndarray]:
        """Get copy of current valid embedding buffer for a track."""
        with self._lock:
            return list(self.buffers.get(track_id, []))

    def verify_identity(
        self,
        track_id: int,
        matcher_func: Any,
        gallery_features: Any,
        gallery_labels: Any,
        metadata: dict | None = None,
    ) -> Tuple[str, float, str]:
        """
        Perform matching across all valid embeddings in rolling buffer and apply majority voting.

        Args:
            track_id: Track ID
            matcher_func: Callable matcher (e.g., matching_step.match)
            gallery_features: Gallery features matrix
            gallery_labels: Gallery labels array
            metadata: Active subject metadata dict

        Returns:
            Tuple[final_identity, mean_score, decision_type]
            - final_identity: str
            - mean_score: float
            - decision_type: "MAJORITY_VOTE" | "SINGLE_MATCH" | "UNCERTAIN" | "PREVIOUS_IDENTITY"
        """
        embeddings = self.get_buffer(track_id)

        if not embeddings:
            return "UNKNOWN", 0.0, "UNCERTAIN"

        match_results: List[Tuple[str, float]] = []

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

        # Check for majority (at least 2 out of 3, or > 50% of buffer)
        required_votes = (len(embeddings) // 2) + 1

        with self._lock:
            prev_identity = self.last_identities.get(track_id, "UNKNOWN")

            if most_common_count >= required_votes and most_common_id != "UNKNOWN":
                final_identity = most_common_id
                decision = "MAJORITY_VOTE" if len(embeddings) > 1 else "SINGLE_MATCH"
                self.last_identities[track_id] = final_identity
            elif prev_identity != "UNKNOWN":
                # If no majority, fallback to previous verified identity
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
        """Map temporal verification outcome to 3-state open-set classification (KNOWN, UNKNOWN, UNCERTAIN)."""
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
        """Clear rolling buffer and state when a track terminates."""
        with self._lock:
            self.buffers.pop(track_id, None)
            self.last_identities.pop(track_id, None)
            self.logger.debug(f"Cleared temporal buffer for track {track_id}")

    def clear_all(self) -> None:
        """Clear all buffers."""
        with self._lock:
            self.buffers.clear()
            self.last_identities.clear()
