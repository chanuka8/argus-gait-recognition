"""
Track-Level Temporal Confidence Accumulation Layer.

Maintains a bounded sliding window of per-frame dual-modal decision outputs (candidate
identity, fused score, modality state) per active track ID. Promotes a track to CONFIRMED
only when bounded window consensus (voting >= M%) and average confidence (score >= 0.72)
are met. Produces REVIEW_REQUIRED near-miss alerts and LOW_CONFIDENCE quiet logs for
unconfirmed tracks.
"""

import threading
import time
from collections import Counter, deque
from typing import Any

from monitoring.logging_config import get_logger


class TrackIdentityAggregator:
    """
    Temporal sliding-window consensus and confidence aggregator for per-track identity verification.

    Sits on top of per-frame decide_identity() outputs without modifying underlying fusion math.
    """

    def __init__(
        self,
        window_size: int = 8,
        consensus_threshold: float = 0.60,
        confirm_threshold: float = 0.72,
        near_miss_margin: float = 0.05,
        min_frames_for_decision: int = 3,
        high_risk_confusion_groups: list[list[str]] | None = None,
    ) -> None:
        self.window_size = int(window_size)
        self.consensus_threshold = float(consensus_threshold)
        self.confirm_threshold = float(confirm_threshold)
        self.near_miss_margin = float(near_miss_margin)
        self.min_frames_for_decision = int(min_frames_for_decision)
        self.high_risk_confusion_groups = [
            set(g) for g in (high_risk_confusion_groups or [["Devhan", "Isuru", "person01"]])
        ]

        self._lock = threading.Lock()
        self._tracks: dict[int, deque] = {}
        self._track_stats: dict[int, dict[str, Any]] = {}
        self._confirmed_identities: dict[int, str] = {}
        self._logger = get_logger("track_identity_aggregator")

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "TrackIdentityAggregator":
        cfg = config or {}
        temp_cfg = cfg.get("temporal_aggregation", cfg.get("temporal_verification", {}))
        confusion_groups = cfg.get("high_risk_confusion_groups", [["Devhan", "Isuru", "person01"]])
        return cls(
            window_size=int(temp_cfg.get("window_size", 8)),
            consensus_threshold=float(temp_cfg.get("consensus_threshold", 0.60)),
            confirm_threshold=float(temp_cfg.get("confirm_threshold", 0.72)),
            near_miss_margin=float(temp_cfg.get("near_miss_margin", 0.05)),
            min_frames_for_decision=int(temp_cfg.get("min_frames", 3)),
            high_risk_confusion_groups=confusion_groups,
        )

    @staticmethod
    def _is_valid_identity(identity: Any) -> bool:
        if identity is None:
            return False
        s = str(identity).strip().upper()
        return s not in ("", "UNKNOWN", "UNKNOWN_PERSON", "NONE", "UNAVAILABLE", "NULL", "REVIEW_REQUIRED")

    def update(
        self,
        track_id: int,
        identity: str | None,
        score: float = 0.0,
        modality_state: str = "UNKNOWN",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a per-frame decide_identity() output and update the sliding window for track_id.

        Returns track-level aggregation result with decision:
        - CONFIRMED: Consensus >= M% AND average score >= confirm_threshold.
        - REVIEW_REQUIRED: Consensus >= M% AND average score in [confirm_threshold - margin, confirm_threshold).
        - LOW_CONFIDENCE: Candidate present but below near-miss margin (logged, no operator alert).
        - UNKNOWN: No valid candidate or insufficient consensus.
        """
        now = time.time()
        record = {
            "identity": str(identity).strip() if identity is not None else "UNKNOWN_PERSON",
            "score": float(score) if (score is not None and score > 0) else 0.0,
            "modality_state": str(modality_state),
            "details": details or {},
            "timestamp": now,
        }

        with self._lock:
            if track_id not in self._tracks:
                self._tracks[track_id] = deque(maxlen=self.window_size)
                self._track_stats[track_id] = {
                    "start_time": now,
                    "total_frames": 0,
                    "confirmed_at_frame": None,
                    "confirmed_identity": None,
                }

            window = self._tracks[track_id]
            window.append(record)
            stats = self._track_stats[track_id]
            stats["total_frames"] += 1
            frame_idx = stats["total_frames"]


            valid_votes = [r["identity"] for r in window if self._is_valid_identity(r["identity"])]
            counts = Counter(valid_votes)

            window_len = len(window)
            if not counts or window_len < self.min_frames_for_decision:

                return {
                    "track_id": track_id,
                    "decision": "UNKNOWN",
                    "status": "UNKNOWN",
                    "identity": "UNKNOWN_PERSON",
                    "confidence": round(float(score), 4),
                    "consensus_fraction": 0.0,
                    "agreeing_frames": 0,
                    "window_size": window_len,
                    "total_track_frames": frame_idx,
                    "modality_state": modality_state,
                    "is_aggregated": False,
                }

            best_candidate, vote_count = counts.most_common(1)[0]
            consensus_fraction = vote_count / window_len


            agreeing_scores = [r["score"] for r in window if r["identity"] == best_candidate]
            avg_score = float(sum(agreeing_scores) / len(agreeing_scores)) if agreeing_scores else 0.0


            if consensus_fraction >= self.consensus_threshold and avg_score >= self.confirm_threshold:

                is_confusion_risk = False
                for group in self.high_risk_confusion_groups:
                    if best_candidate in group:
                        is_confusion_risk = True
                        break

                if is_confusion_risk:
                    return {
                        "track_id": track_id,
                        "decision": "REVIEW_REQUIRED",
                        "status": "REVIEW_REQUIRED",
                        "identity": best_candidate,
                        "confidence": round(avg_score, 4),
                        "consensus_fraction": round(consensus_fraction, 4),
                        "agreeing_frames": vote_count,
                        "window_size": window_len,
                        "total_track_frames": frame_idx,
                        "modality_state": "CONFUSION_SAFEGUARD_REVIEW",
                        "is_aggregated": True,
                        "alert_reason": f"High-risk confusion pair safeguard active for candidate '{best_candidate}'",
                    }

                if stats["confirmed_identity"] is None:
                    stats["confirmed_identity"] = best_candidate
                    stats["confirmed_at_frame"] = frame_idx
                    self._confirmed_identities[track_id] = best_candidate
                    self._logger.info(
                        f"Track {track_id} CONFIRMED as '{best_candidate}' at frame {frame_idx} "
                        f"(Consensus: {vote_count}/{window_len} = {consensus_fraction*100:.1f}%, Avg Score: {avg_score:.4f})"
                    )

                return {
                    "track_id": track_id,
                    "decision": "CONFIRMED",
                    "status": "CONFIRMED",
                    "identity": best_candidate,
                    "confidence": round(avg_score, 4),
                    "consensus_fraction": round(consensus_fraction, 4),
                    "agreeing_frames": vote_count,
                    "window_size": window_len,
                    "total_track_frames": frame_idx,
                    "modality_state": "TEMPORAL_CONFIRMED",
                    "is_aggregated": True,
                }


            review_lower_bound = self.confirm_threshold - self.near_miss_margin
            if consensus_fraction >= self.consensus_threshold and avg_score >= review_lower_bound:
                return {
                    "track_id": track_id,
                    "decision": "REVIEW_REQUIRED",
                    "status": "REVIEW_REQUIRED",
                    "identity": best_candidate,
                    "confidence": round(avg_score, 4),
                    "consensus_fraction": round(consensus_fraction, 4),
                    "agreeing_frames": vote_count,
                    "window_size": window_len,
                    "total_track_frames": frame_idx,
                    "modality_state": "NEAR_MISS_REVIEW",
                    "is_aggregated": True,
                    "alert_reason": f"Score {avg_score:.4f} within near-miss margin of threshold {self.confirm_threshold:.2f}",
                }


            if consensus_fraction >= self.consensus_threshold:
                return {
                    "track_id": track_id,
                    "decision": "LOW_CONFIDENCE",
                    "status": "UNKNOWN",
                    "identity": best_candidate,
                    "confidence": round(avg_score, 4),
                    "consensus_fraction": round(consensus_fraction, 4),
                    "agreeing_frames": vote_count,
                    "window_size": window_len,
                    "total_track_frames": frame_idx,
                    "modality_state": "LOW_CONFIDENCE",
                    "is_aggregated": True,
                }


            return {
                "track_id": track_id,
                "decision": "UNKNOWN",
                "status": "UNKNOWN",
                "identity": "UNKNOWN_PERSON",
                "confidence": round(avg_score, 4),
                "consensus_fraction": round(consensus_fraction, 4),
                "agreeing_frames": vote_count,
                "window_size": window_len,
                "total_track_frames": frame_idx,
                "modality_state": "INSUFFICIENT_CONSENSUS",
                "is_aggregated": True,
            }

    def on_track_lost(self, track_id: int) -> dict[str, Any] | None:
        """
        Handle track loss/eviction. Logs track summary and clears buffer to prevent cross-track carryover.
        """
        with self._lock:
            if track_id not in self._tracks:
                return None

            window = self._tracks.pop(track_id, deque())
            stats = self._track_stats.pop(track_id, {})
            self._confirmed_identities.pop(track_id, None)

            total_frames = stats.get("total_frames", len(window))
            confirmed_id = stats.get("confirmed_identity")

            valid_votes = [r["identity"] for r in window if self._is_valid_identity(r["identity"])]
            counts = Counter(valid_votes)

            if counts:
                best_candidate, vote_count = counts.most_common(1)[0]
                consensus_fraction = vote_count / max(1, len(window))
                agreeing_scores = [r["score"] for r in window if r["identity"] == best_candidate]
                avg_score = float(sum(agreeing_scores) / len(agreeing_scores)) if agreeing_scores else 0.0
            else:
                best_candidate = "UNKNOWN_PERSON"
                vote_count = 0
                consensus_fraction = 0.0
                avg_score = 0.0

            summary = {
                "track_id": track_id,
                "total_frames": total_frames,
                "final_candidate": best_candidate,
                "average_score": round(avg_score, 4),
                "consensus_fraction": round(consensus_fraction, 4),
                "confirmed_identity": confirmed_id,
                "outcome": (
                    "CONFIRMED"
                    if confirmed_id is not None
                    else (
                        "REVIEW_REQUIRED"
                        if (avg_score >= (self.confirm_threshold - self.near_miss_margin) and consensus_fraction >= self.consensus_threshold)
                        else "LOW_CONFIDENCE"
                    )
                ),
            }

            self._logger.info(
                f"Track {track_id} ENDED after {total_frames} frames. Outcome: {summary['outcome']}, "
                f"Candidate: '{best_candidate}', Avg Score: {avg_score:.4f}"
            )
            return summary

    def clear_track(self, track_id: int) -> None:
        """Explicitly clear track state on re-acquisition or tracker reset."""
        with self._lock:
            self._tracks.pop(track_id, None)
            self._track_stats.pop(track_id, None)
            self._confirmed_identities.pop(track_id, None)

    def reset_all(self) -> None:
        """Clear all active tracks."""
        with self._lock:
            self._tracks.clear()
            self._track_stats.clear()
            self._confirmed_identities.clear()
