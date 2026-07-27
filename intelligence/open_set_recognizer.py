"""
Open-Set Gait Recognition Engine.

Classifies recognition candidates into a clear three-state decision model:
  - KNOWN: Confirmed enrolled identity with high confidence and distinct margin.
  - UNKNOWN: Unenrolled identity or probe score below the rejection threshold.
  - UNCERTAIN: Inconclusive match requiring further evidence or human review.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class OpenSetState(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class OpenSetDecisionResult:
    state: OpenSetState
    identity: str
    score: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


class OpenSetRecognizer:
    """
    Open-Set Recognizer.

    Applies three-state decision logic (KNOWN, UNKNOWN, UNCERTAIN) on gallery matches,
    feature quality metrics, and candidate margins.
    """

    def __init__(
        self,
        known_threshold: float = 0.85,
        unknown_threshold: float = 0.70,
        margin_threshold: float = 0.05,
        quality_threshold: float = 0.60,
    ) -> None:
        self.known_threshold = known_threshold
        self.unknown_threshold = unknown_threshold
        self.margin_threshold = margin_threshold
        self.quality_threshold = quality_threshold

    def evaluate_open_set_decision(
        self,
        top_matches: List[Tuple[str, float]],
        quality_score: float = 1.0,
        temporal_decision: Optional[str] = None,
    ) -> OpenSetDecisionResult:
        """
        Evaluates open-set status for a list of candidate matches.

        Args:
            top_matches: Ranked list of (identity_label, similarity_score) tuples.
            quality_score: GEI feature quality score [0.0, 1.0].
            temporal_decision: Decision status from TemporalGaitVerifier if available.

        Returns:
            OpenSetDecisionResult containing state, identity, score, confidence, and details.
        """
        if not top_matches:
            return OpenSetDecisionResult(
                state=OpenSetState.UNKNOWN,
                identity="UNKNOWN",
                score=0.0,
                confidence=0.0,
                details={"reason": "No matches returned"},
            )

        top_id, top_score = str(top_matches[0][0]), float(top_matches[0][1])

        if top_id == "UNKNOWN" or top_score <= 0.0:
            return OpenSetDecisionResult(
                state=OpenSetState.UNKNOWN,
                identity="UNKNOWN",
                score=max(0.0, top_score),
                confidence=0.0,
                details={"reason": "Identity labeled UNKNOWN or non-positive score"},
            )

        # Check for quality failure
        if quality_score < self.quality_threshold:
            return OpenSetDecisionResult(
                state=OpenSetState.UNCERTAIN,
                identity=top_id,
                score=top_score,
                confidence=float(quality_score * top_score),
                details={"reason": "GEI quality below threshold", "quality_score": quality_score},
            )

        # Calculate candidate margin (top-1 score minus top-2 score)
        margin = 1.0
        if len(top_matches) > 1:
            margin = top_score - float(top_matches[1][1])

        # 1. Definite UNKNOWN: Score below unknown_threshold
        if top_score < self.unknown_threshold:
            return OpenSetDecisionResult(
                state=OpenSetState.UNKNOWN,
                identity="UNKNOWN",
                score=top_score,
                confidence=float(1.0 - top_score),
                details={"reason": "Score below unknown_threshold", "margin": margin},
            )

        # 2. Definite KNOWN: High score and sufficient margin
        if top_score >= self.known_threshold:
            if margin < self.margin_threshold:
                return OpenSetDecisionResult(
                    state=OpenSetState.UNCERTAIN,
                    identity=top_id,
                    score=top_score,
                    confidence=float(top_score * (margin / max(1e-5, self.margin_threshold))),
                    details={"reason": "Insufficient margin between candidates", "margin": margin},
                )

            if temporal_decision == "UNCERTAIN":
                return OpenSetDecisionResult(
                    state=OpenSetState.UNCERTAIN,
                    identity=top_id,
                    score=top_score,
                    confidence=float(top_score * 0.8),
                    details={"reason": "Temporal verifier inconclusive", "temporal_decision": temporal_decision},
                )

            return OpenSetDecisionResult(
                state=OpenSetState.KNOWN,
                identity=top_id,
                score=top_score,
                confidence=float(top_score),
                details={"reason": "Confirmed match", "margin": margin},
            )

        # 3. UNCERTAIN: Score in gray zone between unknown_threshold and known_threshold
        return OpenSetDecisionResult(
            state=OpenSetState.UNCERTAIN,
            identity=top_id,
            score=top_score,
            confidence=float(top_score * 0.7),
            details={"reason": "Score in intermediate gray zone", "margin": margin},
        )
