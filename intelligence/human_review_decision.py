from dataclasses import dataclass


@dataclass
class ReviewDecision:
    identity: str
    score: float
    source: str
    severity: str
    decision: str


class HumanReviewDecisionEngine:
    def decide(
        self,
        gait_identity: str,
        gait_score: float,
        appearance_identity: str,
        appearance_score: float,
    ) -> ReviewDecision:
        if gait_identity != "UNKNOWN":
            return ReviewDecision(
                identity=gait_identity,
                score=float(gait_score),
                source="GAIT_CONFIRMED",
                severity="HIGH",
                decision="CONFIRMED_MATCH",
            )

        if appearance_identity != "UNKNOWN":
            return ReviewDecision(
                identity=appearance_identity,
                score=float(appearance_score),
                source="PHOTO_REVIEW",
                severity="MEDIUM",
                decision="REVIEW_REQUIRED",
            )

        return ReviewDecision(
            identity="UNKNOWN",
            score=float(max(gait_score, appearance_score)),
            source="UNKNOWN",
            severity="INFO",
            decision="UNKNOWN_PERSON",
        )