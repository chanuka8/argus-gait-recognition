import numpy as np

from preprocessing.video_quality_gate import (
    DeterministicVideoQualityGate,
    VideoQualityAssessmentResult,
)


class EnrollmentSafeguardEvaluator:
    """
    Evaluates enrollment pre-deletion quality gates and runtime confusion safeguards.
    """

    def __init__(self, quality_gate: DeterministicVideoQualityGate | None = None) -> None:
        self.quality_gate = quality_gate or DeterministicVideoQualityGate()

    def assess_subject_data(
        self,
        subject_crops: list[np.ndarray],
        subject_silhouettes: list[np.ndarray],
    ) -> VideoQualityAssessmentResult:
        return self.quality_gate.assess_video_clip(subject_crops, subject_silhouettes)
