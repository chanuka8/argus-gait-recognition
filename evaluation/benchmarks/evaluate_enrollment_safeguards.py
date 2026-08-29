
from typing import Any
import numpy as np
import cv2
from preprocessing.video_quality_gate import DeterministicVideoQualityGate, VideoQualityAssessmentResult
from intelligence.dual_modal_fusion import DualModalFusion
from intelligence.track_identity_aggregator import TrackIdentityAggregator


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
