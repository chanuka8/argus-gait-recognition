"""ARGUS AI Intelligence Package."""

from intelligence.appearance_embedding import AppearanceEmbeddingExtractor
from intelligence.camera_topology_learner import CameraTopologyLearner, LearnedEdgeStats
from intelligence.camera_transition_model import CameraTransitionModel
from intelligence.confidence_scorer import ConfidenceScorer
from intelligence.cross_camera_tracker import CrossCameraTracker
from intelligence.crowd_density_estimator import CrowdDensityEstimator, CrowdDensityLevel, CrowdDensityResult
from intelligence.crowd_intelligence_system import CrowdIntelligenceEvaluation, CrowdIntelligenceSystem
from intelligence.crowd_occlusion_analyzer import CrowdOcclusionAnalyzer, FrameCrowdAnalysis
from intelligence.crowd_robustness_manager import CrowdRobustnessManager
from intelligence.dual_modal_fusion import DualModalFusion
from intelligence.fusion_weights import DynamicFusionWeights
from intelligence.missing_person_workflow import MissingPersonWorkflow, WatchlistEntry, WatchlistManager
from intelligence.multi_camera_evidence_fusion import FusionState, MultiCameraEvidenceFusion, MultiCameraFusionResult
from intelligence.open_set_recognizer import OpenSetDecisionResult, OpenSetRecognizer, OpenSetState
from intelligence.quality_assessment import QualityAssessment
from intelligence.recognition_deferral_engine import DeferralResult, RecognitionDeferralEngine, RecognitionState
from intelligence.score_normalizer import ScoreNormalizer
from intelligence.track_recovery_manager import LostTrackRecord, TrackRecoveryManager
from intelligence.track_reliability_scorer import TrackReliabilityScorer

__all__ = [
    "AppearanceEmbeddingExtractor",
    "CameraTopologyLearner",
    "CameraTransitionModel",
    "ConfidenceScorer",
    "CrossCameraTracker",
    "CrowdDensityEstimator",
    "CrowdDensityLevel",
    "CrowdDensityResult",
    "CrowdIntelligenceEvaluation",
    "CrowdIntelligenceSystem",
    "CrowdOcclusionAnalyzer",
    "CrowdRobustnessManager",
    "DeferralResult",
    "DualModalFusion",
    "DynamicFusionWeights",
    "FrameCrowdAnalysis",
    "FusionState",
    "LearnedEdgeStats",
    "LostTrackRecord",
    "MissingPersonWorkflow",
    "MultiCameraEvidenceFusion",
    "MultiCameraFusionResult",
    "OpenSetDecisionResult",
    "OpenSetRecognizer",
    "OpenSetState",
    "QualityAssessment",
    "RecognitionDeferralEngine",
    "RecognitionState",
    "ScoreNormalizer",
    "TrackRecoveryManager",
    "TrackReliabilityScorer",
    "WatchlistEntry",
    "WatchlistManager",
]
