"""ARGUS AI Intelligence Package."""

try:
    from intelligence.appearance_embedding import AppearanceEmbeddingExtractor
except ImportError:
    AppearanceEmbeddingExtractor = None

try:
    from intelligence.camera_topology_learner import CameraTopologyLearner, LearnedEdgeStats
except ImportError:
    CameraTopologyLearner = None
    LearnedEdgeStats = None

try:
    from intelligence.camera_transition_model import CameraTransitionModel
except ImportError:
    CameraTransitionModel = None

try:
    from intelligence.confidence_scorer import ConfidenceScorer
except ImportError:
    ConfidenceScorer = None

try:
    from intelligence.cross_camera_tracker import CrossCameraTracker
except ImportError:
    CrossCameraTracker = None

try:
    from intelligence.crowd_density_estimator import CrowdDensityEstimator, CrowdDensityLevel, CrowdDensityResult
except ImportError:
    CrowdDensityEstimator = None
    CrowdDensityLevel = None
    CrowdDensityResult = None

try:
    from intelligence.crowd_intelligence_system import CrowdIntelligenceEvaluation, CrowdIntelligenceSystem
except ImportError:
    CrowdIntelligenceEvaluation = None
    CrowdIntelligenceSystem = None

try:
    from intelligence.crowd_occlusion_analyzer import CrowdOcclusionAnalyzer, FrameCrowdAnalysis
except ImportError:
    CrowdOcclusionAnalyzer = None
    FrameCrowdAnalysis = None

try:
    from intelligence.crowd_robustness_manager import CrowdRobustnessManager
except ImportError:
    CrowdRobustnessManager = None

try:
    from intelligence.dual_modal_fusion import DualModalFusion
except ImportError:
    DualModalFusion = None

try:
    from intelligence.fusion_weights import DynamicFusionWeights
except ImportError:
    DynamicFusionWeights = None

try:
    from intelligence.missing_person_workflow import MissingPersonWorkflow, WatchlistEntry, WatchlistManager
except ImportError:
    MissingPersonWorkflow = None
    WatchlistEntry = None
    WatchlistManager = None

try:
    from intelligence.multi_camera_evidence_fusion import (
        FusionState,
        MultiCameraEvidenceFusion,
        MultiCameraFusionResult,
    )
except ImportError:
    FusionState = None
    MultiCameraEvidenceFusion = None
    MultiCameraFusionResult = None

try:
    from intelligence.open_set_recognizer import OpenSetDecisionResult, OpenSetRecognizer, OpenSetState
except ImportError:
    OpenSetDecisionResult = None
    OpenSetRecognizer = None
    OpenSetState = None

try:
    from intelligence.quality_assessment import QualityAssessment
except ImportError:
    QualityAssessment = None

try:
    from intelligence.recognition_deferral_engine import DeferralResult, RecognitionDeferralEngine, RecognitionState
except ImportError:
    DeferralResult = None
    RecognitionDeferralEngine = None
    RecognitionState = None

try:
    from intelligence.score_normalizer import ScoreNormalizer
except ImportError:
    ScoreNormalizer = None

try:
    from intelligence.track_recovery_manager import LostTrackRecord, TrackRecoveryManager
except ImportError:
    LostTrackRecord = None
    TrackRecoveryManager = None

try:
    from intelligence.track_reliability_scorer import TrackReliabilityScorer
except ImportError:
    TrackReliabilityScorer = None

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
