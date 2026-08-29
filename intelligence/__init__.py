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
    from intelligence.candidate_validator import CandidateValidator, ValidationGateResult
except ImportError:
    CandidateValidator = None
    ValidationGateResult = None

try:
    from intelligence.drift_detector import DriftDetector, DriftReport
except ImportError:
    DriftDetector = None
    DriftReport = None

try:
    from intelligence.operational_embedding_collector import (
        ObservationState,
        OperationalEmbeddingCollector,
        OperationalObservation,
    )
except ImportError:
    ObservationState = None
    OperationalEmbeddingCollector = None
    OperationalObservation = None

try:
    from intelligence.date_aware_learning_scheduler import (
        DateAwareLearningScheduler,
        LearningJobRecord,
        LearningJobStatus,
    )
except ImportError:
    DateAwareLearningScheduler = None
    LearningJobRecord = None
    LearningJobStatus = None

try:
    from intelligence.background_learning_worker import BackgroundLearningWorker
except ImportError:
    BackgroundLearningWorker = None

try:
    from intelligence.continuous_improvement_engine import ContinuousImprovementEngine
except ImportError:
    ContinuousImprovementEngine = None

try:
    from intelligence.track_reliability_scorer import TrackReliabilityScorer
except ImportError:
    TrackReliabilityScorer = None

try:
    from intelligence.track_identity_aggregator import TrackIdentityAggregator
except ImportError:
    TrackIdentityAggregator = None

try:
    from intelligence.score_calibrator import ScoreCalibrator
except ImportError:
    ScoreCalibrator = None

try:
    from intelligence.confusion_detector import ConfusionDetector
except ImportError:
    ConfusionDetector = None

try:
    from intelligence.nn_fine_tuner import NNFineTuner
except ImportError:
    NNFineTuner = None

try:
    from intelligence.learned_fusion import LearnedFusion
except ImportError:
    LearnedFusion = None

try:
    from intelligence.fusion_diagnostics import FusionDiagnostics
except ImportError:
    FusionDiagnostics = None

__all__ = [
    "AppearanceEmbeddingExtractor",
    "BackgroundLearningWorker",
    "CameraTopologyLearner",
    "CameraTransitionModel",
    "CandidateValidator",
    "ConfidenceScorer",
    "ConfusionDetector",
    "ContinuousImprovementEngine",
    "CrossCameraTracker",
    "CrowdDensityEstimator",
    "CrowdDensityLevel",
    "CrowdDensityResult",
    "CrowdIntelligenceEvaluation",
    "CrowdIntelligenceSystem",
    "CrowdOcclusionAnalyzer",
    "CrowdRobustnessManager",
    "DateAwareLearningScheduler",
    "DeferralResult",
    "DriftDetector",
    "DriftReport",
    "DualModalFusion",
    "DynamicFusionWeights",
    "FrameCrowdAnalysis",
    "FusionDiagnostics",
    "FusionState",
    "LearnedEdgeStats",
    "LearnedFusion",
    "LearningJobRecord",
    "LearningJobStatus",
    "LostTrackRecord",
    "MissingPersonWorkflow",
    "MultiCameraEvidenceFusion",
    "MultiCameraFusionResult",
    "NNFineTuner",
    "ObservationState",
    "OpenSetDecisionResult",
    "OpenSetRecognizer",
    "OpenSetState",
    "OperationalEmbeddingCollector",
    "OperationalObservation",
    "QualityAssessment",
    "RecognitionDeferralEngine",
    "RecognitionState",
    "ScoreCalibrator",
    "ScoreNormalizer",
    "TrackIdentityAggregator",
    "TrackRecoveryManager",
    "TrackReliabilityScorer",
    "ValidationGateResult",
    "WatchlistEntry",
    "WatchlistManager",
]
