"""ARGUS AI Pipeline Steps Package."""

try:
    from pipeline.steps.centroid_matching_step import CentroidMatchingStep
except ImportError:
    CentroidMatchingStep = None

try:
    from pipeline.steps.feature_extraction import FeatureExtractionStep
except ImportError:
    FeatureExtractionStep = None

try:
    from pipeline.steps.live_gei import LiveGEI
except ImportError:
    LiveGEI = None

try:
    from pipeline.steps.matching_step import MatchingStep
except ImportError:
    MatchingStep = None

try:
    from pipeline.steps.quality_estimator import QualityEstimator
except ImportError:
    QualityEstimator = None

try:
    from pipeline.steps.reid_feature_extraction import ReIDFeatureExtractionStep
except ImportError:
    ReIDFeatureExtractionStep = None

try:
    from pipeline.steps.reid_matching_step import ReIDMatchingStep
except ImportError:
    ReIDMatchingStep = None

try:
    from pipeline.steps.silhouette_step import SilhouetteStep
except ImportError:
    SilhouetteStep = None

try:
    from pipeline.steps.temporal_gait_verifier import TemporalGaitVerifier
except ImportError:
    TemporalGaitVerifier = None

try:
    from pipeline.steps.tracking import TrackingStep
except ImportError:
    TrackingStep = None

__all__ = [
    "CentroidMatchingStep",
    "FeatureExtractionStep",
    "LiveGEI",
    "MatchingStep",
    "QualityEstimator",
    "ReIDFeatureExtractionStep",
    "ReIDMatchingStep",
    "SilhouetteStep",
    "TemporalGaitVerifier",
    "TrackingStep",
]
