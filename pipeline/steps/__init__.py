import importlib
from typing import Any

_STEP_MODULE_MAP = {
    "CentroidMatchingStep": "pipeline.steps.centroid_matching_step",
    "FeatureExtractionStep": "pipeline.steps.feature_extraction",
    "Gait3DStep": "pipeline.steps.gait_3d_step",
    "LiveGEI": "pipeline.steps.live_gei",
    "MatchingStep": "pipeline.steps.matching_step",
    "QualityEstimator": "pipeline.steps.quality_estimator",
    "ReIDFeatureExtractionStep": "pipeline.steps.reid_feature_extraction",
    "ReIDMatchingStep": "pipeline.steps.reid_matching_step",
    "SilhouetteStep": "pipeline.steps.silhouette_step",
    "TemporalGaitVerifier": "pipeline.steps.temporal_gait_verifier",
    "TrackingStep": "pipeline.steps.tracking",
}


def __getattr__(name: str) -> Any:
    if name in _STEP_MODULE_MAP:
        mod = importlib.import_module(_STEP_MODULE_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_STEP_MODULE_MAP.keys())
