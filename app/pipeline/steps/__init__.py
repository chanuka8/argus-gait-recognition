import importlib
from typing import Any

_STEP_MODULE_MAP = {
    "CentroidMatchingStep": "app.pipeline.steps.centroid_matching_step",
    "FeatureExtractionStep": "app.pipeline.steps.feature_extraction",
    "Gait3DStep": "app.pipeline.steps.gait_3d_step",
    "LiveGEI": "app.pipeline.steps.live_gei",
    "MatchingStep": "app.pipeline.steps.matching_step",
    "QualityEstimator": "app.pipeline.steps.quality_estimator",
    "ReIDFeatureExtractionStep": "app.pipeline.steps.reid_feature_extraction",
    "ReIDMatchingStep": "app.pipeline.steps.reid_matching_step",
    "SilhouetteStep": "app.pipeline.steps.silhouette_step",
    "TemporalGaitVerifier": "app.pipeline.steps.temporal_gait_verifier",
    "TrackingStep": "app.pipeline.steps.tracking",
}


def __getattr__(name: str) -> Any:
    if name in _STEP_MODULE_MAP:
        mod = importlib.import_module(_STEP_MODULE_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_STEP_MODULE_MAP.keys())
