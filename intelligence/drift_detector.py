import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from intelligence.operational_embedding_collector import (
    OperationalEmbeddingCollector,
)
from monitoring.logging_config import get_logger
from storage.vector_store import VectorStore


@dataclass
class DriftReport:
    timestamp: float = field(default_factory=time.time)
    observation_count: int = 0
    mean_confidence: float = 0.0
    low_confidence_ratio: float = 0.0
    mean_gallery_similarity: float = 0.0
    drift_detected: bool = False
    drift_severity: str = "NONE"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DriftDetector:
    def __init__(
        self,
        collector: OperationalEmbeddingCollector | None = None,
        gait_gallery_dir: str = "models/live_gallery",
        confidence_threshold: float = 0.85,
        drift_cosine_drop_threshold: float = 0.15,
    ) -> None:
        self.collector = collector or OperationalEmbeddingCollector()
        self.gait_store = VectorStore(gallery_dir=gait_gallery_dir)
        self.confidence_threshold = confidence_threshold
        self.drift_cosine_drop_threshold = drift_cosine_drop_threshold
        self._logger = get_logger("drift_detector")

    def evaluate_drift(self, window_size: int = 50) -> DriftReport:
        observations = self.collector.get_recent_observations(limit=window_size)
        if not observations:
            return DriftReport(
                observation_count=0,
                drift_detected=False,
                drift_severity="NONE",
                recommendations=["Insufficient operational data to assess drift."],
            )

        confidences = [obs.confidence for obs in observations]
        mean_conf = float(np.mean(confidences))
        low_conf_count = sum(1 for c in confidences if c < self.confidence_threshold)
        low_conf_ratio = float(low_conf_count / len(observations))


        gallery_data = self.gait_store.load()
        mean_sim = 0.0

        if gallery_data is not None:
            g_features, _, _ = gallery_data
            if len(g_features) > 0:
                sims = []
                for obs in observations:
                    if obs.embedding_dim == 256:
                        vec = np.asarray(obs.vector, dtype=np.float32)

                        dot_prods = np.dot(g_features, vec)
                        sims.append(float(np.max(dot_prods)))
                if sims:
                    mean_sim = float(np.mean(sims))


        drift_detected = False
        severity = "NONE"
        recommendations = []

        if low_conf_ratio > 0.40 or (
            mean_sim > 0 and mean_sim < (self.confidence_threshold - self.drift_cosine_drop_threshold)
        ):
            drift_detected = True
            severity = "HIGH" if low_conf_ratio > 0.60 else "MODERATE"
            recommendations.append(
                "Trigger background candidate calibration to adapt to new camera/lighting conditions."
            )
            recommendations.append("Request human verification for low-confidence clusters.")
        elif low_conf_ratio > 0.20:
            severity = "LOW"
            recommendations.append("Continue monitoring operational observations.")
        else:
            recommendations.append("Model operating within nominal calibration bounds.")

        report = DriftReport(
            observation_count=len(observations),
            mean_confidence=round(mean_conf, 4),
            low_confidence_ratio=round(low_conf_ratio, 4),
            mean_gallery_similarity=round(mean_sim, 4),
            drift_detected=drift_detected,
            drift_severity=severity,
            recommendations=recommendations,
        )

        self._logger.debug(
            f"Drift assessment: severity={severity}, mean_conf={mean_conf:.3f}, low_conf_ratio={low_conf_ratio:.2%}"
        )
        return report
