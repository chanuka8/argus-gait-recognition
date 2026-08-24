"""
GEI Feature Quality Estimator.

Evaluates generated Gait Energy Images (GEI) prior to embedding extraction.
Computes metrics for Blur, Noise, Shadow, GEI Completeness, and Bounding Box Stability.
"""

from typing import Any, Dict

import cv2
import numpy as np

from monitoring.logging_config import get_logger


class QualityEstimator:
    """
    Lightweight, real-time GEI Quality Estimator.

    Evaluates GEI quality against a configurable threshold to skip low-quality frames.
    """

    def __init__(
        self,
        quality_threshold: float = 0.6,
        weights: Dict[str, float] | None = None,
    ) -> None:
        self.quality_threshold = quality_threshold
        self.logger = get_logger("quality_estimator")

        default_weights = {
            "blur": 0.25,
            "noise": 0.20,
            "shadow": 0.15,
            "completeness": 0.25,
            "stability": 0.15,
        }

        if weights:
            default_weights.update(weights)

        total_w = sum(default_weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in default_weights.items()}
        else:
            self.weights = default_weights

    def compute_blur_score(
        self,
        gei: np.ndarray,
    ) -> float:
        """
        Compute blur score [0.0, 1.0] using Laplacian variance.
        Higher is sharper / higher quality.
        """
        if gei is None or gei.size == 0:
            return 0.0

        if len(gei.shape) == 3:
            gray = cv2.cvtColor(gei, cv2.COLOR_BGR2GRAY)
        else:
            gray = gei

        if gray.dtype != np.uint8:
            gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        norm_score = min(1.0, lap_var / 300.0)
        return float(max(0.0, norm_score))

    def compute_noise_score(
        self,
        gei: np.ndarray,
    ) -> float:
        """
        Compute noise score [0.0, 1.0].
        Higher means lower background noise.
        """
        if gei is None or gei.size == 0:
            return 0.0

        if len(gei.shape) == 3:
            gray = cv2.cvtColor(gei, cv2.COLOR_BGR2GRAY)
        else:
            gray = gei

        h, w = gray.shape
        margin_h = max(1, h // 10)
        margin_w = max(1, w // 10)

        top = gray[:margin_h, :]
        bottom = gray[-margin_h:, :]
        left = gray[:, :margin_w]
        right = gray[:, -margin_w:]

        bg_std = (np.std(top) + np.std(bottom) + np.std(left) + np.std(right)) / 4.0

        noise_penalty = min(1.0, bg_std / 50.0)
        return float(max(0.0, 1.0 - noise_penalty))

    def compute_shadow_score(
        self,
        gei: np.ndarray,
    ) -> float:
        """
        Compute shadow score [0.0, 1.0].
        Detects ground shadow artifacts in the bottom portion of GEI.
        """
        if gei is None or gei.size == 0:
            return 0.0

        if len(gei.shape) == 3:
            gray = cv2.cvtColor(gei, cv2.COLOR_BGR2GRAY)
        else:
            gray = gei

        h, w = gray.shape
        bottom_region = gray[int(h * 0.85) :, :]


        shadow_pixels = np.logical_and(bottom_region > 10, bottom_region < 100)
        shadow_ratio = float(np.sum(shadow_pixels)) / float(max(1, bottom_region.size))

        score = max(0.0, 1.0 - shadow_ratio * 3.0)
        return float(min(1.0, score))

    def compute_completeness_score(
        self,
        gei: np.ndarray,
    ) -> float:
        """
        Compute GEI completeness score [0.0, 1.0].
        Evaluates body portion coverage and non-zero density.
        """
        if gei is None or gei.size == 0:
            return 0.0

        if len(gei.shape) == 3:
            gray = cv2.cvtColor(gei, cv2.COLOR_BGR2GRAY)
        else:
            gray = gei

        h, w = gray.shape
        non_zero_ratio = float(np.count_nonzero(gray > 15)) / float(gray.size)

        if 0.10 <= non_zero_ratio <= 0.45:
            coverage_score = 1.0
        elif non_zero_ratio < 0.10:
            coverage_score = non_zero_ratio / 0.10
        else:
            coverage_score = max(0.0, 1.0 - (non_zero_ratio - 0.45) * 2.0)

        head_present = np.any(gray[: int(h * 0.3), :] > 15)
        torso_present = np.any(gray[int(h * 0.3) : int(h * 0.7), :] > 15)
        legs_present = np.any(gray[int(h * 0.7) :, :] > 15)

        structural_integrity = (float(head_present) + float(torso_present) + float(legs_present)) / 3.0

        completeness = 0.6 * coverage_score + 0.4 * structural_integrity
        return float(max(0.0, min(1.0, completeness)))

    def compute_stability_score(
        self,
        gei: np.ndarray,
        box_aspect_ratio: float | None = None,
    ) -> float:
        """
        Compute bounding box stability score [0.0, 1.0].
        Evaluates aspect ratio consistency (standard person ratio is ~2:1 height:width).
        """
        if gei is None or gei.size == 0:
            return 0.0

        h, w = gei.shape[:2]
        ratio = float(h) / float(max(1, w)) if box_aspect_ratio is None else box_aspect_ratio

        if 1.5 <= ratio <= 3.0:
            stability = 1.0
        elif ratio < 1.5:
            stability = max(0.0, ratio / 1.5)
        else:
            stability = max(0.0, 1.0 - (ratio - 3.0) * 0.5)

        return float(min(1.0, stability))

    def evaluate(
        self,
        gei: np.ndarray,
        box_aspect_ratio: float | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate GEI quality across all metrics.

        Returns:
            Dict containing:
                - overall_quality: float [0.0, 1.0]
                - metrics: Dict[str, float]
                - accepted: bool
                - reason: str | None
        """
        if gei is None or gei.size == 0:
            self.logger.warning("QualityEstimator rejected: GEI is empty or None")
            return {
                "overall_quality": 0.0,
                "metrics": {
                    "blur": 0.0,
                    "noise": 0.0,
                    "shadow": 0.0,
                    "completeness": 0.0,
                    "stability": 0.0,
                },
                "accepted": False,
                "reason": "GEI is empty or None",
            }

        blur = self.compute_blur_score(gei)
        noise = self.compute_noise_score(gei)
        shadow = self.compute_shadow_score(gei)
        completeness = self.compute_completeness_score(gei)
        stability = self.compute_stability_score(gei, box_aspect_ratio=box_aspect_ratio)

        metrics = {
            "blur": blur,
            "noise": noise,
            "shadow": shadow,
            "completeness": completeness,
            "stability": stability,
        }

        overall_quality = sum(metrics[k] * self.weights[k] for k in metrics)
        overall_quality = float(max(0.0, min(1.0, overall_quality)))

        accepted = overall_quality >= self.quality_threshold
        reason = None

        if not accepted:
            lowest_metric = min(metrics, key=metrics.get)
            reason = (
                f"GEI quality score ({overall_quality:.3f}) below threshold "
                f"({self.quality_threshold:.3f}). Lowest component: {lowest_metric} ({metrics[lowest_metric]:.3f})"
            )
            self.logger.info(f"Skipping GEI matching: {reason}")
        else:
            self.logger.debug(f"GEI quality score accepted: {overall_quality:.3f}")

        return {
            "overall_quality": overall_quality,
            "metrics": metrics,
            "accepted": accepted,
            "reason": reason,
        }
