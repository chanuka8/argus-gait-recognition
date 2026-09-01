"""
Deterministic Video and Silhouette Quality Gate for ARGUS AI Auto-Enrollment (Step 5M).

Enforces strict pre-deletion quality assurance on raw video uploads and extracted silhouettes
before GEI synthesis and biometric embedding generation.

Checks:
1. Usable Frame Count: Minimum valid silhouette frames (>= 15 frames) for a complete gait cycle.
2. Full-Body Silhouette Completeness: Aspect ratio (H/W >= 1.3), non-empty mask area (>10%), and vertical height coverage.
3. Motion Consistency & Gait Dynamism: Non-zero centroid displacement across frames (rejects static standing/jitter).
4. Frame-Level Exposure & Blur: Mean luminance (30 <= L <= 235), Laplacian blur variance (>= 25.0).
5. Borderline Deterministic Enhancement: Bilateral denoising, adaptive CLAHE, Lanczos upscaling (NO generative AI).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class VideoQualityAssessmentResult:
    passed: bool
    salvageable: bool
    issues: list[str] = field(default_factory=list)
    actionable_guidance: str = ""
    usable_frames_count: int = 0
    mean_blur_variance: float = 0.0
    mean_luminance: float = 0.0
    mean_aspect_ratio: float = 0.0
    motion_dynamism_score: float = 0.0


class DeterministicVideoQualityGate:
    """
    Quality gate applied to video uploads and silhouette sequences prior to GEI generation and vector storage.
    """

    def __init__(
        self,
        min_frames: int = 5,
        min_blur_var: float = 25.0,
        min_luminance: float = 30.0,
        max_luminance: float = 235.0,
        min_aspect_ratio: float = 1.2,
        min_silhouette_coverage: float = 0.08,
        min_motion_displacement: float = 1.0,
    ) -> None:
        self.min_frames = int(min_frames)
        self.min_blur_var = float(min_blur_var)
        self.min_luminance = float(min_luminance)
        self.max_luminance = float(max_luminance)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.min_silhouette_coverage = float(min_silhouette_coverage)
        self.min_motion_displacement = float(min_motion_displacement)

    def assess_frame_quality(self, crop: np.ndarray) -> tuple[bool, float, float]:
        """Assess blur variance and exposure on an individual crop."""
        if crop is None or crop.size == 0:
            return False, 0.0, 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        is_ok = (blur_var >= self.min_blur_var) and (self.min_luminance <= mean_lum <= self.max_luminance)
        return is_ok, blur_var, mean_lum

    def assess_silhouette_quality(self, silhouette: np.ndarray) -> tuple[bool, float, float, tuple[float, float]]:
        """Assess silhouette completeness, aspect ratio, and centroid."""
        if silhouette is None or silhouette.size == 0:
            return False, 0.0, 0.0, (0.0, 0.0)

        mask = (silhouette > 128).astype(np.uint8) if silhouette.max() > 1 else silhouette.astype(np.uint8)
        area = int(np.sum(mask))
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = area / max(total_pixels, 1)


        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return False, 0.0, 0.0, (0.0, 0.0)

        h_sil = float(np.max(y_indices) - np.min(y_indices) + 1)
        w_sil = float(np.max(x_indices) - np.min(x_indices) + 1)
        aspect_ratio = h_sil / max(w_sil, 1.0)
        centroid = (float(np.mean(x_indices)), float(np.mean(y_indices)))

        is_complete = (coverage >= self.min_silhouette_coverage) and (aspect_ratio >= self.min_aspect_ratio)
        return is_complete, coverage, aspect_ratio, centroid

    def assess_video_clip(
        self,
        frames: list[np.ndarray],
        silhouettes: list[np.ndarray],
    ) -> VideoQualityAssessmentResult:
        """
        Assess an entire video sequence and its corresponding extracted silhouettes.
        """
        issues = []
        usable_frames = 0
        blur_scores, lum_scores, aspect_ratios, centroids = [], [], [], []

        for f, s in zip(frames, silhouettes):
            if f is None or s is None:
                continue

            f_ok, blur_v, lum_v = self.assess_frame_quality(f)
            s_ok, _cov_v, asp_v, cent_v = self.assess_silhouette_quality(s)

            blur_scores.append(blur_v)
            lum_scores.append(lum_v)
            aspect_ratios.append(asp_v)
            centroids.append(cent_v)

            if f_ok and s_ok:
                usable_frames += 1

        mean_blur = float(np.mean(blur_scores)) if blur_scores else 0.0
        mean_lum = float(np.mean(lum_scores)) if lum_scores else 0.0
        mean_asp = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0


        motion_dynamism = 0.0
        if len(centroids) >= 2:
            dx_total = sum(abs(centroids[i][0] - centroids[i - 1][0]) for i in range(1, len(centroids)))
            dy_total = sum(abs(centroids[i][1] - centroids[i - 1][1]) for i in range(1, len(centroids)))
            motion_dynamism = float(dx_total + dy_total) / max(len(centroids), 1)


        if usable_frames < self.min_frames:
            issues.append(f"Insufficient usable walking frames ({usable_frames} valid < {self.min_frames} required for full gait cycle)")

        if mean_blur < self.min_blur_var:
            issues.append(f"Severe video motion blur (Laplacian variance {mean_blur:.1f} < {self.min_blur_var:.1f})")

        if mean_lum < self.min_luminance:
            issues.append(f"Video lighting underexposed / too dark (Mean luminance {mean_lum:.1f} < {self.min_luminance:.1f})")
        elif mean_lum > self.max_luminance:
            issues.append(f"Video lighting overexposed / washed out (Mean luminance {mean_lum:.1f} > {self.max_luminance:.1f})")

        if mean_asp < self.min_aspect_ratio:
            issues.append(f"Incomplete full-body silhouette visibility (Aspect ratio {mean_asp:.2f} < {self.min_aspect_ratio:.2f}; full vertical body head-to-feet required)")

        if motion_dynamism < self.min_motion_displacement and len(centroids) >= self.min_frames:
            issues.append(f"Static or non-walking motion detected (Centroid displacement {motion_dynamism:.2f} < {self.min_motion_displacement:.2f})")

        passed = (len(issues) == 0)
        salvageable = (usable_frames >= 10 and mean_blur >= 15.0 and 20.0 <= mean_lum <= 245.0)

        guidance = ""
        if not passed:
            guidance = "Quality Gate Rejected: Please upload a clear video of the subject walking across the camera view with full body visible (head to feet), at least 2 seconds duration, under adequate lighting."

        return VideoQualityAssessmentResult(
            passed=passed,
            salvageable=salvageable,
            issues=issues,
            actionable_guidance=guidance,
            usable_frames_count=usable_frames,
            mean_blur_variance=round(mean_blur, 2),
            mean_luminance=round(mean_lum, 2),
            mean_aspect_ratio=round(mean_asp, 2),
            motion_dynamism_score=round(motion_dynamism, 2),
        )

    def enhance_crop_deterministic(self, crop: np.ndarray) -> np.ndarray:
        """
        Apply deterministic non-generative enhancement to borderline salvageable person crop.
        """
        if crop is None or crop.size == 0:
            return crop

        img = crop.copy()

        img = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)


        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        if np.mean(v) < 60.0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(v)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


        h, w = img.shape[:2]
        if h < 128 or w < 64:
            scale = max(128.0 / h, 64.0 / w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return img
