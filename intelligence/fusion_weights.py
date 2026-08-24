"""
Dynamic Fusion Weight Allocation for Dual-Modal (Gait + ReID) Biometrics.

Enforces constraint: gait_weight + reid_weight = 1.0.
Supports default static weights and quality-adaptive dynamic weighting.
"""

from typing import Tuple


class DynamicFusionWeights:
    """
    Dual-modal weight allocator.

    Maintains gait_weight + reid_weight = 1.0.
    Dynamically adjusts weights based on modal quality factors.
    """

    def __init__(
        self,
        default_gait_weight: float = 0.7,
        default_reid_weight: float = 0.3,
    ) -> None:
        total = default_gait_weight + default_reid_weight
        if total <= 0:
            self.base_gait_weight = 0.7
            self.base_reid_weight = 0.3
        else:
            self.base_gait_weight = default_gait_weight / total
            self.base_reid_weight = default_reid_weight / total

    def compute_weights(
        self,
        gait_available: bool = True,
        reid_available: bool = True,
        gait_quality: float = 1.0,
        reid_quality: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute normalized dual-modal weights (w_gait, w_reid).

        Returns:
            Tuple[w_gait, w_reid] such that w_gait + w_reid = 1.0.
        """
        g_avail = bool(gait_available and gait_quality > 0.0)
        r_avail = bool(reid_available and reid_quality > 0.0)

        if g_avail and not r_avail:
            return 1.0, 0.0
        if r_avail and not g_avail:
            return 0.0, 1.0
        if not g_avail and not r_avail:
            return self.base_gait_weight, self.base_reid_weight

        eff_gait_w = self.base_gait_weight * max(0.0, min(1.0, gait_quality))
        eff_reid_w = self.base_reid_weight * max(0.0, min(1.0, reid_quality))

        total_w = eff_gait_w + eff_reid_w
        if total_w <= 0:
            return self.base_gait_weight, self.base_reid_weight

        w_gait = eff_gait_w / total_w
        w_reid = eff_reid_w / total_w

        return float(w_gait), float(w_reid)
