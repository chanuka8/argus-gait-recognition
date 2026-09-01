"""
Runtime Biometric Confusion Risk Detector for ARGUS AI.

Evaluates cross-subject similarity upon new enrollment and automatically tags high-risk
confusion groups to prevent cross-identity false confirmations in production.
"""

from typing import Any

import numpy as np


class RuntimeConfusionDetector:
    """
    Automated similarity analyzer for enrolled identity galleries.
    Flags cross-subject similarity clusters exceeding safe operating thresholds.
    Supports advisory mode and configurable feature flagging.
    """

    def __init__(
        self,
        gait_risk_thresh: float = 0.92,
        app_risk_thresh: float = 0.65,
        app_co_risk_thresh: float = 0.55,
        enabled: bool = False,
        mode: str = "advisory",
    ) -> None:
        self.gait_risk_thresh = float(gait_risk_thresh)
        self.app_risk_thresh = float(app_risk_thresh)
        self.app_co_risk_thresh = float(app_co_risk_thresh)
        self.enabled = bool(enabled)
        self.mode = str(mode).lower()

    def analyze_new_enrollment(
        self,
        new_subject: str,
        new_gait_embs: list[np.ndarray],
        new_app_embs: list[np.ndarray],
        gallery_gait: dict[str, list[np.ndarray]],
        gallery_app: dict[str, list[np.ndarray]],
    ) -> dict[str, Any]:
        """
        Scan a newly enrolled subject against all existing gallery identities.
        """
        flagged_confusions = []
        max_g_sims = {}
        max_a_sims = {}

        for existing_subj, ex_g_list in gallery_gait.items():
            if existing_subj == new_subject or not ex_g_list:
                continue


            g_sims = []
            for ng in new_gait_embs:
                ng_norm = np.linalg.norm(ng)
                if ng_norm == 0:
                    continue
                for eg in ex_g_list:
                    eg_norm = np.linalg.norm(eg)
                    if eg_norm == 0:
                        continue
                    g_sims.append(float(np.dot(ng, eg) / (ng_norm * eg_norm)))
            max_g = float(np.max(g_sims)) if g_sims else 0.0
            max_g_sims[existing_subj] = round(max_g, 4)


            ex_a_list = gallery_app.get(existing_subj, [])
            a_sims = []
            for na in new_app_embs:
                na_norm = np.linalg.norm(na)
                if na_norm == 0:
                    continue
                for ea in ex_a_list:
                    ea_norm = np.linalg.norm(ea)
                    if ea_norm == 0:
                        continue
                    a_sims.append(float(np.dot(na, ea) / (na_norm * ea_norm)))
            max_a = float(np.max(a_sims)) if a_sims else 0.0
            max_a_sims[existing_subj] = round(max_a, 4)




            is_pair_risk = (max_a >= self.app_risk_thresh) or (max_g >= 0.92 and max_a >= 0.55)

            if is_pair_risk:
                flagged_confusions.append({
                    "confusable_with": existing_subj,
                    "max_gait_sim": round(max_g, 4),
                    "max_app_sim": round(max_a, 4),
                    "reason": f"Cross-similarity exceeds dual-modal co-risk gate (Gait {max_g:.4f} >= 0.92 & App {max_a:.4f} >= 0.55, or App {max_a:.4f} >= {self.app_risk_thresh})",
                })

        is_risk = len(flagged_confusions) > 0
        return {
            "subject": new_subject,
            "confusion_risk_detected": is_risk,
            "flagged_pairs": flagged_confusions,
            "max_gait_similarities": max_g_sims,
            "max_app_similarities": max_a_sims,
            "recommended_action": "AUTO_ADD_TO_CONFUSION_SAFEGUARD_GROUP" if is_risk else "AUTO_CONFIRM_ENABLED",
        }
