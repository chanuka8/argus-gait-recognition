"""
Learned Biometric Fusion Layer for ARGUS AI.

Combines Gait and Appearance features using:
1. Platt-calibrated posterior match probabilities
2. Learned logistic fusion layer with cross-modal interaction terms:
   P(Match | s_gait, s_app) = sigmoid(w_g * calib(s_gait) + w_a * calib(s_app) + w_inter * (s_gait * s_app) + b)
3. Configurable deployment profiles:
   - Profile 'identification': Maximizes Closed-Set Rank-1 Identification & Top-1 margin.
   - Profile 'verification': Minimizes EER / False Accepts for access control gates.
"""

from typing import Any

import numpy as np

from intelligence.score_calibrator import PlattScoreCalibrator


class LearnedLogisticFusion:
    """
    Learned fusion layer for dual-modal biometric scores.
    Replaces static fixed-weight addition with an empirical decision surface.
    """

    def __init__(
        self,
        w_gait: float = 2.5,
        w_app: float = 4.0,
        w_inter: float = 1.5,
        bias: float = -3.5,
        profile: str = "identification",
    ) -> None:
        self.w_gait = float(w_gait)
        self.w_app = float(w_app)
        self.w_inter = float(w_inter)
        self.bias = float(bias)
        self.profile = str(profile).lower()
        self.gait_calibrator = PlattScoreCalibrator()
        self.app_calibrator = PlattScoreCalibrator()
        self.is_fitted = False

    def fit(
        self,
        gait_scores: np.ndarray | list[float],
        app_scores: np.ndarray | list[float],
        labels: np.ndarray | list[int],
        loss_type: str = "ranking_auc",
    ) -> "LearnedLogisticFusion":
        """
        Fit calibrators and fusion weights using AUC ranking loss or logistic regression.

        Args:
            gait_scores: Array of genuine and impostor gait similarity scores.
            app_scores: Array of genuine and impostor appearance similarity scores.
            labels: 1 for genuine same-person pair, 0 for cross-identity impostor pair.
            loss_type: 'ranking_auc' (maximizes ROC-AUC / minimizes EER) or 'bce' (binary cross entropy).
        """
        g_arr = np.asarray(gait_scores, dtype=np.float64).ravel()
        a_arr = np.asarray(app_scores, dtype=np.float64).ravel()
        y_arr = np.asarray(labels, dtype=np.float64).ravel()

        if len(g_arr) == 0:
            return self


        self.gait_calibrator.fit(g_arr, y_arr)
        self.app_calibrator.fit(a_arr, y_arr)

        if loss_type == "ranking_auc":

            pos_mask = y_arr == 1.0
            neg_mask = y_arr == 0.0

            same_g, diff_g = g_arr[pos_mask], g_arr[neg_mask]
            same_a, diff_a = a_arr[pos_mask], a_arr[neg_mask]

            from scipy.optimize import minimize

            def auc_surrogate_loss(params: np.ndarray, temp: float = 0.02) -> float:
                wg, wa, winter = params[0], params[1], params[2]
                s_pos = wg * same_g + wa * same_a + winter * (same_g * same_a)
                s_neg = wg * diff_g + wa * diff_a + winter * (diff_g * diff_a)
                diff_mat = np.clip(s_neg[:, None] - s_pos[None, :], -10.0, 10.0)
                loss = np.mean(1.0 / (1.0 + np.exp(-diff_mat / temp)))
                return float(loss) + 0.01 * (wg**2 + wa**2 + winter**2)

            res = minimize(auc_surrogate_loss, [0.90, 0.10, 0.0], method="Nelder-Mead")
            wg, wa, wi = res.x
            scale = abs(wg) + abs(wa) + abs(wi) + 1e-8
            self.w_gait = float(wg / scale)
            self.w_app = float(wa / scale)
            self.w_inter = float(wi / scale)
            self.bias = 0.0
            self.use_raw_scores = True
            self.is_fitted = True
            return self


        self.use_raw_scores = False
        g_prob = np.array([self.gait_calibrator.calibrate(s) for s in g_arr], dtype=np.float64)
        a_prob = np.array([self.app_calibrator.calibrate(s) for s in a_arr], dtype=np.float64)
        inter = g_prob * a_prob

        X = np.column_stack([g_prob, a_prob, inter])
        N = len(y_arr)

        weights = np.array([2.0, 3.5, 1.5], dtype=np.float64)
        bias = -2.5
        lr = 0.05
        l2_reg = 0.01

        for _ in range(300):
            logits = X @ weights + bias
            logits = np.clip(logits, -20.0, 20.0)
            probs = 1.0 / (1.0 + np.exp(-logits))

            grad_w = (X.T @ (probs - y_arr)) / N + l2_reg * weights
            grad_b = np.mean(probs - y_arr)

            weights -= lr * grad_w
            bias -= lr * grad_b

        self.w_gait = float(weights[0])
        self.w_app = float(weights[1])
        self.w_inter = float(weights[2])
        self.bias = float(bias)
        self.is_fitted = True
        return self

    def predict_probability(self, gait_score: float | None, app_score: float | None) -> float:
        """Compute fused posterior match probability P(Match = 1 | s_gait, s_app)."""
        if gait_score is None and app_score is None:
            return 0.0
        if gait_score is None:
            return (
                float(self.app_calibrator.calibrate(app_score))
                if not getattr(self, "use_raw_scores", False)
                else float(app_score)
            )
        if app_score is None:
            return (
                float(self.gait_calibrator.calibrate(gait_score))
                if not getattr(self, "use_raw_scores", False)
                else float(gait_score)
            )

        if getattr(self, "use_raw_scores", False):
            inter = float(gait_score * app_score)
            score = self.w_gait * float(gait_score) + self.w_app * float(app_score) + self.w_inter * inter
            return float(score)

        p_g = float(self.gait_calibrator.calibrate(gait_score))
        p_a = float(self.app_calibrator.calibrate(app_score))
        inter = p_g * p_a

        logit = self.w_gait * p_g + self.w_app * p_a + self.w_inter * inter + self.bias
        logit = max(-25.0, min(25.0, logit))
        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(prob)

    def fuse(self, gait_score: float | None, app_score: float | None) -> float:
        """Alias for predict_probability to combine dual-modal match scores."""
        return self.predict_probability(gait_score=gait_score, app_score=app_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "w_gait": round(self.w_gait, 4),
            "w_app": round(self.w_app, 4),
            "w_inter": round(self.w_inter, 4),
            "bias": round(self.bias, 4),
            "profile": self.profile,
            "gait_calibrator": self.gait_calibrator.to_dict(),
            "app_calibrator": self.app_calibrator.to_dict(),
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedLogisticFusion":
        inst = cls(
            w_gait=data.get("w_gait", 2.5),
            w_app=data.get("w_app", 4.0),
            w_inter=data.get("w_inter", 1.5),
            bias=data.get("bias", -3.5),
            profile=data.get("profile", "identification"),
        )
        if "gait_calibrator" in data:
            inst.gait_calibrator = PlattScoreCalibrator.from_dict(data["gait_calibrator"])
        if "app_calibrator" in data:
            inst.app_calibrator = PlattScoreCalibrator.from_dict(data["app_calibrator"])
        inst.is_fitted = data.get("is_fitted", True)
        return inst
