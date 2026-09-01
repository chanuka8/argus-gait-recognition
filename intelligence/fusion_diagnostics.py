from typing import Any

import numpy as np
from scipy.optimize import minimize


class FusionDiagnostics:
    @staticmethod
    def analyze_score_distributions(
        gait_scores: np.ndarray | list[float],
        app_scores: np.ndarray | list[float],
        labels: np.ndarray | list[int],
    ) -> dict[str, Any]:
        g = np.asarray(gait_scores, dtype=np.float64).ravel()
        a = np.asarray(app_scores, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.int32).ravel()

        pos_mask = (y == 1)
        neg_mask = (y == 0)

        return {
            "num_genuine_pairs": int(np.sum(pos_mask)),
            "num_impostor_pairs": int(np.sum(neg_mask)),
            "gait_genuine_mean": round(float(np.mean(g[pos_mask])), 4) if np.any(pos_mask) else 0.0,
            "gait_genuine_std": round(float(np.std(g[pos_mask])), 4) if np.any(pos_mask) else 0.0,
            "gait_impostor_max": round(float(np.max(g[neg_mask])), 4) if np.any(neg_mask) else 0.0,
            "app_genuine_mean": round(float(np.mean(a[pos_mask])), 4) if np.any(pos_mask) else 0.0,
            "app_genuine_std": round(float(np.std(a[pos_mask])), 4) if np.any(pos_mask) else 0.0,
            "app_impostor_max": round(float(np.max(a[neg_mask])), 4) if np.any(neg_mask) else 0.0,
        }

    @staticmethod
    def fit_auc_optimal_weights(
        same_g: np.ndarray | list[float],
        same_a: np.ndarray | list[float],
        diff_g: np.ndarray | list[float],
        diff_a: np.ndarray | list[float],
        temp: float = 0.02,
    ) -> tuple[float, float, float]:
        sg = np.asarray(same_g, dtype=np.float64)
        sa = np.asarray(same_a, dtype=np.float64)
        dg = np.asarray(diff_g, dtype=np.float64)
        da = np.asarray(diff_a, dtype=np.float64)

        def loss_fn(params: np.ndarray) -> float:
            wg, wa, wi = params[0], params[1], params[2]
            s_pos = wg * sg + wa * sa + wi * (sg * sa)
            s_neg = wg * dg + wa * da + wi * (dg * da)
            diff_mat = np.clip(s_neg[:, None] - s_pos[None, :], -10.0, 10.0)
            loss = np.mean(1.0 / (1.0 + np.exp(-diff_mat / temp)))
            return float(loss) + 0.01 * (wg**2 + wa**2 + wi**2)

        res = minimize(loss_fn, [0.90, 0.10, 0.0], method="Nelder-Mead")
        scale = abs(res.x[0]) + abs(res.x[1]) + abs(res.x[2]) + 1e-8
        return float(res.x[0] / scale), float(res.x[1] / scale), float(res.x[2] / scale)
