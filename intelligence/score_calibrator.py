"""
Statistical Score Calibration for Biometric Fusion (Platt Scaling & Isotonic Calibration).

Converts raw cosine similarities (Gait and Appearance) into calibrated posterior match probabilities:
P(Match = 1 | score) in [0.0, 1.0].
"""

from typing import Any

import numpy as np


class PlattScoreCalibrator:
    """
    Parametric Platt Scaling score calibrator.
    Fits a logistic sigmoid mapping: P(y=1 | s) = 1 / (1 + exp(-(A * s + B))).
    """

    def __init__(self, A: float = 10.0, B: float = -5.0) -> None:
        self.A = float(A)
        self.B = float(B)
        self.is_fitted = False

    def fit(self, scores: np.ndarray | list[float], labels: np.ndarray | list[int]) -> "PlattScoreCalibrator":
        """
        Fit Platt scaling parameters (A, B) via maximum likelihood logistic regression.
        
        Args:
            scores: Array of raw similarity scores.
            labels: Binary array (1 for genuine same-person, 0 for impostor).
        """
        s_arr = np.asarray(scores, dtype=np.float64).ravel()
        y_arr = np.asarray(labels, dtype=np.float64).ravel()

        if len(s_arr) == 0 or len(y_arr) == 0:
            return self

        # Target smoothing to prevent overfitting on small sample sizes (Platt 1999 standard)
        n_pos = np.sum(y_arr == 1.0)
        n_neg = np.sum(y_arr == 0.0)
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        targets = np.where(y_arr == 1.0, t_pos, t_neg)

        # Simple Newton-Raphson or gradient descent for (A, B)
        a, b = 5.0, -2.5
        lr = 0.05
        for _ in range(200):
            logits = a * s_arr + b
            logits = np.clip(logits, -20.0, 20.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            
            grad_a = np.mean((probs - targets) * s_arr)
            grad_b = np.mean(probs - targets)
            
            a -= lr * grad_a
            b -= lr * grad_b

        self.A = float(a)
        self.B = float(b)
        self.is_fitted = True
        return self

    def calibrate(self, score: float | np.ndarray) -> float | np.ndarray:
        """Transform raw cosine similarity into calibrated probability."""
        if score is None:
            return None
        s = np.asarray(score, dtype=np.float64)
        logits = self.A * s + self.B
        logits = np.clip(logits, -25.0, 25.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        if np.isscalar(score) or s.ndim == 0:
            return float(prob)
        return prob.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"A": round(self.A, 6), "B": round(self.B, 6), "is_fitted": self.is_fitted}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlattScoreCalibrator":
        inst = cls(A=data.get("A", 10.0), B=data.get("B", -5.0))
        inst.is_fitted = data.get("is_fitted", True)
        return inst
