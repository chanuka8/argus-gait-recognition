"""
Authoritative Recognition Threshold Manager for ARGUS AI.

Loads, validates, and resolves semantic recognition thresholds from configuration
(configs/inference.yaml) and optional evaluation calibration metadata (threshold_calibration.json).
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class RecognitionThresholds:
    confirmed_threshold: float = 0.92
    known_threshold: float = 0.85
    verify_low: float = 0.85
    verify_high: float = 0.92
    low_confidence_low: float = 0.70
    low_confidence_high: float = 0.85
    unknown_ceiling: float = 0.70
    unknown_threshold: float = 0.70
    centroid_threshold: float = 0.85
    margin_threshold: float = 0.05
    calibrated: bool = False
    calibration_source: Optional[str] = None


class ThresholdManager:
    """Manages recognition score thresholds and calibration resolution."""

    DEFAULT_CONFIG_PATH = Path("configs/inference.yaml")

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH

    def load_thresholds(self, config_override: Optional[Dict[str, Any]] = None) -> RecognitionThresholds:
        config = config_override if config_override is not None else self._load_config()

        policy = config.get("matching_policy", {})
        open_set_cfg = config.get("open_set", {})

        known_th = float(
            open_set_cfg.get(
                "known_threshold",
                policy.get("known_threshold", policy.get("verify_low", 0.85)),
            )
        )
        unknown_th = float(
            open_set_cfg.get(
                "unknown_threshold",
                policy.get("unknown_threshold", policy.get("unknown_ceiling", 0.70)),
            )
        )

        confirmed_th = float(policy.get("confirmed_threshold", 0.92))
        verify_low = float(policy.get("verify_low", known_th))
        verify_high = float(policy.get("verify_high", confirmed_th))
        low_conf_low = float(policy.get("low_confidence_low", unknown_th))
        low_conf_high = float(policy.get("low_confidence_high", known_th))
        unknown_ceil = float(policy.get("unknown_ceiling", unknown_th))
        centroid_th = float(policy.get("centroid_threshold", known_th))
        margin_th = float(open_set_cfg.get("margin_threshold", policy.get("margin", 0.05)))

        use_calibrated = bool(policy.get("use_calibrated_threshold", False))
        calib_file = policy.get(
            "calibration_file",
            "runs/exp_001/evaluation_subject_disjoint/threshold_calibration.json",
        )
        calibrated = False
        calib_source = None

        if use_calibrated and calib_file:
            calib_path = Path(calib_file)
            if calib_path.exists():
                try:
                    with open(calib_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    val_th = meta.get("selected_threshold")
                    if isinstance(val_th, (int, float)) and 0.0 <= val_th <= 1.0:
                        known_th = float(val_th)
                        confirmed_th = max(confirmed_th, known_th)
                        verify_low = known_th
                        verify_high = max(verify_high, confirmed_th)
                        centroid_th = known_th
                        calibrated = True
                        calib_source = str(calib_path.as_posix())
                except Exception:
                    pass

        if unknown_th >= known_th:
            raise ValueError(
                f"Invalid threshold ordering: unknown_threshold ({unknown_th}) "
                f"must be strictly less than known_threshold ({known_th})"
            )

        return RecognitionThresholds(
            confirmed_threshold=confirmed_th,
            known_threshold=known_th,
            verify_low=verify_low,
            verify_high=verify_high,
            low_confidence_low=low_conf_low,
            low_confidence_high=low_conf_high,
            unknown_ceiling=unknown_ceil,
            unknown_threshold=unknown_th,
            centroid_threshold=centroid_th,
            margin_threshold=margin_th,
            calibrated=calibrated,
            calibration_source=calib_source,
        )

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
