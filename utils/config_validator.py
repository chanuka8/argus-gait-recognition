"""
Reusable Configuration Validator and Credential Sanitizer for ARGUS AI.

Validates deployment settings across YAML configurations and provides human-readable
error messages while ensuring sensitive RTSP credentials are masked in logs and reports.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union
import yaml


def sanitize_rtsp_url(text: Optional[str]) -> str:
    """
    Sanitize all RTSP credentials from string or connection URL.

    Handles:
    - rtsp://user:pass@host:port/path -> rtsp://user:***@host:port/path
    - Multiple RTSP URLs in single error message
    - Encoded characters (%40, %21, etc.)
    - Query parameters (?channel=1&stream=main)
    """
    if not text or not isinstance(text, str):
        return ""

    pattern = r"(rtsp://[^\s:@]+):([^\s@]+)@([^\s,;\)]+)"
    return re.sub(pattern, r"\1:***@\3", text, flags=re.IGNORECASE)


class ConfigValidationError(ValueError):
    """Human-readable configuration validation error suppressing traceback spam."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        message = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        super().__init__(message)


class ConfigValidator:
    """Validator for ARGUS deployment YAML configurations."""

    VALID_BACKENDS = {"pytorch", "onnxruntime", "auto"}
    VALID_DEVICES = {"cpu", "cuda", "gpu", "auto"}
    VALID_PRECISIONS = {"fp32", "fp16"}

    def __init__(self, configs_dir: Union[str, Path] = "configs") -> None:
        self.configs_dir = Path(configs_dir)

    def load_yaml(self, file_path: Union[str, Path]) -> Tuple[Optional[dict], Optional[str]]:
        """Safely load YAML file without raising unhandled exceptions."""
        path = Path(file_path)
        if not path.exists():
            return None, f"Configuration file not found: {path.as_posix()}"

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return None, f"Invalid YAML structure in {path.as_posix()}: expected dictionary"
            return data, None
        except yaml.YAMLError as e:
            return None, f"YAML syntax error in {path.as_posix()}: {e}"
        except Exception as e:
            return None, f"Error reading configuration file {path.as_posix()}: {e}"

    def validate_inference_config(self, config: dict) -> List[str]:
        """Validate inference.yaml parameters."""
        errors = []
        backend_cfg = config.get("inference_backend", {})
        if not isinstance(backend_cfg, dict):
            errors.append("inference_backend section must be a dictionary")
            return errors

        backend = str(backend_cfg.get("backend", "pytorch")).lower()
        if backend not in self.VALID_BACKENDS and backend != "tensorrt":
            errors.append(f"Invalid backend '{backend}'. Supported backends: {sorted(list(self.VALID_BACKENDS))}")

        device = str(backend_cfg.get("device", "auto")).lower()
        if device not in self.VALID_DEVICES:
            errors.append(f"Invalid device '{device}'. Allowed: {sorted(list(self.VALID_DEVICES))}")

        precision = str(backend_cfg.get("precision", "fp32")).lower()
        if precision not in self.VALID_PRECISIONS:
            errors.append(f"Invalid precision '{precision}'. Allowed: {sorted(list(self.VALID_PRECISIONS))}")

        max_batch = backend_cfg.get("max_batch_size", 1)
        if not isinstance(max_batch, int) or max_batch < 1:
            errors.append(f"max_batch_size must be a positive integer, got: {max_batch}")

        # Validate threshold parameters if present
        for key in ("evaluation_threshold", "live_threshold", "security_threshold"):
            val = config.get(key)
            if val is not None and not (isinstance(val, (int, float)) and 0.0 <= val <= 1.0):
                errors.append(f"{key} must be a float between 0.0 and 1.0, got: {val}")

        return errors

    def validate_cameras_config(self, config: dict) -> List[str]:
        """Validate cameras.yaml parameters."""
        errors = []
        cameras = config.get("cameras", {})
        if not isinstance(cameras, dict):
            errors.append("cameras section must be a dictionary")
            return errors

        for cam_key, cam in cameras.items():
            if not isinstance(cam, dict):
                errors.append(f"Camera '{cam_key}' entry must be a dictionary")
                continue

            cam_id = cam.get("id", cam_key)
            cam_type = str(cam.get("type", "rtsp")).lower()
            if cam_type not in {"rtsp", "usb", "file"}:
                errors.append(f"Camera '{cam_id}' has invalid type '{cam_type}'. Allowed: rtsp, usb, file")

            width = cam.get("width", 640)
            height = cam.get("height", 480)
            if not (isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0):
                errors.append(f"Camera '{cam_id}' invalid frame dimensions ({width}x{height})")

            url = cam.get("url")
            if url:
                sanitized = sanitize_rtsp_url(url)
                if cam_type == "rtsp" and not (url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://")):
                    errors.append(f"Camera '{cam_id}' has invalid stream URL format: '{sanitized}'")

        return errors

    def validate_system_config(self, config: dict) -> List[str]:
        """Validate system.yaml parameters."""
        errors = []

        rec_cfg = config.get("recognition", {})
        if isinstance(rec_cfg, dict):
            threshold = rec_cfg.get("threshold")
            if threshold is not None and not (isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0):
                errors.append(f"recognition.threshold must be between 0.0 and 1.0, got {threshold}")

        log_cfg = config.get("logging", {})
        if isinstance(log_cfg, dict):
            level = str(log_cfg.get("level", "INFO")).upper()
            if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
                errors.append(f"Invalid log level '{level}' in logging.level")

        return errors

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all standard configuration files in self.configs_dir."""
        results = {}

        inf_data, err = self.load_yaml(self.configs_dir / "inference.yaml")
        if err:
            results["inference.yaml"] = [err]
        elif inf_data:
            results["inference.yaml"] = self.validate_inference_config(inf_data)

        cam_data, err = self.load_yaml(self.configs_dir / "cameras.yaml")
        if err:
            results["cameras.yaml"] = [err]
        elif cam_data:
            results["cameras.yaml"] = self.validate_cameras_config(cam_data)

        sys_data, err = self.load_yaml(self.configs_dir / "system.yaml")
        if err:
            results["system.yaml"] = [err]
        elif sys_data:
            results["system.yaml"] = self.validate_system_config(sys_data)

        return results
