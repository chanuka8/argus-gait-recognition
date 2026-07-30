"""
Lightweight Deployment Startup Validator for ARGUS AI Pipeline.

Executes fast pre-flight validation before launching real-time video or camera streams.
Reuses shared validators without modifying thresholds, loading cameras, or altering models.
"""

from pathlib import Path
from typing import Dict, List, Union

from models.inference.backend import BackendValidator, get_inference_backend
from storage.vector_store import validate_gallery_files
from utils.config_validator import ConfigValidator, sanitize_rtsp_url



class StartupValidationError(RuntimeError):
    """Exception raised when pre-pipeline deployment validation fails."""

    def __init__(self, blocking_issues: List[str]) -> None:
        self.blocking_issues = blocking_issues
        msg = "Pipeline startup blocked due to deployment defects:\n  - " + "\n  - ".join(blocking_issues)
        super().__init__(msg)


class DeploymentStartupValidator:
    """Pre-flight validator for live/video recognition pipeline startup."""

    def __init__(self, configs_dir: str = "configs") -> None:
        self.configs_dir = Path(configs_dir)
        self.config_validator = ConfigValidator(configs_dir=self.configs_dir)
        self._backend = None

    def validate_startup(self, raise_on_failure: bool = True) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate backend, config, model files, gallery, and storage writability.

        Returns summary dictionary containing status, backend instance, and list of blocking issues.
        """
        blocking_issues: List[str] = []
        warnings: List[str] = []

        # 1. Configuration Validation
        cfg_results = self.config_validator.validate_all()
        for cfg_file, errors in cfg_results.items():
            for err in errors:
                sanitized = sanitize_rtsp_url(err)
                blocking_issues.append(f"Config error ({cfg_file}): {sanitized}")

        # 2. Inference Backend & Model Files (Startup-once initialization)
        if self._backend is None:
            try:
                self._backend = get_inference_backend()
                backend_validator = BackendValidator(self._backend.config)
                smoke_pass = backend_validator.run_smoke_test(self._backend)
                if not smoke_pass:
                    blocking_issues.append(f"Configured backend '{self._backend.active_backend}' failed smoke test")
            except Exception as e:
                blocking_issues.append(f"Failed to initialize inference backend: {e}")

        # 3. Gallery Validation
        g_dir = Path("models/gallery")
        g_valid, g_err, g_count = validate_gallery_files(gallery_dir=g_dir, expected_dim=256)
        if not g_valid and "files missing" not in (g_err or "").lower():
            blocking_issues.append(f"Gallery defect: {g_err}")
        elif not g_valid:
            warnings.append(f"Gallery state notice: {g_err}")

        # 4. Output Storage Writability
        out_dir = Path("outputs/reports")
        out_dir.mkdir(parents=True, exist_ok=True)

        test_file = out_dir / ".startup_check.tmp"
        try:
            test_file.write_text("ok", encoding="utf-8")
            if test_file.exists():
                test_file.unlink()
        except Exception as e:
            blocking_issues.append(f"Output directory '{out_dir.as_posix()}' is not writable: {e}")

        success = len(blocking_issues) == 0

        summary = {
            "success": success,
            "backend": self._backend,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
        }

        if not success and raise_on_failure:
            raise StartupValidationError(blocking_issues)

        return summary

    def get_backend(self):
        """Return cached initialized backend instance."""
        if self._backend is None:
            self.validate_startup(raise_on_failure=True)
        return self._backend

