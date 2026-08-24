"""
Lightweight Deployment Startup Validator for ARGUS AI Pipeline.

Executes fast pre-flight validation before launching real-time video or camera
streams. Reuses shared validators without modifying thresholds, loading cameras,
or altering models.
"""

from pathlib import Path
from typing import Any

from deployment.runtime_manifest import get_runtime_manifest
from models.inference.backend import BackendValidator, get_inference_backend
from monitoring.logging_config import get_logger
from storage.vector_store import validate_gallery_files
from utils.config_validator import ConfigValidator, sanitize_rtsp_url


class StartupValidationError(RuntimeError):
    """Exception raised when pre-pipeline deployment validation fails."""

    def __init__(self, blocking_issues: list[str]) -> None:
        self.blocking_issues = blocking_issues
        message = (
            "Pipeline startup blocked due to deployment defects:\n  - "
            + "\n  - ".join(blocking_issues)
        )
        super().__init__(message)


class DeploymentStartupValidator:
    """Perform deployment pre-flight checks before recognition startup."""

    STATUS_READY = "READY_FOR_CONTROLLED_CCTV_TESTING"
    STATUS_READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    STATUS_NOT_READY = "NOT_READY"
    STATUS_UNABLE_TO_VERIFY = "UNABLE_TO_VERIFY"

    def __init__(self, configs_dir: str = "configs") -> None:
        self.configs_dir = Path(configs_dir)
        self.config_validator = ConfigValidator(configs_dir=self.configs_dir)
        self._backend: Any | None = None

    @staticmethod
    def _sanitize_error(error: object) -> str:
        """Return a credential-sanitized error message."""

        return sanitize_rtsp_url(str(error))

    def _validate_storage_path(
        self,
        target_dir: Path,
        blocking_issues: list[str],
    ) -> None:
        """
        Verify that a runtime storage directory can be created and written.

        A temporary probe file is always removed when possible, including after
        partial failures.
        """

        test_file = target_dir / ".startup_check.tmp"

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            test_file.write_text("ok", encoding="utf-8")
        except Exception as exc:
            sanitized_error = self._sanitize_error(exc)
            blocking_issues.append(
                f"Storage path '{target_dir.as_posix()}' is not writable: "
                f"{sanitized_error}"
            )
        finally:
            try:
                test_file.unlink(missing_ok=True)
            except OSError:
                pass

    def validate_startup(
        self,
        raise_on_failure: bool = True,
        override_backend: Any | None = None,
    ) -> dict[str, Any]:
        """
        Validate deployment requirements before pipeline startup.

        Checks include:

        - Logging initialization
        - Configuration validity
        - Runtime manifest assets
        - Backend initialization and smoke testing
        - Gallery integrity
        - Output and report directory writability

        The method does not open cameras, connect to RTSP streams, start the
        recognition pipeline, or modify model, gallery, or configuration files.

        Args:
            raise_on_failure:
                Raise ``StartupValidationError`` when blocking issues exist.
            override_backend:
                Optional pre-initialized backend used primarily by tests or
                controlled startup integrations.

        Returns:
            A structured deployment readiness result.
        """

        blocking_issues: list[str] = []
        warnings: list[str] = []
        unable_to_verify: list[str] = []
        backend_metadata: dict[str, Any] = {}

        try:
            logger = get_logger("system")
            logger.debug("Startup validator verifying system log stream.")
        except Exception as exc:
            sanitized_error = self._sanitize_error(exc)
            blocking_issues.append(
                f"Logging initialization failed: {sanitized_error}"
            )

        try:
            config_results = self.config_validator.validate_all()

            for config_file, errors in config_results.items():
                for error in errors:
                    sanitized_error = self._sanitize_error(error)
                    blocking_issues.append(
                        f"Config error ({config_file}): {sanitized_error}"
                    )
        except Exception as exc:
            sanitized_error = self._sanitize_error(exc)
            blocking_issues.append(
                f"Configuration validation failed: {sanitized_error}"
            )

        try:
            manifest = get_runtime_manifest()
            manifest_result = manifest.validate_runtime_assets()

            if not manifest_result.get("valid", False):
                for missing_asset in manifest_result.get("missing", []):
                    blocking_issues.append(
                        f"Runtime manifest missing asset: {missing_asset}"
                    )

                for issue in manifest_result.get("errors", []):
                    sanitized_issue = self._sanitize_error(issue)
                    blocking_issues.append(
                        f"Runtime manifest defect: {sanitized_issue}"
                    )

            for notice in manifest_result.get("warnings", []):
                warnings.append(
                    f"Runtime manifest notice: {self._sanitize_error(notice)}"
                )
        except Exception as exc:
            sanitized_error = self._sanitize_error(exc)
            blocking_issues.append(
                f"Runtime manifest validation failed: {sanitized_error}"
            )

        if override_backend is not None:
            self._backend = override_backend

        if self._backend is None:
            try:
                self._backend = get_inference_backend()
            except Exception as exc:
                sanitized_error = self._sanitize_error(exc)
                blocking_issues.append(
                    f"Failed to initialize inference backend: {sanitized_error}"
                )

        if self._backend is not None:
            try:
                backend_config = getattr(self._backend, "config", None)
                backend_validator = BackendValidator(backend_config)

                smoke_passed = backend_validator.run_smoke_test(self._backend)

                if not smoke_passed:
                    active_backend = getattr(
                        self._backend,
                        "active_backend",
                        "unknown",
                    )
                    blocking_issues.append(
                        f"Configured backend '{active_backend}' failed smoke test"
                    )

                metadata = getattr(self._backend, "metadata", {})
                if isinstance(metadata, dict):
                    backend_metadata = dict(metadata)
                elif hasattr(metadata, "__dict__"):
                    backend_metadata = dict(vars(metadata))

                fallback_used = bool(
                    getattr(self._backend, "fallback_used", False)
                )

                if fallback_used:
                    requested_backend = getattr(
                        self._backend,
                        "requested_backend",
                        "unknown",
                    )
                    active_backend = getattr(
                        self._backend,
                        "active_backend",
                        "unknown",
                    )
                    fallback_reason = self._sanitize_error(
                        getattr(
                            self._backend,
                            "fallback_reason",
                            "No fallback reason reported",
                        )
                    )

                    warnings.append(
                        "Backend fallback active "
                        f"({requested_backend} -> {active_backend}): "
                        f"{fallback_reason}"
                    )

            except Exception as exc:
                sanitized_error = self._sanitize_error(exc)
                blocking_issues.append(
                    f"Backend verification failed: {sanitized_error}"
                )

        gallery_dir = Path("models/gallery")

        try:
            gallery_valid, gallery_error, gallery_count = (
                validate_gallery_files(
                    gallery_dir=gallery_dir,
                    expected_dim=256,
                )
            )

            if not gallery_valid:
                normalized_error = (gallery_error or "").lower()
                sanitized_error = self._sanitize_error(
                    gallery_error or "Unknown gallery validation error"
                )

                if "files missing" in normalized_error:
                    warnings.append(
                        f"Gallery state notice: {sanitized_error}"
                    )
                else:
                    blocking_issues.append(
                        f"Gallery defect: {sanitized_error}"
                    )
            elif gallery_count <= 0:
                warnings.append(
                    "Gallery validation succeeded but contains no embeddings"
                )

        except Exception as exc:
            sanitized_error = self._sanitize_error(exc)
            blocking_issues.append(
                f"Gallery validation failed: {sanitized_error}"
            )

        for directory_name in ("outputs", "outputs/reports"):
            self._validate_storage_path(
                target_dir=Path(directory_name),
                blocking_issues=blocking_issues,
            )

        unable_to_verify.append(
            "Live RTSP camera streams "
            "(network check deferred to runtime pipeline launch)"
        )

        if blocking_issues:
            status = self.STATUS_NOT_READY
            success = False
        elif warnings:
            status = self.STATUS_READY_WITH_WARNINGS
            success = True
        else:
            status = self.STATUS_READY
            success = True

        summary: dict[str, Any] = {
            "success": success,
            "status": status,
            "backend": self._backend,
            "backend_metadata": backend_metadata,
            "blocking_issues": blocking_issues,
            "warnings": warnings,
            "unable_to_verify": unable_to_verify,
        }

        if not success and raise_on_failure:
            raise StartupValidationError(blocking_issues)

        return summary

    def get_backend(self) -> Any:
        """Return the cached, initialized backend instance."""

        if self._backend is None:
            self.validate_startup(raise_on_failure=True)

        if self._backend is None:
            raise StartupValidationError(
                ["Inference backend was not initialized"]
            )

        return self._backend
