"""
Backend Startup Summary Formatter for ARGUS AI.

Emits concise, structured startup summaries after inference engine initialization.
Reuses authoritative backend metadata without reinitializing models or exposing credentials.
"""

from pathlib import Path
from typing import Optional

from monitoring.logging_config import get_logger

_SUMMARY_EMITTED = False


class BackendStartupSummary:
    """Formats and emits structured backend initialization summaries."""

    def __init__(
        self,
        backend,
        startup_status: str = "READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING",
        model_path: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.startup_status = startup_status

        # Resolve repository-relative model path
        if model_path:
            m_path = model_path
        elif getattr(backend, "active_backend", None) == "onnxruntime":
            m_path = str(backend.config.get("onnx_path", "models/engines/bygait_light.onnx"))
        else:
            m_path = str(backend.config.get("model_path", "runs/exp_001/best_model.pth"))

        self.model_path = Path(m_path).as_posix()
        self.logger = get_logger("system")

    def format_summary(self) -> str:
        """Format human-readable CLI summary block."""
        req = getattr(self.backend, "requested_backend", "pytorch")
        act = getattr(self.backend, "active_backend", "pytorch")
        prov = getattr(self.backend, "execution_provider", "PyTorch-CPU")
        allow_fb = str(getattr(self.backend, "allow_fallback", True)).lower()
        fb_used = str(getattr(self.backend, "fallback_used", False) or getattr(self.backend, "selection_fallback_used", False)).lower()
        fb_reason = getattr(self.backend, "fallback_reason", None) or "None"
        attempted = ", ".join(getattr(self.backend, "attempted_backends", [req]))

        lines = [
            "==================================================",
            "ARGUS Backend Startup Summary",
            "==================================================",
            f"Requested Backend : {req}",
            f"Active Backend    : {act}",
            f"Provider          : {prov}",
            f"Allow Fallback    : {allow_fb}",
            f"Fallback Used     : {fb_used}",
            f"Fallback Reason   : {fb_reason}",
            f"Attempted Engines : {attempted}",
            f"Model Path        : {self.model_path}",
            f"Startup Status    : {self.startup_status}",
            "==================================================",
        ]
        return "\n".join(lines)

    def emit(self, force: bool = False, print_cli: bool = True) -> str:
        """
        Emit backend summary to system logs and optional stdout CLI once.

        Returns formatted summary string.
        """
        global _SUMMARY_EMITTED
        summary_text = self.format_summary()

        # Check sanitization
        if "C:\\Users" in summary_text or "/home/" in summary_text:
            raise ValueError("Absolute user-home path detected in backend summary")
        if "SECRET" in summary_text.upper() and "PATTERNS" not in summary_text.upper():
            raise ValueError("Credential leak detected in backend summary")

        if not _SUMMARY_EMITTED or force:
            self.logger.info("\n" + summary_text)
            if print_cli:
                print(summary_text)
            _SUMMARY_EMITTED = True

        return summary_text


def reset_summary_emitted_flag() -> None:
    """Reset summary single-emit flag for unit testing."""
    global _SUMMARY_EMITTED
    _SUMMARY_EMITTED = False
