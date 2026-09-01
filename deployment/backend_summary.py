from pathlib import Path

from monitoring.logging_config import get_logger

_SUMMARY_EMITTED = False


class BackendStartupSummary:
    def __init__(
        self,
        backend,
        startup_status: str = "READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING",
        model_path: str | None = None,
    ) -> None:
        self.backend = backend
        self.startup_status = startup_status

        if model_path:
            m_path = model_path
        elif getattr(backend, "active_backend", None) == "onnxruntime":
            cfg = getattr(backend, "config", {})
            m_path = (
                str(cfg.get("onnx_path", "models/engines/bygait_light.onnx"))
                if isinstance(cfg, dict)
                else "models/engines/bygait_light.onnx"
            )
        else:
            cfg = getattr(backend, "config", {})
            m_path = (
                str(cfg.get("model_path", "runs/exp_001/best_model.pth"))
                if isinstance(cfg, dict)
                else "runs/exp_001/best_model.pth"
            )

        self.model_path = Path(m_path).as_posix()
        self.logger = get_logger("system")

    def format_summary(self) -> str:
        req = getattr(self.backend, "requested_backend", "pytorch")
        act = getattr(self.backend, "active_backend", "pytorch")
        prov = getattr(self.backend, "execution_provider", "PyTorch-CPU")
        allow_fb = str(getattr(self.backend, "allow_fallback", True)).lower()
        fb_used = str(
            getattr(self.backend, "fallback_used", False) or getattr(self.backend, "selection_fallback_used", False)
        ).lower()
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
        global _SUMMARY_EMITTED
        summary_text = self.format_summary()

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
    global _SUMMARY_EMITTED
    _SUMMARY_EMITTED = False
