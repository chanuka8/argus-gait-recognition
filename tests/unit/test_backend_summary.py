"""
Unit tests for BackendStartupSummary formatter and single-emit enforcement.
"""

from unittest.mock import MagicMock

from deployment.backend_summary import BackendStartupSummary, reset_summary_emitted_flag


def test_backend_summary_emitted_once(capsys):
    reset_summary_emitted_flag()
    mock_backend = MagicMock()
    mock_backend.requested_backend = "pytorch"
    mock_backend.active_backend = "pytorch"
    mock_backend.execution_provider = "PyTorch-CPU"
    mock_backend.allow_fallback = True
    mock_backend.fallback_used = False
    mock_backend.attempted_backends = ["pytorch"]
    mock_backend.config = {"model_path": "runs/exp_001/best_model.pth"}

    summary_obj = BackendStartupSummary(mock_backend, startup_status="READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING")

    text1 = summary_obj.emit(print_cli=True)
    captured1 = capsys.readouterr().out
    assert "ARGUS Backend Startup Summary" in captured1
    assert "Active Backend    : pytorch" in text1

    summary_obj.emit(print_cli=True)
    captured2 = capsys.readouterr().out
    assert captured2 == ""


def test_backend_summary_pytorch_fallback_formatting():
    reset_summary_emitted_flag()
    mock_backend = MagicMock()
    mock_backend.requested_backend = "onnxruntime"
    mock_backend.active_backend = "pytorch"
    mock_backend.execution_provider = "PyTorch-CPU"
    mock_backend.allow_fallback = True
    mock_backend.fallback_used = True
    mock_backend.selection_fallback_used = True
    mock_backend.fallback_reason = "onnxruntime package missing"
    mock_backend.attempted_backends = ["onnxruntime", "pytorch"]
    mock_backend.config = {"model_path": "runs/exp_001/best_model.pth"}

    summary_obj = BackendStartupSummary(mock_backend, startup_status="READY_WITH_WARNINGS")
    text = summary_obj.format_summary()

    assert "Requested Backend : onnxruntime" in text
    assert "Active Backend    : pytorch" in text
    assert "Fallback Used     : true" in text
    assert "Fallback Reason   : onnxruntime package missing" in text
    assert "Startup Status    : READY_WITH_WARNINGS" in text


def test_backend_summary_no_secrets_or_absolute_user_paths():
    reset_summary_emitted_flag()
    mock_backend = MagicMock()
    mock_backend.requested_backend = "pytorch"
    mock_backend.active_backend = "pytorch"
    mock_backend.execution_provider = "PyTorch-CPU"
    mock_backend.config = {"model_path": "models/engines/bygait_light.onnx"}

    summary_obj = BackendStartupSummary(mock_backend)
    text = summary_obj.format_summary()

    assert "C:\\Users" not in text
    assert "/home/" not in text
    assert "SECRET" not in text.upper()
