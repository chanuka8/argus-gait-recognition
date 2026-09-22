"""Regression test suite for CWD-independent runtime path resolution.

Verifies that all application assets (configs, models, manifests, galleries)
and runtime writable outputs correctly resolve against the canonical application
root and runtime root when the process is executed from an arbitrary external working directory.

Enforces zero-network policy:
Any socket.connect or socket.create_connection attempt must raise AssertionError and fail the test.
"""

from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.paths import (
    get_app_root,
    get_config_path,
    get_model_path,
    get_runtime_root,
    resolve_app_path,
    resolve_runtime_path,
    set_app_root,
    set_runtime_root,
)


@pytest.fixture(autouse=True)
def guard_zero_network(monkeypatch):
    """Enforce strict zero-network invariant across all path tests."""
    def blocked_connect(*args, **kwargs):
        raise AssertionError("TEST ISOLATION FAILURE: Outbound socket.connect attempted!")

    def blocked_create_connection(*args, **kwargs):
        raise AssertionError("TEST ISOLATION FAILURE: Outbound socket.create_connection attempted!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket, "create_connection", blocked_create_connection)


@pytest.fixture(autouse=True)
def clean_path_overrides():
    """Ensure clean override state before and after each test."""
    set_app_root(None)
    set_runtime_root(None)
    yield
    set_app_root(None)
    set_runtime_root(None)


class TestCorePathsResolver:
    """Validate core/paths.py resolution semantics and precedence."""

    def test_get_app_root_stable_after_chdir(self, monkeypatch, tmp_path: Path):
        """1. get_app_root() is deterministic and unchanged after chdir to external temp dir."""
        orig_root = get_app_root()
        assert orig_root.is_dir()
        assert (orig_root / "main.py").is_file()

        monkeypatch.chdir(tmp_path)
        assert Path.cwd() == tmp_path.resolve()
        assert get_app_root() == orig_root
        assert get_app_root() != tmp_path.resolve()

    def test_argus_app_root_env_override(self, monkeypatch, tmp_path: Path):
        """2. ARGUS_APP_ROOT environment variable overrides the application root."""
        custom_root = tmp_path / "custom_app"
        custom_root.mkdir()
        monkeypatch.setenv("ARGUS_APP_ROOT", str(custom_root))

        assert get_app_root() == custom_root.resolve()
        resolved = resolve_app_path("configs/base.yaml")
        assert resolved == (custom_root / "configs/base.yaml").resolve()

    def test_argus_runtime_root_env_override(self, monkeypatch, tmp_path: Path):
        """3. ARGUS_RUNTIME_ROOT environment variable overrides the runtime writable root."""
        custom_runtime = tmp_path / "custom_runtime"
        custom_runtime.mkdir()
        monkeypatch.setenv("ARGUS_RUNTIME_ROOT", str(custom_runtime))

        assert get_runtime_root() == custom_runtime.resolve()
        resolved = resolve_runtime_path("outputs/reports")
        assert resolved == (custom_runtime / "outputs/reports").resolve()

    def test_absolute_paths_remain_unchanged(self, tmp_path: Path):
        """4. Absolute paths supplied by operators remain completely unchanged."""
        abs_path = (tmp_path / "custom/operator/config.yaml").resolve()
        assert resolve_app_path(abs_path) == abs_path
        assert resolve_runtime_path(abs_path) == abs_path
        assert get_config_path(abs_path) == abs_path
        assert get_model_path(abs_path) == abs_path

    def test_resolver_never_mutates_cwd(self, monkeypatch, tmp_path: Path):
        """5. Path resolution helpers never call os.chdir() or mutate process CWD."""
        monkeypatch.chdir(tmp_path)
        initial_cwd = Path.cwd()

        _ = resolve_app_path("configs/inference.yaml")
        _ = resolve_runtime_path("outputs/reports")
        _ = get_config_path("system.yaml")
        _ = get_model_path("ml_platform/models/model_store/weights/yolov8n.pt")

        assert Path.cwd() == initial_cwd


class TestExternalCwdSubsystemResolution:
    """Validate that production subsystems resolve correctly from external process CWD."""

    def test_runtime_manifest_outside_repo_cwd(self, monkeypatch, tmp_path: Path):
        """6. RuntimeManifest validates cleanly from external CWD without false missing assets."""
        monkeypatch.chdir(tmp_path)
        from ops.deployment.runtime_manifest import get_runtime_manifest

        manifest = get_runtime_manifest()
        result = manifest.validate_runtime_assets()

        assert result["valid"] is True
        assert len(result["missing"]) == 0
        assert "main.py" in result["checked"]
        assert "VERSION" in result["checked"]

    def test_core_config_loads_app_root_base_yaml(self, monkeypatch, tmp_path: Path):
        """7. core.config.Config loads configs/base.yaml from app root when CWD is external."""
        monkeypatch.chdir(tmp_path)
        from app.core.config import Config

        cfg = Config()
        assert cfg.base_config != {}
        assert "project_name" in cfg.base_config

    def test_logging_loads_app_root_system_yaml(self, monkeypatch, tmp_path: Path):
        """8 & 9. logging loads configs/system.yaml from app root and resolves log_dir to runtime root."""
        monkeypatch.chdir(tmp_path)
        from app.monitoring.logging_config import _load_logging_config

        cfg = _load_logging_config()
        assert isinstance(cfg, dict)
        assert "log_dir" in cfg
        resolved_log_dir = resolve_runtime_path(cfg["log_dir"])
        assert resolved_log_dir == (get_runtime_root() / "outputs/logs/system").resolve()

    def test_inference_config_loads_from_app_root(self, monkeypatch, tmp_path: Path):
        """10. load_inference_backend_config loads configs/inference.yaml from app root."""
        monkeypatch.chdir(tmp_path)
        from ml_platform.models.inference.backend import load_inference_backend_config

        cfg = load_inference_backend_config()
        assert isinstance(cfg, dict)
        assert cfg.get("backend") is not None
        assert "model_path" in cfg

    def test_detector_config_loads_from_app_root(self, monkeypatch, tmp_path: Path):
        """11. PersonDetector._load_config resolves configs/detection.yaml from app root."""
        monkeypatch.chdir(tmp_path)
        from app.pipeline.detection.person_detector import PersonDetector

        cfg = PersonDetector._load_config("configs/detection.yaml")
        assert isinstance(cfg, dict)
        assert "model_path" in cfg
        assert cfg.get("model_path") == "ml_platform/models/model_store/weights/yolov8n.pt"

    def test_pytorch_relative_model_resolves_to_app_root(self, monkeypatch, tmp_path: Path):
        """12. PyTorchBackend resolves model path against app root."""
        monkeypatch.chdir(tmp_path)
        from ml_platform.models.inference.pytorch_backend import PyTorchBackend

        backend = PyTorchBackend(config={"model_path": "runs/exp_001/best_model.pth"})
        assert backend.model_path == (get_app_root() / "runs/exp_001/best_model.pth").resolve()
        assert backend.model_path != (tmp_path / "runs/exp_001/best_model.pth").resolve()

    def test_onnx_relative_engine_resolves_to_app_root(self, monkeypatch, tmp_path: Path):
        """13. ONNXBackend resolves onnx_path against app root."""
        monkeypatch.chdir(tmp_path)
        from ml_platform.models.inference.onnx_backend import ONNXBackend

        backend = ONNXBackend(
            config={
                "onnx_path": "ml_platform/models/model_store/engines/bygait_light.onnx",
                "allow_fallback": True,
                "warmup_iterations": 0,
            }
        )
        assert backend.onnx_path == (get_app_root() / "ml_platform/models/model_store/engines/bygait_light.onnx").resolve()

    def test_tensorrt_relative_engine_resolves_to_app_root(self, monkeypatch, tmp_path: Path):
        """14. TensorRTBackend resolves engine_path against app root."""
        monkeypatch.chdir(tmp_path)
        from ml_platform.models.inference.tensorrt_backend import TensorRTBackend

        backend = TensorRTBackend(config={"engine_path": "ml_platform/models/model_store/engines/bygait_light_fp16.engine", "allow_fallback": True})
        assert backend.engine_path == (get_app_root() / "ml_platform/models/model_store/engines/bygait_light_fp16.engine").resolve()

    def test_person_detector_local_model_zero_download_on_external_cwd(self, monkeypatch, tmp_path: Path):
        """15 & 16. Local YOLO asset under app root is resolved; ZERO download/network attempts."""
        import ultralytics

        from app.pipeline.detection.person_detector import PersonDetector

        fake_app_root = tmp_path / "fake_app"
        external_cwd = tmp_path / "external_cwd"
        fake_app_root.mkdir()
        external_cwd.mkdir()

        # Synthetic detection config
        configs_dir = fake_app_root / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        detection_yaml_content = (
            "model_path: models/model_store/weights/yolov8n.pt\n"
            "confidence: 0.4\n"
            "iou_threshold: 0.45\n"
            "classes:\n"
            "  - 0\n"
            "device: cpu\n"
            "img_size: 640\n"
        )
        (configs_dir / "detection.yaml").write_text(detection_yaml_content, encoding="utf-8")

        # Synthetic placeholder weight (tiny placeholder, not real model)
        weights_dir = fake_app_root / "models" / "model_store" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        placeholder_weight = weights_dir / "yolov8n.pt"
        placeholder_weight.write_bytes(b"synthetic_placeholder_yolov8n_weight")

        # Process CWD and App Root isolation
        monkeypatch.setenv("ARGUS_APP_ROOT", str(fake_app_root))
        monkeypatch.chdir(external_cwd)
        assert get_app_root() == fake_app_root.resolve()
        assert Path.cwd() == external_cwd.resolve()
        assert get_app_root() != Path.cwd()

        # Isolate model-integrity loading for synthetic weight
        monkeypatch.setattr(
            "app.security_layer.model_integrity.verify_model",
            lambda model_path, *args, **kwargs: Path(model_path).resolve(),
        )

        # Mock YOLO model loading to verify path resolution without model parsing / network
        captured_yolo_paths: list[str] = []

        class FakeYOLO:
            def __init__(self, model: str | Path, *args, **kwargs):
                captured_yolo_paths.append(str(model))

            def to(self, *args, **kwargs):
                return self

        monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)

        mock_device_mgr = MagicMock()
        mock_device_mgr.resolve_component_device.return_value = "cpu"
        monkeypatch.setattr(
            "app.pipeline.detection.person_detector.DeviceManager.get_instance",
            lambda *args, **kwargs: mock_device_mgr,
        )

        # Initialize PersonDetector from external CWD
        detector = PersonDetector()
        assert detector.model is not None

        # Assertions:
        # 1. Exactly one model instantiation occurred
        assert len(captured_yolo_paths) == 1
        resolved_yolo_path = Path(captured_yolo_paths[0]).resolve()

        # 2. YOLO received exactly fake_app_root/models/model_store/weights/yolov8n.pt as resolved absolute path
        expected_model_path = placeholder_weight.resolve()
        assert resolved_yolo_path == expected_model_path

        # 3. Explicitly assert fallback download path ("yolov8n.pt") was not selected
        assert captured_yolo_paths[0] != "yolov8n.pt"
        assert "yolov8n.pt" not in captured_yolo_paths

        # 4. Confirms zero files / artifacts created in external temp CWD
        temp_files = list(external_cwd.iterdir())
        assert len(temp_files) == 0, f"Unexpected artifacts created in external CWD: {temp_files}"

    def test_model_manifest_and_sig_resolve_to_app_root(self, monkeypatch, tmp_path: Path):
        """17, 18, 19. ModelVerifier default paths resolve against app root."""
        monkeypatch.chdir(tmp_path)
        from app.security_layer.model_integrity import ModelVerifier

        verifier = ModelVerifier()
        assert verifier.default_manifest_path == (get_app_root() / "ml_platform/models/model_manifest.json").resolve()
        assert verifier.default_signature_path == (get_app_root() / "ml_platform/models/model_manifest.sig").resolve()

    def test_gallery_default_does_not_follow_process_cwd(self, monkeypatch, tmp_path: Path):
        """20. VectorStore and validate_gallery_files anchor to app root, not external CWD."""
        monkeypatch.chdir(tmp_path)
        from app.storage.vector_store import VectorStore, validate_gallery_files

        store = VectorStore()
        assert store.gallery_dir == (get_app_root() / "ml_platform/models/galleries/gallery").resolve()
        assert not (tmp_path / "models").exists()

        _is_valid, _err, _count = validate_gallery_files()
        assert not (tmp_path / "models").exists()

    def test_outputs_reports_does_not_follow_process_cwd(self, monkeypatch, tmp_path: Path):
        """21. Reports and diagnostic outputs anchor to runtime root, not external CWD."""
        monkeypatch.chdir(tmp_path)
        resolved_reports = resolve_runtime_path("outputs/reports")
        assert resolved_reports == (get_runtime_root() / "outputs/reports").resolve()
        assert resolved_reports != (tmp_path / "outputs/reports").resolve()

    def test_health_check_does_not_report_missing_because_of_cwd(self, monkeypatch, tmp_path: Path):
        """22. core.health_check.HealthCheck passes from external CWD."""
        monkeypatch.chdir(tmp_path)
        from app.core.health_check import HealthCheck

        hc = HealthCheck()
        res = hc.run()
        assert res["healthy"] is True
        assert len(res["missing_directories"]) == 0
        assert len(res["missing_files"]) == 0

    def test_readiness_reporter_does_not_report_missing_because_of_cwd(self, monkeypatch, tmp_path: Path):
        """23. DeploymentReadinessReporter finds models and configs from external CWD."""
        monkeypatch.chdir(tmp_path)
        from ops.deployment.readiness_reporter import DeploymentReadinessReporter

        reporter = DeploymentReadinessReporter()
        report = reporter.evaluate_readiness()
        assert report["configuration_readiness"]["status"] == "READY"
        assert report["camera_configuration_readiness"]["camera_yaml_exists"] is True

    def test_startup_validator_zero_false_missing_assets_from_external_cwd(self, monkeypatch, tmp_path: Path):
        """24 & 25. DeploymentStartupValidator has zero missing runtime manifest assets from external CWD.
        U4 plaintext camera transport correctly blocks camera_01 and camera_02 under strict transport security.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "true")
        from ops.deployment.startup_validator import DeploymentStartupValidator

        mock_backend = MagicMock()
        mock_backend.active_backend = "pytorch"
        mock_backend.requested_backend = "pytorch"
        mock_backend.fallback_used = False
        mock_backend.config = {}
        mock_backend.predict.return_value = np.zeros((1, 256), dtype=np.float32)
        mock_backend.metadata = {}

        validator = DeploymentStartupValidator()
        summary = validator.validate_startup(raise_on_failure=False, override_backend=mock_backend)

        # Confirm ZERO manifest missing asset defects
        manifest_defects = [i for i in summary["blocking_issues"] if "manifest" in i.lower()]
        assert len(manifest_defects) == 0, f"Unexpected manifest defects: {manifest_defects}"

        # U4 plaintext camera rejection is preserved
        assert any("camera_01" in i for i in summary["blocking_issues"])
        assert any("camera_02" in i for i in summary["blocking_issues"])

        # Confirm no artifacts were created in temp CWD
        assert not (tmp_path / "outputs").exists()
        assert not (tmp_path / "models").exists()
        assert not (tmp_path / "configs").exists()
