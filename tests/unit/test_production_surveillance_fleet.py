import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from intelligence.dual_modal_fusion import DualModalFusion
from intelligence.operational_embedding_collector import (
    OperationalEmbeddingCollector,
)
from intelligence.track_identity_aggregator import TrackIdentityAggregator
from services.camera_manager import CameraManager
from services.camera_worker import CameraWorker
from streaming.deployment_readiness import (
    AdmissionDecision,
    CameraAdmissionController,
    DeploymentReadinessManager,
)
from streaming.production_multicamera_engine import (
    HardwareProfile,
    ProductionMultiCameraEngine,
)


class TestCameraFleetAdmission:
    def test_admission_controller_normal_resources(self):
        controller = CameraAdmissionController(
            max_cpu_percent=85.0,
            max_vram_percent=90.0,
            max_ram_percent=85.0,
        )
        result = controller.evaluate_admission(
            camera_id="cam-01",
            current_active_cameras=1,
            sustainable_capacity=8,
            cpu_percent=35.0,
            ram_percent=40.0,
            vram_allocated_mb=1000.0,
            vram_total_mb=6000.0,
            network_headroom_pct=80.0,
            target_fps=15.0,
        )
        assert result.admitted is True
        assert result.decision == AdmissionDecision.ADMITTED
        assert result.effective_fps == 15.0

    def test_admission_controller_saturated_cpu_rejection(self):
        controller = CameraAdmissionController(max_cpu_percent=85.0)
        result = controller.evaluate_admission(
            camera_id="cam-09",
            current_active_cameras=8,
            sustainable_capacity=8,
            cpu_percent=92.0,
            ram_percent=50.0,
            vram_allocated_mb=2000.0,
            vram_total_mb=6000.0,
        )
        assert result.admitted is False
        assert result.decision == AdmissionDecision.REJECTED_COMPUTE_CAPACITY
        assert "CPU saturated" in result.reason

    def test_deployment_readiness_manager_admission(self):
        mgr = DeploymentReadinessManager()
        adm = mgr.request_camera_admission(
            camera_id="cam-test",
            current_active_cameras=0,
            target_fps=15.0,
        )
        assert adm.camera_id == "cam-test"
        assert isinstance(adm.admitted, bool)


class TestCameraWorkerInferenceEngineIntegration:
    def test_camera_worker_dispatches_to_inference_engine(self):
        mock_engine = MagicMock()
        mock_engine.is_running.return_value = True
        mock_engine.cache = MagicMock()
        mock_engine.cache.get_active_tracks.return_value = []

        cfg = {
            "type": "file",
            "file_path": "nonexistent.mp4",
            "width": 320,
            "height": 240,
            "target_fps": 10,
        }
        worker = CameraWorker(
            camera_id="cam-mock",
            camera_config=cfg,
            inference_engine=mock_engine,
        )

        assert worker.inference_engine is mock_engine
        assert worker.camera_id == "cam-mock"


class TestMultiCameraFusionAndAggregation:
    def test_engine_processes_frame_with_fusion_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            collector = OperationalEmbeddingCollector(output_dir=tmp_dir)
            fusion = DualModalFusion(enabled=True)
            aggregator = TrackIdentityAggregator(window_size=8, min_frames_for_decision=2)

            profile = HardwareProfile(
                device_type="cpu",
                cpu_cores=2,
                total_ram_mb=2048.0,
            )


            gallery_features = np.random.randn(2, 256).astype(np.float32)
            gallery_labels = ["Subject_A", "Subject_B"]
            app_gallery_features = np.random.randn(2, 512).astype(np.float32)
            app_gallery_labels = ["Subject_A", "Subject_B"]

            engine = ProductionMultiCameraEngine(
                hardware_profile=profile,
                fusion_engine=fusion,
                track_aggregator=aggregator,
                operational_collector=collector,
                gallery_features=gallery_features,
                gallery_labels=gallery_labels,
                appearance_gallery_features=app_gallery_features,
                appearance_gallery_labels=app_gallery_labels,
            )

            q = engine.register_camera("cam-01", priority=5)
            assert q is not None
            assert engine.track_aggregator is aggregator
            assert engine.fusion_engine is fusion


            dummy_frame = (np.ones((240, 320, 3), dtype=np.uint8) * 128)
            success = engine.put_frame("cam-01", dummy_frame, frame_id=1)
            assert success is True

            telemetry = engine.get_telemetry()
            assert telemetry["registered_cameras_count"] == 1
            assert "cam-01" in telemetry["cameras"]


class TestCameraManagerDynamicConfig:
    def test_camera_manager_save_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "cameras.yaml"
            cfg_path.write_text("cameras: {}\ndefaults: {}\n", encoding="utf-8")

            manager = CameraManager(config_path=str(cfg_path))
            assert len(manager.cameras_config) == 0


            with patch.object(
                DeploymentReadinessManager,
                "request_camera_admission",
                return_value=MagicMock(admitted=True),
            ):

                added = manager.add_camera("cam-dyn-01", {"type": "webcam", "device_index": 0})
                assert added is True
                assert "cam-dyn-01" in manager.cameras_config


                saved = manager.save_config()
                assert saved is True
                assert cfg_path.exists()
                content = cfg_path.read_text(encoding="utf-8")
                assert "cam-dyn-01" in content


                removed = manager.remove_camera("cam-dyn-01")
                assert removed is True
                assert "cam-dyn-01" not in manager.cameras_config

    def test_camera_manager_enforce_admission_rejection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "cameras.yaml"
            cfg_path.write_text("cameras: {}\ndefaults: {}\n", encoding="utf-8")

            manager = CameraManager(config_path=str(cfg_path))
            with patch.object(
                DeploymentReadinessManager,
                "request_camera_admission",
                return_value=MagicMock(admitted=False, reason="CPU overloaded"),
            ):
                added = manager.add_camera(
                    "cam-dyn-02",
                    {"type": "webcam", "device_index": 0, "enforce_admission": True},
                )
                assert added is False
