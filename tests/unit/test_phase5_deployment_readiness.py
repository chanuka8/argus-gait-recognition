import pytest

from services.camera_source_resolver import CameraSourceResolver
from streaming.deployment_readiness import (
    AdaptiveInferencePolicy,
    AdmissionDecision,
    CameraAdmissionController,
    CPUInfo,
    DeploymentReadinessManager,
    GPUInfo,
    GPUMemoryGuard,
    HardwareCapabilityDetector,
    HardwareCapabilityReport,
    InferenceQualityMode,
    ModelProfileRegistry,
    ModelResourceProfile,
    NetworkBandwidthEstimator,
    ProductionCapacityEstimator,
    RAMInfo,
    SecurityAuditor,
    StorageSafetyAuditor,
    SystemProfile,
    SystemProfileEngine,
)
from streaming.production_runtime import (
    CameraState,
    ProductionSurveillanceRuntime,
)


class TestHardwareCapabilityDiscovery:
    def test_discover_returns_valid_report(self):
        detector = HardwareCapabilityDetector()
        report = detector.discover()
        assert isinstance(report, HardwareCapabilityReport)
        assert report.cpu.physical_cores >= 1
        assert report.cpu.logical_cores >= report.cpu.physical_cores
        assert report.ram.total_mb > 0
        assert report.ram.available_mb > 0
        assert report.storage.total_gb > 0
        assert report.storage.writable is True
        assert len(report.network.hostname) > 0

    def test_report_serialization(self):
        detector = HardwareCapabilityDetector()
        report = detector.discover()
        d = report.to_dict()
        assert "cpu" in d
        assert "gpu" in d
        assert "ram" in d
        assert "cuda" in d
        assert "storage" in d
        assert "network" in d







class TestSystemProfileEngine:
    def test_auto_profile_cpu_only(self):
        report = HardwareCapabilityReport(
            cpu=CPUInfo(physical_cores=4, logical_cores=8),
            ram=RAMInfo(total_mb=8192.0, available_mb=4096.0),
            gpu=GPUInfo(available=False),
        )
        params = SystemProfileEngine.select_profile(report, SystemProfile.AUTO)
        assert params.profile_name in ("CPU_ONLY", "LOW_RESOURCE")
        assert params.enable_gpu is False
        assert params.worker_count >= 1

    def test_auto_profile_gpu_small(self):
        report = HardwareCapabilityReport(
            cpu=CPUInfo(physical_cores=6, logical_cores=12),
            ram=RAMInfo(total_mb=8192.0, available_mb=4096.0),
            gpu=GPUInfo(available=True, model="RTX 3050", vram_total_mb=4096.0),
        )
        params = SystemProfileEngine.select_profile(report, SystemProfile.AUTO)
        assert params.profile_name == "GPU_SMALL"
        assert params.enable_gpu is True
        assert params.osnet_batch_size >= 4

    def test_auto_profile_server(self):
        report = HardwareCapabilityReport(
            cpu=CPUInfo(physical_cores=32, logical_cores=64),
            ram=RAMInfo(total_mb=65536.0, available_mb=32768.0),
            gpu=GPUInfo(available=True, model="A100", vram_total_mb=81920.0),
        )
        params = SystemProfileEngine.select_profile(report, SystemProfile.AUTO)
        assert params.profile_name == "SERVER"
        assert params.worker_count >= 4
        assert params.detector_batch_size >= 8

    def test_explicit_profile_override(self):
        report = HardwareCapabilityReport()
        params = SystemProfileEngine.select_profile(report, SystemProfile.LOW_RESOURCE)
        assert params.profile_name == "LOW_RESOURCE"
        assert params.worker_count == 1







class TestProductionCapacityEstimator:
    def test_sustainable_capacity_calculation(self):
        estimator = ProductionCapacityEstimator(target_camera_fps=15.0)
        res = estimator.estimate_capacity(
            measured_throughput_fps=300.0,
            current_active_cameras=5,
            cpu_percent=40.0,
            vram_allocated_mb=1000.0,
            vram_total_mb=6000.0,
            p95_latency_ms=10.0,
            drop_rate=0.0,
        )
        assert res["estimated_sustainable_cameras"] == 20
        assert res["headroom_cameras"] == 15
        assert res["confidence_level"] == "HIGH"
        assert res["constraints_met"] is True

    def test_capacity_bottlenecked_by_cpu(self):
        estimator = ProductionCapacityEstimator(target_camera_fps=10.0)
        res = estimator.estimate_capacity(
            measured_throughput_fps=200.0,
            current_active_cameras=10,
            cpu_percent=95.0,
            vram_allocated_mb=500.0,
            vram_total_mb=4000.0,
            p95_latency_ms=15.0,
            drop_rate=0.0,
        )
        assert res["estimated_sustainable_cameras"] < 20
        assert res["limiting_factor"] == "cpu"
        assert res["constraints_met"] is False

    def test_capacity_bottlenecked_by_vram(self):
        estimator = ProductionCapacityEstimator(target_camera_fps=10.0)
        res = estimator.estimate_capacity(
            measured_throughput_fps=200.0,
            current_active_cameras=5,
            cpu_percent=30.0,
            vram_allocated_mb=5800.0,
            vram_total_mb=6000.0,
            p95_latency_ms=15.0,
            drop_rate=0.0,
        )
        assert res["limiting_factor"] == "vram"
        assert res["constraints_met"] is False







class TestCameraAdmissionController:
    def test_safe_admission(self):
        adm = CameraAdmissionController()
        res = adm.evaluate_admission(
            camera_id="cam_01",
            current_active_cameras=2,
            sustainable_capacity=10,
            cpu_percent=30.0,
            ram_percent=40.0,
            vram_allocated_mb=1000.0,
            vram_total_mb=6000.0,
            network_headroom_pct=80.0,
        )
        assert res.admitted is True
        assert res.decision == AdmissionDecision.ADMITTED

    def test_cpu_saturation_rejection(self):
        adm = CameraAdmissionController(max_cpu_percent=85.0)
        res = adm.evaluate_admission(
            camera_id="cam_overflow",
            current_active_cameras=5,
            sustainable_capacity=10,
            cpu_percent=92.0,
            ram_percent=40.0,
            vram_allocated_mb=1000.0,
            vram_total_mb=6000.0,
        )
        assert res.admitted is False
        assert res.decision == AdmissionDecision.REJECTED_COMPUTE_CAPACITY

    def test_vram_saturation_rejection(self):
        adm = CameraAdmissionController(max_vram_percent=90.0)
        res = adm.evaluate_admission(
            camera_id="cam_vram_overflow",
            current_active_cameras=5,
            sustainable_capacity=10,
            cpu_percent=30.0,
            ram_percent=40.0,
            vram_allocated_mb=5800.0,
            vram_total_mb=6000.0,
        )
        assert res.admitted is False
        assert res.decision == AdmissionDecision.REJECTED_VRAM_CAPACITY

    def test_network_headroom_exhaustion_rejection(self):
        adm = CameraAdmissionController(min_network_headroom_pct=10.0)
        res = adm.evaluate_admission(
            camera_id="cam_net_overflow",
            current_active_cameras=15,
            sustainable_capacity=30,
            cpu_percent=30.0,
            ram_percent=40.0,
            vram_allocated_mb=1000.0,
            vram_total_mb=6000.0,
            network_headroom_pct=5.0,
        )
        assert res.admitted is False
        assert res.decision == AdmissionDecision.REJECTED_NETWORK_CAPACITY

    def test_degraded_admission_near_capacity(self):
        adm = CameraAdmissionController()
        res = adm.evaluate_admission(
            camera_id="cam_degraded",
            current_active_cameras=10,
            sustainable_capacity=10,
            cpu_percent=50.0,
            ram_percent=50.0,
            vram_allocated_mb=2000.0,
            vram_total_mb=6000.0,
            target_fps=15.0,
        )
        assert res.admitted is True
        assert res.decision == AdmissionDecision.ADMITTED_DEGRADED
        assert res.effective_fps < 15.0

    def test_admission_rejection_does_not_affect_active_cameras(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("cam_A")
        rt.start_camera("cam_A")
        rt.connect_camera("cam_A")

        adm = CameraAdmissionController()

        res = adm.evaluate_admission(
            camera_id="cam_B",
            current_active_cameras=10,
            sustainable_capacity=5,
            cpu_percent=95.0,
            ram_percent=50.0,
            vram_allocated_mb=2000.0,
            vram_total_mb=4000.0,
        )
        assert res.admitted is False


        cam_a = rt.camera_state_machine.get_camera("cam_A")
        assert cam_a.connection_state == CameraState.CONNECTED







class TestAdaptiveInferencePolicy:
    def test_full_quality_under_low_load(self):
        policy = AdaptiveInferencePolicy()
        mode = policy.evaluate_quality_mode(cpu_percent=30.0, vram_percent=25.0, p95_latency_ms=10.0)
        assert mode == InferenceQualityMode.FULL_QUALITY

    def test_reduced_fps_under_moderate_load(self):
        policy = AdaptiveInferencePolicy()
        mode = policy.evaluate_quality_mode(cpu_percent=80.0, vram_percent=50.0, p95_latency_ms=20.0)
        assert mode == InferenceQualityMode.REDUCED_PROCESSING_FPS

    def test_aggressive_skipping_under_heavy_load(self):
        policy = AdaptiveInferencePolicy()
        mode = policy.evaluate_quality_mode(cpu_percent=95.0, vram_percent=95.0, p95_latency_ms=400.0)
        assert mode == InferenceQualityMode.AGGRESSIVE_FRAME_SKIPPING

    def test_automatic_recovery(self):
        policy = AdaptiveInferencePolicy()

        policy.evaluate_quality_mode(cpu_percent=95.0, vram_percent=95.0, p95_latency_ms=400.0)

        mode = policy.evaluate_quality_mode(cpu_percent=30.0, vram_percent=30.0, p95_latency_ms=10.0)
        assert mode == InferenceQualityMode.AUTOMATIC_RECOVERY







class TestGPUMemoryGuard:
    def test_vram_state_structure(self):
        guard = GPUMemoryGuard()
        state = guard.get_vram_state()
        assert "allocated_mb" in state
        assert "reserved_mb" in state
        assert "total_mb" in state
        assert "free_mb" in state

    def test_handle_cuda_oom_recovers_gracefully(self):
        guard = GPUMemoryGuard()
        fake_oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        recovered = guard.handle_cuda_oom(fake_oom, context="OSNet_inference_batch")
        assert guard.oom_recoveries_count == 1

        assert isinstance(recovered, bool)







class TestNetworkBandwidthEstimator:
    def test_estimate_camera_bandwidth(self):
        est = NetworkBandwidthEstimator()
        bw_720p = est.estimate_camera_bandwidth(resolution="720p", fps=15.0, codec="h264")
        assert bw_720p == pytest.approx(1.5, abs=0.1)

        bw_1080p_h265 = est.estimate_camera_bandwidth(resolution="1080p", fps=15.0, codec="h265")
        assert bw_1080p_h265 == pytest.approx(1.8, abs=0.1)

    def test_evaluate_link_capacity(self):
        est = NetworkBandwidthEstimator()
        eval_res = est.evaluate_link_capacity(
            active_camera_count=4,
            target_camera_count=16,
            link_speed_mbps=1000.0,
            resolution="720p",
            fps=15.0,
            codec="h264",
        )
        assert eval_res["is_network_capacity_sufficient"] is True
        assert eval_res["total_ingress_mbps"] == pytest.approx(24.0, abs=1.0)
        assert eval_res["headroom_pct"] > 90.0







class TestModelProfileRegistry:
    def test_default_profiles_exist(self):
        reg = ModelProfileRegistry()
        osnet = reg.get_profile("OSNet-x0.25")
        assert osnet is not None
        assert osnet.embedding_dimension == 512
        assert osnet.modality == "appearance"

        gait = reg.get_profile("ByGaitLight")
        assert gait is not None
        assert gait.embedding_dimension == 256
        assert gait.modality == "gait"

    def test_register_custom_profile(self):
        reg = ModelProfileRegistry()
        prof = ModelResourceProfile(
            model_name="CustomCNN",
            version="v2",
            embedding_dimension=128,
            modality="custom",
            device="cpu",
            precision="fp32",
            typical_batch_size=1,
            estimated_vram_mb=0.0,
            measured_throughput_fps=100.0,
            p50_latency_ms=10.0,
            p95_latency_ms=15.0,
        )
        reg.register_profile(prof)
        assert reg.get_profile("CustomCNN") is not None







class TestStorageSafetyAuditor:
    def test_audit_storage_healthy(self):
        auditor = StorageSafetyAuditor(storage_dir="data")
        res = auditor.audit_storage()
        assert res["free_space_gb"] > 0
        assert res["atomic_write_verified"] is True
        assert res["status"] in ("HEALTHY", "DEGRADED")







class TestSecurityAuditor:
    def test_rtsp_sanitization_verification(self):
        raw_url = "rtsp://admin:supersecretpassword123@192.168.1.100:554/stream1"
        assert SecurityAuditor.verify_rtsp_sanitization(raw_url) is True

    def test_audit_system_security(self):
        sec = SecurityAuditor.audit_system_security()
        assert sec["security_status"] == "SECURE"
        assert "VERIFIED" in sec["model_provenance_gating"]
        assert "PROHIBITED" in sec["face_recognition_prohibited"]







class TestDeploymentReadinessManager:
    def test_get_deployment_summary(self):
        mgr = DeploymentReadinessManager()
        summary = mgr.get_deployment_summary()
        assert summary["status"] == "DEPLOYMENT_READY"
        assert "system_profile" in summary
        assert "capacity" in summary
        assert "storage_safety" in summary
        assert "security" in summary
        assert "models" in summary







class TestWebcamRegressionAndProbing:
    def test_camera_source_resolver_probing(self):
        resolver = CameraSourceResolver()

        is_index_0 = resolver.probe_usb_webcam(0)
        assert isinstance(is_index_0, bool)

        resolver.reserve_source("usb:0", "cam_owner")
        assert resolver.probe_usb_webcam(0) is False
        resolver.release_source_by_camera_id("cam_owner")

    def test_webcam_initial_state_invariant(self):
        rt = ProductionSurveillanceRuntime()
        cam = rt.register_camera("webcam_test", source_type="webcam")
        assert cam.connection_state == CameraState.STOPPED
        assert cam.actual_state == CameraState.STOPPED

    def test_webcam_lifecycle_transitions(self):
        rt = ProductionSurveillanceRuntime()
        rt.register_camera("webcam_live", source_type="webcam")
        assert rt.start_camera("webcam_live") is True
        assert rt.connect_camera("webcam_live") is True
        cam = rt.camera_state_machine.get_camera("webcam_live")
        assert cam.connection_state == CameraState.CONNECTED
        assert rt.stop_camera("webcam_live") is True
        cam = rt.camera_state_machine.get_camera("webcam_live")
        assert cam.connection_state == CameraState.STOPPED







@pytest.mark.parametrize("num_cameras", [1, 2, 4, 8, 16, 32, 64, 128])
def test_simulated_stream_scaling_1_to_128(num_cameras):
    rt = ProductionSurveillanceRuntime()

    for i in range(num_cameras):
        cid = f"scale_cam_{i:03d}"
        rt.register_camera(cid, source_type="rtsp", fps_target=15)
        rt.start_camera(cid)
        rt.connect_camera(cid)

    health = rt.get_system_health()
    assert health["cameras"]["total"] == num_cameras
    assert health["cameras"]["connected"] == num_cameras


    for i in range(num_cameras):
        rt.stop_camera(f"scale_cam_{i:03d}")

    health = rt.get_system_health()
    assert health["cameras"]["connected"] == 0
