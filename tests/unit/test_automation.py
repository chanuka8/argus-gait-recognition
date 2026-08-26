"""
Comprehensive Unit Tests for ARGUS AI Automation & Environment Management Subsystem.
"""

import unittest
from unittest.mock import MagicMock, patch

from automation.device_manager import DeviceManager
from automation.dll_manager import setup_cuda_dll_paths
from automation.download_manager import DownloadManager
from automation.environment_validator import (
    ComputeBackend,
    EnvironmentState,
    EnvironmentValidator,
)
from automation.hardware_detector import HardwareDetector, HostSystemInfo, NvidiaGpuInfo
from automation.onnx_manager import OnnxManager
from automation.pytorch_manager import PyTorchManager
from api.schemas import StatusResponse, ComputeInfo


class TestAutomationSubsystem(unittest.TestCase):
    """Test suite covering hardware detection, device arbitration, and fallback logic."""

    def test_hardware_detector_system(self):
        sys_info = HardwareDetector.detect_system()
        self.assertIsInstance(sys_info, HostSystemInfo)
        self.assertGreater(sys_info.cpu_cores, 0)
        self.assertIn("3.", sys_info.python_version)

    def test_hardware_detector_gpu(self):
        gpu_info = HardwareDetector.detect_nvidia_gpu()
        self.assertIsInstance(gpu_info, NvidiaGpuInfo)
        if gpu_info.present:
            self.assertIsNotNone(gpu_info.gpu_name)
            self.assertIsNotNone(gpu_info.driver_version)
            self.assertGreater(gpu_info.vram_mb, 0)

    def test_dll_manager(self):
        paths = setup_cuda_dll_paths()
        self.assertIsInstance(paths, list)

    def test_download_manager_formatting(self):
        self.assertEqual(DownloadManager._format_size(1024), "1.00 KB")
        self.assertEqual(DownloadManager._format_size(1024 * 1024), "1.00 MB")
        self.assertEqual(DownloadManager._format_size(1024 * 1024 * 1024), "1.00 GB")
        self.assertEqual(DownloadManager._format_time(65), "00:01:05")

    def test_cpu_pipeline_validation(self):
        ok, details, errors = EnvironmentValidator.validate_cpu_pipeline()
        self.assertTrue(ok)
        self.assertGreater(len(details), 0)
        self.assertEqual(len(errors), 0)

    def test_device_manager_singleton(self):
        dm1 = DeviceManager.get_instance()
        dm2 = DeviceManager.get_instance()
        self.assertIs(dm1, dm2)
        self.assertIn(dm1.backend, ("cuda", "cpu"))
        self.assertIn(dm1.device, ("cuda:0", "cpu"))
        summary = dm1.summary()
        self.assertIn("backend", summary)
        self.assertIn("device", summary)
        self.assertIn("status", summary)

    def test_device_manager_resolution(self):
        dm = DeviceManager.get_instance()
        self.assertEqual(dm.resolve_component_device("cpu"), "cpu")
        if dm.is_cuda:
            self.assertEqual(dm.resolve_component_device("auto"), "cuda:0")
            self.assertEqual(dm.resolve_component_device("cuda"), "cuda:0")
        else:
            self.assertEqual(dm.resolve_component_device("auto"), "cpu")
            self.assertEqual(dm.resolve_component_device("cuda"), "cpu")

    def test_pytorch_manager_inspect(self):
        mgr = PyTorchManager()
        info = mgr.inspect_current_pytorch()
        self.assertTrue(info["installed"])
        self.assertTrue(info["tensor_probe_passed"])

    def test_onnx_manager_inspect(self):
        mgr = OnnxManager()
        info = mgr.inspect_current_onnx()
        self.assertTrue(info["installed"])
        self.assertGreater(len(info["providers"]), 0)

    def test_status_response_schema_compatibility(self):
        # Backward-compatible initialization
        resp = StatusResponse(
            status="operational",
            device="cuda",
            active_cameras=1,
        )
        self.assertEqual(resp.status, "operational")
        self.assertEqual(resp.device, "cuda")
        self.assertIsNone(resp.compute)

        # Extended initialization with compute info
        compute_data = ComputeInfo(
            backend="cuda",
            device="cuda:0",
            gpu="RTX 3050",
            vram_mb=6144.0,
            cuda_available=True,
            pytorch_version="2.5.1+cu121",
            cuda_version="12.1",
            onnx_provider="CUDAExecutionProvider",
        )
        resp_with_compute = StatusResponse(
            status="operational",
            device="cuda",
            compute=compute_data,
        )
        self.assertEqual(resp_with_compute.compute.device, "cuda:0")
        self.assertEqual(resp_with_compute.compute.backend, "cuda")

    def test_force_cpu_device_manager(self):
        # Force CPU mode
        dm_cpu = DeviceManager.get_instance(force_refresh=True, force_cpu=True)
        self.assertEqual(dm_cpu.backend, "cpu")
        self.assertEqual(dm_cpu.device, "cpu")
        self.assertFalse(dm_cpu.is_cuda)
        self.assertTrue(dm_cpu.is_cpu)
        self.assertFalse(dm_cpu.cuda_available)
        self.assertEqual(dm_cpu.onnx_provider, "CPUExecutionProvider")
        self.assertEqual(dm_cpu.status, EnvironmentState.CPU_READY)
        self.assertEqual(dm_cpu.resolve_component_device("auto"), "cpu")
        self.assertEqual(dm_cpu.resolve_component_device("cuda"), "cpu")
        self.assertEqual(dm_cpu.resolve_component_device("cpu"), "cpu")

        # Restore default auto hardware mode
        dm_auto = DeviceManager.get_instance(force_refresh=True, force_cpu=False)
        self.assertIsNotNone(dm_auto.backend)

    def test_yolo_forced_cpu_validation(self):
        import numpy as np
        from pipeline.detection.person_detector import PersonDetector

        # Force CPU mode
        DeviceManager.get_instance(force_refresh=True, force_cpu=True)
        detector = PersonDetector()
        self.assertEqual(detector.runtime_device, "cpu")

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = detector.detect(dummy)
        param_dev = next(detector.model.model.parameters()).device
        self.assertEqual(param_dev.type, "cpu")

        # Restore default auto hardware mode
        DeviceManager.get_instance(force_refresh=True, force_cpu=False)

    def test_automation_module_getattr(self):
        import automation
        boot_cls = getattr(automation, "EnvironmentBootstrap")
        self.assertIsNotNone(boot_cls)
        with self.assertRaises(AttributeError):
            _ = getattr(automation, "NonExistentClassXYZ")


if __name__ == "__main__":
    unittest.main()

