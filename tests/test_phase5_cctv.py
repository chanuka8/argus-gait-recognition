"""Unit tests for Phase 5 enterprise CCTV integration."""

import unittest

from services.onvif_client import ONVIFClient, ONVIFCapabilities, ONVIFProfile
from services.vendor_adapters import (
    VENDOR_REGISTRY,
    AxisAdapter,
    CameraAdapter,
    DahuaAdapter,
    GenericRTSPAdapter,
    HikvisionAdapter,
    UniviewAdapter,
    get_adapter,
)
from services.camera_discovery import CameraDiscoveryService, DiscoveredCamera


class TestONVIFClient(unittest.TestCase):
    """Test ONVIF XML parsing and URL building."""

    def test_soap_envelope_with_auth(self):
        client = ONVIFClient("192.168.1.100", username="admin", password="pass")
        env = client.build_soap_envelope("<GetCapabilities/>")
        self.assertIn("UsernameToken", env)
        self.assertIn("admin", env)

    def test_soap_envelope_without_auth(self):
        client = ONVIFClient("192.168.1.100")
        env = client.build_soap_envelope("<GetCapabilities/>")
        self.assertNotIn("UsernameToken", env)

    def test_parse_capabilities_ptz(self):
        client = ONVIFClient("10.0.0.1")
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
<s:Body>
<tds:GetCapabilitiesResponse xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
<tds:Capabilities>
  <tt:PTZ xmlns:tt="http://www.onvif.org/ver10/schema"><tt:XAddr>http://10.0.0.1/onvif/ptz</tt:XAddr></tt:PTZ>
  <tt:Media xmlns:tt="http://www.onvif.org/ver10/schema"><tt:XAddr>http://10.0.0.1/onvif/media</tt:XAddr></tt:Media>
</tds:Capabilities>
</tds:GetCapabilitiesResponse>
</s:Body>
</s:Envelope>"""
        caps = client.parse_capabilities_response(xml)
        self.assertTrue(caps.has_ptz)
        self.assertTrue(caps.has_media)
        self.assertFalse(caps.has_analytics)

    def test_parse_empty_capabilities(self):
        client = ONVIFClient("10.0.0.1")
        caps = client.parse_capabilities_response("<not-xml")
        self.assertFalse(caps.has_ptz)

    def test_parse_device_info(self):
        client = ONVIFClient("10.0.0.1")
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
<s:Body>
<tds:GetDeviceInformationResponse xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <tds:Manufacturer>Hikvision</tds:Manufacturer>
  <tds:Model>DS-2CD2143G2</tds:Model>
  <tds:SerialNumber>ABC123</tds:SerialNumber>
</tds:GetDeviceInformationResponse>
</s:Body>
</s:Envelope>"""
        info = client.parse_device_info_response(xml)
        self.assertEqual(info["manufacturer"], "Hikvision")
        self.assertEqual(info["model"], "DS-2CD2143G2")

    def test_build_rtsp_url(self):
        url = ONVIFClient.build_rtsp_url("10.0.0.1", 554, "/stream", "admin", "pass")
        self.assertEqual(url, "rtsp://admin:pass@10.0.0.1:554/stream")

    def test_build_rtsp_url_no_auth(self):
        url = ONVIFClient.build_rtsp_url("10.0.0.1")
        self.assertEqual(url, "rtsp://10.0.0.1:554/Streaming/Channels/101")

    def test_extract_host_from_xaddr(self):
        host = ONVIFClient.extract_host_from_xaddr("http://192.168.1.50:80/onvif/device_service")
        self.assertEqual(host, "192.168.1.50")

    def test_onvif_profile_to_dict(self):
        p = ONVIFProfile(name="Main", token="tok1", resolution=(1920, 1080))
        d = p.to_dict()
        self.assertEqual(d["name"], "Main")
        self.assertEqual(d["resolution"], (1920, 1080))

    def test_capabilities_to_dict(self):
        caps = ONVIFCapabilities()
        caps.has_ptz = True
        d = caps.to_dict()
        self.assertTrue(d["has_ptz"])
        self.assertIsInstance(d["profiles"], list)


class TestVendorAdapters(unittest.TestCase):
    """Test vendor-specific RTSP URL adapters."""

    def test_hikvision_url(self):
        adapter = HikvisionAdapter("10.0.0.1", username="admin", password="pass")
        url = adapter.get_rtsp_url(channel=1, subtype=0)
        self.assertIn("Streaming/Channels", url)
        self.assertIn("admin:pass@", url)

    def test_dahua_url(self):
        adapter = DahuaAdapter("10.0.0.2", username="admin", password="test")
        url = adapter.get_rtsp_url(channel=2, subtype=1)
        self.assertIn("realmonitor", url)
        self.assertIn("channel=2", url)
        self.assertIn("subtype=1", url)

    def test_uniview_url(self):
        adapter = UniviewAdapter("10.0.0.3")
        url = adapter.get_rtsp_url()
        self.assertIn("/media/video1", url)

    def test_axis_url(self):
        adapter = AxisAdapter("10.0.0.4")
        url = adapter.get_rtsp_url()
        self.assertIn("axis-media", url)

    def test_generic_rtsp_url(self):
        adapter = GenericRTSPAdapter("10.0.0.5", path="/live/ch1")
        url = adapter.get_rtsp_url()
        self.assertIn("/live/ch1", url)

    def test_adapter_factory(self):
        adapter = get_adapter("hikvision", host="10.0.0.1")
        self.assertIsInstance(adapter, HikvisionAdapter)

        adapter2 = get_adapter("unknown_vendor", host="10.0.0.1")
        self.assertIsInstance(adapter2, CameraAdapter)

    def test_get_camera_config(self):
        adapter = HikvisionAdapter("10.0.0.1", username="admin", password="pass")
        cfg = adapter.get_camera_config("cam_hik_01", width=1920, height=1080)
        self.assertEqual(cfg["id"], "cam_hik_01")
        self.assertEqual(cfg["type"], "rtsp")
        self.assertEqual(cfg["vendor"], "hikvision")
        self.assertIn("Streaming/Channels", cfg["url"])

    def test_vendor_registry(self):
        self.assertIn("hikvision", VENDOR_REGISTRY)
        self.assertIn("dahua", VENDOR_REGISTRY)
        self.assertIn("uniview", VENDOR_REGISTRY)
        self.assertIn("axis", VENDOR_REGISTRY)
        self.assertIn("generic_rtsp", VENDOR_REGISTRY)

    def test_hikvision_metadata(self):
        adapter = HikvisionAdapter("10.0.0.1")
        meta = adapter.get_metadata()
        self.assertEqual(meta["vendor"], "hikvision")
        self.assertIn("onvif_port", meta)


class TestCameraDiscovery(unittest.TestCase):
    """Test camera discovery and health validation."""

    def test_discover_from_config(self):
        service = CameraDiscoveryService()
        config = {
            "cam1": {"url": "rtsp://10.0.0.1:554/stream", "vendor": "hikvision"},
            "cam2": {"url": "rtsp://10.0.0.2:554/live", "vendor": "dahua"},
        }
        discovered = service.discover_from_config(config)
        self.assertEqual(len(discovered), 2)
        self.assertEqual(discovered[0].host, "10.0.0.1")
        self.assertEqual(discovered[1].vendor, "dahua")

    def test_discovered_camera_to_dict(self):
        cam = DiscoveredCamera("10.0.0.1", vendor="axis")
        d = cam.to_dict()
        self.assertEqual(d["host"], "10.0.0.1")
        self.assertEqual(d["vendor"], "axis")

    def test_get_all_discovered(self):
        service = CameraDiscoveryService()
        service.discover_from_config({"cam1": {"url": "rtsp://10.0.0.1/s", "vendor": "generic"}})
        all_cams = service.get_all_discovered()
        self.assertIn("cam1", all_cams)

    def test_detect_vendor_from_url(self):
        self.assertEqual(CameraDiscoveryService.detect_vendor_from_url("rtsp://10.0.0.1/Streaming/Channels/101"), "hikvision")
        self.assertEqual(CameraDiscoveryService.detect_vendor_from_url("rtsp://10.0.0.1/cam/realmonitor?channel=1"), "dahua")
        self.assertEqual(CameraDiscoveryService.detect_vendor_from_url("rtsp://10.0.0.1/media/video1"), "uniview")
        self.assertEqual(CameraDiscoveryService.detect_vendor_from_url("rtsp://10.0.0.1/axis-media/media.amp"), "axis")
        self.assertEqual(CameraDiscoveryService.detect_vendor_from_url("rtsp://10.0.0.1/live"), "generic")

    def test_parse_ws_discovery_xaddrs(self):
        service = CameraDiscoveryService()
        xml = '<ProbeMatch><XAddrs>http://10.0.0.1:80/onvif/device_service http://10.0.0.2:80/onvif/device_service</XAddrs></ProbeMatch>'
        xaddrs = service.parse_ws_discovery_response(xml)
        self.assertEqual(len(xaddrs), 2)
        self.assertIn("http://10.0.0.1:80/onvif/device_service", xaddrs)

    def test_parse_rtsp_url(self):
        host, port = CameraDiscoveryService._parse_rtsp_url("rtsp://admin:pass@192.168.1.10:554/stream")
        self.assertEqual(host, "192.168.1.10")
        self.assertEqual(port, 554)

    def test_health_check_unknown_camera(self):
        service = CameraDiscoveryService()
        result = service.health_check_camera("nonexistent")
        self.assertEqual(result["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
