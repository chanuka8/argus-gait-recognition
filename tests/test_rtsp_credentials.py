import os
import unittest
from pathlib import Path

from security_layer.credentials import (
    CredentialManager,
    extract_url_credentials,
    resolve_camera_config,
    sanitize_rtsp_url,
)
from services.camera_manager import CameraManager


class TestRTSPCredentials(unittest.TestCase):
    def setUp(self):
        self.test_enc_file = Path("configs/test_credentials.enc")
        if self.test_enc_file.exists():
            self.test_enc_file.unlink()

    def tearDown(self):
        if self.test_enc_file.exists():
            self.test_enc_file.unlink()
        # Clean environment variables
        for key in list(os.environ.keys()):
            if key.startswith("ARGUS_CAMERA_") or key.startswith("ARGUS_RTSP_") or key.startswith("ARGUS_LEGACY_"):
                del os.environ[key]

    def test_sanitize_rtsp_url(self):
        raw_url = "rtsp://admin:secret123@192.168.1.100:554/stream1"
        sanitized = sanitize_rtsp_url(raw_url)
        self.assertEqual(sanitized, "rtsp://***:***@192.168.1.100:554/stream1")
        self.assertNotIn("secret123", sanitized)
        self.assertNotIn("admin", sanitized)

        no_auth_url = "rtsp://192.168.1.100:554/stream1"
        self.assertEqual(sanitize_rtsp_url(no_auth_url), no_auth_url)

    def test_extract_url_credentials(self):
        user, passwd, clean_url = extract_url_credentials("rtsp://user:pass@host:554/path")
        self.assertEqual(user, "user")
        self.assertEqual(passwd, "pass")
        self.assertEqual(clean_url, "rtsp://host:554/path")

    def test_priority1_environment_variables(self):
        os.environ["ARGUS_CAMERA_CAM01_USERNAME"] = "env_user"
        os.environ["ARGUS_CAMERA_CAM01_PASSWORD"] = "env_pass"

        cam_config = {
            "id": "cam01",
            "host": "10.0.0.1",
            "port": 554,
            "path": "/live",
        }

        resolved = resolve_camera_config(cam_config)
        self.assertEqual(resolved["username"], "env_user")
        self.assertEqual(resolved["password"], "env_pass")
        self.assertEqual(resolved["url"], "rtsp://env_user:env_pass@10.0.0.1:554/live")

    def test_priority1_explicit_env_keys(self):
        os.environ["CUSTOM_USER"] = "custom_u"
        os.environ["CUSTOM_PASS"] = "custom_p"

        cam_config = {
            "id": "cam_custom",
            "host": "10.0.0.2",
            "username_env": "CUSTOM_USER",
            "password_env": "CUSTOM_PASS",
        }

        resolved = resolve_camera_config(cam_config)
        self.assertEqual(resolved["username"], "custom_u")
        self.assertEqual(resolved["password"], "custom_p")
        self.assertEqual(resolved["url"], "rtsp://custom_u:custom_p@10.0.0.2:554")

    def test_priority2_encrypted_credential_store(self):
        key = CredentialManager.generate_key()
        cm = CredentialManager(credentials_file=str(self.test_enc_file), key=key)

        creds = {
            "cam_enc": {"username": "enc_user", "password": "enc_password"},
        }
        cm.encrypt_credentials(creds)
        self.assertTrue(self.test_enc_file.exists())

        cam_config = {
            "id": "cam_enc",
            "host": "10.0.0.3",
            "port": 554,
            "path": "/ch1",
        }

        resolved = resolve_camera_config(cam_config, credential_manager=cm)
        self.assertEqual(resolved["username"], "enc_user")
        self.assertEqual(resolved["password"], "enc_password")
        self.assertEqual(resolved["url"], "rtsp://enc_user:enc_password@10.0.0.3:554/ch1")

    def test_priority3_legacy_plaintext_default_rejection(self):
        cam_config = {
            "id": "cam_plain",
            "url": "rtsp://admin:secret_pass@10.0.0.4:554/stream",
        }

        with self.assertRaises(ValueError) as ctx:
            resolve_camera_config(cam_config)
        self.assertIn("Plaintext RTSP passwords in configuration are rejected by default", str(ctx.exception))

    def test_priority3_legacy_plaintext_allowed(self):
        os.environ["ARGUS_LEGACY_ALLOW_PLAINTEXT_CREDS"] = "true"

        cam_config = {
            "id": "cam_plain",
            "url": "rtsp://admin:secret_pass@10.0.0.4:554/stream",
        }

        resolved = resolve_camera_config(cam_config)
        self.assertEqual(resolved["username"], "admin")
        self.assertEqual(resolved["password"], "secret_pass")
        self.assertEqual(resolved["url"], "rtsp://admin:secret_pass@10.0.0.4:554/stream")

    def test_camera_manager_integration(self):
        os.environ["ARGUS_CAMERA_CAMERA_01_USERNAME"] = "cm_user"
        os.environ["ARGUS_CAMERA_CAMERA_01_PASSWORD"] = "cm_pass"

        mgr = CameraManager("configs/cameras.yaml")
        cam01 = mgr.cameras_config.get("camera_01")
        self.assertIsNotNone(cam01)
        self.assertEqual(cam01["username"], "cm_user")
        self.assertEqual(cam01["password"], "cm_pass")
        self.assertIn("rtsp://cm_user:cm_pass@192.168.1.100:554/stream1", cam01["url"])


if __name__ == "__main__":
    unittest.main()
