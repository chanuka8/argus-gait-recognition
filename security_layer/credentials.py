"""Secure RTSP credential storage, resolution, and sanitization.

Supports 3-tier priority credential resolution:
1. Environment variables
2. Encrypted local credential file (Fernet)
3. Legacy plaintext configuration fallback (only when explicitly enabled)
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def sanitize_rtsp_url(url: str) -> str:
    """Mask RTSP username and password in a URL string for secure logging."""
    if not isinstance(url, str):
        return str(url)
    return re.sub(r"rtsp://([^:@]+):([^@]+)@", r"rtsp://***:***@", url)


def derive_fernet_key(passphrase: str, salt: bytes = b"argus_rtsp_salt") -> bytes:
    """Derive a valid 32-byte Fernet key from a passphrase using standard PBKDF2HMAC."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


class CredentialManager:
    """Manages encrypted credential storage and resolution."""

    def __init__(
        self,
        credentials_file: str = "configs/credentials.enc",
        key: Optional[str] = None,
    ) -> None:
        self.credentials_file = Path(os.environ.get("ARGUS_CREDENTIALS_FILE", credentials_file))
        raw_key = key or os.environ.get("ARGUS_CREDENTIALS_KEY")
        if not raw_key and Path(".credentials.key").exists():
            try:
                raw_key = Path(".credentials.key").read_text(encoding="utf-8").strip()
            except Exception:
                raw_key = None

        self._fernet: Optional[Fernet] = None
        if raw_key:
            try:
                fernet_key = raw_key.encode("utf-8")
                # Try loading key directly as Fernet key
                self._fernet = Fernet(fernet_key)
            except Exception:
                # Derive key using PBKDF2HMAC if string is a passphrase
                derived = derive_fernet_key(raw_key)
                self._fernet = Fernet(derived)

    @staticmethod
    def generate_key() -> str:
        """Generate a random Fernet key string."""
        return Fernet.generate_key().decode("utf-8")

    def encrypt_credentials(self, credentials_data: Dict[str, Dict[str, str]], output_path: Optional[str] = None) -> Path:
        """Encrypt credentials dictionary to file."""
        if not self._fernet:
            raise ValueError("CredentialManager initialized without a valid key")
        path = Path(output_path) if output_path else self.credentials_file
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_json = json.dumps(credentials_data, indent=2).encode("utf-8")
        encrypted = self._fernet.encrypt(raw_json)
        path.write_bytes(encrypted)
        return path

    def load_encrypted_credentials(self) -> Dict[str, Dict[str, str]]:
        """Load and decrypt credentials from file."""
        if not self._fernet or not self.credentials_file.exists():
            return {}
        try:
            encrypted_data = self.credentials_file.read_bytes()
            decrypted_data = self._fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode("utf-8"))
        except Exception:
            return {}

    def get_credentials(self, camera_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve username and password for camera_id from encrypted file."""
        creds = self.load_encrypted_credentials()
        cam_data = creds.get(camera_id) or creds.get("default") or {}
        return cam_data.get("username"), cam_data.get("password")


def extract_url_credentials(url: str) -> Tuple[Optional[str], Optional[str], str]:
    """Extract embedded username/password from RTSP URL and return clean URL."""
    match = re.search(r"rtsp://([^:@]+):([^@]+)@(.+)", url)
    if match:
        user = match.group(1)
        passwd = match.group(2)
        clean_url = f"rtsp://{match.group(3)}"
        return user, passwd, clean_url
    return None, None, url


def is_legacy_plaintext_allowed(config: Dict[str, Any], override: Optional[bool] = None) -> bool:
    """Check if legacy plaintext fallback is enabled via config, env, or override."""
    if override is not None:
        return override
    if config.get("allow_plaintext_credentials") is True:
        return True
    env_val = os.environ.get("ARGUS_LEGACY_ALLOW_PLAINTEXT_CREDS", "").strip().lower()
    return env_val in ("true", "1", "yes")


def resolve_camera_config(
    camera_config: Dict[str, Any],
    credential_manager: Optional[CredentialManager] = None,
    legacy_allow_plaintext: Optional[bool] = None,
) -> Dict[str, Any]:
    """Resolve camera credentials adhering to priority order.

    Priority order:
    1. Environment variables (username_env/password_env, ARGUS_CAMERA_<ID>_*, ARGUS_RTSP_*)
    2. Encrypted local credential file
    3. Existing plaintext configuration (only when legacy fallback is enabled)
    """
    res = dict(camera_config)
    camera_id = str(res.get("id") or res.get("name") or "camera_default")

    username: Optional[str] = None
    password: Optional[str] = None

    # Priority 1: Environment variables
    u_env_key = res.get("username_env")
    p_env_key = res.get("password_env")

    if u_env_key and u_env_key in os.environ:
        username = os.environ[u_env_key]
    if p_env_key and p_env_key in os.environ:
        password = os.environ[p_env_key]

    clean_id = re.sub(r"\W+", "_", camera_id).upper()
    if not username:
        username = os.environ.get(f"ARGUS_CAMERA_{clean_id}_USERNAME") or os.environ.get("ARGUS_RTSP_USERNAME")
    if not password:
        password = os.environ.get(f"ARGUS_CAMERA_{clean_id}_PASSWORD") or os.environ.get("ARGUS_RTSP_PASSWORD")

    # Priority 2: Encrypted local credential file
    if not (username and password):
        cm = credential_manager or CredentialManager()
        enc_u, enc_p = cm.get_credentials(camera_id)
        if not username:
            username = enc_u
        if not password:
            password = enc_p

    # Priority 3: Plaintext configuration fallback
    raw_url = res.get("url", "")
    url_u, url_p, clean_url = extract_url_credentials(raw_url)
    plain_u = res.get("username") or url_u
    plain_p = res.get("password") or url_p

    has_plaintext = bool(plain_p)

    if not (username and password) and has_plaintext:
        if not is_legacy_plaintext_allowed(res, legacy_allow_plaintext):
            raise ValueError(
                f"Camera '{camera_id}': Plaintext RTSP passwords in configuration are rejected by default. "
                f"Use environment variables (username_env/password_env), an encrypted credential file, "
                f"or set ARGUS_LEGACY_ALLOW_PLAINTEXT_CREDS=true to enable explicit legacy fallback."
            )
        username = plain_u
        password = plain_p

    # Reconstruct clean RTSP URL or populate fields
    if username or password:
        res["username"] = username or ""
        res["password"] = password or ""

        host = res.get("host")
        port = res.get("port", 554)
        path = res.get("path", "")

        if host:
            if path and not path.startswith("/"):
                path = "/" + path
            auth = f"{username}:{password}@" if username else ""
            res["url"] = f"rtsp://{auth}{host}:{port}{path}"
        elif raw_url:
            auth = f"{username}:{password}@" if username else ""
            clean_base = re.sub(r"^rtsp://(?:[^@]+@)?", "rtsp://", raw_url)
            res["url"] = re.sub(r"^rtsp://", f"rtsp://{auth}", clean_base)
    elif "url" in res:
        res["url"] = clean_url

    return res
