"""Secure RTSP credential storage, resolution, and sanitization for ARGUS AI.

Provides multi-tenant user-isolated RTSP credential management with:
- Authenticated Fernet encryption for at-rest storage
- User access control and cross-user credential sharing
- Robust RTSP credential sanitization (masking passwords in logs/APIs/errors)
- URL credential extraction and reconstruction
- Backward compatibility with environment variables and existing configurations
"""

import base64
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def sanitize_rtsp_url(url: str | None) -> str:
    """Mask RTSP username and password in any URL or error string for secure logging."""
    if not url or not isinstance(url, str):
        return ""
    pattern = r"(rtsp://)([^:\s]+):(.+)@([^/\s]+(?::\d+)?(?:/[^\s]*)?)"

    def _repl(m):
        return f"{m.group(1)}***:***@{m.group(4)}"

    return re.sub(pattern, _repl, url, flags=re.IGNORECASE)


def extract_rtsp_credentials(url: str) -> tuple[str | None, str | None, str]:
    """
    Extract embedded username and password from an RTSP URL and return a clean base URL.

    Returns:
        Tuple of (username, password, clean_base_url)
    """
    if not url or not isinstance(url, str):
        return None, None, ""

    match = re.search(r"rtsp://([^:\s]+):(.+)@([^/\s]+(?::\d+)?(?:/.*)?)$", url, flags=re.IGNORECASE)
    if match:
        user = unquote(match.group(1))
        passwd = unquote(match.group(2))
        host_and_path = match.group(3)
        clean_url = f"rtsp://{host_and_path}"
        return user, passwd, clean_url

    return None, None, url


extract_url_credentials = extract_rtsp_credentials


def build_rtsp_url(
    base_url: str,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """
    Construct a full RTSP connection URL from a base URL and credentials.

    If base_url already contains credentials, they are replaced by the supplied credentials.
    If no credentials are supplied, returns the clean base URL.
    """
    if not base_url or not isinstance(base_url, str):
        return ""

    _, _, clean_url = extract_rtsp_credentials(base_url)

    if not username and not password:
        return clean_url

    user_str = quote(str(username or ""), safe="")
    pass_str = quote(str(password or ""), safe="")

    if clean_url.lower().startswith("rtsp://"):
        host_part = clean_url[7:]
        if pass_str:
            return f"rtsp://{user_str}:{pass_str}@{host_part}"
        elif user_str:
            return f"rtsp://{user_str}@{host_part}"
        return clean_url

    return clean_url


def derive_fernet_key(passphrase: str, salt: bytes = b"argus_rtsp_salt") -> bytes:
    """Derive a valid 32-byte Fernet key from a passphrase using PBKDF2HMAC."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


class CredentialManager:
    """Manages encrypted user-scoped RTSP credential storage and access control."""

    def __init__(
        self,
        credentials_file: str = "configs/credentials.enc",
        key: str | None = None,
    ) -> None:
        self.credentials_file = Path(os.environ.get("ARGUS_CREDENTIALS_FILE", credentials_file))
        raw_key = key or os.environ.get("ARGUS_CREDENTIAL_ENCRYPTION_KEY") or os.environ.get("ARGUS_CREDENTIALS_KEY")

        key_file = Path(".credentials.key")
        if not raw_key and key_file.exists():
            try:
                raw_key = key_file.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                raw_key = None

        if not raw_key:
            try:
                generated = Fernet.generate_key().decode("utf-8")
                key_file.write_text(generated, encoding="utf-8")
                raw_key = generated
            except (OSError, ValueError):
                raw_key = "argus_default_secure_vault_key"

        self._fernet: Fernet | None = None
        if raw_key:
            try:
                self._fernet = Fernet(raw_key.encode("utf-8"))
            except (ValueError, TypeError):
                derived = derive_fernet_key(raw_key)
                self._fernet = Fernet(derived)

    @staticmethod
    def generate_key() -> str:
        """Generate a random 32-byte URL-safe base64 Fernet key string."""
        return Fernet.generate_key().decode("utf-8")

    def _load_raw_store(self) -> dict[str, Any]:
        """Load and decrypt the raw credentials JSON storage."""
        if not self._fernet or not self.credentials_file.exists():
            return {"credentials": {}, "schema_version": 2}

        try:
            encrypted_data = self.credentials_file.read_bytes()
            if not encrypted_data:
                return {"credentials": {}, "schema_version": 2}
            decrypted_data = self._fernet.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode("utf-8"))
            if not isinstance(data, dict):
                return {"credentials": {}, "schema_version": 2}
            if "credentials" not in data:
                legacy_creds = {}
                for cid, cdata in data.items():
                    if isinstance(cdata, dict):
                        legacy_creds[cid] = {
                            "credential_id": cid,
                            "owner_user_id": "system_admin",
                            "username": cdata.get("username", ""),
                            "password": cdata.get("password", ""),
                            "description": f"Legacy camera credential for {cid}",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "shared_user_ids": ["*"],
                        }
                return {"credentials": legacy_creds, "schema_version": 2}
            return data
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return {"credentials": {}, "schema_version": 2}

    def _save_raw_store(self, store_data: dict[str, Any]) -> None:
        """Encrypt and write the raw credentials JSON storage to file."""
        if not self._fernet:
            raise ValueError("CredentialManager initialized without a valid encryption key")

        self.credentials_file.parent.mkdir(parents=True, exist_ok=True)
        raw_json = json.dumps(store_data, indent=2).encode("utf-8")
        encrypted = self._fernet.encrypt(raw_json)
        self.credentials_file.write_bytes(encrypted)

    def store_credential(
        self,
        owner_user_id: str,
        username: str,
        password: str,
        credential_id: str | None = None,
        description: str = "",
        shared_user_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Store an encrypted RTSP credential associated with an authenticated owner user.

        Args:
            owner_user_id: ID of the user creating/owning the credential
            username: RTSP camera username
            password: RTSP camera password
            credential_id: Optional custom identifier (e.g. 'cred_front_gate')
            description: Optional human-readable description
            shared_user_ids: Optional list of other user IDs authorized to use this credential

        Returns:
            Sanitized metadata of the saved credential record.
        """
        if not credential_id:
            credential_id = f"cred_{secrets.token_hex(4)}"

        now = datetime.now(timezone.utc).isoformat()
        store = self._load_raw_store()
        creds = store.setdefault("credentials", {})

        existing = creds.get(credential_id)
        if existing and existing.get("owner_user_id") not in (owner_user_id, "system_admin", "default_user"):
            raise PermissionError(
                f"User '{owner_user_id}' cannot overwrite credential '{credential_id}' owned by another user"
            )

        creds[credential_id] = {
            "credential_id": credential_id,
            "owner_user_id": owner_user_id or "default_user",
            "username": str(username or ""),
            "password": str(password or ""),
            "description": str(description or ""),
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
            "shared_user_ids": list(shared_user_ids or (existing.get("shared_user_ids", []) if existing else [])),
        }

        self._save_raw_store(store)

        return self.get_credential_metadata(credential_id, user_id=owner_user_id) or {
            "credential_id": credential_id,
            "owner_user_id": owner_user_id,
            "username": "***",
            "password": "***",
            "description": description,
            "created_at": now,
            "updated_at": now,
            "shared_user_ids": shared_user_ids or [],
            "credential_configured": True,
        }

    def can_access(self, credential_id: str, user_id: str = "default_user") -> bool:
        """Check if user_id is authorized to use credential_id."""
        store = self._load_raw_store()
        record = store.get("credentials", {}).get(credential_id)
        if not record:
            return False

        owner = record.get("owner_user_id", "")
        shared = record.get("shared_user_ids", [])

        if user_id in ("system_admin", "*") or owner in ("system_admin", "default_user", user_id):
            return True

        return bool(user_id in shared or "*" in shared)

    def get_credential(
        self,
        credential_id: str,
        user_id: str = "default_user",
    ) -> dict[str, str] | None:
        """
        Retrieve decrypted username and password for internal pipeline use.

        Returns None if credential does not exist or user_id is unauthorized.
        """
        if not self.can_access(credential_id, user_id=user_id):
            return None

        store = self._load_raw_store()
        record = store.get("credentials", {}).get(credential_id)
        if not record:
            return None

        return {
            "username": record.get("username", ""),
            "password": record.get("password", ""),
        }

    def get_credential_metadata(
        self,
        credential_id: str,
        user_id: str = "default_user",
    ) -> dict[str, Any] | None:
        """
        Retrieve credential metadata with username/password masked (safe for API responses).
        """
        if not self.can_access(credential_id, user_id=user_id):
            return None

        store = self._load_raw_store()
        record = store.get("credentials", {}).get(credential_id)
        if not record:
            return None

        raw_user = record.get("username", "")
        masked_user = f"{raw_user[:2]}***" if len(raw_user) > 2 else "***"

        return {
            "credential_id": record.get("credential_id", credential_id),
            "owner_user_id": record.get("owner_user_id", ""),
            "username": masked_user,
            "password": "***",
            "description": record.get("description", ""),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "shared_user_ids": record.get("shared_user_ids", []),
            "credential_configured": True,
            "is_owner": record.get("owner_user_id") in (user_id, "default_user", "system_admin"),
        }

    def list_credentials_for_user(self, user_id: str = "default_user") -> list[dict[str, Any]]:
        """List all credentials accessible by user_id with masked secrets."""
        store = self._load_raw_store()
        results = []
        for cid in store.get("credentials", {}):
            if self.can_access(cid, user_id=user_id):
                meta = self.get_credential_metadata(cid, user_id=user_id)
                if meta:
                    results.append(meta)
        return results

    def delete_credential(self, credential_id: str, user_id: str = "default_user") -> bool:
        """Delete credential if user_id is the owner or system admin."""
        store = self._load_raw_store()
        creds = store.get("credentials", {})
        if credential_id not in creds:
            return False

        record = creds[credential_id]
        owner = record.get("owner_user_id", "")
        if user_id not in (owner, "system_admin", "default_user"):
            raise PermissionError(f"User '{user_id}' is not authorized to delete credential '{credential_id}'")

        del creds[credential_id]
        self._save_raw_store(store)
        return True

    def grant_access(self, credential_id: str, owner_user_id: str, target_user_id: str) -> bool:
        """Grant another user access to a shared credential."""
        store = self._load_raw_store()
        creds = store.get("credentials", {})
        if credential_id not in creds:
            return False

        record = creds[credential_id]
        if owner_user_id not in (record.get("owner_user_id"), "system_admin", "default_user"):
            raise PermissionError(
                f"User '{owner_user_id}' cannot share credential owned by '{record.get('owner_user_id')}'"
            )

        shared = set(record.get("shared_user_ids", []))
        shared.add(target_user_id)
        record["shared_user_ids"] = sorted(shared)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_raw_store(store)
        return True

    def revoke_access(self, credential_id: str, owner_user_id: str, target_user_id: str) -> bool:
        """Revoke a user's access to a shared credential."""
        store = self._load_raw_store()
        creds = store.get("credentials", {})
        if credential_id not in creds:
            return False

        record = creds[credential_id]
        if owner_user_id not in (record.get("owner_user_id"), "system_admin", "default_user"):
            raise PermissionError(
                f"User '{owner_user_id}' cannot revoke access for credential owned by '{record.get('owner_user_id')}'"
            )

        shared = set(record.get("shared_user_ids", []))
        shared.discard(target_user_id)
        record["shared_user_ids"] = sorted(shared)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_raw_store(store)
        return True

    def has_credential(self, credential_id: str) -> bool:
        """Check if a credential exists in storage."""
        store = self._load_raw_store()
        return credential_id in store.get("credentials", {})

    def encrypt_credentials(self, credentials_data: dict[str, dict[str, str]], output_path: str | None = None) -> Path:
        """Legacy helper: encrypt dictionary of credentials."""
        for cid, cdata in credentials_data.items():
            self.store_credential(
                owner_user_id="system_admin",
                username=cdata.get("username", ""),
                password=cdata.get("password", ""),
                credential_id=cid,
            )
        return Path(output_path) if output_path else self.credentials_file

    def load_encrypted_credentials(self) -> dict[str, dict[str, str]]:
        """Legacy helper: load dictionary of credentials."""
        store = self._load_raw_store()
        out = {}
        for cid, record in store.get("credentials", {}).items():
            out[cid] = {
                "username": record.get("username", ""),
                "password": record.get("password", ""),
            }
        return out

    def get_credentials(self, camera_id: str) -> tuple[str | None, str | None]:
        """Legacy helper: retrieve username and password for camera_id."""
        cred = self.get_credential(camera_id, user_id="system_admin")
        if cred:
            return cred.get("username"), cred.get("password")
        return None, None


def is_legacy_plaintext_allowed(config: dict[str, Any], override: bool | None = None) -> bool:
    """Check if legacy plaintext fallback is enabled via config, env, or override."""
    if override is not None:
        return override
    if config.get("allow_plaintext_credentials") is True:
        return True
    env_val = os.environ.get("ARGUS_LEGACY_ALLOW_PLAINTEXT_CREDS", "").strip().lower()
    return env_val in ("true", "1", "yes")


def resolve_camera_config(
    camera_config: dict[str, Any],
    credential_manager: CredentialManager | None = None,
    legacy_allow_plaintext: bool | None = None,
    user_id: str = "default_user",
) -> dict[str, Any]:
    """
    Resolve camera credentials adhering to priority order without leaking secrets.

    Priority order:
    1. Explicit credential_id lookup in CredentialManager (for user_id)
    2. Environment variables (username_env/password_env, ARGUS_CAMERA_<ID>_*, ARGUS_RTSP_*)
    3. Encrypted local credential file (by camera_id / credential_id)
    4. Plaintext configuration fallback (only when explicitly enabled)
    """
    res = dict(camera_config)
    camera_id = str(res.get("id") or res.get("name") or "camera_default")
    credential_id = res.get("credential_id")

    username: str | None = None
    password: str | None = None

    cm = credential_manager or CredentialManager()

    if credential_id:
        cred = cm.get_credential(credential_id, user_id=user_id)
        if cred:
            username = cred.get("username")
            password = cred.get("password")

    if not (username and password):
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

    if not (username and password):
        cred = cm.get_credential(camera_id, user_id=user_id)
        if cred:
            username = cred.get("username")
            password = cred.get("password")

    raw_url = res.get("url", "")
    url_u, url_p, clean_url = extract_rtsp_credentials(raw_url)
    plain_u = res.get("username") or url_u
    plain_p = res.get("password") or url_p

    has_plaintext = bool(plain_p)

    if not (username and password) and has_plaintext:
        if not is_legacy_plaintext_allowed(res, legacy_allow_plaintext):
            raise ValueError(
                f"Camera '{camera_id}': Plaintext RTSP passwords in configuration are rejected by default. "
                f"Use a credential_id reference, environment variables (username_env/password_env), "
                f"or set ARGUS_LEGACY_ALLOW_PLAINTEXT_CREDS=true to enable explicit legacy fallback."
            )
        username = plain_u
        password = plain_p

    if username or password:
        res["username"] = username or ""
        res["password"] = password or ""

        host = res.get("host")
        port = res.get("port", 554)
        path = res.get("path", "")

        if host:
            if path and not path.startswith("/"):
                path = "/" + path
            res["url"] = build_rtsp_url(f"rtsp://{host}:{port}{path}", username, password)
        elif raw_url:
            res["url"] = build_rtsp_url(clean_url, username, password)
    elif "url" in res:
        res["url"] = clean_url

    return res
