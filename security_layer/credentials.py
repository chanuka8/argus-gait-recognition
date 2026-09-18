import base64
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from core.paths import resolve_app_path


class CameraTransportSecurityError(ValueError):
    """Raised when camera stream transport does not comply with security policy."""


SENSITIVE_QUERY_KEYS: set[str] = {
    "password",
    "pass",
    "token",
    "access_token",
    "api_key",
    "apikey",
    "secret",
    "auth",
    "key",
}

STREAM_URL_REGEX = re.compile(r"((?:rtsp|rtsps|https?)://[^\s\"'<>,;]+)", re.IGNORECASE)


def sanitize_stream_url(url: str | None) -> str:
    """Sanitize credentials and sensitive query parameters from stream URLs and log strings.

    Redacts:
    - User credentials (username:password in URL authority) -> '***:***@'
    - Sensitive query parameters (token, access_token, api_key, password, secret, etc.) -> '***'

    Malformed URLs fail safely without echoing raw sensitive tokens.
    """
    if not url or not isinstance(url, str):
        return ""

    def _sanitize_url_match(match: re.Match) -> str:
        target = match.group(1)
        try:
            parsed = urlsplit(target)
            scheme = parsed.scheme.lower()
            if scheme not in ("rtsp", "rtsps", "http", "https"):
                return target

            netloc = parsed.netloc
            if "@" in netloc:
                _userinfo, hostport = netloc.rsplit("@", 1)
                new_netloc = f"***:***@{hostport}"
            else:
                new_netloc = netloc

            new_query = ""
            if parsed.query:
                pairs = parse_qsl(parsed.query, keep_blank_values=True)
                sanitized_pairs = []
                for k, v in pairs:
                    k_lower = k.lower()
                    if k_lower in SENSITIVE_QUERY_KEYS or any(
                        s in k_lower for s in ("password", "secret", "token", "apikey", "api_key")
                    ):
                        sanitized_pairs.append((k, "***"))
                    else:
                        sanitized_pairs.append((k, v))
                new_query = urlencode(sanitized_pairs, safe="*:")

            return urlunsplit((parsed.scheme, new_netloc, parsed.path, new_query, parsed.fragment))
        except Exception:  # noqa: BLE001
            # Fallback for malformed URLs
            safe = re.sub(r"://([^:\s@]+):([^\s@]+)@", r"://***:***@", target)
            for k in SENSITIVE_QUERY_KEYS:
                safe = re.sub(rf"([?&]{k}=)[^&\s]+", r"\1***", safe, flags=re.IGNORECASE)
            return safe

    return STREAM_URL_REGEX.sub(_sanitize_url_match, url)


def sanitize_rtsp_url(url: str | None) -> str:
    """Backward-compatible wrapper for sanitize_stream_url."""
    return sanitize_stream_url(url)


def extract_stream_credentials(url: str) -> tuple[str | None, str | None, str]:
    """Extract username and password from a stream URL (rtsp://, rtsps://), returning clean URL."""
    if not url or not isinstance(url, str):
        return None, None, ""

    match = re.search(
        r"^(rtsp|rtsps)://([^:\s]+):(.+)@([^/\s]+(?::\d+)?(?:/.*)?)$",
        url.strip(),
        flags=re.IGNORECASE,
    )
    if match:
        scheme = match.group(1).lower()
        user = unquote(match.group(2))
        passwd = unquote(match.group(3))
        host_and_path = match.group(4)
        clean_url = f"{scheme}://{host_and_path}"
        return user, passwd, clean_url

    return None, None, url


def extract_rtsp_credentials(url: str) -> tuple[str | None, str | None, str]:
    """Backward-compatible wrapper for extract_stream_credentials."""
    return extract_stream_credentials(url)


extract_url_credentials = extract_rtsp_credentials


def build_stream_url(
    base_url: str,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Inject credentials into an RTSP or RTSPS stream URL."""
    if not base_url or not isinstance(base_url, str):
        return ""

    _, _, clean_url = extract_stream_credentials(base_url)

    if not username and not password:
        return clean_url

    user_str = quote(str(username or ""), safe="")
    pass_str = quote(str(password or ""), safe="")

    scheme_match = re.match(r"^(rtsp|rtsps)://", clean_url, flags=re.IGNORECASE)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        host_part = clean_url[len(scheme) + 3 :]
        if pass_str:
            return f"{scheme}://{user_str}:{pass_str}@{host_part}"
        elif user_str:
            return f"{scheme}://{user_str}@{host_part}"
        return clean_url

    return clean_url


def build_rtsp_url(
    base_url: str,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Backward-compatible wrapper for build_stream_url."""
    return build_stream_url(base_url, username, password)


def get_deployment_environment(
    override: str | None = None,
    config_path: str | Path | None = None,
) -> str:
    """Resolve canonical deployment environment.

    Precedence:
    1. Explicit override
    2. ARGUS_ENVIRONMENT env var
    3. configs/production.yaml deployment.target_environment
    4. 'development' default
    """
    if override is not None and str(override).strip():
        return str(override).strip().lower()

    env_val = os.environ.get("ARGUS_ENVIRONMENT", "").strip().lower()
    if env_val:
        return env_val

    if config_path is not None:
        prod_yaml_path = resolve_app_path(config_path)
    else:
        prod_yaml_path = resolve_app_path("configs/production.yaml")


    if prod_yaml_path.is_file():
        try:
            import yaml

            with open(prod_yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                target_env = data.get("deployment", {}).get("target_environment")
                if target_env and str(target_env).strip():
                    return str(target_env).strip().lower()
        except (OSError, ValueError, TypeError):
            pass

    return "development"


def is_secure_camera_transport_required(
    override: bool | None = None,
    config_path: str | Path | None = None,
) -> bool:
    """Check if strict camera transport security is enforced.

    Strict U4 is enabled when:
    - explicit override=True
    - ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true
    - canonical deployment environment == 'production'
    """
    if override is not None:
        return bool(override)

    env_val = os.environ.get("ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT", "").strip().lower()
    if env_val in ("true", "1", "yes"):
        return True

    return get_deployment_environment(config_path=config_path) == "production"


def validate_camera_transport(
    source: Any,
    config: dict[str, Any] | None = None,
    strict_mode: bool | None = None,
    enforce: bool = True,
) -> dict[str, Any]:
    """Classify and validate camera stream transport security.

    Distinguishes 7 distinct transport categories:
    1. 'local' - USB webcams, DirectShow devices, local video files.
    2. 'protected_tunnel' - RTSP over WireGuard, IPSec, or authenticated VPN tunnel (operator-asserted).
    3. 'rtsps_candidate' - Native RTSPS (rtsps://). Note: OpenCV FFmpeg recognizes scheme, but application
       cannot cryptographically prove CA trust, SAN, or revocation without operator confirmation.
    4. 'https_candidate' - HTTPS stream (https://).
    5. 'plaintext_rtsp' - Unencrypted RTSP (rtsp://).
    6. 'plaintext_http' - Unencrypted HTTP (http://).
    7. 'unsupported' - Unknown or invalid stream scheme.

    In strict mode (ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true):
    - 'local' and 'protected_tunnel' are allowed.
    - 'rtsps_candidate' is allowed ONLY if operator explicitly asserts verified transport
      (e.g., config['allow_rtsps_candidate'] is True or config['transport_security'] in ('rtsps_verified', 'verified_native_tls')
      or ARGUS_ALLOW_RTSPS_CANDIDATE=true).
    - 'plaintext_rtsp' and 'plaintext_http' are REJECTED.

    Returns dict with classification, allowed flag, reason, and transport security metadata.
    If enforce is True and not allowed, raises CameraTransportSecurityError.
    """
    cfg = config or {}
    strict = is_secure_camera_transport_required(strict_mode)

    # 1. Local device check
    if isinstance(source, int) or (isinstance(source, str) and (source.isdigit() or source.startswith("usb:"))):
        return {
            "allowed": True,
            "category": "local",
            "transport_scheme": "local",
            "is_encrypted": True,
            "verified_by_argus": True,
            "reason": "Local camera hardware device",
            "description": "Local USB or V4L2/DirectShow capture",
        }

    source_str = str(source).strip() if source is not None else ""

    # Check for local video file
    if source_str and (
        Path(source_str).exists() or any(source_str.lower().endswith(ext) for ext in (".mp4", ".avi", ".mkv", ".mov"))
    ):
        return {
            "allowed": True,
            "category": "local",
            "transport_scheme": "file",
            "is_encrypted": True,
            "verified_by_argus": True,
            "reason": "Local media file",
            "description": "Local video file playback",
        }

    # 2. Check for operator-asserted protected tunnel
    transport_sec_cfg = str(cfg.get("transport_security", "")).strip().lower()
    is_tunnel_cfg = bool(cfg.get("is_tunnel") or cfg.get("protected_tunnel"))
    tunnel_type = cfg.get("tunnel_type") or "operator_asserted"

    if is_tunnel_cfg or transport_sec_cfg in ("protected_tunnel", "vpn", "ipsec", "wireguard", "tls_proxy"):
        return {
            "allowed": True,
            "category": "protected_tunnel",
            "transport_scheme": "rtsp",
            "is_encrypted": True,
            "verified_by_argus": False,  # ARGUS cannot cryptographically verify external network tunnel
            "tunnel_type": tunnel_type,
            "reason": "Operator-asserted protected tunnel (WireGuard/IPSec/VPN)",
            "description": "Network stream encapsulated in verified external encrypted tunnel",
        }

    # Scheme inspection
    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+-.]*)://", source_str)
    scheme = scheme_match.group(1).lower() if scheme_match else ""

    # 3. RTSPS candidate
    if scheme == "rtsps":
        operator_confirmed = bool(
            cfg.get("allow_rtsps_candidate")
            or transport_sec_cfg in ("rtsps_verified", "rtsps_confirmed", "verified_native_tls")
            or os.environ.get("ARGUS_ALLOW_RTSPS_CANDIDATE", "").strip().lower() in ("true", "1", "yes")
        )

        if strict and not operator_confirmed:
            reason = (
                "RTSPS scheme recognized ('rtsps://'), but native certificate validation is not "
                "independently proven by OpenCV runtime. Strict mode requires explicit operator "
                "confirmation of verified transport (set transport_security='rtsps_verified' or "
                "allow_rtsps_candidate=True)."
            )
            if enforce:
                raise CameraTransportSecurityError(reason)
            return {
                "allowed": False,
                "category": "rtsps_candidate",
                "transport_scheme": "rtsps",
                "is_encrypted": True,
                "verified_by_argus": False,
                "reason": reason,
                "description": "RTSPS candidate stream requiring operator verification confirmation",
            }

        return {
            "allowed": True,
            "category": "rtsps_candidate",
            "transport_scheme": "rtsps",
            "is_encrypted": True,
            "verified_by_argus": False,  # Explicitly NOT claimed to be verified by ARGUS runtime
            "operator_confirmed": operator_confirmed,
            "reason": (
                "RTSPS transport candidate accepted (operator confirmation configured)"
                if operator_confirmed
                else "RTSPS transport candidate accepted (permissive mode)"
            ),
            "description": "Native RTSPS stream candidate",
        }

    # 4. HTTPS candidate
    if scheme == "https":
        operator_confirmed = bool(
            cfg.get("allow_https_candidate")
            or transport_sec_cfg in ("https_verified", "verified_native_tls")
            or os.environ.get("ARGUS_ALLOW_HTTPS_CANDIDATE", "").strip().lower() in ("true", "1", "yes")
        )
        if strict and not operator_confirmed:
            reason = "HTTPS stream candidate requires operator verification confirmation in strict mode."
            if enforce:
                raise CameraTransportSecurityError(reason)
            return {
                "allowed": False,
                "category": "https_candidate",
                "transport_scheme": "https",
                "is_encrypted": True,
                "verified_by_argus": False,
                "reason": reason,
            }
        return {
            "allowed": True,
            "category": "https_candidate",
            "transport_scheme": "https",
            "is_encrypted": True,
            "verified_by_argus": False,
            "reason": "HTTPS candidate accepted",
        }

    # 5. Plaintext RTSP
    if scheme == "rtsp":
        if strict:
            reason = (
                "Plaintext RTSP transport ('rtsp://') is rejected under strict camera transport security "
                "(ARGUS_REQUIRE_SECURE_CAMERA_TRANSPORT=true). Stream traffic and credentials are vulnerable "
                "to network sniffing and MITM tampering. Remediate with a protected tunnel (WireGuard/IPSec) "
                "or native verified RTSPS."
            )
            if enforce:
                raise CameraTransportSecurityError(reason)
            return {
                "allowed": False,
                "category": "plaintext_rtsp",
                "transport_scheme": "rtsp",
                "is_encrypted": False,
                "verified_by_argus": True,
                "reason": reason,
                "description": "Unencrypted RTSP stream",
            }

        return {
            "allowed": True,
            "category": "plaintext_rtsp",
            "transport_scheme": "rtsp",
            "is_encrypted": False,
            "verified_by_argus": True,
            "reason": "Plaintext RTSP accepted in permissive mode (transport remediation pending)",
            "description": "Unencrypted RTSP stream (vulnerable to eavesdropping)",
        }

    # 6. Plaintext HTTP
    if scheme == "http":
        if strict:
            reason = "Plaintext HTTP transport ('http://') is rejected under strict transport security."
            if enforce:
                raise CameraTransportSecurityError(reason)
            return {
                "allowed": False,
                "category": "plaintext_http",
                "transport_scheme": "http",
                "is_encrypted": False,
                "verified_by_argus": True,
                "reason": reason,
            }
        return {
            "allowed": True,
            "category": "plaintext_http",
            "transport_scheme": "http",
            "is_encrypted": False,
            "verified_by_argus": True,
            "reason": "Plaintext HTTP accepted in permissive mode",
        }

    # 7. Unsupported scheme
    reason = f"Unsupported camera source or stream scheme: '{scheme or source_str}'"
    if enforce:
        raise CameraTransportSecurityError(reason)
    return {
        "allowed": False,
        "category": "unsupported",
        "transport_scheme": scheme or "unknown",
        "is_encrypted": False,
        "verified_by_argus": True,
        "reason": reason,
    }


def derive_fernet_key(passphrase: str, salt: bytes = b"argus_rtsp_salt") -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


class CredentialManager:
    def __init__(
        self,
        credentials_file: str | Path = "configs/credentials.enc",
        key: str | None = None,
    ) -> None:
        self.credentials_file = resolve_app_path(os.environ.get("ARGUS_CREDENTIALS_FILE", credentials_file))
        raw_key = key or os.environ.get("ARGUS_CREDENTIAL_ENCRYPTION_KEY") or os.environ.get("ARGUS_CREDENTIALS_KEY")

        key_file = resolve_app_path(".credentials.key")

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
        return Fernet.generate_key().decode("utf-8")

    def _load_raw_store(self) -> dict[str, Any]:
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
        store = self._load_raw_store()
        results = []
        for cid in store.get("credentials", {}):
            if self.can_access(cid, user_id=user_id):
                meta = self.get_credential_metadata(cid, user_id=user_id)
                if meta:
                    results.append(meta)
        return results

    def delete_credential(self, credential_id: str, user_id: str = "default_user") -> bool:
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
        store = self._load_raw_store()
        return credential_id in store.get("credentials", {})

    def encrypt_credentials(self, credentials_data: dict[str, dict[str, str]], output_path: str | None = None) -> Path:
        for cid, cdata in credentials_data.items():
            self.store_credential(
                owner_user_id="system_admin",
                username=cdata.get("username", ""),
                password=cdata.get("password", ""),
                credential_id=cid,
            )
        return Path(output_path) if output_path else self.credentials_file

    def load_encrypted_credentials(self) -> dict[str, dict[str, str]]:
        store = self._load_raw_store()
        out = {}
        for cid, record in store.get("credentials", {}).items():
            out[cid] = {
                "username": record.get("username", ""),
                "password": record.get("password", ""),
            }
        return out

    def get_credentials(self, camera_id: str) -> tuple[str | None, str | None]:
        cred = self.get_credential(camera_id, user_id="system_admin")
        if cred:
            return cred.get("username"), cred.get("password")
        return None, None


def is_legacy_plaintext_allowed(config: dict[str, Any], override: bool | None = None) -> bool:
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
        protocol = str(res.get("protocol") or "").lower()
        if not protocol:
            if raw_url.lower().startswith("rtsps://"):
                protocol = "rtsps"
            else:
                protocol = "rtsp"
        scheme = "rtsps" if protocol == "rtsps" else "rtsp"
        default_port = 322 if scheme == "rtsps" else 554
        port = res.get("port", default_port)
        path = res.get("path", "")

        if host:
            if path and not path.startswith("/"):
                path = "/" + path
            res["url"] = build_stream_url(f"{scheme}://{host}:{port}{path}", username, password)
        elif raw_url:
            res["url"] = build_stream_url(clean_url, username, password)
    elif "url" in res:
        res["url"] = clean_url

    target_url = res.get("url")
    if target_url:
        strict = is_secure_camera_transport_required()
        validation = validate_camera_transport(target_url, config=res, strict_mode=strict, enforce=False)
        res["transport_category"] = validation.get("category")
        res["transport_scheme"] = validation.get("transport_scheme")
        res["transport_allowed"] = validation.get("allowed")
        res["transport_reason"] = validation.get("reason")
        if strict and not validation.get("allowed", False):
            raise CameraTransportSecurityError(f"Camera '{camera_id}': {validation.get('reason')}")

    return res
