"""Model Checkpoint Authenticity, Integrity, and Safe Deserialization Module.

Finding U5 — Model Checkpoint / AI Model Asset Protection (Phase 1).
Provides:
- Canonical manifest parsing and serialization
- Asymmetric Ed25519 digital signature verification
- Streaming SHA-256 digest calculation
- Model-role and model-ID binding
- Strict fail-closed verification gating for model loaders
- Operator-controlled development/legacy mode support
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger("argus.security.model_integrity")

# Stable model role identifiers
ROLE_PERSON_DETECTOR = "person_detector"
ROLE_SILHOUETTE_SEGMENTER = "silhouette_segmenter"
ROLE_GAIT_EMBEDDING = "gait_embedding"
ROLE_APPEARANCE_EMBEDDING = "appearance_embedding"

VALID_ROLES: frozenset[str] = frozenset(
    {
        ROLE_PERSON_DETECTOR,
        ROLE_SILHOUETTE_SEGMENTER,
        ROLE_GAIT_EMBEDDING,
        ROLE_APPEARANCE_EMBEDDING,
    }
)

ROLE_SYNONYMS: dict[str, str] = {
    "gait_encoder": ROLE_GAIT_EMBEDDING,
    "appearance_reid": ROLE_APPEARANCE_EMBEDDING,
    "reid": ROLE_APPEARANCE_EMBEDDING,
    "detector": ROLE_PERSON_DETECTOR,
    "segmenter": ROLE_SILHOUETTE_SEGMENTER,
}

SUPPORTED_MANIFEST_VERSIONS: frozenset[str] = frozenset({"1.0"})


class ModelIntegrityError(Exception):
    """Base exception for all model authenticity and integrity failures."""


class ModelManifestError(ModelIntegrityError):
    """Raised when model manifest is missing, malformed, or unparseable."""


class ModelSignatureError(ModelIntegrityError):
    """Raised when digital signature is missing, invalid, or verification fails."""


class ModelDigestMismatchError(ModelIntegrityError):
    """Raised when model SHA-256 hash or byte size does not match manifest."""


class ModelRoleMismatchError(ModelIntegrityError):
    """Raised when model role or model ID does not match expected binding."""


class ModelTrustConfigurationError(ModelIntegrityError):
    """Raised when trusted verification key is missing or misconfigured in strict mode."""


def normalize_role(role: str) -> str:
    """Normalize model role string to canonical identifier."""
    cleaned = role.strip().lower()
    return ROLE_SYNONYMS.get(cleaned, cleaned)


def canonical_manifest_bytes(manifest_data: dict[str, Any]) -> bytes:
    """Serialize manifest dictionary to deterministic canonical JSON bytes.

    Uses sorted keys, no whitespace separators, and utf-8 encoding.
    """
    return json.dumps(
        manifest_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_file_sha256(file_path: str | Path, chunk_size: int = 65536) -> tuple[str, int]:
    """Calculate streaming SHA-256 hex digest and byte length of a file."""
    path = Path(file_path)
    if not path.is_file():
        raise ModelIntegrityError(f"Model file not found or not a regular file: {path.name}")

    hasher = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            total_bytes += len(chunk)

    return hasher.hexdigest().lower(), total_bytes


def load_public_key(key_material: str | bytes | ed25519.Ed25519PublicKey) -> ed25519.Ed25519PublicKey:
    """Load an Ed25519 public key from bytes, PEM string, Base64, raw hex, or existing object."""
    if isinstance(key_material, ed25519.Ed25519PublicKey):
        return key_material

    if isinstance(key_material, str):
        key_str = key_material.strip()
        # Check if it's a file path
        if len(key_str) < 512 and Path(key_str).is_file():
            key_material = Path(key_str).read_bytes()
        elif key_str.startswith("-----BEGIN"):
            key_material = key_str.encode("utf-8")
        elif len(key_str) == 64 and all(c in "0123456789abcdefABCDEF" for c in key_str):
            key_material = bytes.fromhex(key_str)
        else:
            try:
                key_material = base64.b64decode(key_str)
            except Exception as e:
                raise ModelTrustConfigurationError(f"Failed to decode public key material: {e}") from e

    if isinstance(key_material, (bytes, bytearray)):
        raw_bytes = bytes(key_material)
        if raw_bytes.startswith(b"-----BEGIN"):
            try:
                loaded = serialization.load_pem_public_key(raw_bytes)
                if not isinstance(loaded, ed25519.Ed25519PublicKey):
                    raise ModelTrustConfigurationError(f"Expected Ed25519 public key, got {type(loaded).__name__}")
                return loaded
            except Exception as e:
                raise ModelTrustConfigurationError(f"Invalid PEM public key: {e}") from e

        if len(raw_bytes) == 32:
            try:
                return ed25519.Ed25519PublicKey.from_public_bytes(raw_bytes)
            except Exception as e:
                raise ModelTrustConfigurationError(f"Invalid raw 32-byte Ed25519 public key: {e}") from e

        # Try base64 decoding if raw_bytes is ASCII
        try:
            decoded = base64.b64decode(raw_bytes)
            if len(decoded) == 32:
                return ed25519.Ed25519PublicKey.from_public_bytes(decoded)
        except (binascii.Error, ValueError, TypeError):
            pass

    raise ModelTrustConfigurationError("Unsupported public key format; expected 32 raw bytes or PEM encoded Ed25519")


def decode_signature(sig_material: str | bytes) -> bytes:
    """Normalize raw or Base64 signature material to 64-byte Ed25519 signature."""
    if isinstance(sig_material, str):
        cleaned = sig_material.strip()
        if len(cleaned) == 128 and all(c in "0123456789abcdefABCDEF" for c in cleaned):
            raw = bytes.fromhex(cleaned)
        else:
            try:
                raw = base64.b64decode(cleaned)
            except Exception as e:
                raise ModelSignatureError(f"Signature is not valid Base64 or hex: {e}") from e
    elif isinstance(sig_material, (bytes, bytearray)):
        raw = bytes(sig_material)
        if len(raw) != 64:
            # Try Base64 decode on stripped bytes
            try:
                decoded = base64.b64decode(raw.strip())
                if len(decoded) == 64:
                    raw = decoded
            except (binascii.Error, ValueError, TypeError):
                pass
    else:
        raise ModelSignatureError(f"Invalid signature type: {type(sig_material).__name__}")

    if len(raw) != 64:
        raise ModelSignatureError(f"Invalid Ed25519 signature length: expected 64 bytes, got {len(raw)}")

    return raw


class ModelVerifier:
    """Central model verification engine enforcing authenticity and integrity."""

    def __init__(
        self,
        strict_mode: bool | None = None,
        default_manifest_path: str | Path = "models/model_manifest.json",
        default_signature_path: str | Path = "models/model_manifest.sig",
        public_key: str | bytes | ed25519.Ed25519PublicKey | None = None,
    ) -> None:
        if strict_mode is not None:
            self._strict_mode = bool(strict_mode)
        else:
            _TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
            _FALSY_VALUES = frozenset({"0", "false", "no", "off", ""})
            env_var_name = "ARGUS_MODEL_STRICT_MODE"
            env_raw = os.getenv("ARGUS_MODEL_STRICT_MODE")
            if env_raw is None:
                env_var_name = "ARGUS_REQUIRE_SIGNED_MODELS"
                env_raw = os.getenv("ARGUS_REQUIRE_SIGNED_MODELS", "")
            env_val = env_raw.strip().lower()
            if env_val in _TRUTHY_VALUES:
                self._strict_mode = True
            elif env_val in _FALSY_VALUES:
                self._strict_mode = False
            else:
                logger.warning(
                    "Unrecognized value for %s=%r; "
                    "defaulting to strict mode DISABLED. "
                    "Accepted values: 1/true/yes/on (enable) or 0/false/no/off/empty (disable).",
                    env_var_name,
                    env_raw,
                )
                self._strict_mode = False

        self.default_manifest_path = Path(default_manifest_path)
        self.default_signature_path = Path(default_signature_path)
        self._public_key = public_key
        self._warned_models: set[str] = set()

    @property
    def strict_mode(self) -> bool:
        return self._strict_mode

    def is_strict_mode(self) -> bool:
        return self._strict_mode

    def set_strict_mode(self, enabled: bool) -> None:
        self._strict_mode = bool(enabled)

    def resolve_public_key(
        self,
        key_override: str | bytes | ed25519.Ed25519PublicKey | None = None,
    ) -> ed25519.Ed25519PublicKey | None:
        """Resolve trusted public key from override, constructor, environment, or default file."""
        if key_override is not None:
            return load_public_key(key_override)

        if self._public_key is not None:
            return load_public_key(self._public_key)

        env_key = os.getenv("ARGUS_MODEL_SIGNING_PUBLIC_KEY")
        if env_key:
            return load_public_key(env_key)

        default_pub_path = Path("security_layer/keys/model_signing_pubkey.pem")
        if default_pub_path.is_file():
            return load_public_key(default_pub_path.read_bytes())

        return None

    def verify_manifest(
        self,
        manifest_data: dict[str, Any],
        signature_material: str | bytes,
        public_key: ed25519.Ed25519PublicKey,
    ) -> bool:
        """Verify the Ed25519 digital signature over the canonical manifest representation."""
        if not isinstance(manifest_data, dict):
            raise ModelManifestError("Manifest data must be a dictionary")

        m_version = str(manifest_data.get("manifest_version", ""))
        if m_version not in SUPPORTED_MANIFEST_VERSIONS:
            raise ModelManifestError(
                f"Unsupported manifest version: {m_version}. Supported: {sorted(SUPPORTED_MANIFEST_VERSIONS)}"
            )

        if "models" not in manifest_data or not isinstance(manifest_data["models"], dict):
            raise ModelManifestError("Manifest missing 'models' dictionary")

        canonical_bytes = canonical_manifest_bytes(manifest_data)
        sig_bytes = decode_signature(signature_material)

        try:
            public_key.verify(sig_bytes, canonical_bytes)
            return True
        except InvalidSignature as e:
            raise ModelSignatureError("Ed25519 signature verification failed") from e
        except Exception as e:
            raise ModelSignatureError(f"Signature verification encountered an error: {e}") from e

    def load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        """Load and parse JSON model manifest."""
        if not manifest_path.is_file():
            raise ModelManifestError(f"Model manifest file not found: {manifest_path.name}")

        try:
            content = manifest_path.read_text(encoding="utf-8")
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ModelManifestError("Model manifest root must be a JSON object")
            return data
        except json.JSONDecodeError as e:
            raise ModelManifestError(f"Malformed JSON in model manifest: {e}") from e
        except OSError as e:
            raise ModelManifestError(f"Failed to read model manifest: {e}") from e

    def verify_model(
        self,
        model_path: str | Path,
        expected_role: str,
        expected_model_id: str | None = None,
        manifest_path: str | Path | None = None,
        signature_path: str | Path | None = None,
        public_key: str | bytes | ed25519.Ed25519PublicKey | None = None,
    ) -> Path:
        """Primary gate: Verifies model authenticity and integrity before loading.

        In strict mode (ARGUS_REQUIRE_SIGNED_MODELS=true):
        - Fails closed if public key, manifest, signature, or digest validation fails.
        In development mode:
        - Allows unsigned model with a single warning if manifest/sig/key is absent.
        - Fails closed if manifest/sig/key ARE present but verification fails.
        """
        canonical_role = normalize_role(expected_role)
        resolved_path = Path(model_path)
        m_path = Path(manifest_path) if manifest_path else self.default_manifest_path
        s_path = Path(signature_path) if signature_path else self.default_signature_path

        trusted_pubkey = self.resolve_public_key(public_key)
        is_strict = self.is_strict_mode()

        # Check if trust assets exist
        manifest_exists = m_path.is_file()
        sig_exists = s_path.is_file()
        key_exists = trusted_pubkey is not None

        if not is_strict and (not manifest_exists or not sig_exists or not key_exists):
            # Development / legacy mode: allow loading with warning
            warn_key = f"{resolved_path.name}:{canonical_role}"
            if warn_key not in self._warned_models:
                logger.warning(
                    "[SECURITY WARNING] Model signature verification is not enforced; "
                    "unsigned legacy model loading is enabled for %s (role: %s).",
                    resolved_path.name,
                    canonical_role,
                )
                self._warned_models.add(warn_key)
            return resolved_path

        # Strict mode or explicitly provided trust material -> full verification
        if not key_exists:
            raise ModelTrustConfigurationError(
                "Model verification requires a trusted public key, but none is configured."
            )

        if not manifest_exists:
            raise ModelManifestError(f"Model manifest not found: {m_path.name}")

        if not sig_exists:
            raise ModelSignatureError(f"Model signature not found: {s_path.name}")

        manifest_data = self.load_manifest(m_path)
        sig_material = s_path.read_bytes()

        # Verify signature over manifest
        self.verify_manifest(manifest_data, sig_material, trusted_pubkey)

        # Look up model entry in manifest
        models_dict: dict[str, Any] = manifest_data.get("models", {})
        matched_entry: dict[str, Any] | None = None
        matched_key: str | None = None

        # Search by role and/or filename
        req_filename = resolved_path.name
        norm_req_path = resolved_path.as_posix()

        for k, entry in models_dict.items():
            if not isinstance(entry, dict):
                continue
            entry_role = normalize_role(str(entry.get("model_role", "")))
            entry_filename = Path(str(entry.get("filename", ""))).name
            entry_path = Path(str(entry.get("filename", ""))).as_posix()

            if entry_role == canonical_role:
                # Match by role
                if entry_filename == req_filename or entry_path == norm_req_path or k == canonical_role:
                    matched_entry = entry
                    matched_key = k
                    break
            elif entry_filename == req_filename or entry_path == norm_req_path:
                # File matches but role does not match
                raise ModelRoleMismatchError(
                    f"Model '{req_filename}' is registered in manifest with role '{entry_role}', "
                    f"which does not match expected role '{canonical_role}'"
                )

        if matched_entry is None:
            raise ModelRoleMismatchError(
                f"No authorized model entry found in manifest for role '{canonical_role}' (target: {req_filename})"
            )

        # Validate role binding
        entry_role = normalize_role(str(matched_entry.get("model_role", "")))
        if entry_role != canonical_role:
            raise ModelRoleMismatchError(
                f"Model role mismatch for {matched_key}: expected '{canonical_role}', found '{entry_role}'"
            )

        # Validate model_id binding if expected
        if expected_model_id is not None:
            entry_model_id = str(matched_entry.get("model_id", ""))
            if entry_model_id != expected_model_id:
                raise ModelRoleMismatchError(
                    f"Model ID mismatch: expected '{expected_model_id}', found '{entry_model_id}'"
                )

        # Validate filename / path binding
        entry_filename = Path(str(matched_entry.get("filename", ""))).name
        if entry_filename != req_filename:
            raise ModelRoleMismatchError(
                f"Model filename mismatch: manifest binds '{entry_filename}', requested '{req_filename}'"
            )

        # Verify physical model file existence
        if not resolved_path.is_file():
            raise ModelIntegrityError(f"Authorized model file does not exist on disk: {resolved_path.name}")

        # Compute digest and size
        actual_sha256, actual_size = compute_file_sha256(resolved_path)

        expected_size = matched_entry.get("size_bytes")
        if expected_size is not None and int(expected_size) != actual_size:
            raise ModelDigestMismatchError(
                f"Model size mismatch for '{resolved_path.name}': expected {expected_size} bytes, got {actual_size} bytes"
            )

        expected_sha256 = str(matched_entry.get("sha256", "")).strip().lower()
        if not expected_sha256 or expected_sha256 != actual_sha256:
            raise ModelDigestMismatchError(
                f"Model SHA-256 digest mismatch for '{resolved_path.name}': "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        logger.info(
            "Model integrity and authenticity verified for %s (role: %s, sha256: %s...)",
            resolved_path.name,
            canonical_role,
            actual_sha256[:12],
        )
        return resolved_path


# Global singleton instance
_default_verifier: ModelVerifier | None = None


def get_model_verifier() -> ModelVerifier:
    """Obtain or initialize the global ModelVerifier instance."""
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = ModelVerifier()
    return _default_verifier


def verify_model(
    model_path: str | Path,
    expected_role: str,
    expected_model_id: str | None = None,
    manifest_path: str | Path | None = None,
    signature_path: str | Path | None = None,
    public_key: str | bytes | ed25519.Ed25519PublicKey | None = None,
) -> Path:
    """Convenience functional interface for model verification."""
    return get_model_verifier().verify_model(
        model_path=model_path,
        expected_role=expected_role,
        expected_model_id=expected_model_id,
        manifest_path=manifest_path,
        signature_path=signature_path,
        public_key=public_key,
    )
