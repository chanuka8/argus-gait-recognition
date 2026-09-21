"""Model Checkpoint Confidentiality and Encryption-at-Rest Module.

Finding U5 — Model Checkpoint / AI Model Asset Protection (Phase 2).
Provides:
- AES-256-GCM authenticated encryption and decryption for model artifacts
- Authenticated container header serialization, parsing, and AAD binding
- Bijective, collision-free key-ID to environment-variable mapping
- Key provider abstractions (Environment, Static, File) with strict routing
- Trusted public-model identity registry and non-path-based classification
- Non-downgradable production encryption enforcement
- Non-downgradable Phase-1 strict authenticity enforcement in production
- In-memory decrypted model loading without persistent disk residues
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import secrets
import struct
from enum import Enum
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.paths import resolve_app_path
from app.security_layer.model_integrity import (
    ROLE_APPEARANCE_EMBEDDING,
    ROLE_PERSON_DETECTOR,
    normalize_role,
)

logger = logging.getLogger("argus.security.model_confidentiality")

# Container header constants
MAGIC_HEADER = b"ARGUSENC"
CONTAINER_VERSION = 1
ALGORITHM_ID_AES256_GCM = 1
NONCE_LENGTH = 12
TAG_LENGTH = 16
KEY_LENGTH = 32
HEADER_BASE_FORMAT = ">8sHHH"  # magic(8), version(uint16), algo_id(uint16), key_id_len(uint16)
HEADER_BASE_SIZE = struct.calcsize(HEADER_BASE_FORMAT)


class ArtifactConfidentiality(str, Enum):
    """Artifact confidentiality classification."""

    PROTECTED = "PROTECTED"
    PUBLIC_UPSTREAM = "PUBLIC_UPSTREAM"


# Canonical trusted public upstream models registered in repository architecture
# Grounded in docs/security/u5_model_checkpoint_security_design.md lines 410, 421
TRUSTED_PUBLIC_UPSTREAM_MODELS: frozenset[tuple[str, str]] = frozenset(
    {
        (ROLE_APPEARANCE_EMBEDDING, "osnet_x0_25_msmt17"),
        (ROLE_PERSON_DETECTOR, "yolov8n"),
    }
)


class ModelConfidentialityError(RuntimeError):
    """Base exception for all model confidentiality and decryption failures."""


class ModelDecryptionError(ModelConfidentialityError):
    """Raised when authenticated decryption fails (invalid tag, corrupt data, or wrong key)."""


class ModelEncryptionRequiredError(ModelConfidentialityError):
    """Raised when an unencrypted protected model is loaded in an environment requiring encryption."""


class ModelKeyUnavailableError(ModelConfidentialityError):
    """Raised when the required model encryption key cannot be retrieved from key providers."""


class ModelSecurityPolicyViolationError(ModelConfidentialityError):
    """Raised when confidentiality policy or logical identity requirements are violated."""


class ModelProvisioningError(ModelConfidentialityError):
    """Raised when model encryption/provisioning fails."""


class ProvisioningLockError(ModelProvisioningError):
    """Raised when destination lock acquisition fails due to concurrent provisioning."""


def key_id_to_env_var(key_id: str) -> str:
    """Bijective, collision-free mapping from key_id to environment variable name."""
    clean_id = key_id.strip()
    if not clean_id or not re.match(r"^[A-Za-z0-9._-]{1,64}$", clean_id):
        raise ValueError(
            f"Invalid key_id format: {key_id!r}. Must be 1-64 characters matching ^[A-Za-z0-9._-]{{1,64}}$."
        )
    if clean_id == "default":
        return "ARGUS_MODEL_ENCRYPTION_KEY"
    hex_suffix = clean_id.encode("utf-8").hex().upper()
    return f"ARGUS_MODEL_KEY_HEX_{hex_suffix}"


class ModelKeyProvider:
    """Abstract interface for model encryption key acquisition."""

    def get_key(self, key_id: str) -> bytes:
        raise NotImplementedError


class EnvModelKeyProvider(ModelKeyProvider):
    """Environment-variable based key provider with collision-free hex mapping."""

    def get_key(self, key_id: str) -> bytes:
        clean_id = key_id.strip()
        env_var = key_id_to_env_var(clean_id)

        raw_val = os.getenv(env_var)
        if raw_val is None and clean_id == "default":
            # Check secondary default hex format
            secondary_var = "ARGUS_MODEL_KEY_HEX_64656661756C74"
            raw_val = os.getenv(secondary_var)

        if not raw_val:
            raise ModelKeyUnavailableError(
                f"Model encryption key for key_id '{clean_id}' is not set in environment variable '{env_var}'."
            )

        val = raw_val.strip()

        # Hex decoding (64 hex characters = 32 bytes)
        if len(val) == 64 and all(c in "0123456789abcdefABCDEF" for c in val):
            try:
                decoded = bytes.fromhex(val)
                if len(decoded) == KEY_LENGTH:
                    return decoded
            except ValueError:
                pass

        # Base64 decoding (44 characters = 32 bytes with padding)
        if len(val) in {43, 44}:
            try:
                decoded = base64.b64decode(val)
                if len(decoded) == KEY_LENGTH:
                    return decoded
            except (binascii.Error, ValueError):
                pass

        # Raw 32 bytes
        raw_bytes = val.encode("utf-8")
        if len(raw_bytes) == KEY_LENGTH:
            return raw_bytes

        raise ModelKeyUnavailableError(
            f"Key material in '{env_var}' must decode to exactly {KEY_LENGTH} bytes (got {len(val)} chars)."
        )


class StaticModelKeyProvider(ModelKeyProvider):
    """Key provider backed by an explicit mapping of key_id -> raw 32 bytes."""

    def __init__(self, key_map: dict[str, bytes]) -> None:
        self._keys: dict[str, bytes] = {}
        for k, v in key_map.items():
            if len(v) != KEY_LENGTH:
                raise ValueError(f"Key for '{k}' must be exactly {KEY_LENGTH} bytes; got {len(v)}.")
            self._keys[k.strip()] = bytes(v)

    def get_key(self, key_id: str) -> bytes:
        clean_id = key_id.strip()
        if clean_id in self._keys:
            return self._keys[clean_id]
        raise ModelKeyUnavailableError(
            f"Model encryption key for key_id '{clean_id}' not found in static key provider."
        )


class FileModelKeyProvider(ModelKeyProvider):
    """Key provider backed by a local key file (PEM or raw 32 bytes)."""

    def __init__(self, key_path: str | Path, key_id: str = "default") -> None:
        self._key_path = resolve_app_path(key_path)
        self._key_id = key_id.strip()

    def get_key(self, key_id: str) -> bytes:
        clean_id = key_id.strip()
        if clean_id != self._key_id:
            raise ModelKeyUnavailableError(
                f"File key provider only configured for key_id '{self._key_id}', requested '{clean_id}'."
            )
        if not self._key_path.is_file():
            raise ModelKeyUnavailableError(f"Key file not found: {self._key_path}")

        raw = self._key_path.read_bytes().strip()
        if len(raw) == KEY_LENGTH:
            return raw
        if len(raw) == 64:
            try:
                decoded = bytes.fromhex(raw.decode("ascii"))
                if len(decoded) == KEY_LENGTH:
                    return decoded
            except ValueError:
                pass
        try:
            decoded = base64.b64decode(raw)
            if len(decoded) == KEY_LENGTH:
                return decoded
        except (binascii.Error, ValueError):
            pass

        raise ModelKeyUnavailableError(f"Key file {self._key_path.name} does not contain valid 32-byte key.")


_default_key_provider: ModelKeyProvider | None = None


def get_default_key_provider() -> ModelKeyProvider:
    """Return the global default environment key provider."""
    global _default_key_provider
    if _default_key_provider is None:
        _default_key_provider = EnvModelKeyProvider()
    return _default_key_provider


def get_deployment_environment() -> str:
    """Return the canonical active deployment environment string."""
    env = os.getenv("ARGUS_ENV", os.getenv("ENVIRONMENT", "development")).strip().lower()
    return env


def resolve_artifact_confidentiality(
    expected_role: str,
    expected_model_id: str | None = None,
    explicit_policy: ArtifactConfidentiality | None = None,
) -> ArtifactConfidentiality:
    """Resolve artifact confidentiality policy strictly from trusted metadata.

    INVARIANTS:
    - Never uses filename, path, directory, or extension to infer public status.
    - Default is ALWAYS ArtifactConfidentiality.PROTECTED.
    - ArtifactConfidentiality.PUBLIC_UPSTREAM is returned ONLY when:
      1. explicit_policy is ArtifactConfidentiality.PUBLIC_UPSTREAM, OR
      2. (canonical_role, expected_model_id) is in TRUSTED_PUBLIC_UPSTREAM_MODELS.
    - Missing, unknown, or synthetic model IDs fail safe to PROTECTED.
    """
    if explicit_policy is not None:
        return explicit_policy

    canonical_role = normalize_role(expected_role)
    mid = str(expected_model_id or "").strip()

    if (canonical_role, mid) in TRUSTED_PUBLIC_UPSTREAM_MODELS:
        return ArtifactConfidentiality.PUBLIC_UPSTREAM

    return ArtifactConfidentiality.PROTECTED


def is_model_encryption_required(
    confidentiality: ArtifactConfidentiality,
    deployment_environment: str | None = None,
) -> bool:
    """Evaluate whether encryption at rest is mandatory for this artifact.

    NON-DOWNGRADE RULE:
    If deployment_environment == 'production', encryption is strictly mandatory
    for all PROTECTED artifacts and cannot be disabled by any configuration or
    permissive environment variable.
    """
    if confidentiality != ArtifactConfidentiality.PROTECTED:
        return False

    env = (deployment_environment or get_deployment_environment()).strip().lower()
    if env == "production":
        return True

    # In development/testing/staging, operators may opt-in to strict encryption
    for env_var in ("ARGUS_REQUIRE_ENCRYPTED_MODELS", "ARGUS_MODEL_ENCRYPTION_STRICT"):
        val = os.getenv(env_var)
        if val is not None and val.strip().lower() in {"1", "true", "yes", "on"}:
            return True

    return False


def encrypt_model_container(
    plaintext_bytes: bytes,
    key_id: str = "default",
    key_provider: ModelKeyProvider | None = None,
    key_material: bytes | None = None,
) -> bytes:
    """Encrypt plaintext model bytes into authenticated AES-256-GCM container format."""
    clean_key_id = key_id.strip()
    key_id_bytes = clean_key_id.encode("utf-8")
    if len(key_id_bytes) > 64:
        raise ValueError(f"Key ID exceeds 64 bytes: {clean_key_id}")

    if key_material is not None:
        if len(key_material) != KEY_LENGTH:
            raise ValueError(f"Explicit key material must be {KEY_LENGTH} bytes; got {len(key_material)}.")
        key = key_material
    else:
        provider = key_provider or get_default_key_provider()
        key = provider.get_key(clean_key_id)

    nonce = secrets.token_bytes(NONCE_LENGTH)

    header = (
        struct.pack(
            HEADER_BASE_FORMAT,
            MAGIC_HEADER,
            CONTAINER_VERSION,
            ALGORITHM_ID_AES256_GCM,
            len(key_id_bytes),
        )
        + key_id_bytes
        + struct.pack(">H", len(nonce))
        + nonce
    )

    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext_bytes, header)

    return header + ciphertext_and_tag


def decrypt_model_container(
    container_bytes: bytes,
    key_provider: ModelKeyProvider | None = None,
    key_material: bytes | None = None,
) -> bytes:
    """Authenticate and decrypt an AES-256-GCM model container.

    Fails closed with generic ModelDecryptionError on any corruption,
    tag mismatch, header invalidity, or key mismatch to prevent oracles.
    """
    if len(container_bytes) < HEADER_BASE_SIZE:
        raise ModelDecryptionError("Authenticated model decryption failed.")

    try:
        magic, version, algo_id, key_id_len = struct.unpack_from(HEADER_BASE_FORMAT, container_bytes, 0)
        if magic != MAGIC_HEADER or version != CONTAINER_VERSION or algo_id != ALGORITHM_ID_AES256_GCM:
            raise ModelDecryptionError("Authenticated model decryption failed.")

        offset = HEADER_BASE_SIZE
        if len(container_bytes) < offset + key_id_len + 2:
            raise ModelDecryptionError("Authenticated model decryption failed.")

        key_id_bytes = container_bytes[offset : offset + key_id_len]
        key_id = key_id_bytes.decode("utf-8")
        offset += key_id_len

        (nonce_len,) = struct.unpack_from(">H", container_bytes, offset)
        offset += 2

        if nonce_len != NONCE_LENGTH or len(container_bytes) < offset + nonce_len + TAG_LENGTH:
            raise ModelDecryptionError("Authenticated model decryption failed.")

        nonce = container_bytes[offset : offset + nonce_len]
        offset += nonce_len

        header_aad = container_bytes[:offset]
        ciphertext_and_tag = container_bytes[offset:]

        if key_material is not None:
            key = key_material
        else:
            provider = key_provider or get_default_key_provider()
            key = provider.get_key(key_id)

        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext_and_tag, header_aad)
        return plaintext

    except ModelConfidentialityError:
        raise
    except (InvalidTag, ValueError, struct.error, UnicodeDecodeError) as e:
        raise ModelDecryptionError("Authenticated model decryption failed.") from e
    except Exception as e:
        raise ModelDecryptionError("Authenticated model decryption failed.") from e


def load_verified_model_bytes(
    model_path: str | Path,
    expected_role: str,
    logical_filename: str | None = None,
    expected_model_id: str | None = None,
    manifest_path: str | Path | None = None,
    signature_path: str | Path | None = None,
    public_key: str | bytes | ed25519.Ed25519PublicKey | None = None,
    key_provider: ModelKeyProvider | None = None,
    key_material: bytes | None = None,
    confidentiality: ArtifactConfidentiality | None = None,
    strict_phase1: bool | None = None,
) -> bytes:
    """Primary loader gate: Enforces authenticated decryption and Phase 1 verification in memory.

    Rules:
    - Never uses file paths to downgrade confidentiality.
    - Physical filename of an encrypted artifact (.enc) must NEVER be passed as the logical filename.
    - In production, protected encrypted artifacts REQUIRE an explicit non-empty logical_filename.
    - In production, Phase 1 strict verification is non-downgradable and MANDATORY.
    - Returns verified plaintext bytes in memory. Creates zero persistent disk files.
    """
    path = resolve_app_path(model_path)
    canonical_role = normalize_role(expected_role)

    resolved_confidentiality = (
        confidentiality
        if confidentiality is not None
        else resolve_artifact_confidentiality(
            expected_role=canonical_role,
            expected_model_id=expected_model_id,
        )
    )

    env = get_deployment_environment()
    encryption_required = is_model_encryption_required(resolved_confidentiality, env)

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    raw_bytes = path.read_bytes()
    is_enc = raw_bytes.startswith(MAGIC_HEADER)

    if encryption_required and not is_enc:
        raise ModelEncryptionRequiredError(
            f"Production policy requires encryption at rest for protected model '{path.name}'."
        )

    # Determine trusted logical filename for Phase 1 verification
    if logical_filename is not None and logical_filename.strip():
        trusted_logical_filename = logical_filename.strip()
    elif not is_enc:
        # Unencrypted plaintext artifact: physical name is the authoritative logical name
        trusted_logical_filename = path.name
    else:
        # Encrypted artifact without an explicit logical_filename
        if env == "production" or resolved_confidentiality == ArtifactConfidentiality.PROTECTED:
            raise ModelSecurityPolicyViolationError(
                f"Protected model '{path.name}' requires an explicit trusted logical_filename."
            )
        # Non-production fallback for tests/development
        if path.name.endswith(".enc"):
            trusted_logical_filename = path.name[:-4]
        else:
            trusted_logical_filename = path.name

    if is_enc:
        decrypted_bytes = decrypt_model_container(
            raw_bytes,
            key_provider=key_provider,
            key_material=key_material,
        )
    else:
        decrypted_bytes = raw_bytes

    # In production for protected artifacts, Phase 1 strict mode is non-downgradable and MANDATORY
    if env == "production" and resolved_confidentiality == ArtifactConfidentiality.PROTECTED:
        effective_strict_phase1 = True
    else:
        effective_strict_phase1 = strict_phase1

    from app.security_layer import model_integrity

    verifier = model_integrity.get_model_verifier()
    verifier.verify_model_bytes(
        model_bytes=decrypted_bytes,
        logical_filename=trusted_logical_filename,
        expected_role=canonical_role,
        expected_model_id=expected_model_id,
        manifest_path=manifest_path,
        signature_path=signature_path,
        public_key=public_key,
        strict_mode=effective_strict_phase1,
    )

    return decrypted_bytes
