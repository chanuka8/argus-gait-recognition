"""Biometric Template Encryption at Rest (Finding U3).

Implements authenticated encryption for biometric vectors stored across
VectorStore gallery containers and EmbeddingDatabase person records.

Cryptographic construction:
- AES-256-GCM (NIST SP 800-38D)
- 256-bit symmetric encryption key
- 96-bit (12-byte) fresh CSPRNG nonce per encryption operation via os.urandom(12)
- AES-GCM uses GHASH as part of its authenticated-encryption construction and
  produces a 128-bit authentication tag appended to the ciphertext.
- Associated Authenticated Data (AAD) cryptographically binds ciphertext to
  gallery type, vector dimension, dtype, record count, and identity labels digest.

Identity Policy:
- Subject identifiers (person_id) remain readable at rest.
- Biometric vectors are cryptographically bound to their identities via AAD,
  preventing template substitution, cross-gallery swapping, and label tampering.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import struct
from typing import Any

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class ConfigurationError(ValueError):
    """Raised when cryptographic keys or configuration parameters are invalid or conflicting."""


class ContainerFormatError(ValueError):
    """Raised when an encrypted gallery container has an invalid header, version, or format."""


class BiometricDecryptionError(RuntimeError):
    """Raised when decryption or AEAD authentication fails due to tampering, wrong key, or AAD mismatch."""


class BiometricEncryptor:
    """Core authenticated encryption engine for biometric feature vectors and templates."""

    MAGIC: bytes = b"AGFE"  # ARGUS Gallery Features Encrypted
    VERSION: int = 1
    CIPHER_SUITE_AES_256_GCM: int = 1
    NONCE_BYTES: int = 12
    TAG_BYTES: int = 16
    MAX_AAD_BYTES: int = 65536
    HEADER_STRUCT: struct.Struct = struct.Struct("!4sHH12sI")  # 24 bytes fixed header
    MIN_CONTAINER_BYTES: int = 24 + 16  # 40 bytes minimum

    def __init__(
        self,
        key: str | bytes | None = None,
        key_id: str | None = None,
        strict_mode: bool = False,
    ) -> None:
        self.strict_mode = strict_mode
        self.key_id = key_id or "argus_v1"
        self._key_bytes: bytes | None = self._resolve_and_validate_key(key)
        if self._key_bytes is not None:
            self._aesgcm: AESGCM | None = AESGCM(self._key_bytes)
        else:
            self._aesgcm = None

    @property
    def is_configured(self) -> bool:
        """Return True if a valid 256-bit encryption key is available."""
        return self._aesgcm is not None

    def _resolve_and_validate_key(self, key: str | bytes | None) -> bytes | None:
        """Resolve and validate 32-byte encryption key from argument or environment."""
        raw_val = key
        if raw_val is None:
            raw_val = os.environ.get("ARGUS_BIOMETRIC_ENCRYPTION_KEY")

        if raw_val is None:
            keyfile_path = os.environ.get("ARGUS_BIOMETRIC_KEYFILE", ".biometric_encryption.key")
            if os.path.isfile(keyfile_path):
                try:
                    with open(keyfile_path, "rb") as kf:
                        file_content = kf.read().strip()
                    if file_content:
                        raw_val = file_content
                except OSError as err:
                    raise ConfigurationError(f"Failed to read biometric keyfile: {err}") from err

        if raw_val is None:
            if self.strict_mode or os.environ.get("ARGUS_STRICT_BIOMETRIC_ENCRYPTION", "").lower() in (
                "true",
                "1",
                "yes",
            ):
                raise ConfigurationError(
                    "ARGUS_BIOMETRIC_ENCRYPTION_KEY is required in strict production mode but is not configured."
                )
            return None

        # Parse string or bytes
        parsed_key: bytes
        if isinstance(raw_val, str):
            cleaned = raw_val.strip()
            if len(cleaned) == 64:
                try:
                    parsed_key = bytes.fromhex(cleaned)
                except ValueError as err:
                    raise ConfigurationError("Biometric encryption key hex encoding is malformed.") from err
            else:
                raw_b = cleaned.encode("utf-8")
                if len(raw_b) != 32:
                    raise ConfigurationError(
                        f"ARGUS_BIOMETRIC_ENCRYPTION_KEY must be exactly 32 bytes (256 bits) "
                        f"or 64 hexadecimal characters. Provided length: {len(raw_b)} bytes."
                    )
                parsed_key = raw_b
        elif isinstance(raw_val, (bytes, bytearray)):
            cleaned_b = bytes(raw_val).strip()
            if len(cleaned_b) == 64:
                try:
                    parsed_key = bytes.fromhex(cleaned_b.decode("ascii"))
                except (ValueError, UnicodeDecodeError):
                    parsed_key = cleaned_b
            else:
                parsed_key = cleaned_b

            if len(parsed_key) != 32:
                raise ConfigurationError(
                    f"ARGUS_BIOMETRIC_ENCRYPTION_KEY must be exactly 32 bytes (256 bits). "
                    f"Provided length: {len(parsed_key)} bytes."
                )
        else:
            raise ConfigurationError(f"Unsupported key type: {type(raw_val).__name__}")

        # Enforce U2 / U3 key separation
        self._enforce_key_separation(parsed_key)
        return parsed_key

    def _enforce_key_separation(self, biometric_key: bytes) -> None:
        """Enforce that U3 biometric encryption key differs from U2 audit log HMAC key."""
        audit_key_env = os.environ.get("ARGUS_AUDIT_HMAC_KEY")
        if not audit_key_env:
            return

        cleaned_audit = audit_key_env.strip()
        parsed_audit: bytes
        if len(cleaned_audit) == 64:
            try:
                parsed_audit = bytes.fromhex(cleaned_audit)
            except ValueError:
                parsed_audit = cleaned_audit.encode("utf-8")
        else:
            parsed_audit = cleaned_audit.encode("utf-8")

        if hmac_compare_safe(biometric_key, parsed_audit):
            raise ConfigurationError(
                "Cryptographic domain separation violation: ARGUS_BIOMETRIC_ENCRYPTION_KEY "
                "must not match ARGUS_AUDIT_HMAC_KEY. Separate keys are required for audit log HMAC "
                "and biometric template encryption."
            )

    @classmethod
    def canonicalize_aad(cls, aad_payload: dict[str, Any]) -> bytes:
        """Deterministically serialize Associated Authenticated Data into UTF-8 JSON bytes."""
        return json.dumps(aad_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    @classmethod
    def compute_labels_sha256(cls, labels: list[str] | np.ndarray) -> str:
        """Compute canonical SHA-256 hex digest of subject labels array."""
        canonical_str = "\n".join(str(lbl) for lbl in labels)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def encrypt_gallery_container(
        self,
        features: np.ndarray,
        labels: list[str] | np.ndarray,
        gallery_type: str,
    ) -> bytes:
        """Encrypt NumPy gallery features array into a versioned authenticated container."""
        if not self.is_configured or self._aesgcm is None:
            raise ConfigurationError("BiometricEncryptor is not configured with an encryption key.")

        feats_arr = np.asarray(features, dtype=np.float32)
        if feats_arr.size == 0 and feats_arr.ndim != 2:
            feats_arr = np.empty((0, 0), dtype=np.float32)

        # Serialize features with allow_pickle=False
        bio = io.BytesIO()
        np.save(bio, feats_arr, allow_pickle=False)
        plaintext_bytes = bio.getvalue()

        # Construct canonical AAD
        dimension = int(feats_arr.shape[1]) if feats_arr.ndim == 2 else 0
        num_records = len(feats_arr)
        labels_digest = self.compute_labels_sha256(labels)

        aad_payload = {
            "dimension": dimension,
            "dtype": str(feats_arr.dtype),
            "gallery_type": str(gallery_type),
            "labels_sha256": labels_digest,
            "num_records": num_records,
        }
        aad_bytes = self.canonicalize_aad(aad_payload)
        if len(aad_bytes) > self.MAX_AAD_BYTES:
            raise ContainerFormatError(f"Serialized AAD length ({len(aad_bytes)}) exceeds limit ({self.MAX_AAD_BYTES})")

        # Generate fresh 96-bit CSPRNG nonce
        nonce = os.urandom(self.NONCE_BYTES)

        # AES-256-GCM encryption (produces ciphertext with 128-bit authentication tag appended)
        ciphertext_and_tag = self._aesgcm.encrypt(nonce, plaintext_bytes, aad_bytes)

        # Pack binary container header
        header = self.HEADER_STRUCT.pack(
            self.MAGIC,
            self.VERSION,
            self.CIPHER_SUITE_AES_256_GCM,
            nonce,
            len(aad_bytes),
        )
        return header + aad_bytes + ciphertext_and_tag

    def decrypt_gallery_container(
        self,
        container_bytes: bytes,
        expected_gallery_type: str,
        expected_labels: list[str] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Validate and decrypt an authenticated gallery container, returning (features, aad)."""
        if not self.is_configured or self._aesgcm is None:
            raise ConfigurationError("BiometricEncryptor is not configured with an encryption key.")

        if len(container_bytes) < self.MIN_CONTAINER_BYTES:
            raise ContainerFormatError(
                f"Corrupted container: length {len(container_bytes)} is less than minimum {self.MIN_CONTAINER_BYTES}."
            )

        # Unpack header
        magic, version, cipher_suite, nonce, aad_len = self.HEADER_STRUCT.unpack_from(container_bytes, 0)
        if magic != self.MAGIC:
            raise ContainerFormatError(f"Invalid container magic: expected {self.MAGIC!r}, got {magic!r}.")
        if version != self.VERSION:
            raise ContainerFormatError(f"Unsupported container version: expected {self.VERSION}, got {version}.")
        if cipher_suite != self.CIPHER_SUITE_AES_256_GCM:
            raise ContainerFormatError(
                f"Unsupported cipher suite: expected {self.CIPHER_SUITE_AES_256_GCM}, got {cipher_suite}."
            )
        if aad_len > self.MAX_AAD_BYTES:
            raise ContainerFormatError(f"Container AAD length ({aad_len}) exceeds maximum ({self.MAX_AAD_BYTES}).")

        expected_min_len = 24 + aad_len + self.TAG_BYTES
        if len(container_bytes) < expected_min_len:
            raise ContainerFormatError(
                f"Truncated container: expected at least {expected_min_len} bytes, got {len(container_bytes)}."
            )

        aad_bytes = container_bytes[24 : 24 + aad_len]
        ciphertext_and_tag = container_bytes[24 + aad_len :]

        try:
            aad_payload = json.loads(aad_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise ContainerFormatError(f"Malformed AAD JSON payload in container: {err}") from err

        # Verify gallery type in AAD
        actual_type = aad_payload.get("gallery_type")
        if actual_type != expected_gallery_type:
            raise BiometricDecryptionError(
                f"Gallery type mismatch in AAD: container was encrypted for '{actual_type}', "
                f"but caller requested '{expected_gallery_type}'."
            )

        # Verify labels SHA-256 if expected_labels provided
        if expected_labels is not None:
            expected_digest = self.compute_labels_sha256(expected_labels)
            actual_digest = aad_payload.get("labels_sha256")
            if actual_digest != expected_digest:
                raise BiometricDecryptionError(
                    "Cryptographic label binding mismatch: gallery labels do not match the digest "
                    "authenticated in the container AAD."
                )

        # Decrypt with AES-256-GCM
        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext_and_tag, aad_bytes)
        except InvalidTag as err:
            raise BiometricDecryptionError(
                "AEAD authentication tag verification failed. Container ciphertext or metadata has been "
                "tampered with, corrupted, or an incorrect encryption key was provided."
            ) from err

        # Deserialize NumPy array
        bio = io.BytesIO(plaintext)
        try:
            features = np.load(bio, allow_pickle=False)
        except Exception as err:
            raise ContainerFormatError(f"Failed to deserialize plaintext biometric array: {err}") from err

        if features.dtype == object or features.dtype.kind == "O":
            raise ContainerFormatError("Deserialized biometric array has disallowed object dtype.")
        if not np.issubdtype(features.dtype, np.number):
            raise ContainerFormatError(f"Biometric array must have numeric dtype, got {features.dtype}.")

        # Validate dimensions match AAD
        expected_dim = aad_payload.get("dimension", 0)
        expected_records = aad_payload.get("num_records", 0)
        if features.ndim == 2:
            if features.shape[1] != expected_dim:
                raise ContainerFormatError(
                    f"Array dimension mismatch: container AAD expected {expected_dim}, got {features.shape[1]}."
                )
            if len(features) != expected_records:
                raise ContainerFormatError(
                    f"Record count mismatch: container AAD expected {expected_records}, got {len(features)}."
                )

        return features, aad_payload

    def encrypt_vector(
        self,
        vector: np.ndarray | list[float],
        person_id: str,
        modality: str,
        model_version: str = "v1.0.0",
    ) -> dict[str, Any]:
        """Encrypt a single biometric vector into a JSON-safe field-level encrypted dictionary."""
        if not self.is_configured or self._aesgcm is None:
            raise ConfigurationError("BiometricEncryptor is not configured with an encryption key.")

        vec = np.asarray(vector, dtype=np.float32).ravel()
        if not np.isfinite(vec).all():
            raise ValueError("Biometric vector contains non-finite values (NaN or Inf).")
        if float(np.linalg.norm(vec)) == 0.0:
            raise ValueError("Biometric vector has zero norm.")

        # Serialize vector deterministically
        bio = io.BytesIO()
        np.save(bio, vec, allow_pickle=False)
        plaintext = bio.getvalue()

        # Construct AAD binding person_id, modality, dimension, dtype, model_version
        aad_payload = {
            "dimension": int(vec.size),
            "dtype": "float32",
            "modality": str(modality),
            "model_version": str(model_version),
            "person_id": str(person_id),
        }
        aad_bytes = self.canonicalize_aad(aad_payload)

        # Generate fresh 12-byte CSPRNG nonce
        nonce = os.urandom(self.NONCE_BYTES)

        # Encrypt
        ciphertext_and_tag = self._aesgcm.encrypt(nonce, plaintext, aad_bytes)

        return {
            "cipher": "AES-256-GCM",
            "ciphertext": base64.b64encode(ciphertext_and_tag).decode("ascii"),
            "dimension": int(vec.size),
            "key_id": self.key_id,
            "modality": str(modality),
            "model_version": str(model_version),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "version": 1,
        }

    def decrypt_vector(
        self,
        payload: dict[str, Any],
        person_id: str,
        modality: str,
        model_version: str = "v1.0.0",
    ) -> list[float]:
        """Decrypt a field-level encrypted vector dictionary, verifying cryptographic identity binding."""
        if not self.is_configured or self._aesgcm is None:
            raise ConfigurationError("BiometricEncryptor is not configured with an encryption key.")

        version = payload.get("version")
        if version != 1:
            raise ContainerFormatError(f"Unsupported vector encryption version: {version}")

        cipher = payload.get("cipher")
        if cipher != "AES-256-GCM":
            raise ContainerFormatError(f"Unsupported vector cipher: {cipher}")

        payload_modality = str(payload.get("modality", modality))
        if payload_modality != modality:
            raise BiometricDecryptionError(
                f"Modality mismatch in vector encryption: payload has '{payload_modality}', requested '{modality}'."
            )

        dimension = int(payload.get("dimension", 0))

        try:
            nonce = base64.b64decode(payload["nonce"])
            ciphertext_and_tag = base64.b64decode(payload["ciphertext"])
        except (KeyError, ValueError) as err:
            raise ContainerFormatError(f"Corrupted base64 encoding in vector encryption container: {err}") from err

        if len(nonce) != self.NONCE_BYTES:
            raise ContainerFormatError(f"Invalid nonce length: expected {self.NONCE_BYTES}, got {len(nonce)}")

        # Reconstruct canonical AAD
        aad_payload = {
            "dimension": dimension,
            "dtype": "float32",
            "modality": str(modality),
            "model_version": str(payload.get("model_version", model_version)),
            "person_id": str(person_id),
        }
        aad_bytes = self.canonicalize_aad(aad_payload)

        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext_and_tag, aad_bytes)
        except InvalidTag as err:
            raise BiometricDecryptionError(
                f"AEAD authentication tag verification failed for vector belonging to '{person_id}' ({modality}). "
                "Ciphertext has been tampered with or context AAD does not match."
            ) from err

        # Deserialize
        bio = io.BytesIO(plaintext)
        try:
            vec = np.load(bio, allow_pickle=False)
        except Exception as err:
            raise ContainerFormatError(f"Failed to deserialize decrypted vector array: {err}") from err

        if not np.issubdtype(vec.dtype, np.number):
            raise ContainerFormatError("Deserialized vector is not numeric.")
        if dimension and vec.size != dimension:
            raise ContainerFormatError(f"Decrypted vector size {vec.size} does not match expected {dimension}.")

        return [float(x) for x in vec.ravel()]


def hmac_compare_safe(val_a: bytes, val_b: bytes) -> bool:
    """Constant-time comparison helper."""
    import hmac

    return hmac.compare_digest(val_a, val_b)
