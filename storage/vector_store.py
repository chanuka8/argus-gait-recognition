import json
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
from filelock import FileLock

from security_layer.biometric_encryption import (
    BiometricDecryptionError,
    BiometricEncryptor,
    ConfigurationError,
    ContainerFormatError,
)

logger = logging.getLogger("ARGUS.VectorStore")


class VectorStore:
    def __init__(
        self,
        gallery_dir: str | Path = "models/gallery",
        gallery_type: str | None = None,
        encryptor: BiometricEncryptor | None = None,
        strict_mode: bool | None = None,
    ) -> None:
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.features_file = self.gallery_dir / "gallery_features.npy"
        self.encrypted_features_file = self.gallery_dir / "gallery_features.enc"
        self.labels_file = self.gallery_dir / "gallery_labels.npy"
        self.metadata_file = self.gallery_dir / "gallery_metadata.json"
        self.lock_file = self.gallery_dir / f".{self.gallery_dir.name}.lock"

        # Determine gallery type for AAD binding
        if gallery_type is not None:
            self.gallery_type = gallery_type
        else:
            dir_name = self.gallery_dir.name.lower()
            if "appearance" in dir_name:
                self.gallery_type = "appearance"
            elif "live" in dir_name:
                self.gallery_type = "live_gait"
            else:
                self.gallery_type = "baseline_gait"

        # Determine strict mode
        if strict_mode is not None:
            self.strict_mode = strict_mode
        else:
            self.strict_mode = os.environ.get("ARGUS_STRICT_BIOMETRIC_ENCRYPTION", "").lower() in (
                "true",
                "1",
                "yes",
            ) or os.environ.get("ARGUS_SECURITY_STRICT", "").lower() in ("true", "1", "yes")

        # Encryption engine
        if encryptor is not None:
            self.encryptor = encryptor
        else:
            try:
                self.encryptor = BiometricEncryptor(strict_mode=self.strict_mode)
            except ConfigurationError as e:
                if self.strict_mode:
                    raise
                logger.warning(f"BiometricEncryptor initialization warning: {e}")
                self.encryptor = BiometricEncryptor(strict_mode=False)

    def _normalize_metadata_entry(
        self,
        value,
    ) -> dict:
        if isinstance(value, dict):
            status = str(
                value.get(
                    "status",
                    "ACTIVE" if value.get("enabled", True) else "DISABLED",
                )
            ).upper()

            return {
                "embeddings": int(value.get("embeddings", 0)),
                "status": status,
                "enabled": status == "ACTIVE",
                "updated_at": float(value.get("updated_at", time.time())),
            }

        if isinstance(value, int):
            return {
                "embeddings": value,
                "status": "ACTIVE",
                "enabled": True,
                "updated_at": time.time(),
            }

        return {
            "embeddings": 0,
            "status": "ACTIVE",
            "enabled": True,
            "updated_at": time.time(),
        }

    def _normalize_metadata(
        self,
        metadata: dict,
    ) -> dict:
        return {str(person_id): self._normalize_metadata_entry(value) for person_id, value in metadata.items()}

    def save(
        self,
        features,
        labels,
        metadata,
    ) -> None:
        """Atomically persist gallery features, labels, and metadata under inter-process lock.

        If a biometric encryption key is configured, features are encrypted with AES-256-GCM
        and written to gallery_features.enc. In development mode without a key, legacy .npy
        writes are performed with an explicit security warning.
        """
        metadata = self._normalize_metadata(
            metadata or {},
        )

        feats_arr = np.asarray(features, dtype=np.float32)
        if feats_arr.size == 0 and feats_arr.ndim != 2:
            feats_arr = np.empty((0, 0), dtype=np.float32)

        lbls_arr = np.asarray(labels)
        if lbls_arr.size == 0 and lbls_arr.ndim != 1:
            lbls_arr = np.empty((0,), dtype=str)

        with FileLock(str(self.lock_file), timeout=10.0):
            # 1. Features persistence (Encrypted or Plaintext)
            if self.encryptor.is_configured:
                # Encrypt to memory buffer
                container_bytes = self.encryptor.encrypt_gallery_container(
                    features=feats_arr,
                    labels=lbls_arr,
                    gallery_type=self.gallery_type,
                )
                tmp_enc = self.gallery_dir / f".tmp_{uuid.uuid4().hex}.enc"
                with open(tmp_enc, "wb") as f:
                    f.write(container_bytes)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                os.replace(tmp_enc, self.encrypted_features_file)
            else:
                if self.strict_mode:
                    raise RuntimeError(
                        "Cannot save unencrypted biometric gallery: strict production encryption mode "
                        "is active but ARGUS_BIOMETRIC_ENCRYPTION_KEY is not configured."
                    )
                logger.warning(
                    f"SECURITY WARNING: Persisting biometric templates in plaintext .npy format at "
                    f"'{self.features_file}'. Configure ARGUS_BIOMETRIC_ENCRYPTION_KEY to protect templates at rest."
                )
                tmp_npy = self.gallery_dir / f".tmp_{uuid.uuid4().hex}.npy"
                with open(tmp_npy, "wb") as f:
                    np.save(f, feats_arr)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass
                os.replace(tmp_npy, self.features_file)

            # 2. Labels persistence (atomic)
            tmp_lbls = self.gallery_dir / f".tmp_{uuid.uuid4().hex}.npy"
            with open(tmp_lbls, "wb") as f:
                np.save(f, lbls_arr)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_lbls, self.labels_file)

            # 3. Metadata persistence (atomic)
            tmp_meta = self.gallery_dir / f".tmp_{uuid.uuid4().hex}.json"
            with open(tmp_meta, "w", encoding="utf-8") as file:
                json.dump(
                    metadata,
                    file,
                    indent=4,
                )
                file.flush()
                try:
                    os.fsync(file.fileno())
                except OSError:
                    pass
            os.replace(tmp_meta, self.metadata_file)

    def load(self):
        """Load gallery features, labels, and metadata with strict downgrade prevention.

        Rules:
          1. If gallery_features.enc exists, encrypted loading is mandatory.
             Any decryption failure (wrong key, tag mismatch, corruption, label tampering)
             raises an exception and strictly NEVER falls back to plaintext .npy.
          2. If both .enc and .npy coexist, .enc takes precedence and an alert is logged.
          3. In strict mode, plaintext .npy is rejected with a security error.
        """
        has_enc = self.encrypted_features_file.exists()
        has_npy = self.features_file.exists()

        if not has_enc and not has_npy:
            return None

        if not self.labels_file.exists():
            return None

        # Load labels first (allow_pickle=False)
        try:
            labels = np.load(
                self.labels_file,
                allow_pickle=False,
            )
        except (OSError, ValueError, RuntimeError, EOFError) as err:
            err_msg = str(err)
            if "pickle" in err_msg.lower() or "object arrays" in err_msg.lower():
                raise ValueError(
                    f"Gallery labels file '{self.labels_file}' requires pickle deserialization, which is prohibited."
                ) from err
            raise ValueError(f"Failed to load gallery labels file '{self.labels_file}': {err}") from err

        if labels.dtype == object or labels.dtype.kind == "O":
            raise ValueError(
                f"Gallery labels array in '{self.labels_file}' has invalid object dtype ({labels.dtype}). Only string or numeric dtypes are allowed."
            )

        if has_enc:
            if has_npy:
                logger.warning(
                    f"SECURITY ALERT: Plaintext residue detected: '{self.features_file}' coexists alongside "
                    f"encrypted container '{self.encrypted_features_file}'. Plaintext file must be purged."
                )

            # Encrypted load is mandatory: do NOT fall back to .npy on failure!
            if not self.encryptor.is_configured:
                raise RuntimeError(
                    f"Encrypted biometric gallery exists at '{self.encrypted_features_file}', but no valid "
                    "ARGUS_BIOMETRIC_ENCRYPTION_KEY is configured to decrypt it. Refusing to fall back to plaintext."
                )

            try:
                container_bytes = self.encrypted_features_file.read_bytes()
                features, _aad = self.encryptor.decrypt_gallery_container(
                    container_bytes=container_bytes,
                    expected_gallery_type=self.gallery_type,
                    expected_labels=labels,
                )
            except (BiometricDecryptionError, ContainerFormatError, ConfigurationError, ValueError) as enc_err:
                logger.error(
                    f"MANDATORY ENCRYPTION FAILURE: Failed to authenticate/decrypt '{self.encrypted_features_file}': {enc_err}. "
                    "Downgrade to plaintext is strictly prohibited."
                )
                raise

        else:
            # Only plaintext .npy exists
            if self.strict_mode:
                raise RuntimeError(
                    f"STRICT SECURITY VIOLATION: Unencrypted biometric gallery detected at '{self.features_file}' "
                    "in strict production mode. Biometric template encryption is mandatory."
                )

            logger.warning(
                f"SECURITY NOTICE: Loading unencrypted legacy biometric gallery at '{self.features_file}'. "
                "Migrate to encrypted container format using tools/security/migrate_gallery_encryption.py."
            )
            try:
                features = np.load(
                    self.features_file,
                    allow_pickle=False,
                )
            except (OSError, ValueError, RuntimeError, EOFError) as err:
                err_msg = str(err)
                if "pickle" in err_msg.lower() or "object arrays" in err_msg.lower():
                    raise ValueError(
                        f"Gallery features file '{self.features_file}' requires pickle deserialization, which is prohibited."
                    ) from err
                raise ValueError(f"Failed to load gallery features file '{self.features_file}': {err}") from err

        if features.dtype == object or features.dtype.kind == "O":
            raise ValueError(
                f"Gallery features array has invalid object dtype ({features.dtype}). Only numeric dtypes are allowed."
            )

        if not np.issubdtype(features.dtype, np.number):
            raise ValueError(f"Gallery features array must be numeric, got {features.dtype}.")

        if features.size == 0 and labels.size == 0:
            if features.ndim != 2:
                features = features.reshape((0, 0))
            if labels.ndim != 1:
                labels = labels.reshape((0,))
        else:
            if features.ndim != 2:
                raise ValueError(f"Gallery features array must be 2-dimensional (N, D), got shape {features.shape}.")

            if labels.ndim != 1:
                raise ValueError(f"Gallery labels array must be 1-dimensional (N,), got shape {labels.shape}.")

            if len(features) != len(labels):
                raise ValueError(
                    f"Mismatch between features length ({len(features)}) and labels length ({len(labels)})."
                )

        if self.metadata_file.exists():
            with open(
                self.metadata_file,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)
        else:
            metadata = {}

        metadata = self._normalize_metadata(
            metadata,
        )

        return (
            features,
            labels,
            metadata,
        )


def validate_gallery_files(
    gallery_dir: str | Path = "models/gallery",
    expected_dim: int = 256,
    strict_mode: bool | None = None,
    encryptor: BiometricEncryptor | None = None,
) -> tuple[bool, str | None, int]:
    """Validate biometric gallery files, supporting both encrypted .enc containers and legacy .npy files.

    Returns:
      (is_valid, error_or_warning_message, template_count)
    """
    g_dir = Path(gallery_dir)
    enc_file = g_dir / "gallery_features.enc"
    feats_file = g_dir / "gallery_features.npy"
    lbls_file = g_dir / "gallery_labels.npy"

    if strict_mode is None:
        strict_mode = os.environ.get("ARGUS_STRICT_BIOMETRIC_ENCRYPTION", "").lower() in (
            "true",
            "1",
            "yes",
        ) or os.environ.get("ARGUS_SECURITY_STRICT", "").lower() in ("true", "1", "yes")

    if not enc_file.exists() and not feats_file.exists():
        return False, f"Gallery features file missing in {g_dir.as_posix()}", 0

    if not lbls_file.exists():
        return False, f"Gallery labels file missing in {g_dir.as_posix()}", 0

    # Load labels
    try:
        labels = np.load(lbls_file, allow_pickle=False)
    except (OSError, ValueError, RuntimeError, EOFError) as err:
        return False, f"Failed to load gallery labels file '{lbls_file.name}': {err}", 0

    if labels.dtype == object or labels.dtype.kind == "O":
        return False, f"Gallery labels array has invalid object dtype ({labels.dtype}).", 0

    if labels.ndim != 1:
        return False, f"Gallery labels array must be 1-dimensional (N,), got shape {labels.shape}.", 0

    # 1. Encrypted container validation
    if enc_file.exists():
        # Check for plaintext residue
        residue_msg = None
        if feats_file.exists():
            residue_msg = f"Plaintext residue detected: '{feats_file.name}' coexists with encrypted '{enc_file.name}'."

        try:
            data = enc_file.read_bytes()
            if len(data) < BiometricEncryptor.MIN_CONTAINER_BYTES:
                return False, f"Encrypted container '{enc_file.name}' is truncated (size {len(data)}).", 0

            magic, version, cipher, _nonce, aad_len = BiometricEncryptor.HEADER_STRUCT.unpack_from(data, 0)
            if magic != BiometricEncryptor.MAGIC:
                return False, f"Invalid container magic in '{enc_file.name}'.", 0
            if version != BiometricEncryptor.VERSION:
                return False, f"Unsupported container version {version} in '{enc_file.name}'.", 0
            if cipher != BiometricEncryptor.CIPHER_SUITE_AES_256_GCM:
                return False, f"Unsupported cipher suite {cipher} in '{enc_file.name}'.", 0
            if aad_len > BiometricEncryptor.MAX_AAD_BYTES or len(data) < 24 + aad_len + 16:
                return False, f"Malformed AAD length in '{enc_file.name}'.", 0

            aad_bytes = data[24 : 24 + aad_len]
            aad = json.loads(aad_bytes.decode("utf-8"))
            dim = aad.get("dimension", 0)
            if dim != expected_dim:
                return (
                    False,
                    f"Gallery feature dimension mismatch in container AAD: expected {expected_dim}, got {dim}.",
                    0,
                )

            count = aad.get("num_records", 0)
            if count != len(labels):
                return (
                    False,
                    f"Mismatch between container AAD records ({count}) and labels count ({len(labels)}).",
                    0,
                )

            # Check labels digest
            expected_digest = BiometricEncryptor.compute_labels_sha256(labels)
            if aad.get("labels_sha256") != expected_digest:
                return False, "Gallery labels digest does not match container AAD authentication tag.", 0

            # If encryptor or key is available, test decryption
            enc_inst = encryptor
            if enc_inst is None and (
                os.environ.get("ARGUS_BIOMETRIC_ENCRYPTION_KEY") or os.path.exists(".biometric_encryption.key")
            ):
                try:
                    enc_inst = BiometricEncryptor()
                except Exception:  # noqa: BLE001
                    enc_inst = None

            if enc_inst is not None and enc_inst.is_configured:
                dir_name = g_dir.name.lower()
                g_type = (
                    "appearance"
                    if "appearance" in dir_name
                    else ("live_gait" if "live" in dir_name else "baseline_gait")
                )
                enc_inst.decrypt_gallery_container(
                    container_bytes=data,
                    expected_gallery_type=g_type,
                    expected_labels=labels,
                )

            return True, residue_msg, count

        except Exception as err:  # noqa: BLE001
            return False, f"Encrypted container validation failed for '{enc_file.name}': {err}", 0

    # 2. Plaintext .npy validation
    if strict_mode:
        return False, f"Plaintext gallery '{feats_file.name}' is prohibited in strict production mode.", 0

    try:
        features = np.load(feats_file, allow_pickle=False)
    except (OSError, ValueError, RuntimeError, EOFError) as err:
        return False, f"Failed to load gallery features file '{feats_file.name}': {err}", 0

    if features.dtype == object or features.dtype.kind == "O":
        return False, f"Gallery features array has invalid object dtype ({features.dtype}).", 0

    if not np.issubdtype(features.dtype, np.number):
        return False, f"Gallery features array must be numeric, got {features.dtype}.", 0

    if features.ndim != 2:
        return False, f"Gallery features array must be 2-dimensional (N, D), got shape {features.shape}.", 0

    if features.shape[1] != expected_dim:
        return False, f"Gallery feature dimension mismatch: expected {expected_dim}, got {features.shape[1]}.", 0

    if len(features) != len(labels):
        return False, f"Mismatch between features count ({len(features)}) and labels count ({len(labels)}).", 0

    if not np.isfinite(features).all():
        return False, f"Gallery features file '{feats_file.name}' contains non-finite values (NaN or Inf).", 0

    return True, None, len(features)
