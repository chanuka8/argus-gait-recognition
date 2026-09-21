"""Offline / Build-Time Model Artifact Encryption and Provisioning Tool.

Finding U5 — Model Checkpoint / AI Model Asset Protection (Phase 2).
Provides:
- Authenticated AES-256-GCM container provisioning for proprietary models
- Cross-platform exclusive destination locking (.lock) preventing race conditions
- Guaranteed complete writes via buffered Python I/O and os.fsync
- Pre-replace in-memory roundtrip decryption and SHA-256 self-verification
- Atomic destination file replacement
- Robust cleanup of locks and temporary files on success and failure
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path

from security_layer.model_confidentiality import (
    EnvModelKeyProvider,
    FileModelKeyProvider,
    ModelKeyProvider,
    ModelProvisioningError,
    ProvisioningLockError,
    decrypt_model_container,
    encrypt_model_container,
)

logger = logging.getLogger("argus.security.provisioning")


def provision_encrypted_artifact(
    source_path: str | Path,
    destination_path: str | Path,
    key_id: str = "default",
    key_provider: ModelKeyProvider | None = None,
    key_material: bytes | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically encrypts and provisions a model file with destination-level locking and complete write semantics.

    Guarantees:
    - Prevents concurrent writers from clobbering destination via exclusive .lock.
    - Writes full encrypted bytes to unique temp file in destination directory.
    - Flushes and fsyncs before self-verification.
    - Self-verifies decryption and SHA-256 before atomic os.replace.
    - Unconditionally unlinks temp file and removes lock on exit or failure.
    """
    src = Path(source_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source model file not found: {src}")

    dest = Path(destination_path).resolve()
    dest_dir = dest.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    lock_path = dest_dir / f".{dest.name}.lock"

    # Acquire exclusive destination lock
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ProvisioningLockError(
            f"Artifact provisioning in progress for '{dest.name}' (lock held at {lock_path.name})."
        )

    temp_path: Path | None = None
    try:
        # Re-check destination under lock
        if dest.exists() and not overwrite:
            raise FileExistsError(f"Destination artifact already exists: {dest}")

        plaintext_bytes = src.read_bytes()
        expected_sha256 = hashlib.sha256(plaintext_bytes).hexdigest().lower()

        # Encrypt in memory
        container_bytes = encrypt_model_container(
            plaintext_bytes=plaintext_bytes,
            key_id=key_id,
            key_provider=key_provider,
            key_material=key_material,
        )

        # Create unique temp file in destination directory
        temp_fd, temp_raw_path = tempfile.mkstemp(
            prefix=f".{dest.name}.tmp_",
            suffix=".enc",
            dir=dest_dir,
        )
        temp_path = Path(temp_raw_path)

        # Complete-write semantics using Python buffered I/O + fsync
        with os.fdopen(temp_fd, "wb", closefd=True) as fh:
            written = fh.write(container_bytes)
            if written != len(container_bytes):
                raise ModelProvisioningError(
                    f"Incomplete encrypted artifact write: expected {len(container_bytes)} bytes, wrote {written}."
                )
            fh.flush()
            os.fsync(fh.fileno())

        # Self-verification of written artifact before replace
        written_bytes = temp_path.read_bytes()
        if len(written_bytes) != len(container_bytes):
            raise ModelProvisioningError(
                f"Written temp artifact size mismatch: expected {len(container_bytes)}, got {len(written_bytes)}."
            )

        decrypted = decrypt_model_container(
            written_bytes,
            key_provider=key_provider,
            key_material=key_material,
        )
        actual_sha256 = hashlib.sha256(decrypted).hexdigest().lower()

        if actual_sha256 != expected_sha256:
            raise ModelProvisioningError(
                f"Post-write verification failed: plaintext SHA-256 mismatch ({actual_sha256} != {expected_sha256})."
            )

        # Atomic replacement into final destination under lock
        os.replace(temp_path, dest)
        temp_path = None  # Replaced successfully, nothing to cleanup

        logger.info(
            "Successfully encrypted and provisioned %s -> %s (size: %d bytes, sha256: %s...)",
            src.name,
            dest.name,
            len(written_bytes),
            expected_sha256[:12],
        )
        return dest

    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            os.close(lock_fd)
        except OSError:
            pass
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encrypt an ARGUS proprietary model artifact using authenticated AES-256-GCM."
    )
    parser.add_argument("--input", required=True, help="Path to input plaintext model file")
    parser.add_argument("--output", help="Path to output encrypted container (.enc)")
    parser.add_argument("--key-id", default="default", help="Key identifier (default: 'default')")
    parser.add_argument("--key", help="Raw 32-byte key or 64-character hex string")
    parser.add_argument("--key-file", help="Path to file containing raw or hex key")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists")

    args = parser.parse_args()

    src = Path(args.input)
    dest = Path(args.output) if args.output else src.with_name(src.name + ".enc")

    key_material: bytes | None = None
    key_provider: ModelKeyProvider | None = None

    if args.key:
        raw_k = args.key.strip()
        if len(raw_k) == 64:
            key_material = bytes.fromhex(raw_k)
        elif len(raw_k.encode("utf-8")) == 32:
            key_material = raw_k.encode("utf-8")
        else:
            print("[ERROR] Key must be 64 hex characters or 32 raw bytes.", file=sys.stderr)
            return 1
    elif args.key_file:
        key_provider = FileModelKeyProvider(args.key_file, key_id=args.key_id)
    else:
        key_provider = EnvModelKeyProvider()

    try:
        out_path = provision_encrypted_artifact(
            source_path=src,
            destination_path=dest,
            key_id=args.key_id,
            key_provider=key_provider,
            key_material=key_material,
            overwrite=args.overwrite,
        )
        print(f"[ARGUS] Provisioned encrypted model: {out_path}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Provisioning failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
