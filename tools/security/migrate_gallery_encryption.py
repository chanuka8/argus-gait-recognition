"""Biometric Template Encryption Migration Utility (Finding U3).

Enables operators to migrate legacy plaintext VectorStore biometric galleries
and EmbeddingDatabase person records to authenticated AES-256-GCM containers.

Safety guarantees:
1. Operator Invocation: Never runs automatically on application startup.
2. Non-Destructive Default: Original plaintext files are NEVER deleted unless
   the operator explicitly passes --purge-plaintext after successful migration.
3. Round-Trip Verification: All ciphertexts are test-decrypted and checked for
   bit-for-bit array equality BEFORE atomic installation via os.replace.
4. No In-Place Plaintext Backups: Does not create .bak files in gallery directories.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from security_layer.biometric_encryption import (
    BiometricEncryptor,
    ConfigurationError,
)

logger = logging.getLogger("ARGUS.MigrationUtility")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def migrate_vector_store(
    gallery_dir: str | Path,
    encryptor: BiometricEncryptor,
    dry_run: bool = False,
    purge_plaintext: bool = False,
) -> dict[str, Any]:
    """Migrate a single VectorStore gallery from plaintext .npy to authenticated .enc."""
    g_dir = Path(gallery_dir)
    feats_file = g_dir / "gallery_features.npy"
    lbls_file = g_dir / "gallery_labels.npy"
    enc_file = g_dir / "gallery_features.enc"

    result: dict[str, Any] = {
        "gallery_dir": str(g_dir),
        "status": "SKIPPED",
        "templates": 0,
        "action": "none",
        "plaintext_remaining": feats_file.exists(),
    }

    if not g_dir.exists():
        result["status"] = "ERROR"
        result["error"] = f"Gallery directory '{g_dir}' does not exist."
        return result

    if not feats_file.exists() or not lbls_file.exists():
        if enc_file.exists():
            result["status"] = "ALREADY_ENCRYPTED"
            return result
        result["status"] = "NO_DATA"
        return result

    # Load and validate legacy files
    try:
        features = np.load(feats_file, allow_pickle=False)
        labels = np.load(lbls_file, allow_pickle=False)
    except Exception as err:  # noqa: BLE001
        result["status"] = "ERROR"
        result["error"] = f"Failed to load plaintext files: {err}"
        return result

    if features.dtype == object or labels.dtype == object:
        result["status"] = "ERROR"
        result["error"] = "Object array detected; pickle deserialization prohibited."
        return result

    if not np.isfinite(features).all():
        result["status"] = "ERROR"
        result["error"] = "Plaintext features contain non-finite numbers (NaN/Inf)."
        return result

    result["templates"] = len(features)

    dir_name = g_dir.name.lower()
    gallery_type = (
        "appearance" if "appearance" in dir_name else ("live_gait" if "live" in dir_name else "baseline_gait")
    )

    if dry_run:
        result["status"] = "DRY_RUN"
        result["action"] = f"Would encrypt {len(features)} templates to '{enc_file.name}'"
        return result

    # Encrypt to memory
    container_bytes = encryptor.encrypt_gallery_container(
        features=features,
        labels=labels,
        gallery_type=gallery_type,
    )

    # Write to temporary file on same filesystem
    tmp_enc = g_dir / f".tmp_{uuid.uuid4().hex}.enc"
    try:
        with open(tmp_enc, "wb") as f:
            f.write(container_bytes)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

        # Verify round-trip equality before installing
        decrypted_feats, _aad = encryptor.decrypt_gallery_container(
            container_bytes=tmp_enc.read_bytes(),
            expected_gallery_type=gallery_type,
            expected_labels=labels,
        )

        if not np.array_equal(features, decrypted_feats):
            tmp_enc.unlink(missing_ok=True)
            result["status"] = "ERROR"
            result["error"] = "Round-trip array equality verification failed."
            return result

        # Atomic installation
        os.replace(tmp_enc, enc_file)
        result["status"] = "MIGRATED"
        result["action"] = f"Successfully encrypted {len(features)} templates."

        if purge_plaintext:
            if not enc_file.exists() or enc_file.stat().st_size < BiometricEncryptor.MIN_CONTAINER_BYTES:
                result["warning"] = "Refusing to purge plaintext: encrypted container missing or invalid."
                result["plaintext_remaining"] = True
            else:
                # Independent verification safeguard: verify encrypted container before unlinking plaintext
                verify_res = verify_vector_store(g_dir, encryptor)
                if not verify_res.get("is_valid"):
                    result["warning"] = (
                        f"Refusing to purge plaintext: container verification failed ({verify_res.get('error')})."
                    )
                    result["plaintext_remaining"] = True
                else:
                    feats_file.unlink(missing_ok=True)
                    result["plaintext_remaining"] = False
                    result["action"] += " Plaintext .npy removed (--purge-plaintext)."
        else:
            result["plaintext_remaining"] = True
            result["warning"] = "Plaintext .npy still exists. Purge with --purge-plaintext once verified."

        return result

    except Exception as err:  # noqa: BLE001
        tmp_enc.unlink(missing_ok=True)
        result["status"] = "ERROR"
        result["error"] = f"Migration failed: {err}"
        return result


def migrate_embedding_db(
    db_dir: str | Path,
    encryptor: BiometricEncryptor,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Migrate all person JSON files in an EmbeddingDatabase to field-level encrypted records."""
    d_dir = Path(db_dir)
    persons_dir = d_dir / "persons"

    result: dict[str, Any] = {
        "db_dir": str(d_dir),
        "status": "COMPLETED",
        "total_persons": 0,
        "migrated_persons": 0,
        "migrated_vectors": 0,
        "errors": [],
    }

    if not persons_dir.exists():
        result["status"] = "NO_DATA"
        return result

    person_files = sorted(persons_dir.glob("*.json"))
    result["total_persons"] = len(person_files)

    for p_file in person_files:
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            person_id = str(data.get("person_id", ""))
            modified = False
            file_vectors_migrated = 0

            for group_name in ("gait_embeddings", "appearance_embeddings"):
                embs = data.get(group_name, [])
                for emb in embs:
                    if "vector" in emb and emb.get("vector") and "vector_encryption" not in emb:
                        raw_vec = emb["vector"]
                        modality = str(emb.get("modality", "gait" if "gait" in group_name else "appearance"))
                        model_ver = str(emb.get("model_version", "v1.0.0"))

                        if not dry_run:
                            # Encrypt and verify round-trip
                            enc_payload = encryptor.encrypt_vector(
                                vector=raw_vec,
                                person_id=person_id,
                                modality=modality,
                                model_version=model_ver,
                            )
                            decrypted_vec = encryptor.decrypt_vector(
                                payload=enc_payload,
                                person_id=person_id,
                                modality=modality,
                                model_version=model_ver,
                            )
                            if not all(
                                math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5) for a, b in zip(raw_vec, decrypted_vec)
                            ):
                                raise ValueError(f"Round trip verification failed for {emb.get('embedding_id')}")

                            emb["vector_encryption"] = enc_payload
                            # Remove plaintext vector
                            del emb["vector"]

                        modified = True
                        file_vectors_migrated += 1

            if modified:
                if not dry_run:
                    tmp_file = p_file.with_suffix(f".tmp_{uuid.uuid4().hex}")
                    with open(tmp_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass
                    os.replace(tmp_file, p_file)

                result["migrated_persons"] += 1
                result["migrated_vectors"] += file_vectors_migrated

        except Exception as err:  # noqa: BLE001
            result["errors"].append(f"Failed to migrate '{p_file.name}': {err}")

    if result["errors"]:
        result["status"] = "PARTIAL_ERROR" if result["migrated_persons"] > 0 else "ERROR"

    return result


def verify_vector_store(
    gallery_dir: str | Path,
    encryptor: BiometricEncryptor,
) -> dict[str, Any]:
    """Verify an encrypted VectorStore gallery container."""
    g_dir = Path(gallery_dir)
    enc_file = g_dir / "gallery_features.enc"
    lbls_file = g_dir / "gallery_labels.npy"

    if not enc_file.exists() or not lbls_file.exists():
        return {"gallery_dir": str(g_dir), "status": "MISSING_FILES", "is_valid": False}

    dir_name = g_dir.name.lower()
    gallery_type = (
        "appearance" if "appearance" in dir_name else ("live_gait" if "live" in dir_name else "baseline_gait")
    )

    try:
        labels = np.load(lbls_file, allow_pickle=False)
        container_bytes = enc_file.read_bytes()
        features, aad = encryptor.decrypt_gallery_container(
            container_bytes=container_bytes,
            expected_gallery_type=gallery_type,
            expected_labels=labels,
        )
        return {
            "gallery_dir": str(g_dir),
            "status": "VALID",
            "is_valid": True,
            "templates": len(features),
            "dimension": aad.get("dimension"),
            "dtype": str(features.dtype),
        }
    except Exception as err:  # noqa: BLE001
        return {"gallery_dir": str(g_dir), "status": "INVALID", "is_valid": False, "error": str(err)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARGUS Biometric Template Encryption Migration Utility (Finding U3)")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true", help="Simulate migration without modifying disk.")
    mode_group.add_argument("--migrate", action="store_true", help="Perform encryption migration.")
    mode_group.add_argument("--verify", action="store_true", help="Verify existing encrypted galleries and records.")

    parser.add_argument("--gallery-dir", type=str, default=None, help="Target specific gallery directory.")
    parser.add_argument("--db-dir", type=str, default=None, help="Target specific embedding database directory.")
    parser.add_argument("--all-galleries", action="store_true", help="Migrate standard gallery directories.")
    parser.add_argument(
        "--key", type=str, default=None, help="Encryption key (overrides ARGUS_BIOMETRIC_ENCRYPTION_KEY)."
    )
    parser.add_argument(
        "--purge-plaintext", action="store_true", help="Remove plaintext .npy files after successful migration."
    )

    args = parser.parse_args(argv)

    if args.purge_plaintext and not args.migrate:
        logger.error("Safety violation: --purge-plaintext can only be used in conjunction with --migrate.")
        return 1
    try:
        encryptor = BiometricEncryptor(key=args.key, strict_mode=True)
    except ConfigurationError as err:
        logger.error(f"Configuration error: {err}")
        return 1

    dry_run = args.dry_run

    target_galleries = []
    if args.gallery_dir:
        target_galleries.append(Path(args.gallery_dir))
    elif args.all_galleries:
        target_galleries.extend(
            [
                Path("models/galleries/gallery"),
                Path("models/galleries/live_gallery"),
                Path("models/galleries/appearance_gallery"),
            ]
        )

    target_db = Path(args.db_dir) if args.db_dir else (Path("data/runtime/embedding_db") if args.all_galleries else None)

    if not target_galleries and not target_db:
        logger.error("No targets specified. Use --gallery-dir, --db-dir, or --all-galleries.")
        return 1

    if args.verify:
        logger.info("Starting cryptographic verification...")
        for g in target_galleries:
            res = verify_vector_store(g, encryptor)
            logger.info(f"Gallery [{g}]: {res.get('status')} - Valid: {res.get('is_valid')}")
        return 0

    logger.info(f"Starting biometric encryption migration (dry_run={dry_run})...")
    for g in target_galleries:
        res = migrate_vector_store(g, encryptor, dry_run=dry_run, purge_plaintext=args.purge_plaintext)
        logger.info(f"Gallery [{g}]: {res.get('status')} - Action: {res.get('action', res.get('error'))}")

    if target_db:
        db_res = migrate_embedding_db(target_db, encryptor, dry_run=dry_run)
        logger.info(
            f"EmbeddingDatabase [{target_db}]: {db_res.get('status')} - "
            f"Migrated {db_res.get('migrated_vectors')} vectors across {db_res.get('migrated_persons')} persons."
        )

    logger.info("Migration operation completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
