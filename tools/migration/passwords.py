#!/usr/bin/env python3
"""ARGUS AI - Secure Password Migration Script.

Migrates legacy plaintext operator credentials in Firestore and local stores to Argon2id.

Safety guarantees:
  - Never logs, prints, or exposes plaintext passwords or hash values.
  - Idempotent: safely skips already-migrated accounts.
  - Fail-safe: only removes legacy plaintext password AFTER the new Argon2id hash is verified as persisted.
  - Supports dry-run inspection mode (--dry-run) and execution mode (--apply).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from security_layer.password_hasher import get_password_hasher

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] ARGUS.PasswordMigration: %(message)s",
)
logger = logging.getLogger("ARGUS.PasswordMigration")


class PasswordMigrator:
    def __init__(self, offline_store_path: str = "data/operator_store.json") -> None:
        self.offline_store_path = Path(offline_store_path)
        self.hasher = get_password_hasher()
        self._firestore_client = None
        self._initialized = False

    def _get_firestore_client(self):
        if self._initialized:
            return self._firestore_client
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            from storage.firebase_embedding_store import validate_service_account_file

            if not firebase_admin._apps:
                raw_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH") or os.environ.get(
                    "GOOGLE_APPLICATION_CREDENTIALS"
                )
                if raw_path and raw_path.strip():
                    cred_path = Path(raw_path.strip())
                elif Path("config/firebase-service-account.json").exists():
                    cred_path = Path("config/firebase-service-account.json")
                else:
                    cred_path = None

                is_valid, reason, meta = validate_service_account_file(cred_path)
                if is_valid:
                    cred = credentials.Certificate(str(cred_path))
                    firebase_admin.initialize_app(
                        cred,
                        {
                            "projectId": meta.get("project_id", "argus-17702"),
                            "storageBucket": "argus-17702.firebasestorage.app",
                        },
                    )
                    logger.info(
                        f"[FIREBASE_ADMIN] Initialized Firebase Admin SDK for project '{meta.get('project_id')}' for migration."
                    )
                else:
                    logger.warning(
                        f"[FIREBASE_ADMIN] Service account credential not configured or invalid: {reason}"
                    )

            if firebase_admin._apps:
                self._firestore_client = firestore.client()
            else:
                self._firestore_client = None
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Firestore client unavailable for migration: {exc}")
            self._firestore_client = None
        self._initialized = True
        return self._firestore_client

    def _load_offline_store(self) -> dict[str, Any]:
        if self.offline_store_path.exists():
            try:
                with open(self.offline_store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to read offline store: {exc}")
        return {"admins": {}, "investigators": {}}

    def _save_offline_store(self, data: dict[str, Any]) -> bool:
        try:
            self.offline_store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.offline_store_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self.offline_store_path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to save offline store: {exc}")
            return False

    def migrate(
        self,
        dry_run: bool = True,
        collections: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute password migration across target collections."""
        target_collections = collections or ["admins", "investigators"]
        stats = {
            "mode": "DRY_RUN" if dry_run else "APPLY",
            "total_scanned": 0,
            "already_migrated": 0,
            "needs_migration": 0,
            "migrated_success": 0,
            "migration_failed": 0,
            "malformed_records": 0,
            "details": [],
        }

        client = self._get_firestore_client()
        mode = os.environ.get("ARGUS_OPERATOR_STORE_MODE", "firebase").strip().lower()

        if mode not in ("offline", "local") and client is not None:
            logger.info("[SOURCE] Operating against live Firestore database.")
            self._migrate_firestore(client, target_collections, dry_run, stats)
        elif mode == "firebase":
            logger.error("[MIGRATION_ERROR] Mode is 'firebase' but Firebase Admin SDK credentials are not configured.")
            stats["details"].append({
                "error": "Firebase Admin SDK credentials not configured in environment. Set FIREBASE_SERVICE_ACCOUNT_PATH or GOOGLE_APPLICATION_CREDENTIALS.",
            })
        else:
            logger.info(f"[SOURCE] Operating against offline store: {self.offline_store_path.as_posix()}")
            self._migrate_offline(target_collections, dry_run, stats)

        return stats

    def _migrate_firestore(
        self,
        client: Any,
        collections: list[str],
        dry_run: bool,
        stats: dict[str, Any],
    ) -> None:
        for col_name in collections:
            try:
                from firebase_admin import firestore

                docs = list(client.collection(col_name).stream())
                for doc in docs:
                    stats["total_scanned"] += 1
                    data = doc.to_dict() or {}
                    username = data.get("username", doc.id)

                    pw_hash = data.get("password_hash")
                    pw_plain = data.get("password")

                    if pw_hash and isinstance(pw_hash, str) and pw_hash.startswith("$argon2"):
                        stats["already_migrated"] += 1
                        stats["details"].append({
                            "collection": col_name,
                            "id": doc.id,
                            "username": username,
                            "status": "ALREADY_MIGRATED",
                        })
                        continue

                    if not pw_plain or not isinstance(pw_plain, str) or not pw_plain.strip():
                        stats["malformed_records"] += 1
                        stats["details"].append({
                            "collection": col_name,
                            "id": doc.id,
                            "username": username,
                            "status": "MALFORMED_NO_PASSWORD",
                        })
                        continue

                    stats["needs_migration"] += 1

                    if dry_run:
                        stats["details"].append({
                            "collection": col_name,
                            "id": doc.id,
                            "username": username,
                            "status": "NEEDS_MIGRATION",
                        })
                        continue

                    # Execute Apply
                    try:
                        new_hash = self.hasher.hash(pw_plain)
                        doc_ref = client.collection(col_name).document(doc.id)
                        # Step 1: Persist password_hash and migrated flag
                        doc_ref.update({
                            "password_hash": new_hash,
                            "password_migrated": True,
                        })

                        # Step 2: Read back and verify persistence
                        snap = doc_ref.get()
                        persisted = snap.to_dict() or {}
                        if (
                            persisted.get("password_hash") != new_hash
                            or persisted.get("password_migrated") is not True
                        ):
                            stats["migration_failed"] += 1
                            logger.error(
                                f"Verification read-back failed for {doc.id} in {col_name}. Retaining legacy password."
                            )
                            stats["details"].append({
                                "collection": col_name,
                                "id": doc.id,
                                "username": username,
                                "status": "MIGRATION_FAILED",
                                "error": "Verification read-back mismatch",
                            })
                            continue

                        # Step 3: Only after verified persistence, remove legacy plaintext
                        doc_ref.update({
                            "password": firestore.DELETE_FIELD,
                        })
                        stats["migrated_success"] += 1
                        stats["details"].append({
                            "collection": col_name,
                            "id": doc.id,
                            "username": username,
                            "status": "MIGRATED_SUCCESS",
                        })
                    except Exception as exc:  # noqa: BLE001
                        stats["migration_failed"] += 1
                        logger.error(f"Failed to migrate account {doc.id} in {col_name}: {exc}. Legacy password preserved.")
                        stats["details"].append({
                            "collection": col_name,
                            "id": doc.id,
                            "username": username,
                            "status": "MIGRATION_FAILED",
                            "error": str(exc),
                        })

            except Exception as exc:  # noqa: BLE001
                logger.error(f"Error accessing collection {col_name} in Firestore: {exc}")

    def _migrate_offline(
        self,
        collections: list[str],
        dry_run: bool,
        stats: dict[str, Any],
    ) -> None:
        offline_data = self._load_offline_store()
        dirty = False

        for col_name in collections:
            col_data = offline_data.get(col_name, {})
            for doc_id, data in col_data.items():
                stats["total_scanned"] += 1
                username = data.get("username", doc_id)
                pw_hash = data.get("password_hash")
                pw_plain = data.get("password")

                if pw_hash and isinstance(pw_hash, str) and pw_hash.startswith("$argon2"):
                    stats["already_migrated"] += 1
                    stats["details"].append({
                        "collection": col_name,
                        "id": doc_id,
                        "username": username,
                        "status": "ALREADY_MIGRATED",
                    })
                    continue

                if not pw_plain or not isinstance(pw_plain, str) or not pw_plain.strip():
                    stats["malformed_records"] += 1
                    stats["details"].append({
                        "collection": col_name,
                        "id": doc_id,
                        "username": username,
                        "status": "MALFORMED_NO_PASSWORD",
                    })
                    continue

                stats["needs_migration"] += 1

                if dry_run:
                    stats["details"].append({
                        "collection": col_name,
                        "id": doc_id,
                        "username": username,
                        "status": "NEEDS_MIGRATION",
                    })
                    continue

                # Execute Apply
                try:
                    new_hash = self.hasher.hash(pw_plain)
                    data["password_hash"] = new_hash
                    data["password_migrated"] = True
                    # Only delete plaintext after new_hash is assigned
                    data.pop("password", None)
                    stats["migrated_success"] += 1
                    dirty = True
                    stats["details"].append({
                        "collection": col_name,
                        "id": doc_id,
                        "username": username,
                        "status": "MIGRATED_SUCCESS",
                    })
                except Exception as exc:  # noqa: BLE001
                    stats["migration_failed"] += 1
                    logger.error(f"Failed to hash account {doc_id}: {exc}. Legacy password preserved.")
                    stats["details"].append({
                        "collection": col_name,
                        "id": doc_id,
                        "username": username,
                        "status": "MIGRATION_FAILED",
                        "error": str(exc),
                    })

        if dirty and not dry_run:
            success = self._save_offline_store(offline_data)
            if not success:
                logger.critical("Failed to write migrated data to disk!")
                stats["migration_failed"] += stats["migrated_success"]
                stats["migrated_success"] = 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARGUS AI - Fail-safe Password Migration to Argon2id",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report accounts needing migration without modifying data",
    )
    group.add_argument(
        "--apply",
        action="store_true",
        help="Execute password migration and persist Argon2id hashes",
    )
    parser.add_argument(
        "--store-path",
        default="data/operator_store.json",
        help="Path to offline operator store JSON file",
    )

    args = parser.parse_args()

    migrator = PasswordMigrator(offline_store_path=args.store_path)
    stats = migrator.migrate(dry_run=args.dry_run)

    print("\n==================================================")
    print(f"ARGUS PASSWORD MIGRATION REPORT [{stats['mode']}]")
    print("==================================================")
    print(f"Total Accounts Scanned : {stats['total_scanned']}")
    print(f"Already Migrated       : {stats['already_migrated']}")
    print(f"Needs Migration        : {stats['needs_migration']}")
    print(f"Migrated Successfully  : {stats['migrated_success']}")
    print(f"Migration Failed       : {stats['migration_failed']}")
    print(f"Malformed Records      : {stats['malformed_records']}")
    print("--------------------------------------------------")

    for item in stats["details"]:
        if "error" in item:
            print(f"[ERROR               ] {item['error']}")
        else:
            status_str = item.get("status", "UNKNOWN")
            print(f"[{status_str:20s}] {item.get('collection', ''):13s} : {item.get('username', '')}")

    print("==================================================")

    if stats["migration_failed"] > 0:
        logger.error("Migration encountered failures. Check logs for details.")
        return 1

    if args.dry_run and stats["needs_migration"] > 0:
        print("\nAction required: Run with '--apply' to execute migration.")
    elif not args.dry_run:
        print("\nMigration complete. All valid credentials migrated to Argon2id.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
