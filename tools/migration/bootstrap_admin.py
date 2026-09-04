#!/usr/bin/env python3
"""ARGUS AI - Secure First-Time Administrator Bootstrap Script.

Initializes the first root administrator account for an ARGUS deployment.

Security guarantees:
  - Zero hardcoded passwords.
  - Password must be supplied via environment variable (ARGUS_BOOTSTRAP_ADMIN_PASSWORD)
    or entered via secure interactive prompt (masked input).
  - Enforces strong password complexity policy (min 12 chars, upper, lower, number, special).
  - Hashes passwords using Argon2id before persistence.
  - Idempotent and fail-closed: aborts safely without modification if any administrator already exists.
  - Never logs or prints password values.
"""

import argparse
import getpass
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from security_layer.password_hasher import get_password_hasher

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] ARGUS.BootstrapAdmin: %(message)s",
)
logger = logging.getLogger("ARGUS.BootstrapAdmin")


def validate_password_complexity(password: str) -> tuple[bool, str]:
    """Validate password against enterprise complexity policy."""
    if len(password) < 12:
        return False, "Password must be at least 12 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>\-_]', password):
        return False, "Password must contain at least one special character."
    return True, "Password meets complexity requirements."


class AdminBootstrapper:
    def __init__(self, offline_store_path: str = "data/operator_store.json") -> None:
        self.offline_store_path = Path(offline_store_path)
        self.hasher = get_password_hasher()
        self._firestore_client = None
        self._initialized = False

    def _get_firestore_client(self):
        mode = os.environ.get("ARGUS_OPERATOR_STORE_MODE", "firebase").strip().lower()
        if mode == "offline":
            return None
        if self._initialized:
            return self._firestore_client
        try:
            import firebase_admin
            from firebase_admin import firestore

            if firebase_admin._apps:
                self._firestore_client = firestore.client()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Firestore client unavailable for bootstrap: {exc}")
            self._firestore_client = None
        self._initialized = True
        return self._firestore_client

    def admin_exists(self) -> bool:
        """Check whether any administrator account already exists."""
        client = self._get_firestore_client()
        if client is not None:
            try:
                docs = list(client.collection("admins").limit(1).stream())
                if docs:
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error checking Firestore admins: {exc}")

        # Check offline store
        if self.offline_store_path.exists():
            try:
                with open(self.offline_store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                admins = data.get("admins", {})
                if admins:
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error reading offline store: {exc}")

        return False

    def bootstrap(
        self,
        username: str,
        password: str,
        name: str = "Root Administrator",
        nic: str = "000000000000",
    ) -> bool:
        """Create the root administrator account if no admin exists."""
        # 1. Enforce idempotency: refuse if admin exists
        if self.admin_exists():
            logger.warning(
                "[BOOTSTRAP_ABORTED] An administrator account already exists. "
                "Bootstrap refuses to overwrite existing administrators."
            )
            return False

        # 2. Enforce password complexity
        is_valid, reason = validate_password_complexity(password)
        if not is_valid:
            logger.error(f"[PASSWORD_REJECTED] {reason}")
            return False

        # 3. Hash with Argon2id
        password_hash = self.hasher.hash(password)
        clean_username = username.strip().lower()

        doc_data: dict[str, Any] = {
            "name": name,
            "username": clean_username,
            "password_hash": password_hash,
            "password_migrated": True,
            "role": "root_admin",
            "nic": nic,
            "image": f"https://api.dicebear.com/7.x/bottts/svg?seed={clean_username}",
            "status": "Active",
            "lastLogin": "Never",
        }

        # 4. Save to Firestore if available
        client = self._get_firestore_client()
        if client is not None:
            try:
                client.collection("admins").document(clean_username).set(doc_data)
                logger.info(f"[BOOTSTRAP_SUCCESS] Root administrator '{clean_username}' created in Firestore.")
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to write admin to Firestore: {exc}")
                return False

        # 5. Otherwise save to offline store
        try:
            self.offline_store_path.parent.mkdir(parents=True, exist_ok=True)
            offline_data = {}
            if self.offline_store_path.exists():
                with open(self.offline_store_path, "r", encoding="utf-8") as f:
                    offline_data = json.load(f)

            offline_data.setdefault("admins", {})[clean_username] = doc_data
            tmp_path = self.offline_store_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(offline_data, f, indent=2)
            tmp_path.replace(self.offline_store_path)

            logger.info(
                f"[BOOTSTRAP_SUCCESS] Root administrator '{clean_username}' created in offline store "
                f"({self.offline_store_path.as_posix()})."
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to write admin to offline store: {exc}")
            return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARGUS AI - Secure First-Time Administrator Bootstrap",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("ARGUS_BOOTSTRAP_ADMIN_USERNAME", "admin_root"),
        help="Root administrator username (default: admin_root)",
    )
    parser.add_argument(
        "--name",
        default=os.environ.get("ARGUS_BOOTSTRAP_ADMIN_NAME", "Root Administrator"),
        help="Administrator full name",
    )
    parser.add_argument(
        "--store-path",
        default="data/operator_store.json",
        help="Path to offline operator store JSON file",
    )

    args = parser.parse_args()

    bootstrapper = AdminBootstrapper(offline_store_path=args.store_path)

    # Check if admin already exists
    if bootstrapper.admin_exists():
        print("[NOTICE] An administrator already exists in the system. No bootstrap needed.")
        return 0

    # Obtain password securely
    password = os.environ.get("ARGUS_BOOTSTRAP_ADMIN_PASSWORD")
    if not password:
        if sys.stdin.isatty():
            password = getpass.getpass("Enter secure password for root administrator: ")
            confirm = getpass.getpass("Confirm root administrator password: ")
            if password != confirm:
                print("[ERROR] Passwords do not match.")
                return 1
        else:
            logger.error(
                "No password provided. Set ARGUS_BOOTSTRAP_ADMIN_PASSWORD environment variable."
            )
            return 1

    success = bootstrapper.bootstrap(
        username=args.username,
        password=password,
        name=args.name,
    )

    if success:
        print("\n==================================================")
        print("ROOT ADMINISTRATOR BOOTSTRAP COMPLETE")
        print(f"Username : {args.username}")
        print("Role     : root_admin")
        print("Security : Password hashed with Argon2id")
        print("==================================================")
        print("\nACTION REQUIRED: Unset ARGUS_BOOTSTRAP_ADMIN_PASSWORD from your environment.")
        return 0

    print("\n[ERROR] Bootstrap initialization failed. Review logs above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
