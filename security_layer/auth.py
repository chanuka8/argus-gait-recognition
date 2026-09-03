import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, status

from security_layer.password_hasher import get_password_hasher

logger = logging.getLogger("ARGUS.Auth")


class AuthenticationInfrastructureError(RuntimeError):
    """Raised when authoritative authentication backend (e.g. Firebase Admin SDK) is misconfigured or unavailable."""



@dataclass
class SessionToken:
    token: str
    operator_id: str
    username: str
    role: str
    name: str = ""
    nic: str = ""
    image: str = ""
    status: str = "Active"
    created_at: float = 0.0
    expires_at: float = 0.0
    last_activity: float = 0.0
    source_ip: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_profile_dict(self) -> dict[str, Any]:
        return {
            "id": self.operator_id,
            "username": self.username,
            "name": self.name,
            "role": self.role,
            "nic": self.nic,
            "image": self.image,
            "status": self.status,
            "last_activity": self.last_activity,
        }


class SessionStore:
    """Thread-safe, in-memory session manager with sliding idle timeout and concurrency limits."""

    def __init__(
        self,
        ttl_seconds: int = 8 * 3600,
        idle_timeout_seconds: int = 30 * 60,
        max_concurrent_sessions: int = 5,
    ) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionToken] = {}
        self._operator_sessions: dict[str, set[str]] = {}
        self.ttl_seconds = ttl_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_concurrent_sessions = max_concurrent_sessions

    def create_session(
        self,
        operator_id: str,
        username: str,
        role: str,
        name: str = "",
        nic: str = "",
        image: str = "",
        status_val: str = "Active",
        source_ip: str = "",
        status: str | None = None,
    ) -> SessionToken:
        with self._lock:
            effective_status = status or status_val
            now = time.time()
            self._cleanup_expired_locked(now)

            # Enforce max concurrent sessions per operator (evict oldest)
            existing_tokens = self._operator_sessions.get(operator_id, set())
            if len(existing_tokens) >= self.max_concurrent_sessions:
                # Find oldest by last_activity
                oldest_token = min(
                    existing_tokens,
                    key=lambda tok: self._sessions.get(tok, SessionToken("", "", "", "")).last_activity,
                    default=None,
                )
                if oldest_token and oldest_token in self._sessions:
                    del self._sessions[oldest_token]
                    existing_tokens.discard(oldest_token)

            token_val = secrets.token_urlsafe(32)
            session = SessionToken(
                token=token_val,
                operator_id=operator_id,
                username=username,
                role=role.lower(),
                name=name,
                nic=nic,
                image=image,
                status=effective_status,
                created_at=now,
                expires_at=now + self.ttl_seconds,
                last_activity=now,
                source_ip=source_ip,
            )

            self._sessions[token_val] = session
            if operator_id not in self._operator_sessions:
                self._operator_sessions[operator_id] = set()
            self._operator_sessions[operator_id].add(token_val)

            return session

    def get_session(self, token: str, update_activity: bool = True) -> SessionToken | None:
        with self._lock:
            now = time.time()
            session = self._sessions.get(token)
            if not session:
                return None

            # Check absolute TTL
            if now > session.expires_at:
                self.revoke_session(token)
                return None

            # Check idle timeout
            if self.idle_timeout_seconds > 0 and (now - session.last_activity) > self.idle_timeout_seconds:
                self.revoke_session(token)
                return None

            if update_activity:
                session.last_activity = now

            return session

    def revoke_session(self, token: str) -> bool:
        with self._lock:
            session = self._sessions.pop(token, None)
            if session:
                op_tokens = self._operator_sessions.get(session.operator_id)
                if op_tokens:
                    op_tokens.discard(token)
                    if not op_tokens:
                        self._operator_sessions.pop(session.operator_id, None)
                return True
            return False

    def revoke_all_for_operator(self, operator_id: str) -> int:
        with self._lock:
            op_tokens = list(self._operator_sessions.get(operator_id, set()))
            count = 0
            for tok in op_tokens:
                if self.revoke_session(tok):
                    count += 1
            return count

    def _cleanup_expired_locked(self, now: float) -> int:
        expired = [
            tok for tok, sess in self._sessions.items()
            if now > sess.expires_at or (self.idle_timeout_seconds > 0 and (now - sess.last_activity) > self.idle_timeout_seconds)
        ]
        for tok in expired:
            self.revoke_session(tok)
        return len(expired)

    def cleanup_expired(self) -> int:
        with self._lock:
            return self._cleanup_expired_locked(time.time())

    def clear(self) -> None:
        """Clear all sessions (useful for tests)."""
        with self._lock:
            self._sessions.clear()
            self._operator_sessions.clear()


_GLOBAL_SESSION_STORE = SessionStore()


def get_session_store() -> SessionStore:
    return _GLOBAL_SESSION_STORE


class OperatorStore:
    """Manages operator records across Firebase Admin SDK (production) and isolated offline storage (dev/test)."""

    def __init__(self, offline_store_path: str = "data/operator_store.json") -> None:
        self.offline_store_path = Path(offline_store_path)
        self.hasher = get_password_hasher()
        self._lock = threading.RLock()
        self._firestore_client = None
        self._initialized = False

    @property
    def mode(self) -> str:
        """Operating mode: 'firebase' (authoritative production) or 'offline' (dev/test)."""
        return os.environ.get("ARGUS_OPERATOR_STORE_MODE", "firebase").strip().lower()

    def _get_firestore_client(self):
        """Thread-safe singleton initialization of Firebase Admin SDK."""
        if self._initialized:
            return self._firestore_client

        with self._lock:
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
                            f"[FIREBASE_ADMIN] Initialized Firebase Admin SDK for project '{meta.get('project_id')}' with credentials."
                        )
                    else:
                        logger.warning(
                            f"[FIREBASE_ADMIN] Service account credential not configured or invalid: {reason}"
                        )

                if firebase_admin._apps:
                    self._firestore_client = firestore.client()
                    logger.info("[FIREBASE_ADMIN] Firestore client connected successfully.")
                else:
                    self._firestore_client = None
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[FIREBASE_ADMIN_INIT_ERROR] Error initializing Firebase Admin: {exc}")
                self._firestore_client = None

            self._initialized = True
            return self._firestore_client

    def reset_client(self) -> None:
        """Reset cached client (for testing configuration changes)."""
        with self._lock:
            self._firestore_client = None
            self._initialized = False

    def _load_offline_store(self) -> dict[str, Any]:
        with self._lock:
            if self.offline_store_path.exists():
                try:
                    with open(self.offline_store_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Failed to read offline operator store: {exc}")
            return {"admins": {}, "investigators": {}}

    def _save_offline_store(self, data: dict[str, Any]) -> bool:
        with self._lock:
            try:
                self.offline_store_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self.offline_store_path.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                tmp_path.replace(self.offline_store_path)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to write offline operator store: {exc}")
                return False

    def get_operator(
        self,
        username: str,
        collection_name: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        """Lookup an operator by username according to active store mode.

        Returns: (operator_data, collection_name, document_id) or (None, None, None)
        Raises: AuthenticationInfrastructureError if production Firebase is unconfigured or unreachable.
        """
        username_clean = username.strip().lower()
        collections_to_search = [collection_name] if collection_name else ["admins", "investigators"]

        if self.mode == "firebase":
            client = self._get_firestore_client()
            if client is None:
                logger.error("[OPERATOR_STORE] Mode is 'firebase' but Firebase Admin SDK is uninitialized/unavailable.")
                raise AuthenticationInfrastructureError("Authentication service infrastructure unavailable")

            try:
                for col in collections_to_search:
                    docs = list(client.collection(col).where("username", "==", username_clean).limit(1).stream())
                    if docs:
                        doc = docs[0]
                        data = doc.to_dict() or {}
                        data["id"] = doc.id
                        return data, col, doc.id
                return None, None, None
            except Exception as exc:
                logger.error(f"[FIRESTORE_QUERY_ERROR] Firestore lookup failed: {exc}")
                raise AuthenticationInfrastructureError("Authentication service infrastructure unavailable") from exc

        elif self.mode == "offline":
            offline_data = self._load_offline_store()
            for col in collections_to_search:
                col_data = offline_data.get(col, {})
                for doc_id, u_data in col_data.items():
                    if str(u_data.get("username", "")).strip().lower() == username_clean:
                        res = dict(u_data)
                        res["id"] = doc_id
                        return res, col, doc_id
            return None, None, None

        else:
            logger.error(f"[OPERATOR_STORE] Invalid ARGUS_OPERATOR_STORE_MODE: '{self.mode}'")
            raise AuthenticationInfrastructureError("Invalid operator store configuration mode")

    def authenticate_operator(
        self,
        username: str,
        password: str,
        role: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Authenticate an operator and perform fail-safe Argon2id migration if needed.

        Returns:
            (operator_dict, error_message)
        Raises:
            AuthenticationInfrastructureError if production backend is unavailable.
        """
        if not username or not password:
            return None, "Username and password are required"

        target_collection = None
        if role:
            role_clean = role.strip().lower()
            if role_clean in ("admin", "root admin", "root_admin"):
                target_collection = "admins"
            elif role_clean in ("investigator",):
                target_collection = "investigators"

        user_data, col, doc_id = self.get_operator(username, target_collection)
        if not user_data or not col or not doc_id:
            return None, "Operator account not found"

        if user_data.get("status") == "Suspended":
            return None, "Account is suspended. Contact administration."

        # Verification check: prefer password_hash, fallback to legacy password
        stored_hash = user_data.get("password_hash")
        stored_plaintext = user_data.get("password")

        credential_to_verify = stored_hash if stored_hash else stored_plaintext
        if not credential_to_verify:
            return None, "Account has no valid credential configured"

        is_valid, needs_rehash = self.hasher.verify(password, credential_to_verify)
        if not is_valid:
            return None, "Invalid credentials"

        # Fail-safe Argon2id migration:
        # If the password verified against legacy plaintext or needs parameter upgrade,
        # hash it with Argon2id and persist. Only delete legacy plaintext on verified persistence.
        if needs_rehash:
            self._migrate_credential_failsafe(col, doc_id, password)

        # Prepare clean return dict (strip passwords)
        clean_user = dict(user_data)
        clean_user.pop("password", None)
        clean_user.pop("password_hash", None)
        return clean_user, None

    def _migrate_credential_failsafe(self, collection_name: str, doc_id: str, password: str) -> bool:
        """Fail-safe credential migration to Argon2id.

        Invariants:
          - Persists password_hash and password_migrated=True first.
          - Verifies persistence via read-back.
          - Only after verified persistence deletes the legacy plaintext password.
          - Never deletes legacy credential if hashing, persistence, or read-back fails.
          - Never logs password values or hashes.
        """
        try:
            new_hash = self.hasher.hash(password)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[MIGRATION_FAILED] Could not compute Argon2id hash: {exc}")
            return False

        if self.mode == "firebase":
            client = self._get_firestore_client()
            if client is None:
                logger.error("[MIGRATION_FAILED] Firebase Admin SDK client unavailable.")
                return False
            try:
                from firebase_admin import firestore

                doc_ref = client.collection(collection_name).document(doc_id)
                # 1. Persist password_hash and migrated flag
                doc_ref.update({
                    "password_hash": new_hash,
                    "password_migrated": True,
                })

                # 2. Verify persistence via read-back
                snap = doc_ref.get()
                persisted_data = snap.to_dict() or {}
                if (
                    persisted_data.get("password_hash") != new_hash
                    or persisted_data.get("password_migrated") is not True
                ):
                    logger.error(
                        f"[MIGRATION_FAILED] Persistence read-back mismatch for {doc_id}. Retaining legacy password."
                    )
                    return False

                # 3. Only after verification, remove legacy plaintext field
                doc_ref.update({
                    "password": firestore.DELETE_FIELD,
                })
                logger.info(f"[MIGRATION_SUCCESS] Account {doc_id} in {collection_name} verified and migrated to Argon2id.")
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[MIGRATION_FAILED] Failed to update Firestore: {exc}. Legacy password preserved.")
                return False

        elif self.mode == "offline":
            offline_data = self._load_offline_store()
            col_dict = offline_data.setdefault(collection_name, {})
            if doc_id in col_dict:
                col_dict[doc_id]["password_hash"] = new_hash
                col_dict[doc_id]["password_migrated"] = True
                col_dict[doc_id].pop("password", None)
                success = self._save_offline_store(offline_data)
                if success:
                    logger.info(f"[MIGRATION_SUCCESS] Offline account {doc_id} in {collection_name} migrated to Argon2id.")
                    return True
                logger.error("[MIGRATION_FAILED] Failed to save offline store. Legacy password preserved.")
                return False

        return False

    def create_or_update_operator(
        self,
        collection_name: str,
        doc_id: str,
        username: str,
        password: str,
        role: str,
        name: str = "",
        nic: str = "",
        image: str = "",
        status_val: str = "Active",
    ) -> bool:
        """Create or update an operator with mandatory Argon2id password hashing."""
        password_hash = self.hasher.hash(password)
        data = {
            "name": name,
            "username": username.strip().lower(),
            "password_hash": password_hash,
            "password_migrated": True,
            "role": role.lower(),
            "nic": nic,
            "image": image,
            "status": status_val,
            "lastLogin": "Never",
            "updated_at": time.time(),
        }

        if self.mode == "firebase":
            client = self._get_firestore_client()
            if client is None:
                logger.error("[OPERATOR_CREATE_FAILED] Firebase Admin SDK unavailable in firebase mode.")
                raise AuthenticationInfrastructureError("Authentication service infrastructure unavailable")
            try:
                client.collection(collection_name).document(doc_id).set(data, merge=True)
                return True
            except Exception as exc:
                logger.error(f"Failed to create operator in Firestore: {exc}")
                raise AuthenticationInfrastructureError("Authentication service infrastructure unavailable") from exc

        elif self.mode == "offline":
            offline_data = self._load_offline_store()
            offline_data.setdefault(collection_name, {})[doc_id] = data
            return self._save_offline_store(offline_data)

        raise AuthenticationInfrastructureError("Invalid operator store configuration mode")

    def delete_operator(self, collection_name: str, doc_id: str) -> bool:
        """Delete an operator document according to mode."""
        if self.mode == "firebase":
            client = self._get_firestore_client()
            if client is None:
                raise AuthenticationInfrastructureError("Authentication service infrastructure unavailable")
            try:
                client.collection(collection_name).document(doc_id).delete()
                return True
            except Exception as exc:
                logger.error(f"Failed to delete operator in Firestore: {exc}")
                raise AuthenticationInfrastructureError("Authentication service infrastructure unavailable") from exc

        elif self.mode == "offline":
            offline_data = self._load_offline_store()
            if collection_name in offline_data and doc_id in offline_data[collection_name]:
                del offline_data[collection_name][doc_id]
                return self._save_offline_store(offline_data)
            return True

        raise AuthenticationInfrastructureError("Invalid operator store configuration mode")


_GLOBAL_OPERATOR_STORE = OperatorStore()


def get_operator_store() -> OperatorStore:
    return _GLOBAL_OPERATOR_STORE


# ---------------------------------------------------------------------------
# FastAPI Authentication Dependencies
# ---------------------------------------------------------------------------

def extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def get_authenticated_operator(request: Request) -> SessionToken:
    """FastAPI dependency enforcing valid server-side session authentication.

    Zero-Trust Policy:
      - Ignores client-supplied X-User-ID as proof of identity.
      - Requires Authorization: Bearer <valid_token>.
      - Raises HTTP 401 Unauthorized if token is missing, expired, or invalid.
    """
    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide Authorization: Bearer <session_token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    store = get_session_store()
    session = store.get_session(token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if session.status == "Suspended":
        store.revoke_session(token)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been suspended",
        )

    # Attach verified session to request state for downstream handlers
    request.state.operator = session
    return session


async def get_optional_operator(request: Request) -> SessionToken | None:
    """FastAPI dependency that extracts an authenticated operator if present, or returns None."""
    token = extract_bearer_token(request)
    if not token:
        return None
    session = get_session_store().get_session(token)
    if session and session.status != "Suspended":
        request.state.operator = session
        return session
    return None
