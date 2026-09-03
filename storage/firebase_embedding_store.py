import hashlib
import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


class PersistenceErrorCategory(str, Enum):
    NONE = "NONE"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    DUPLICATE_REJECTED = "DUPLICATE_REJECTED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


def generate_deterministic_id(
    person_id: str,
    modality: str,
    capture_timestamp: float,
    track_id: int | str = 0,
    camera_id: str = "cctv-01",
) -> str:
    raw_key = f"{person_id}:{modality}:{int(capture_timestamp)}:{track_id}:{camera_id}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"emb_{modality[:4]}_{person_id}_{int(capture_timestamp)}_{digest}"


@dataclass
class FirebaseEmbeddingDocument:
    embedding_id: str
    person_id: str
    modality: str = "gait"
    embedding_dim: int = 256
    vector: list[float] = field(default_factory=list)
    model_version: str = "v1.0.0"
    model_name: str = ""
    model_architecture: str = ""
    identity_type: str = "LIVE_OPERATIONAL"
    source: str = "cctv_live"
    source_type: str = "live_surveillance"
    feature_version: int = 1
    embedding_version: int = 1
    observation_date: str = ""
    event_date: str = ""
    capture_timestamp: float = 0.0
    camera_id: str = ""
    track_id: int = 0
    confidence: float = 1.0
    quality_score: float = 1.0
    verification_state: str = "PREDICTED"
    operational_state: str = "PREDICTED"
    training_state: str = "NOT_ELIGIBLE"
    training_eligibility: str = "NOT_ELIGIBLE"
    dataset_split: str = "UNASSIGNED"
    training_consumed: bool = False
    consumed_by_model_version: str = ""
    consumed_in_training_job: str = ""
    lineage_id: str = ""
    parent_embedding_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    verification_metadata: dict[str, Any] = field(default_factory=dict)
    case_id: str = ""
    created_by: str = "argus_system"
    verification_status: str = "ACTIVE"
    status: str = "ACTIVE"
    source_session_id: str = ""
    source_camera_id: str = ""
    candidate_training_job_id: str = ""
    production_model_version: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.observation_date:
            ts = self.capture_timestamp if self.capture_timestamp > 0 else self.created_at
            self.observation_date = time.strftime("%Y-%m-%d", time.gmtime(ts))
        if not self.event_date:
            self.event_date = self.observation_date
        if self.capture_timestamp == 0.0:
            self.capture_timestamp = self.created_at
        if not self.source_camera_id and self.camera_id:
            self.source_camera_id = self.camera_id
        if not self.camera_id and self.source_camera_id:
            self.camera_id = self.source_camera_id

    @property
    def identity_id(self) -> str:
        return self.person_id

    @identity_id.setter
    def identity_id(self, val: str) -> None:
        self.person_id = str(val)

    @property
    def embedding_type(self) -> str:
        return self.modality

    @embedding_type.setter
    def embedding_type(self, val: str) -> None:
        self.modality = str(val)

    @property
    def embedding_dimension(self) -> int:
        return self.embedding_dim

    @embedding_dimension.setter
    def embedding_dimension(self, val: int) -> None:
        self.embedding_dim = int(val)

    @property
    def embedding(self) -> list[float]:
        return self.vector

    @embedding.setter
    def embedding(self, val: list[float]) -> None:
        self.vector = [float(v) for v in val]

    @property
    def capture_date(self) -> str:
        return self.observation_date

    @capture_date.setter
    def capture_date(self, val: str) -> None:
        self.observation_date = str(val)
        self.event_date = str(val)

    @property
    def training_eligible(self) -> bool:
        return self.training_eligibility == "ELIGIBLE" or self.operational_state == "TRAINING_ELIGIBLE"

    def validate_schema(self) -> tuple[bool, str]:
        if not self.embedding_id or not str(self.embedding_id).strip():
            return False, "Missing or empty embedding_id"
        if not self.person_id or not str(self.person_id).strip():
            return False, "Missing or empty person_id"
        if self.modality not in ("gait", "appearance"):
            return False, f"Invalid modality '{self.modality}'; must be 'gait' or 'appearance'"
        if self.modality == "gait" and self.embedding_dim != 256:
            return False, f"Gait embedding dimension mismatch: expected 256, got {self.embedding_dim}"
        if self.modality == "appearance" and self.embedding_dim != 512:
            return False, f"Appearance embedding dimension mismatch: expected 512, got {self.embedding_dim}"
        if len(self.vector) != self.embedding_dim:
            return False, f"Vector length {len(self.vector)} != declared embedding_dim {self.embedding_dim}"

        vec_arr = np.asarray(self.vector, dtype=np.float32)
        if not np.isfinite(vec_arr).all():
            return False, "Vector contains non-finite values (NaN or Inf)"
        norm = float(np.linalg.norm(vec_arr))
        if norm <= 0.0:
            return False, "Vector norm must be positive (zero vector invalid)"

        valid_states = (
            "PREDICTED",
            "VERIFIED",
            "TRAINING_ELIGIBLE",
            "TRAINING_CONSUMED",
            "REFERENCE",
            "REJECTED",
        )
        if self.operational_state not in valid_states:
            return False, f"Invalid operational_state '{self.operational_state}'"

        return True, "Schema valid"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["identity_id"] = self.person_id
        d["embedding_type"] = self.modality
        d["embedding_dimension"] = self.embedding_dim
        d["observation_date"] = self.observation_date
        d["event_date"] = self.event_date
        d["capture_date"] = self.observation_date
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FirebaseEmbeddingDocument":
        person_id = str(data.get("person_id") or data.get("identity_id") or "")
        modality = str(data.get("modality") or data.get("embedding_type") or "gait")
        embedding_dim = int(data.get("embedding_dim") or data.get("embedding_dimension") or len(data.get("vector") or data.get("embedding") or []))
        vector = [float(v) for v in (data.get("vector") or data.get("embedding") or [])]
        c_at = float(data.get("created_at", time.time()))
        cap_ts = float(data.get("capture_timestamp", c_at))
        obs_date = str(data.get("observation_date") or data.get("capture_date") or data.get("event_date") or "")

        return cls(
            embedding_id=str(data.get("embedding_id", "")),
            person_id=person_id,
            modality=modality,
            embedding_dim=embedding_dim,
            vector=vector,
            model_version=str(data.get("model_version", "v1.0.0")),
            model_name=str(data.get("model_name", "")),
            model_architecture=str(data.get("model_architecture", "")),
            identity_type=str(data.get("identity_type", "LIVE_OPERATIONAL")),
            source=str(data.get("source", "cctv_live")),
            source_type=str(data.get("source_type", "live_surveillance")),
            feature_version=int(data.get("feature_version", 1)),
            embedding_version=int(data.get("embedding_version", 1)),
            observation_date=obs_date,
            event_date=str(data.get("event_date", obs_date)),
            capture_timestamp=cap_ts,
            camera_id=str(data.get("camera_id") or data.get("source_camera_id") or ""),
            track_id=int(data.get("track_id", 0)),
            confidence=float(data.get("confidence", 1.0)),
            quality_score=float(data.get("quality_score", 1.0)),
            verification_state=str(data.get("verification_state", "PREDICTED")),
            operational_state=str(data.get("operational_state", data.get("status", "PREDICTED"))),
            training_state=str(data.get("training_state", "NOT_ELIGIBLE")),
            training_eligibility=str(data.get("training_eligibility", "NOT_ELIGIBLE")),
            dataset_split=str(data.get("dataset_split", "UNASSIGNED")),
            training_consumed=bool(data.get("training_consumed", False)),
            consumed_by_model_version=str(data.get("consumed_by_model_version", "")),
            consumed_in_training_job=str(data.get("consumed_in_training_job", "")),
            lineage_id=str(data.get("lineage_id", "")),
            parent_embedding_id=str(data.get("parent_embedding_id", "")),
            provenance=dict(data.get("provenance", {})),
            verification_metadata=dict(data.get("verification_metadata", {})),
            case_id=str(data.get("case_id", "")),
            created_by=str(data.get("created_by", "argus_system")),
            verification_status=str(data.get("verification_status", "ACTIVE")),
            status=str(data.get("status", "ACTIVE")),
            source_session_id=str(data.get("source_session_id", "")),
            source_camera_id=str(data.get("source_camera_id") or data.get("camera_id") or ""),
            candidate_training_job_id=str(data.get("candidate_training_job_id", "")),
            production_model_version=str(data.get("production_model_version", "")),
            created_at=c_at,
            updated_at=float(data.get("updated_at", c_at)),
        )


@dataclass
class PersistenceResult:
    success: bool
    embedding_id: str
    firebase_verified: bool = False
    error_message: str | None = None
    error_category: PersistenceErrorCategory = PersistenceErrorCategory.NONE
    retry_queued: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["error_category"] = self.error_category.value
        return d


def validate_service_account_file(file_path: Path | str | None) -> tuple[bool, str, dict[str, str]]:
    """Safely validate service account JSON file without exposing secret keys.

    Returns:
        (is_valid, reason, safe_metadata)
    """
    if not file_path:
        return False, "CREDENTIAL_PATH_MISSING", {}

    p = Path(file_path)
    if not p.exists():
        return False, f"FILE_NOT_FOUND: '{p.as_posix()}'", {}
    if not p.is_file():
        return False, f"NOT_A_FILE: '{p.as_posix()}'", {}

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        return False, f"JSON_PARSE_ERROR: {exc}", {}

    if not isinstance(data, dict):
        return False, "ROOT_NOT_A_JSON_OBJECT", {}

    sa_type = data.get("type")
    if sa_type != "service_account":
        return False, f"INVALID_TYPE: expected 'service_account', got '{sa_type}'", {}

    project_id = data.get("project_id")
    if not project_id or not isinstance(project_id, str):
        return False, "MISSING_OR_EMPTY_PROJECT_ID", {}

    private_key = data.get("private_key")
    if not private_key or not isinstance(private_key, str) or "-----BEGIN PRIVATE KEY-----" not in private_key:
        return False, "MISSING_OR_MALFORMED_PRIVATE_KEY", {}

    client_email = data.get("client_email")
    if not client_email or not isinstance(client_email, str) or "@" not in client_email:
        return False, "MISSING_OR_MALFORMED_CLIENT_EMAIL", {}

    # Public safe metadata only - NEVER contains private_key or access tokens
    safe_metadata = {
        "project_id": str(project_id),
        "client_email": str(client_email),
        "type": str(sa_type),
    }
    return True, "VALID", safe_metadata


class FirebaseEmbeddingStore:
    COLLECTION_NAME = "biometric_embeddings"
    PERSONS_COLLECTION = "biometric_persons"

    def __init__(
        self,
        mode: str = "auto",
        offline_store_path: str = "data/firebase_offline_store.json",
        max_retry_queue_size: int = 100,
        max_retries: int = 3,
    ) -> None:
        self._logger = get_logger("firebase_embedding_store")
        self.offline_store_path = Path(offline_store_path)
        self.offline_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

        self._firestore_client = None
        self._storage_bucket = None
        self._retry_queue: deque[dict[str, Any]] = deque(maxlen=max_retry_queue_size)
        self._lock = threading.RLock()
        self._offline_data: dict[str, Any] = {}

        self.credential_status: str = "MISSING"
        self.firestore_status: str = "UNINITIALIZED"
        self.storage_status: str = "UNINITIALIZED"
        self.project_id: str = "argus-17702"
        self.client_email: str | None = None
        self._resolved_cred_path: Path | None = None

        if mode == "auto":
            self.mode, self._resolved_cred_path = self._detect_mode()
        elif mode == "live":
            self.mode = "live"
            self._resolved_cred_path = self._resolve_credential_path()
        else:
            self.mode = "offline"
            self._resolved_cred_path = None

        if self.mode == "live":
            self._initialize_firebase()
        else:
            self._load_offline_store()
            self._logger.info(
                f"[FIREBASE_OFFLINE] Running in offline/mock mode (credential: {self.credential_status}). "
                "Local inference and offline embedding queue remain fully operational."
            )

    def _resolve_credential_path(self) -> Path | None:
        raw_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH") or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        if raw_path and raw_path.strip():
            return Path(raw_path.strip())
        default_path = Path("config/firebase-service-account.json")
        if default_path.exists() and default_path.is_file():
            return default_path
        return None

    def _detect_mode(self) -> tuple[str, Path | None]:
        cred_path = self._resolve_credential_path()
        if not cred_path:
            self.credential_status = "MISSING"
            return "offline", None

        is_valid, reason, meta = validate_service_account_file(cred_path)
        if is_valid:
            self.credential_status = "FOUND"
            self.project_id = meta.get("project_id", "argus-17702")
            self.client_email = meta.get("client_email")
            return "live", cred_path
        else:
            self.credential_status = "INVALID"
            self._logger.warning(
                f"[FIREBASE_CREDENTIAL_INVALID] Service account at '{cred_path.as_posix()}' is invalid: {reason}. "
                "Safely falling back to offline mode."
            )
            return "offline", None

    def _initialize_firebase(self) -> None:
        try:
            cred_path = self._resolved_cred_path
            if not cred_path:
                self.mode = "offline"
                self.credential_status = "MISSING"
                self.firestore_status = "UNINITIALIZED"
                self.storage_status = "UNINITIALIZED"
                self._load_offline_store()
                return

            is_valid, reason, meta = validate_service_account_file(cred_path)
            if not is_valid:
                self._logger.warning(
                    f"[FIREBASE_INIT_SKIPPED] Service account credential validation failed: {reason}. "
                    "Safely using offline mode."
                )
                self.mode = "offline"
                self.credential_status = "INVALID"
                self.firestore_status = "UNINITIALIZED"
                self.storage_status = "UNINITIALIZED"
                self._load_offline_store()
                return

            self.project_id = meta.get("project_id", "argus-17702")
            self.client_email = meta.get("client_email")
            self.credential_status = "FOUND"

            import firebase_admin
            from firebase_admin import credentials, firestore, storage

            if not firebase_admin._apps:
                cred = credentials.Certificate(str(cred_path))
                firebase_admin.initialize_app(
                    cred,
                    {
                        "projectId": self.project_id,
                        "storageBucket": "argus-17702.firebasestorage.app",
                    },
                )
                self._logger.info(
                    f"[FIREBASE_LIVE] Firebase Admin SDK initialized for project '{self.project_id}'."
                )

            self._firestore_client = firestore.client()
            self.firestore_status = "CONNECTED"

            try:
                self._storage_bucket = storage.bucket()
                self.storage_status = "CONNECTED"
            except Exception as s_err:  # noqa: BLE001
                self._storage_bucket = None
                self.storage_status = f"UNAVAILABLE ({s_err})"
                self._logger.warning(f"[FIREBASE_STORAGE_WARNING] Storage bucket unavailable: {s_err}")

            self._logger.info(
                f"[FIREBASE_STATUS] Mode: LIVE | Project: {self.project_id} | "
                f"Firestore: {self.firestore_status} | Storage: {self.storage_status}"
            )

        except Exception as err:  # noqa: BLE001
            self._logger.warning(
                f"[FIREBASE_INIT_FAILED] {err}. Safely falling back to offline mode. Local inference is unaffected."
            )
            self.mode = "offline"
            self.firestore_status = "FAILED"
            self.storage_status = "FAILED"
            self._load_offline_store()

    def check_connection_health(self) -> tuple[bool, dict[str, Any]]:
        if self.mode == "offline":
            with self._lock:
                return True, {
                    "mode": "offline",
                    "status": "HEALTHY",
                    "credential": self.credential_status,
                    "firestore": self.firestore_status,
                    "storage": self.storage_status,
                    "project_id": self.project_id,
                    "offline_store_exists": self.offline_store_path.exists(),
                    "total_embeddings": len(self._offline_data.get("embeddings", {})),
                    "total_persons": len(self._offline_data.get("persons", {})),
                    "retry_queue_size": len(self._retry_queue),
                }

        try:
            t0 = time.time()
            if self._firestore_client is None:
                return False, {
                    "mode": "live",
                    "status": "UNINITIALIZED",
                    "credential": self.credential_status,
                    "firestore": "UNINITIALIZED",
                    "storage": self.storage_status,
                    "project_id": self.project_id,
                    "error": "Firestore client is None",
                }
            # Lightweight health query
            _ = list(self._firestore_client.collection(self.COLLECTION_NAME).limit(1).stream())
            latency_ms = round((time.time() - t0) * 1000, 2)
            return True, {
                "mode": "live",
                "status": "HEALTHY",
                "credential": "FOUND",
                "firestore": "CONNECTED",
                "storage": self.storage_status,
                "project_id": self.project_id,
                "latency_ms": latency_ms,
                "retry_queue_size": len(self._retry_queue),
            }
        except Exception as err:  # noqa: BLE001
            return False, {
                "mode": "live",
                "status": "UNHEALTHY",
                "credential": self.credential_status,
                "firestore": "FAILED",
                "storage": self.storage_status,
                "project_id": self.project_id,
                "error": str(err),
            }

    def _load_offline_store(self) -> None:
        with self._lock:
            if self.offline_store_path.exists():
                try:
                    with open(self.offline_store_path, "r", encoding="utf-8") as f:
                        self._offline_data = json.load(f)
                except (OSError, json.JSONDecodeError, ValueError) as err:
                    self._logger.warning(f"Failed to load offline store: {err}")
                    self._offline_data = {"embeddings": {}, "persons": {}}
            else:
                self._offline_data = {"embeddings": {}, "persons": {}}

    def _save_offline_store(self) -> bool:
        with self._lock:
            tmp = self.offline_store_path.with_suffix(".tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._offline_data, f, indent=2)
                tmp.replace(self.offline_store_path)
                return True
            except (OSError, ValueError) as err:
                self._logger.error(f"Failed to save offline store: {err}")
                return False

    def persist_embedding(self, doc: FirebaseEmbeddingDocument) -> PersistenceResult:
        is_valid, validation_err = doc.validate_schema()
        if not is_valid:
            self._logger.warning(f"[PERSISTENCE_REJECTED] {doc.embedding_id}: {validation_err}")
            return PersistenceResult(
                success=False,
                embedding_id=doc.embedding_id,
                error_message=validation_err,
                error_category=PersistenceErrorCategory.SCHEMA_VALIDATION_ERROR,
            )

        if self.mode == "live":
            return self._persist_live(doc)
        return self._persist_offline(doc)

    def _persist_live(self, doc: FirebaseEmbeddingDocument) -> PersistenceResult:
        try:
            doc_ref = self._firestore_client.collection(self.COLLECTION_NAME).document(doc.embedding_id)
            doc_ref.set(doc.to_dict())

            self._logger.info(
                f"[FIREBASE_WRITE] Persisted embedding '{doc.embedding_id}' "
                f"({doc.modality}/{doc.embedding_dim}D) for person '{doc.person_id}'"
            )
            return PersistenceResult(
                success=True,
                embedding_id=doc.embedding_id,
                firebase_verified=False,
                error_category=PersistenceErrorCategory.NONE,
            )
        except Exception as err:  # noqa: BLE001
            err_str = str(err)
            category = PersistenceErrorCategory.UNKNOWN_ERROR
            if "deadline" in err_str.lower() or "timeout" in err_str.lower():
                category = PersistenceErrorCategory.NETWORK_TIMEOUT
            elif "permission" in err_str.lower() or "denied" in err_str.lower():
                category = PersistenceErrorCategory.PERMISSION_DENIED
            elif "unauthenticated" in err_str.lower() or "auth" in err_str.lower():
                category = PersistenceErrorCategory.AUTHENTICATION_FAILURE

            self._logger.warning(
                f"[FIREBASE_WRITE_FAILED] {doc.embedding_id} ({category.value}): {err}. Queuing for retry."
            )
            self._enqueue_retry(doc)
            return PersistenceResult(
                success=False,
                embedding_id=doc.embedding_id,
                error_message=err_str,
                error_category=category,
                retry_queued=True,
            )

    def _persist_offline(self, doc: FirebaseEmbeddingDocument) -> PersistenceResult:
        with self._lock:
            self._offline_data.setdefault("embeddings", {})[doc.embedding_id] = doc.to_dict()

            self._offline_data.setdefault("persons", {}).setdefault(doc.person_id, [])
            if doc.embedding_id not in self._offline_data["persons"][doc.person_id]:
                self._offline_data["persons"][doc.person_id].append(doc.embedding_id)
            saved = self._save_offline_store()

        return PersistenceResult(
            success=saved,
            embedding_id=doc.embedding_id,
            firebase_verified=saved,
            error_message=None if saved else "Offline store write failed",
            error_category=PersistenceErrorCategory.NONE if saved else PersistenceErrorCategory.UNKNOWN_ERROR,
        )

    def persist_embeddings(self, docs: list[FirebaseEmbeddingDocument]) -> list[PersistenceResult]:
        return [self.persist_embedding(doc) for doc in docs]

    def verify_persistence(self, embedding_id: str) -> tuple[bool, str]:
        if self.mode == "live":
            return self._verify_live(embedding_id)
        return self._verify_offline(embedding_id)

    def _verify_live(self, embedding_id: str) -> tuple[bool, str]:
        try:
            doc_ref = self._firestore_client.collection(self.COLLECTION_NAME).document(embedding_id)
            doc_snap = doc_ref.get()
            if not doc_snap.exists:
                return False, f"Document '{embedding_id}' not found in Firestore"
            data = doc_snap.to_dict()
            vec = data.get("vector", [])
            dim = data.get("embedding_dim", 0)
            if len(vec) != dim:
                return False, f"Vector length {len(vec)} != declared dim {dim}"
            vec_arr = np.asarray(vec, dtype=np.float32)
            if not np.isfinite(vec_arr).all():
                return False, "Stored vector contains non-finite values"
            return True, "Persistence verified"
        except Exception as err:  # noqa: BLE001
            return False, f"Verification failed: {err}"

    def _verify_offline(self, embedding_id: str) -> tuple[bool, str]:
        with self._lock:
            doc_data = self._offline_data.get("embeddings", {}).get(embedding_id)
        if doc_data is None:
            return False, f"Document '{embedding_id}' not found in offline store"
        vec = doc_data.get("vector", [])
        dim = doc_data.get("embedding_dim", 0)
        if len(vec) != dim:
            return False, f"Vector length {len(vec)} != declared dim {dim}"
        vec_arr = np.asarray(vec, dtype=np.float32)
        if not np.isfinite(vec_arr).all():
            return False, "Stored vector contains non-finite values"
        return True, "Persistence verified"

    def get_embeddings_by_person(
        self, person_id: str, modality: str | None = None
    ) -> list[FirebaseEmbeddingDocument]:
        if self.mode == "live":
            return self._query_live_by_person(person_id, modality)
        return self._query_offline_by_person(person_id, modality)

    def _query_live_by_person(
        self, person_id: str, modality: str | None = None
    ) -> list[FirebaseEmbeddingDocument]:
        try:
            query = self._firestore_client.collection(self.COLLECTION_NAME).where(
                "person_id", "==", person_id
            )
            if modality:
                query = query.where("modality", "==", modality)
            results = []
            for doc_snap in query.stream():
                results.append(FirebaseEmbeddingDocument.from_dict(doc_snap.to_dict()))
            return results
        except Exception as err:  # noqa: BLE001
            self._logger.warning(f"[FIREBASE_QUERY_FAILED] person={person_id}: {err}")
            return []

    def _query_offline_by_person(
        self, person_id: str, modality: str | None = None
    ) -> list[FirebaseEmbeddingDocument]:
        with self._lock:
            emb_ids = self._offline_data.get("persons", {}).get(person_id, [])
            results = []
            for eid in emb_ids:
                doc_data = self._offline_data.get("embeddings", {}).get(eid)
                if doc_data:
                    if modality and doc_data.get("modality") != modality:
                        continue
                    results.append(FirebaseEmbeddingDocument.from_dict(doc_data))
            return results

    def get_embeddings_by_date(
        self, observation_date: str, modality: str | None = None
    ) -> list[FirebaseEmbeddingDocument]:
        if self.mode == "live":
            return self._query_live_by_date(observation_date, modality)
        return self._query_offline_by_date(observation_date, modality)

    def _query_live_by_date(
        self, observation_date: str, modality: str | None = None
    ) -> list[FirebaseEmbeddingDocument]:
        try:
            query = self._firestore_client.collection(self.COLLECTION_NAME).where(
                "observation_date", "==", observation_date
            )
            if modality:
                query = query.where("modality", "==", modality)
            results = []
            for doc_snap in query.stream():
                results.append(FirebaseEmbeddingDocument.from_dict(doc_snap.to_dict()))
            return results
        except Exception as err:  # noqa: BLE001
            self._logger.warning(f"[FIREBASE_QUERY_FAILED] date={observation_date}: {err}")
            return []

    def _query_offline_by_date(
        self, observation_date: str, modality: str | None = None
    ) -> list[FirebaseEmbeddingDocument]:
        with self._lock:
            results = []
            for doc_data in self._offline_data.get("embeddings", {}).values():
                if doc_data.get("observation_date") != observation_date:
                    continue
                if modality and doc_data.get("modality") != modality:
                    continue
                results.append(FirebaseEmbeddingDocument.from_dict(doc_data))
            return results

    def get_all_embeddings(self) -> list[FirebaseEmbeddingDocument]:
        if self.mode == "live":
            try:
                results = []
                for doc_snap in self._firestore_client.collection(self.COLLECTION_NAME).stream():
                    results.append(FirebaseEmbeddingDocument.from_dict(doc_snap.to_dict()))
                return results
            except Exception as err:  # noqa: BLE001
                self._logger.warning(f"[FIREBASE_QUERY_FAILED] get_all: {err}")
                return []
        else:
            with self._lock:
                return [
                    FirebaseEmbeddingDocument.from_dict(d)
                    for d in self._offline_data.get("embeddings", {}).values()
                ]

    def _enqueue_retry(self, doc: FirebaseEmbeddingDocument) -> None:
        with self._lock:
            self._retry_queue.append(
                {"doc": doc.to_dict(), "retries": 0, "queued_at": time.time()}
            )

    def process_retry_queue(self) -> list[PersistenceResult]:
        results = []
        with self._lock:
            pending = list(self._retry_queue)
            self._retry_queue.clear()

        for item in pending:
            if item["retries"] >= self.max_retries:
                self._logger.warning(
                    f"[RETRY_EXHAUSTED] Embedding '{item['doc']['embedding_id']}' "
                    f"failed after {self.max_retries} retries."
                )
                results.append(
                    PersistenceResult(
                        success=False,
                        embedding_id=item["doc"]["embedding_id"],
                        error_message=f"Max retries ({self.max_retries}) exhausted",
                        error_category=PersistenceErrorCategory.RESOURCE_EXHAUSTED,
                    )
                )
                continue

            doc = FirebaseEmbeddingDocument.from_dict(item["doc"])
            result = self.persist_embedding(doc)
            if not result.success and result.retry_queued:
                with self._lock:
                    if self._retry_queue:
                        last = self._retry_queue[-1]
                        last["retries"] = item["retries"] + 1
            results.append(result)

        return results

    def get_retry_queue_size(self) -> int:
        with self._lock:
            return len(self._retry_queue)

    def delete_temporary_media(self, case_id: str) -> tuple[bool, str]:
        if self.mode != "live" or self._storage_bucket is None:
            return True, "No Firebase Storage cleanup needed (offline mode)"

        try:
            prefixes = [f"cases/{case_id}/images/", f"cases/{case_id}/videos/"]
            deleted_count = 0
            for prefix in prefixes:
                blobs = list(self._storage_bucket.list_blobs(prefix=prefix))
                for blob in blobs:
                    blob.delete()
                    deleted_count += 1

            self._logger.info(
                f"[FIREBASE_STORAGE_CLEANUP] Deleted {deleted_count} temporary media files for case '{case_id}'"
            )
            return True, f"Deleted {deleted_count} files"
        except Exception as err:  # noqa: BLE001
            self._logger.warning(f"[FIREBASE_STORAGE_CLEANUP_FAILED] case={case_id}: {err}")
            return False, str(err)

    def rebuild_local_from_firebase(self) -> dict[str, Any]:
        all_docs = self.get_all_embeddings()
        if not all_docs:
            return {}

        persons: dict[str, dict[str, Any]] = {}
        for doc in all_docs:
            if doc.person_id not in persons:
                persons[doc.person_id] = {
                    "gait_embeddings": [],
                    "appearance_embeddings": [],
                    "metadata": {"case_id": doc.case_id, "source": "firebase_recovery"},
                }

            emb_data = {
                "embedding_id": doc.embedding_id,
                "person_id": doc.person_id,
                "modality": doc.modality,
                "embedding_dim": doc.embedding_dim,
                "vector": doc.vector,
                "model_version": doc.model_version,
                "embedding_version": doc.embedding_version,
                "observation_date": doc.observation_date,
                "created_at": doc.created_at,
                "quality_score": doc.quality_score,
                "status": doc.status,
                "source_session_id": doc.source_session_id,
                "identity_type": doc.identity_type,
                "operational_state": doc.operational_state,
                "training_eligibility": doc.training_eligibility,
            }

            if doc.modality == "gait":
                persons[doc.person_id]["gait_embeddings"].append(emb_data)
            elif doc.modality == "appearance":
                persons[doc.person_id]["appearance_embeddings"].append(emb_data)

        self._logger.info(
            f"[FIREBASE_RECOVERY] Retrieved {len(all_docs)} embeddings for "
            f"{len(persons)} persons from Firebase."
        )
        return persons

    def get_persisted_embedding_ids(self) -> set[str]:
        if self.mode == "live":
            try:
                ids = set()
                for doc_snap in self._firestore_client.collection(self.COLLECTION_NAME).stream():
                    ids.add(doc_snap.id)
                return ids
            except Exception:  # noqa: BLE001
                return set()
        else:
            with self._lock:
                return set(self._offline_data.get("embeddings", {}).keys())

    def clear_offline_store(self) -> None:
        with self._lock:
            self._offline_data = {"embeddings": {}, "persons": {}}
            self._save_offline_store()
