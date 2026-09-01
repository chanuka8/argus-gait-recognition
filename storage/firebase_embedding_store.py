import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger


@dataclass
class FirebaseEmbeddingDocument:
    embedding_id: str
    person_id: str
    modality: str
    embedding_dim: int
    vector: list[float]
    model_version: str
    embedding_version: int = 1
    observation_date: str = ""
    created_at: float = field(default_factory=time.time)
    quality_score: float = 1.0
    verification_status: str = "ACTIVE"
    training_eligibility: str = "NOT_ELIGIBLE"
    source_session_id: str = ""
    source_camera_id: str = ""
    candidate_training_job_id: str = ""
    production_model_version: str = ""
    status: str = "ACTIVE"
    case_id: str = ""
    created_by: str = "argus_enrollment"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["observation_date"] = self.observation_date
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FirebaseEmbeddingDocument":
        return cls(
            embedding_id=str(data.get("embedding_id", "")),
            person_id=str(data.get("person_id", "")),
            modality=str(data.get("modality", "gait")),
            embedding_dim=int(data.get("embedding_dim", 0)),
            vector=[float(v) for v in data.get("vector", [])],
            model_version=str(data.get("model_version", "v1.0.0")),
            embedding_version=int(data.get("embedding_version", 1)),
            observation_date=str(data.get("observation_date", "")),
            created_at=float(data.get("created_at", time.time())),
            quality_score=float(data.get("quality_score", 1.0)),
            verification_status=str(data.get("verification_status", "ACTIVE")),
            training_eligibility=str(data.get("training_eligibility", "NOT_ELIGIBLE")),
            source_session_id=str(data.get("source_session_id", "")),
            source_camera_id=str(data.get("source_camera_id", "")),
            candidate_training_job_id=str(data.get("candidate_training_job_id", "")),
            production_model_version=str(data.get("production_model_version", "")),
            status=str(data.get("status", "ACTIVE")),
            case_id=str(data.get("case_id", "")),
            created_by=str(data.get("created_by", "argus_enrollment")),
        )


@dataclass
class PersistenceResult:
    success: bool
    embedding_id: str
    firebase_verified: bool = False
    error_message: str | None = None
    retry_queued: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


        if mode == "auto":
            self.mode = self._detect_mode()
        else:
            self.mode = mode

        if self.mode == "live":
            self._initialize_firebase()
        else:
            self._load_offline_store()
            self._logger.info(
                "[FIREBASE_OFFLINE] Running in offline/mock mode. "
                "Set FIREBASE_SERVICE_ACCOUNT_PATH or GOOGLE_APPLICATION_CREDENTIALS for live mode."
            )

    def _detect_mode(self) -> str:
        cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH") or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        if cred_path and Path(cred_path).exists():
            return "live"
        return "offline"

    def _initialize_firebase(self) -> None:
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, storage

            cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH") or os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )
            if not cred_path or not Path(cred_path).exists():
                self._logger.warning(
                    "[FIREBASE_INIT] Credential file not found. Falling back to offline mode."
                )
                self.mode = "offline"
                self._load_offline_store()
                return

            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(
                    cred,
                    {"storageBucket": "argus-17702.firebasestorage.app"},
                )

            self._firestore_client = firestore.client()
            self._storage_bucket = storage.bucket()
            self._logger.info("[FIREBASE_LIVE] Firebase Admin SDK initialized successfully.")

        except Exception as err:  # noqa: BLE001
            self._logger.warning(f"[FIREBASE_INIT_FAILED] {err}. Falling back to offline mode.")
            self.mode = "offline"
            self._load_offline_store()

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
        if doc.modality == "gait" and doc.embedding_dim != 256:
            return PersistenceResult(
                success=False,
                embedding_id=doc.embedding_id,
                error_message=f"Gait embedding dimension mismatch: expected 256, got {doc.embedding_dim}",
            )
        if doc.modality == "appearance" and doc.embedding_dim != 512:
            return PersistenceResult(
                success=False,
                embedding_id=doc.embedding_id,
                error_message=f"Appearance embedding dimension mismatch: expected 512, got {doc.embedding_dim}",
            )


        if len(doc.vector) != doc.embedding_dim:
            return PersistenceResult(
                success=False,
                embedding_id=doc.embedding_id,
                error_message=f"Vector length {len(doc.vector)} != declared dim {doc.embedding_dim}",
            )


        vec_arr = np.asarray(doc.vector, dtype=np.float32)
        if not np.isfinite(vec_arr).all():
            return PersistenceResult(
                success=False,
                embedding_id=doc.embedding_id,
                error_message="Vector contains non-finite values (NaN/Inf)",
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
            )
        except Exception as err:  # noqa: BLE001
            self._logger.warning(
                f"[FIREBASE_WRITE_FAILED] {doc.embedding_id}: {err}. Queuing for retry."
            )
            self._enqueue_retry(doc)
            return PersistenceResult(
                success=False,
                embedding_id=doc.embedding_id,
                error_message=str(err),
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
