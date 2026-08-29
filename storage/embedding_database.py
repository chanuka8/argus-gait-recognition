"""
Versioned Embedding Database for ARGUS AI.

Manages durable, structured, version-aware embedding records for enrolled persons.
Maintains individual embedding lineage, metadata, model version compatibility,
and provides bidirectional synchronization with VectorStore fast-path matrix files.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from monitoring.logging_config import get_logger
from storage.vector_store import VectorStore

# Lazy import to avoid circular dependency and allow offline-only operation
_FirebaseEmbeddingStore = None
_FirebaseEmbeddingDocument = None


def _get_firebase_classes():
    """Lazy-load Firebase store classes to avoid import-time dependency."""
    global _FirebaseEmbeddingStore, _FirebaseEmbeddingDocument
    if _FirebaseEmbeddingStore is None:
        from storage.firebase_embedding_store import (
            FirebaseEmbeddingDocument,
            FirebaseEmbeddingStore,
        )

        _FirebaseEmbeddingStore = FirebaseEmbeddingStore
        _FirebaseEmbeddingDocument = FirebaseEmbeddingDocument
    return _FirebaseEmbeddingStore, _FirebaseEmbeddingDocument


@dataclass
class EmbeddingRecord:
    """Represents an individual versioned biometric embedding vector and its metadata."""

    embedding_id: str
    person_id: str
    modality: str  # "gait" (256D) or "appearance" (512D)
    embedding_dim: int
    vector: list[float]  # L2-normalized float values
    model_version: str  # e.g., "v1.0.0"
    embedding_version: int = 1
    status: str = "ACTIVE"  # "ACTIVE", "DISABLED", "ARCHIVED"
    quality_score: float = 1.0
    source_session_id: str = ""
    created_at: float = field(default_factory=time.time)
    iso_created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    observation_date: str = ""

    def __post_init__(self) -> None:
        if not self.observation_date:
            self.observation_date = time.strftime("%Y-%m-%d", time.gmtime(self.created_at))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["observation_date"] = self.observation_date
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingRecord":
        c_at = float(data.get("created_at", time.time()))
        obs_date = str(data.get("observation_date", "")) or time.strftime("%Y-%m-%d", time.gmtime(c_at))
        return cls(
            embedding_id=str(data["embedding_id"]),
            person_id=str(data["person_id"]),
            modality=str(data.get("modality", "gait")),
            embedding_dim=int(data.get("embedding_dim", len(data.get("vector", [])))),
            vector=[float(v) for v in data.get("vector", [])],
            model_version=str(data.get("model_version", "v1.0.0")),
            embedding_version=int(data.get("embedding_version", 1)),
            status=str(data.get("status", "ACTIVE")),
            quality_score=float(data.get("quality_score", 1.0)),
            source_session_id=str(data.get("source_session_id", "")),
            created_at=c_at,
            iso_created_at=str(data.get("iso_created_at", "")),
            observation_date=obs_date,
        )


@dataclass
class PersonRecord:
    """Represents an enrolled person and all associated versioned embeddings."""

    person_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "ACTIVE"
    metadata: dict[str, Any] = field(default_factory=dict)
    gait_embeddings: list[EmbeddingRecord] = field(default_factory=list)
    appearance_embeddings: list[EmbeddingRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "metadata": self.metadata,
            "gait_embeddings": [e.to_dict() for e in self.gait_embeddings],
            "appearance_embeddings": [e.to_dict() for e in self.appearance_embeddings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonRecord":
        gait_list = [EmbeddingRecord.from_dict(e) for e in data.get("gait_embeddings", [])]
        app_list = [EmbeddingRecord.from_dict(e) for e in data.get("appearance_embeddings", [])]
        return cls(
            person_id=str(data["person_id"]),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            status=str(data.get("status", "ACTIVE")),
            metadata=dict(data.get("metadata", {})),
            gait_embeddings=gait_list,
            appearance_embeddings=app_list,
        )


class EmbeddingDatabase:
    """
    Thread-safe versioned embedding storage manager.

    Maintains JSON-serializable records for all enrolled identities with full
    lineage and model compatibility guarantees, and mirrors data into VectorStore
    files for high-speed inference.
    """

    def __init__(
        self,
        db_dir: str = "data/embedding_db",
        gait_gallery_dir: str = "models/live_gallery",
        appearance_gallery_dir: str = "models/appearance_gallery",
        firebase_store: Any | None = None,
    ) -> None:
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.persons_dir = self.db_dir / "persons"
        self.persons_dir.mkdir(parents=True, exist_ok=True)

        self.gait_store = VectorStore(gallery_dir=gait_gallery_dir)
        self.appearance_store = VectorStore(gallery_dir=appearance_gallery_dir)
        self.firebase_store = firebase_store
        self._logger = get_logger("embedding_database")

    def _person_file(self, person_id: str) -> Path:
        sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(person_id))
        return self.persons_dir / f"{sanitized}.json"

    def get_person(self, person_id: str) -> PersonRecord | None:
        p_file = self._person_file(person_id)
        if not p_file.exists():
            return None
        try:
            with open(p_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PersonRecord.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError) as err:
            self._logger.warning(f"Failed to read person record for {person_id}: {err}")
            return None

    def save_person(self, record: PersonRecord) -> bool:
        p_file = self._person_file(record.person_id)
        try:
            record.updated_at = time.time()
            data = record.to_dict()
            tmp_file = p_file.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp_file.replace(p_file)
            return True
        except (OSError, ValueError) as err:
            self._logger.error(f"Failed to persist person record for {record.person_id}: {err}")
            return False

    def add_embeddings(
        self,
        person_id: str,
        gait_embeddings: list[np.ndarray | list[float]] | None = None,
        appearance_embeddings: list[np.ndarray | list[float]] | None = None,
        model_version: str = "v1.0.0",
        source_session_id: str = "",
        quality_scores: list[float] | None = None,
        created_at: float | None = None,
        observation_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate, version, and persist embeddings for a subject into the database,
        then synchronize with the VectorStore inference galleries.

        Returns:
            Summary dict with persistence verification results.
        """
        person = self.get_person(person_id)
        if person is None:
            person = PersonRecord(person_id=person_id)

        added_gait = 0
        added_app = 0
        now = created_at if created_at is not None else time.time()
        obs_date = observation_date or time.strftime("%Y-%m-%d", time.gmtime(now))

        # 1. Process and validate Gait Embeddings (Expected: 256D)
        if gait_embeddings:
            for i, raw_vec in enumerate(gait_embeddings):
                vec = np.asarray(raw_vec, dtype=np.float32).ravel()
                if vec.size != 256:
                    raise ValueError(f"Gait embedding dimension mismatch: expected 256, got {vec.size}")
                if not np.isfinite(vec).all():
                    raise ValueError(f"Gait embedding at index {i} contains non-finite values (NaN/Inf)")
                norm = float(np.linalg.norm(vec))
                if norm == 0.0:
                    raise ValueError(f"Gait embedding at index {i} has zero norm")
                vec = (vec / norm).astype(np.float32)

                q_score = quality_scores[i] if quality_scores and i < len(quality_scores) else 1.0
                emb_id = f"gait_{person_id}_{int(now)}_{uuid.uuid4().hex[:6]}"
                rec = EmbeddingRecord(
                    embedding_id=emb_id,
                    person_id=person_id,
                    modality="gait",
                    embedding_dim=256,
                    vector=vec.tolist(),
                    model_version=model_version,
                    embedding_version=len(person.gait_embeddings) + 1,
                    quality_score=q_score,
                    source_session_id=source_session_id,
                    created_at=now,
                    observation_date=obs_date,
                )
                person.gait_embeddings.append(rec)
                added_gait += 1

        # 2. Process and validate Appearance Embeddings (Expected: 512D)
        if appearance_embeddings:
            for i, raw_vec in enumerate(appearance_embeddings):
                vec = np.asarray(raw_vec, dtype=np.float32).ravel()
                if vec.size != 512:
                    raise ValueError(f"Appearance embedding dimension mismatch: expected 512, got {vec.size}")
                if not np.isfinite(vec).all():
                    raise ValueError(f"Appearance embedding at index {i} contains non-finite values (NaN/Inf)")
                norm = float(np.linalg.norm(vec))
                if norm == 0.0:
                    raise ValueError(f"Appearance embedding at index {i} has zero norm")
                vec = (vec / norm).astype(np.float32)

                emb_id = f"app_{person_id}_{int(now)}_{uuid.uuid4().hex[:6]}"
                rec = EmbeddingRecord(
                    embedding_id=emb_id,
                    person_id=person_id,
                    modality="appearance",
                    embedding_dim=512,
                    vector=vec.tolist(),
                    model_version=model_version,
                    embedding_version=len(person.appearance_embeddings) + 1,
                    quality_score=1.0,
                    source_session_id=source_session_id,
                    created_at=now,
                    observation_date=obs_date,
                )
                person.appearance_embeddings.append(rec)
                added_app += 1

        # 3. Persist Person Record
        saved = self.save_person(person)
        if not saved:
            raise RuntimeError(f"Failed to write person database record for {person_id}")

        # 4. Synchronize with VectorStore Galleries
        self._sync_vector_stores()

        # 5. Verify Persistence Integrity
        verified, msg = self.verify_persistence(
            person_id=person_id,
            expected_gait_count=len(person.gait_embeddings),
            expected_app_count=len(person.appearance_embeddings),
        )

        if not verified:
            raise RuntimeError(f"Persistence verification failed for {person_id}: {msg}")

        # 6. Async Firebase Durable Persistence (non-blocking; failure never fails local op)
        firebase_results = []
        if self.firebase_store is not None:
            firebase_results = self._persist_to_firebase(
                person=person,
                model_version=model_version,
                source_session_id=source_session_id or "",
                observation_date=obs_date,
            )

        self._logger.info(
            f"Successfully persisted versioned embeddings for '{person_id}': "
            f"+{added_gait} gait (total {len(person.gait_embeddings)}), "
            f"+{added_app} appearance (total {len(person.appearance_embeddings)})"
        )

        return {
            "success": True,
            "person_id": person_id,
            "gait_embeddings_added": added_gait,
            "appearance_embeddings_added": added_app,
            "total_gait_embeddings": len(person.gait_embeddings),
            "total_appearance_embeddings": len(person.appearance_embeddings),
            "persistence_verified": True,
            "firebase_results": [r.to_dict() for r in firebase_results] if firebase_results else [],
        }

    def verify_persistence(
        self,
        person_id: str,
        expected_gait_count: int | None = None,
        expected_app_count: int | None = None,
    ) -> tuple[bool, str]:
        """
        Strictly verify that database records and VectorStore files are durably written
        and match expected counts before allowing raw media deletion.
        """
        person = self.get_person(person_id)
        if person is None:
            return False, f"Person record for '{person_id}' missing in database"

        if expected_gait_count is not None and len(person.gait_embeddings) < expected_gait_count:
            return (
                False,
                f"Gait embedding count mismatch: expected {expected_gait_count}, found {len(person.gait_embeddings)}",
            )

        if expected_app_count is not None and len(person.appearance_embeddings) < expected_app_count:
            return (
                False,
                f"Appearance embedding count mismatch: expected {expected_app_count}, found {len(person.appearance_embeddings)}",
            )

        # Verify VectorStore contains person_id
        if person.gait_embeddings:
            g_data = self.gait_store.load()
            if g_data is None:
                return False, "Gait VectorStore file missing or empty"
            _, g_labels, _ = g_data
            if person_id not in list(g_labels):
                return False, f"Person '{person_id}' missing in Gait VectorStore labels"

        if person.appearance_embeddings:
            a_data = self.appearance_store.load()
            if a_data is None:
                return False, "Appearance VectorStore file missing or empty"
            _, a_labels, _ = a_data
            if person_id not in list(a_labels):
                return False, f"Person '{person_id}' missing in Appearance VectorStore labels"

        return True, "Persistence verified"

    def _sync_vector_stores(self) -> None:
        """Re-index all active person records into fast-path VectorStore numpy files."""
        all_persons = self.list_all_persons()

        # 1. Sync Gait Gallery (256D)
        gait_feats = []
        gait_lbls = []
        gait_meta = {}

        for p in all_persons:
            if p.status != "ACTIVE":
                continue
            active_gait = [e for e in p.gait_embeddings if e.status == "ACTIVE"]
            for e in active_gait:
                gait_feats.append(e.vector)
                gait_lbls.append(p.person_id)
            if active_gait:
                gait_meta[p.person_id] = {
                    "embeddings": len(active_gait),
                    "status": "ACTIVE",
                    "enabled": True,
                    "updated_at": p.updated_at,
                }

        if gait_feats:
            self.gait_store.save(
                np.asarray(gait_feats, dtype=np.float32),
                np.asarray(gait_lbls, dtype=str),
                gait_meta,
            )

        # 2. Sync Appearance Gallery (512D)
        app_feats = []
        app_lbls = []
        app_meta = {}

        for p in all_persons:
            if p.status != "ACTIVE":
                continue
            active_app = [e for e in p.appearance_embeddings if e.status == "ACTIVE"]
            for e in active_app:
                app_feats.append(e.vector)
                app_lbls.append(p.person_id)
            if active_app:
                app_meta[p.person_id] = {
                    "embeddings": len(active_app),
                    "status": "ACTIVE",
                    "enabled": True,
                    "source": "PHOTO",
                    "updated_at": p.updated_at,
                }

        if app_feats:
            self.appearance_store.save(
                np.asarray(app_feats, dtype=np.float32),
                np.asarray(app_lbls, dtype=str),
                app_meta,
            )

    def list_all_persons(self) -> list[PersonRecord]:
        results = []
        for p_file in sorted(self.persons_dir.glob("*.json")):
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append(PersonRecord.from_dict(data))
            except (OSError, json.JSONDecodeError, ValueError) as err:
                self._logger.warning(f"Error loading {p_file.name}: {err}")
        return results

    def get_distinct_observation_dates(self) -> list[str]:
        """Return sorted unique observation dates (YYYY-MM-DD) across all enrolled embeddings."""
        dates = set()
        for p in self.list_all_persons():
            for e in p.gait_embeddings + p.appearance_embeddings:
                if e.status == "ACTIVE" and e.observation_date:
                    dates.add(e.observation_date)
        return sorted(dates)

    def get_embeddings_by_date(self, observation_date: str, modality: str | None = None) -> list[EmbeddingRecord]:
        """Return all active embeddings associated with a specific observation date."""
        matched = []
        for p in self.list_all_persons():
            embeddings = []
            if modality is None or modality == "gait":
                embeddings.extend(p.gait_embeddings)
            if modality is None or modality == "appearance":
                embeddings.extend(p.appearance_embeddings)

            for e in embeddings:
                if e.status == "ACTIVE" and e.observation_date == observation_date:
                    matched.append(e)
        return matched

    def check_model_compatibility(self, model_version: str, expected_dim: int, modality: str) -> bool:
        """
        Verify that a candidate model's output specification is compatible with the database.
        """
        if modality == "gait" and expected_dim != 256:
            return False
        return not (modality == "appearance" and expected_dim != 512)

    # ──────────────────────────────────────────────────────────────────────
    # FIREBASE INTEGRATION
    # ──────────────────────────────────────────────────────────────────────

    def _persist_to_firebase(
        self,
        person: "PersonRecord",
        model_version: str,
        source_session_id: str,
        observation_date: str,
    ) -> list:
        """
        Asynchronously persist embeddings to Firebase.
        Non-blocking: failures are logged and queued for retry,
        but NEVER fail the local persistence or inference.
        """
        if self.firebase_store is None:
            return []

        _, FBDoc = _get_firebase_classes()
        results = []
        try:
            for emb in person.gait_embeddings + person.appearance_embeddings:
                fb_doc = FBDoc(
                    embedding_id=emb.embedding_id,
                    person_id=emb.person_id,
                    modality=emb.modality,
                    embedding_dim=emb.embedding_dim,
                    vector=emb.vector,
                    model_version=model_version,
                    embedding_version=emb.embedding_version,
                    observation_date=observation_date or emb.observation_date,
                    created_at=emb.created_at,
                    quality_score=emb.quality_score,
                    status=emb.status,
                    source_session_id=source_session_id or emb.source_session_id,
                    production_model_version=model_version,
                )
                result = self.firebase_store.persist_embedding(fb_doc)
                results.append(result)
        except Exception as err:  # noqa: BLE001
            self._logger.warning(
                f"[FIREBASE_SYNC] Non-blocking Firebase persistence error for "
                f"'{person.person_id}': {err}. Local persistence is unaffected."
            )
        return results

    def sync_from_firebase(self, person_id: str) -> dict[str, Any]:
        """
        Pull embeddings for a person from Firebase and merge into local records.
        Used for reconciliation and recovery.
        """
        if self.firebase_store is None:
            return {"success": False, "error": "No Firebase store configured"}

        try:
            fb_docs = self.firebase_store.get_embeddings_by_person(person_id)
            if not fb_docs:
                return {"success": True, "synced": 0, "message": "No Firebase documents found"}

            gait_vecs = []
            app_vecs = []
            model_ver = "v1.0.0"
            obs_date = ""

            for fb_doc in fb_docs:
                model_ver = fb_doc.model_version
                obs_date = fb_doc.observation_date
                if fb_doc.modality == "gait" and len(fb_doc.vector) == 256:
                    gait_vecs.append(fb_doc.vector)
                elif fb_doc.modality == "appearance" and len(fb_doc.vector) == 512:
                    app_vecs.append(fb_doc.vector)

            result = self.add_embeddings(
                person_id=person_id,
                gait_embeddings=gait_vecs if gait_vecs else None,
                appearance_embeddings=app_vecs if app_vecs else None,
                model_version=model_ver,
                source_session_id="firebase_sync",
                observation_date=obs_date,
            )
            return {"success": True, "synced": len(fb_docs), **result}
        except Exception as err:  # noqa: BLE001
            self._logger.error(f"[FIREBASE_SYNC] Sync from Firebase failed for '{person_id}': {err}")
            return {"success": False, "error": str(err)}

    def rebuild_from_firebase(self) -> dict[str, Any]:
        """
        Disaster recovery: Rebuild entire local EmbeddingDatabase and VectorStore
        galleries from Firebase durable store.
        """
        if self.firebase_store is None:
            return {"success": False, "error": "No Firebase store configured"}

        try:
            persons_data = self.firebase_store.rebuild_local_from_firebase()
            if not persons_data:
                return {"success": True, "rebuilt_persons": 0, "message": "No data in Firebase"}

            rebuilt = 0
            for pid, pdata in persons_data.items():
                gait_vecs = [e["vector"] for e in pdata.get("gait_embeddings", [])]
                app_vecs = [e["vector"] for e in pdata.get("appearance_embeddings", [])]
                model_ver = "v1.0.0"
                if pdata.get("gait_embeddings"):
                    model_ver = pdata["gait_embeddings"][0].get("model_version", "v1.0.0")

                self.add_embeddings(
                    person_id=pid,
                    gait_embeddings=gait_vecs if gait_vecs else None,
                    appearance_embeddings=app_vecs if app_vecs else None,
                    model_version=model_ver,
                    source_session_id="firebase_disaster_recovery",
                )
                rebuilt += 1

            self._logger.info(f"[DISASTER_RECOVERY] Rebuilt {rebuilt} persons from Firebase.")
            return {"success": True, "rebuilt_persons": rebuilt}
        except Exception as err:  # noqa: BLE001
            self._logger.error(f"[DISASTER_RECOVERY] Rebuild from Firebase failed: {err}")
            return {"success": False, "error": str(err)}

    def get_firebase_sync_status(self) -> dict[str, Any]:
        """
        Report which local embeddings have confirmed Firebase persistence vs. pending.
        """
        if self.firebase_store is None:
            return {"firebase_configured": False}

        try:
            firebase_ids = self.firebase_store.get_persisted_embedding_ids()
            local_ids = set()
            for p in self.list_all_persons():
                for e in p.gait_embeddings + p.appearance_embeddings:
                    if e.status == "ACTIVE":
                        local_ids.add(e.embedding_id)

            synced = local_ids & firebase_ids
            pending = local_ids - firebase_ids
            return {
                "firebase_configured": True,
                "mode": getattr(self.firebase_store, "mode", "unknown"),
                "local_embeddings": len(local_ids),
                "firebase_confirmed": len(synced),
                "pending_sync": len(pending),
                "retry_queue_size": self.firebase_store.get_retry_queue_size(),
            }
        except Exception as err:  # noqa: BLE001
            return {"firebase_configured": True, "error": str(err)}
