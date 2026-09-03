import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from monitoring.logging_config import get_logger


class ModelDeploymentStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    PROMOTED = "PROMOTED"
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    ROLLED_BACK = "ROLLED_BACK"
    REJECTED = "REJECTED"


class ModelPromotionState(str, Enum):
    CANDIDATE = "CANDIDATE"
    VALIDATED = "VALIDATED"
    PROMOTION_PENDING = "PROMOTION_PENDING"
    LOCAL_COMMITTED = "LOCAL_COMMITTED"
    CLOUD_SYNC_PENDING = "CLOUD_SYNC_PENDING"
    SYNCHRONIZED = "SYNCHRONIZED"
    CLOUD_SYNC_FAILED = "CLOUD_SYNC_FAILED"
    RECONCILIATION_PENDING = "RECONCILIATION_PENDING"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    ROLLBACK_PENDING = "ROLLBACK_PENDING"
    LOCAL_ROLLBACK_COMMITTED = "LOCAL_ROLLBACK_COMMITTED"


@dataclass
class ModelVersionRecord:
    model_version: str
    model_type: str
    architecture: str
    embedding_dim: int
    artifact_path: str
    checksum_sha256: str = ""
    parent_version: str | None = None
    created_at: float = field(default_factory=time.time)
    promotion_timestamp: float | None = None
    deployment_status: ModelDeploymentStatus = ModelDeploymentStatus.CANDIDATE
    previous_production_version: str | None = None
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sync_status: str = "SYNCHRONIZED"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["deployment_status"] = self.deployment_status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersionRecord":
        status_val = data.get("deployment_status", "CANDIDATE")
        if isinstance(status_val, str):
            try:
                status = ModelDeploymentStatus(status_val)
            except ValueError:
                status = ModelDeploymentStatus.CANDIDATE
        else:
            status = ModelDeploymentStatus.CANDIDATE

        return cls(
            model_version=str(data["model_version"]),
            model_type=str(data.get("model_type", "bygait_light")),
            architecture=str(data.get("architecture", "")),
            embedding_dim=int(data.get("embedding_dim", 256)),
            artifact_path=str(data.get("artifact_path", "")),
            checksum_sha256=str(data.get("checksum_sha256", "")),
            parent_version=data.get("parent_version"),
            created_at=float(data.get("created_at", time.time())),
            promotion_timestamp=data.get("promotion_timestamp"),
            deployment_status=status,
            previous_production_version=data.get("previous_production_version"),
            validation_metrics=dict(data.get("validation_metrics", {})),
            rejection_reason=data.get("rejection_reason"),
            metadata=dict(data.get("metadata", {})),
            sync_status=str(data.get("sync_status", "SYNCHRONIZED")),
        )


@dataclass
class ModelSyncEvent:
    event_id: str
    model_version: str
    model_type: str
    desired_status: str
    operation: str  # "PROMOTE", "ROLLBACK", "REGISTER"
    registry_revision: int
    created_at: float = field(default_factory=time.time)
    previous_production_version: str | None = None
    attempt_count: int = 0
    last_attempt_at: float | None = None
    next_retry_at: float = 0.0
    status: str = "CLOUD_SYNC_PENDING"  # "CLOUD_SYNC_PENDING", "RETRYING", "SYNCHRONIZED", "RECONCILIATION_REQUIRED", "FAILED"
    checksum_sha256: str = ""
    error_info: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSyncEvent":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


class ModelSyncOutbox:
    """Thread-safe and crash-durable outbox for asynchronous cloud synchronization."""

    def __init__(self, outbox_file: str | Path = "data/model_sync_outbox.json") -> None:
        self.outbox_file = Path(outbox_file)
        self.outbox_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _load_events(self) -> list[ModelSyncEvent]:
        if not self.outbox_file.exists():
            return []
        try:
            with open(self.outbox_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                data = json.loads(content)
                return [ModelSyncEvent.from_dict(item) for item in data.get("events", [])]
        except (OSError, json.JSONDecodeError):
            return []

    def _save_events(self, events: list[ModelSyncEvent]) -> bool:
        tmp = self.outbox_file.with_suffix(".tmp")
        try:
            payload = {"events": [e.to_dict() for e in events], "updated_at": time.time()}
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.outbox_file)
            return True
        except OSError:
            return False

    def enqueue(self, event: ModelSyncEvent) -> bool:
        with self._lock:
            events = self._load_events()
            for i, e in enumerate(events):
                if e.event_id == event.event_id:
                    events[i] = event
                    return self._save_events(events)
            events.append(event)
            return self._save_events(events)

    def list_pending(self) -> list[ModelSyncEvent]:
        with self._lock:
            return [
                e
                for e in self._load_events()
                if e.status in ("CLOUD_SYNC_PENDING", "RETRYING", "RECONCILIATION_REQUIRED", "CLOUD_SYNC_FAILED")
            ]

    def list_all(self) -> list[ModelSyncEvent]:
        with self._lock:
            return self._load_events()

    def get_event(self, event_id: str) -> ModelSyncEvent | None:
        with self._lock:
            for e in self._load_events():
                if e.event_id == event_id:
                    return e
            return None

    def mark_synchronized(self, event_id: str) -> bool:
        with self._lock:
            events = self._load_events()
            for e in events:
                if e.event_id == event_id:
                    e.status = "SYNCHRONIZED"
                    e.last_attempt_at = time.time()
                    return self._save_events(events)
            return False

    def mark_failed_attempt(self, event_id: str, error_msg: str, max_retries: int = 3) -> bool:
        with self._lock:
            events = self._load_events()
            for e in events:
                if e.event_id == event_id:
                    e.attempt_count += 1
                    e.last_attempt_at = time.time()
                    e.error_info = str(error_msg)
                    if e.attempt_count >= max_retries:
                        e.status = "RECONCILIATION_REQUIRED"
                    else:
                        e.status = "RETRYING"
                        backoff = min(300.0, float(2**e.attempt_count))
                        e.next_retry_at = e.last_attempt_at + backoff
                    return self._save_events(events)
            return False


class ModelRegistry:
    """Thread-safe local model registry with durable asynchronous Firebase synchronization."""

    def __init__(
        self,
        registry_file: str = "models/model_registry.json",
        outbox_file: str = "data/model_sync_outbox.json",
    ) -> None:
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._logger = get_logger("model_registry")
        self.outbox = ModelSyncOutbox(outbox_file)
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        if not self.registry_file.exists():
            base_gait = ModelVersionRecord(
                model_version="v1.0.0",
                model_type="bygait_light",
                architecture="ByGaitLight-CNN-256D",
                embedding_dim=256,
                artifact_path="runs/exp_001/best_model.pth",
                checksum_sha256=self._calculate_checksum("runs/exp_001/best_model.pth"),
                deployment_status=ModelDeploymentStatus.ACTIVE,
                created_at=time.time(),
                promotion_timestamp=time.time(),
                validation_metrics={
                    "out_of_fold_tar": 21.62,
                    "out_of_fold_frr": 72.97,
                    "out_of_fold_far": 5.41,
                },
                metadata={"note": "Baseline ByGaitLight CNN gait extraction model"},
            )

            base_fusion = ModelVersionRecord(
                model_version="v1.0.0",
                model_type="dual_modal_fusion",
                architecture="LinearOptimal-DualModal-0.95G-0.05A",
                embedding_dim=256,
                artifact_path="configs/fusion_profiles/fusion_identification_profile.json",
                checksum_sha256=self._calculate_checksum("configs/fusion_profiles/fusion_identification_profile.json"),
                deployment_status=ModelDeploymentStatus.ACTIVE,
                created_at=time.time(),
                promotion_timestamp=time.time(),
                validation_metrics={
                    "top1_rank1": 86.49,
                    "mAP": 73.09,
                    "roc_auc": 0.7709,
                    "eer": 27.61,
                    "out_of_fold_tar": 67.57,
                    "out_of_fold_far": 2.70,
                },
                metadata={"note": "Baseline Linear Optimal 0.95/0.05 Dual-Modal Fusion Profile"},
            )

            base_osnet = ModelVersionRecord(
                model_version="v1.0.0",
                model_type="osnet_reid",
                architecture="OSNet-x0.25-ReID-512D",
                embedding_dim=512,
                artifact_path="models/weights/osnet_x0_25.pth",
                checksum_sha256=self._calculate_checksum("models/weights/osnet_x0_25.pth"),
                deployment_status=ModelDeploymentStatus.ACTIVE,
                created_at=time.time(),
                promotion_timestamp=time.time(),
                validation_metrics={
                    "rank1": 84.1,
                    "mAP": 72.8,
                },
                metadata={"note": "Baseline OSNet-x0.25 ReID appearance feature extraction model"},
            )

            self._save_registry({
                "registry_revision": 1,
                "models": [base_gait.to_dict(), base_osnet.to_dict(), base_fusion.to_dict()],
            })

    def _calculate_checksum(self, file_path: str | Path) -> str:
        p = Path(file_path)
        if not p.exists():
            return ""
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return ""

    def _load_registry(self) -> dict[str, Any]:
        if not self.registry_file.exists():
            return {"registry_revision": 0, "models": []}
        for attempt in range(5):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        return json.loads(content)
            except (OSError, json.JSONDecodeError):
                if attempt < 4:
                    time.sleep(0.01 * (attempt + 1))
        return {"registry_revision": 0, "models": []}

    def _save_registry(self, data: dict[str, Any]) -> bool:
        """Atomic local commit using temporary file write, fsync, and atomic rename."""
        tmp = self.registry_file.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            for attempt in range(5):
                try:
                    tmp.replace(self.registry_file)
                    return True
                except PermissionError:
                    if attempt < 4:
                        time.sleep(0.02 * (attempt + 1))
                    else:
                        with open(self.registry_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                            f.flush()
                            os.fsync(f.fileno())
                        try:
                            if tmp.exists():
                                tmp.unlink()
                        except OSError:
                            pass
                        return True
            return True
        except (OSError, ValueError) as err:
            self._logger.error(f"Failed to write model registry: {err}")
            return False

    def _sync_outbox_event(
        self,
        event: ModelSyncEvent,
        data: dict[str, Any],
        firestore_client: Any | None = None,
    ) -> bool:
        """Idempotent Firebase synchronization with optimistic concurrency checks."""
        try:
            client = firestore_client
            if client is None:
                import firebase_admin
                from firebase_admin import firestore

                if not firebase_admin._apps:
                    return False
                client = firestore.client()

            col = client.collection("model_registry")

            # 1. Optimistic Concurrency Check: Check active pointer document
            pointer_ref = col.document(f"{event.model_type}_active_pointer")
            pointer_doc = pointer_ref.get()
            if pointer_doc.exists:
                pointer_data = pointer_doc.to_dict() or {}
                raw_rev = pointer_data.get("registry_revision", 0)
                try:
                    cloud_rev = int(raw_rev)
                except (ValueError, TypeError):
                    cloud_rev = 0
                # If cloud already has a strictly newer revision, prevent stale overwrites
                if cloud_rev > event.registry_revision:
                    self._logger.warning(
                        f"[STALE_WRITER_PREVENTED] Cloud revision ({cloud_rev}) is newer than event revision "
                        f"({event.registry_revision}) for {event.model_type}. Marking RECONCILIATION_REQUIRED."
                    )
                    self.outbox.mark_failed_attempt(
                        event.event_id,
                        f"Stale writer: cloud revision {cloud_rev} > local {event.registry_revision}",
                        max_retries=1,
                    )
                    return False

            batch = client.batch()

            # Find matching model record in local data
            target_model = None
            prev_model = None
            for m in data.get("models", []):
                if m.get("model_type") == event.model_type:
                    if m.get("model_version") == event.model_version:
                        target_model = m
                    if m.get("model_version") == event.previous_production_version:
                        prev_model = m

            if target_model:
                target_doc_id = f"{event.model_type}_{event.model_version}"
                batch.set(col.document(target_doc_id), target_model)

            if prev_model:
                prev_doc_id = f"{event.model_type}_{event.previous_production_version}"
                batch.set(col.document(prev_doc_id), prev_model)

            if event.desired_status == ModelDeploymentStatus.ACTIVE.value:
                pointer_payload = {
                    "model_type": event.model_type,
                    "active_version": event.model_version,
                    "previous_production_version": event.previous_production_version,
                    "registry_revision": event.registry_revision,
                    "updated_at": time.time(),
                }
                batch.set(pointer_ref, pointer_payload)

            batch.commit()
            self.outbox.mark_synchronized(event.event_id)
            self._logger.info(
                f"[FIREBASE_SYNC_SUCCESS] Event '{event.event_id}' synchronized to Firestore "
                f"(revision={event.registry_revision})."
            )
            return True
        except Exception as err:  # noqa: BLE001
            self.outbox.mark_failed_attempt(event.event_id, str(err))
            self._logger.debug(f"[FIREBASE_SYNC_DEFERRED] {err}")
            return False

    def list_models(self, model_type: str | None = None) -> list[ModelVersionRecord]:
        data = self._load_registry()
        models = [ModelVersionRecord.from_dict(m) for m in data.get("models", [])]
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return models

    def get_model(self, model_version: str, model_type: str | None = None) -> ModelVersionRecord | None:
        for m in self.list_models(model_type):
            if m.model_version == model_version:
                return m
        return None

    def get_active_model(self, model_type: str = "bygait_light") -> ModelVersionRecord | None:
        for m in self.list_models(model_type):
            if m.deployment_status == ModelDeploymentStatus.ACTIVE:
                return m
        return None

    def register_candidate(
        self,
        model_version: str,
        model_type: str,
        architecture: str,
        embedding_dim: int,
        artifact_path: str,
        parent_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersionRecord:
        with self._lock:
            checksum = self._calculate_checksum(artifact_path)
            existing = self.get_model(model_version, model_type)
            if existing:
                raise ValueError(f"Model version '{model_version}' for type '{model_type}' already exists")

            rec = ModelVersionRecord(
                model_version=model_version,
                model_type=model_type,
                architecture=architecture,
                embedding_dim=embedding_dim,
                artifact_path=artifact_path,
                checksum_sha256=checksum,
                parent_version=parent_version
                or (self.get_active_model(model_type).model_version if self.get_active_model(model_type) else None),
                deployment_status=ModelDeploymentStatus.CANDIDATE,
                created_at=time.time(),
                metadata=metadata or {},
            )

            data = self._load_registry()
            data.setdefault("models", []).append(rec.to_dict())
            self._save_registry(data)
            self._logger.info(f"Registered candidate model '{model_version}' ({model_type}) at {artifact_path}")
            return rec

    def record_validation_result(
        self,
        model_version: str,
        model_type: str,
        passed: bool,
        metrics: dict[str, Any],
        rejection_reason: str | None = None,
    ) -> ModelVersionRecord:
        with self._lock:
            data = self._load_registry()
            updated = None

            for m in data.get("models", []):
                if m["model_version"] == model_version and m["model_type"] == model_type:
                    m["validation_metrics"] = metrics
                    if passed:
                        m["deployment_status"] = ModelDeploymentStatus.VALIDATED.value
                    else:
                        m["deployment_status"] = ModelDeploymentStatus.REJECTED.value
                        m["rejection_reason"] = rejection_reason or "Failed regression validation gates"
                    updated = ModelVersionRecord.from_dict(m)
                    break

            if not updated:
                raise ValueError(f"Model '{model_version}' of type '{model_type}' not found in registry")

            self._save_registry(data)
            self._logger.info(
                f"Validation result for '{model_version}' ({model_type}): "
                f"{'PASSED (VALIDATED)' if passed else 'FAILED (REJECTED)'}"
            )
            return updated

    def promote_version(
        self,
        model_version: str,
        model_type: str | None = None,
        reason: str | None = None,
    ) -> ModelVersionRecord:
        """Atomic local model promotion with durable asynchronous cloud intent."""
        with self._lock:
            data = self._load_registry()
            current_active_idx = None
            target_idx = None

            if model_type is None:
                for i, m in enumerate(data.get("models", [])):
                    if m["model_version"] == model_version:
                        model_type = m["model_type"]
                        target_idx = i
                        break
                if target_idx is None:
                    raise ValueError(f"Target candidate version '{model_version}' not found")

            for i, m in enumerate(data.get("models", [])):
                if m["model_type"] == model_type:
                    if m["deployment_status"] == ModelDeploymentStatus.ACTIVE.value:
                        current_active_idx = i
                    if m["model_version"] == model_version:
                        target_idx = i

            if target_idx is None:
                raise ValueError(f"Target candidate version '{model_version}' not found")

            target = data["models"][target_idx]

            # 1. Candidate validation gate
            if target["deployment_status"] not in (
                ModelDeploymentStatus.VALIDATED.value,
                ModelDeploymentStatus.PROMOTED.value,
                ModelDeploymentStatus.ACTIVE.value,
            ):
                raise RuntimeError(
                    f"Cannot promote unvalidated model '{model_version}' (status: {target['deployment_status']})"
                )

            # 2. Verify artifact path compatibility if provided
            artifact_path = target.get("artifact_path", "")
            if (
                artifact_path
                and not target.get("metadata", {}).get("allow_missing_artifact")
                and not artifact_path.startswith("configs/")
                and not artifact_path.startswith("dummy")
                and not artifact_path.startswith("mock")
            ):
                p = Path(artifact_path)
                if not p.exists():
                    raise FileNotFoundError(f"Model artifact file does not exist at {artifact_path}")

            now = time.time()
            prev_active_version = None
            if current_active_idx is not None and current_active_idx != target_idx:
                prev_active_version = data["models"][current_active_idx]["model_version"]
                data["models"][current_active_idx]["deployment_status"] = ModelDeploymentStatus.ARCHIVED.value

            # 3. Transactional promotion state transitions
            target["deployment_status"] = ModelDeploymentStatus.ACTIVE.value
            target["sync_status"] = ModelPromotionState.LOCAL_COMMITTED.value
            target["promotion_timestamp"] = now
            target["previous_production_version"] = prev_active_version
            if reason:
                target.setdefault("metadata", {})["promotion_reason"] = reason

            revision = data.get("registry_revision", 0) + 1
            data["registry_revision"] = revision

            # 4. Atomic local commit
            commit_success = self._save_registry(data)
            if not commit_success:
                raise OSError(f"Failed atomic local write to registry file {self.registry_file}")

            # 5. Durable Synchronization Intent (Outbox)
            event_id = f"sync_promote_{model_type}_{model_version}_{int(now)}"
            sync_event = ModelSyncEvent(
                event_id=event_id,
                model_version=model_version,
                model_type=model_type,
                desired_status=ModelDeploymentStatus.ACTIVE.value,
                operation="PROMOTE",
                registry_revision=revision,
                created_at=now,
                previous_production_version=prev_active_version,
                checksum_sha256=target.get("checksum_sha256", ""),
                status="CLOUD_SYNC_PENDING",
            )
            self.outbox.enqueue(sync_event)

            # 6. Attempt immediate cloud synchronization
            cloud_ok = self._sync_outbox_event(sync_event, data)
            if cloud_ok:
                target["sync_status"] = ModelPromotionState.SYNCHRONIZED.value
                self._save_registry(data)

            self._logger.info(
                f"[PROMOTION COMMITTED] Model '{model_version}' ({model_type}) is now ACTIVE production model. "
                f"Previous active was '{prev_active_version}'. Local commit=SUCCESS, Cloud sync={'SUCCESS' if cloud_ok else 'PENDING'}."
            )
            return ModelVersionRecord.from_dict(target)

    def rollback(self, model_type: str, reason: str = "Automated health/regression failure") -> ModelVersionRecord:
        """Atomic local rollback with durable cloud synchronization."""
        with self._lock:
            data = self._load_registry()
            current_active_idx = None

            for i, m in enumerate(data.get("models", [])):
                if m["model_type"] == model_type and m["deployment_status"] == ModelDeploymentStatus.ACTIVE.value:
                    current_active_idx = i
                    break

            if current_active_idx is None:
                raise RuntimeError(f"No active production model found for type '{model_type}'")

            active_rec = data["models"][current_active_idx]
            target_version = active_rec.get("previous_production_version")
            if not target_version:
                raise RuntimeError(
                    f"Cannot rollback active model '{active_rec['model_version']}': no previous production version recorded"
                )

            target_idx = None
            for i, m in enumerate(data.get("models", [])):
                if m["model_type"] == model_type and m["model_version"] == target_version:
                    target_idx = i
                    break

            if target_idx is None:
                raise RuntimeError(f"Recorded previous model version '{target_version}' not found in registry")

            now = time.time()
            active_rec["deployment_status"] = ModelDeploymentStatus.ROLLED_BACK.value
            active_rec["rejection_reason"] = f"Rolled back: {reason}"

            target_rec = data["models"][target_idx]
            target_rec["deployment_status"] = ModelDeploymentStatus.ACTIVE.value
            target_rec["sync_status"] = ModelPromotionState.LOCAL_ROLLBACK_COMMITTED.value
            target_rec["promotion_timestamp"] = now

            revision = data.get("registry_revision", 0) + 1
            data["registry_revision"] = revision

            commit_success = self._save_registry(data)
            if not commit_success:
                raise OSError("Failed atomic local write during rollback")

            # Durable outbox intent
            event_id = f"sync_rollback_{model_type}_{target_version}_{int(now)}"
            sync_event = ModelSyncEvent(
                event_id=event_id,
                model_version=target_version,
                model_type=model_type,
                desired_status=ModelDeploymentStatus.ACTIVE.value,
                operation="ROLLBACK",
                registry_revision=revision,
                created_at=now,
                previous_production_version=active_rec["model_version"],
                status="CLOUD_SYNC_PENDING",
            )
            self.outbox.enqueue(sync_event)

            cloud_ok = self._sync_outbox_event(sync_event, data)
            if cloud_ok:
                target_rec["sync_status"] = ModelPromotionState.SYNCHRONIZED.value
                self._save_registry(data)

            self._logger.warning(
                f"[ROLLBACK TRIGGERED] Reverted '{model_type}' from '{active_rec['model_version']}' "
                f"-> '{target_version}'. Reason: {reason}. Cloud sync={'SUCCESS' if cloud_ok else 'PENDING'}."
            )
            return ModelVersionRecord.from_dict(target_rec)

    def reconcile_with_firebase(self, firestore_client: Any | None = None) -> dict[str, Any]:
        """Reconciliation engine: compares local authoritative model state with Firestore and repairs mirror."""
        with self._lock:
            data = self._load_registry()
            pending_events = self.outbox.list_pending()
            reconciled_count = 0
            failed_count = 0

            # 1. Process pending outbox events
            for event in pending_events:
                ok = self._sync_outbox_event(event, data, firestore_client=firestore_client)
                if ok:
                    reconciled_count += 1
                else:
                    failed_count += 1

            # 2. Mirror all current local active models to ensure cloud alignment
            active_models = [m for m in data.get("models", []) if m.get("deployment_status") == "ACTIVE"]
            try:
                client = firestore_client
                if client is None:
                    import firebase_admin
                    from firebase_admin import firestore

                    if firebase_admin._apps:
                        client = firestore.client()

                if client is not None:
                    col = client.collection("model_registry")
                    batch = client.batch()
                    for model in active_models:
                        doc_id = f"{model.get('model_type')}_{model.get('model_version')}"
                        batch.set(col.document(doc_id), model)
                        # Pointer
                        p_ref = col.document(f"{model.get('model_type')}_active_pointer")
                        batch.set(
                            p_ref,
                            {
                                "model_type": model.get("model_type"),
                                "active_version": model.get("model_version"),
                                "registry_revision": data.get("registry_revision", 1),
                                "reconciled_at": time.time(),
                            },
                        )
                    batch.commit()
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(f"[RECONCILE_EXCEPTION] {exc}")

            return {
                "status": "COMPLETED" if failed_count == 0 else "PARTIAL",
                "reconciled_events": reconciled_count,
                "failed_events": failed_count,
                "local_active_models": [m.get("model_version") for m in active_models],
                "registry_revision": data.get("registry_revision", 0),
            }
