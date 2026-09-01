import hashlib
import json
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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["deployment_status"] = self.deployment_status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersionRecord":
        status_val = data.get("deployment_status", "CANDIDATE")
        if isinstance(status_val, str):
            status = ModelDeploymentStatus(status_val)
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
        )


class ModelRegistry:
    def __init__(self, registry_file: str = "models/model_registry.json") -> None:
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("model_registry")
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
                "models": [base_gait.to_dict(), base_osnet.to_dict(), base_fusion.to_dict()]
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
            return {"models": []}
        for attempt in range(5):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        return json.loads(content)
            except (OSError, json.JSONDecodeError):
                if attempt < 4:
                    time.sleep(0.01 * (attempt + 1))
        return {"models": []}

    def _save_registry(self, data: dict[str, Any]) -> bool:
        tmp = self.registry_file.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
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

    def get_active_model(self, model_type: str) -> ModelVersionRecord | None:
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
        if target["deployment_status"] not in (
            ModelDeploymentStatus.VALIDATED.value,
            ModelDeploymentStatus.PROMOTED.value,
            ModelDeploymentStatus.ACTIVE.value,
        ):
            raise RuntimeError(
                f"Cannot promote unvalidated model '{model_version}' (status: {target['deployment_status']})"
            )

        now = time.time()
        prev_active_version = None
        if current_active_idx is not None and current_active_idx != target_idx:
            prev_active_version = data["models"][current_active_idx]["model_version"]
            data["models"][current_active_idx]["deployment_status"] = ModelDeploymentStatus.ARCHIVED.value

        target["deployment_status"] = ModelDeploymentStatus.ACTIVE.value
        target["promotion_timestamp"] = now
        target["previous_production_version"] = prev_active_version
        if reason:
            target.setdefault("metadata", {})["promotion_reason"] = reason

        self._save_registry(data)
        reason_str = f" Reason: '{reason}'." if reason else ""
        self._logger.info(
            f"[PROMOTION SUCCESS] Model '{model_version}' ({model_type}) is now ACTIVE production model. "
            f"Previous active was '{prev_active_version}'.{reason_str}"
        )
        return ModelVersionRecord.from_dict(target)

    def rollback(self, model_type: str, reason: str = "Automated health/regression failure") -> ModelVersionRecord:
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


        active_rec["deployment_status"] = ModelDeploymentStatus.ROLLED_BACK.value
        active_rec["rejection_reason"] = f"Rolled back: {reason}"


        target_rec = data["models"][target_idx]
        target_rec["deployment_status"] = ModelDeploymentStatus.ACTIVE.value
        target_rec["promotion_timestamp"] = time.time()

        self._save_registry(data)
        self._logger.warning(
            f"[ROLLBACK TRIGGERED] Reverted '{model_type}' from '{active_rec['model_version']}' "
            f"-> '{target_version}'. Reason: {reason}"
        )
        return ModelVersionRecord.from_dict(target_rec)
