import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from monitoring.logging_config import get_logger


@dataclass
class ContinualLearningEvent:
    event_id: str
    event_type: str
    timestamp: float
    trigger_date: str
    model_type: str
    dataset_id: str
    baseline_version: str
    candidate_version: str
    baseline_sha256: str = ""
    candidate_sha256: str = ""
    parameters_changed: int = 0
    total_parameters: int = 0
    training_duration_seconds: float = 0.0
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    candidate_metrics: dict[str, Any] = field(default_factory=dict)
    metric_deltas: dict[str, float] = field(default_factory=dict)
    validation_passed: bool = False
    promotion_status: str = "PENDING"
    rejection_reasons: list[str] = field(default_factory=list)
    verdict: str = "UNPROVEN"
    hardware_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinualLearningEvent":
        return cls(**data)


class ContinualLearningAuditTrail:
    def __init__(self, audit_file: str = "data/continual_learning_audit_trail.json") -> None:
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._logger = get_logger("continual_learning_audit_trail")

    def record_event(self, event: ContinualLearningEvent) -> bool:
        with self._lock:
            events = self.list_events()
            events.append(event)
            return self._save_events(events)

    def create_and_record(
        self,
        event_type: str,
        trigger_date: str,
        model_type: str,
        dataset_id: str,
        baseline_version: str,
        candidate_version: str,
        **kwargs: Any,
    ) -> ContinualLearningEvent:
        event = ContinualLearningEvent(
            event_id=f"CLE-{int(time.time())}-{uuid.uuid4().hex[:6]}",
            event_type=event_type,
            timestamp=time.time(),
            trigger_date=trigger_date,
            model_type=model_type,
            dataset_id=dataset_id,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            **kwargs,
        )
        self.record_event(event)
        return event

    def list_events(self, model_type: str | None = None) -> list[ContinualLearningEvent]:
        with self._lock:
            if not self.audit_file.exists():
                return []
            try:
                with open(self.audit_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_events = data.get("events", [])
                events = [ContinualLearningEvent.from_dict(e) for e in raw_events]
                if model_type:
                    events = [e for e in events if e.model_type == model_type]
                return events
            except (OSError, json.JSONDecodeError) as err:
                self._logger.warning(f"Could not read continual learning audit trail: {err}")
                return []

    def get_event(self, event_id: str) -> ContinualLearningEvent | None:
        for e in self.list_events():
            if e.event_id == event_id:
                return e
        return None

    def _save_events(self, events: list[ContinualLearningEvent]) -> bool:
        tmp = self.audit_file.with_suffix(".tmp")
        try:
            payload = {"events": [e.to_dict() for e in events], "updated_at": time.time()}
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
            tmp.replace(self.audit_file)
            return True
        except (OSError, ValueError) as err:
            self._logger.error(f"Failed to persist continual learning audit trail: {err}")
            return False
