import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.model_registry import (
    ModelDeploymentStatus,
    ModelRegistry,
    ModelSyncEvent,
)


@pytest.fixture
def temp_registry(tmp_path: Path):
    reg_file = tmp_path / "model_registry.json"
    outbox_file = tmp_path / "model_sync_outbox.json"
    # Ensure baseline files exist for the default models
    runs_dir = tmp_path / "runs" / "exp_001"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "best_model.pth").write_bytes(b"dummy_weights_1")

    registry = ModelRegistry(registry_file=str(reg_file), outbox_file=str(outbox_file))
    return registry, tmp_path


def _register_and_validate_candidate(
    registry: ModelRegistry,
    version: str,
    model_type: str = "bygait_light",
    weights_path: str = "",
) -> None:
    candidate = registry.register_candidate(
        model_version=version,
        model_type=model_type,
        architecture="ByGaitLight-CNN-256D",
        embedding_dim=256,
        artifact_path=weights_path,
        metadata={"allow_missing_artifact": True},
    )
    assert candidate.deployment_status == ModelDeploymentStatus.CANDIDATE
    validated = registry.record_validation_result(
        model_version=version,
        model_type=model_type,
        passed=True,
        metrics={"rank1": 89.5, "mAP": 78.0},
    )
    assert validated.deployment_status == ModelDeploymentStatus.VALIDATED


def test_scenario_a_local_write_succeeds_firebase_fails(temp_registry):
    """SCENARIO A: Local write succeeds, Firebase write fails.

    Expected: LOCAL = new active model, OUTBOX = pending/retrying, INFERENCE = continues.
    """
    registry, _ = temp_registry
    _register_and_validate_candidate(registry, "v2.0.0")

    # Mock Firestore sync failure
    registry._sync_outbox_event = MagicMock(return_value=False)

    promoted = registry.promote_version("v2.0.0", model_type="bygait_light", reason="Better accuracy")
    assert promoted.deployment_status == ModelDeploymentStatus.ACTIVE
    assert promoted.model_version == "v2.0.0"
    assert promoted.previous_production_version == "v1.0.0"

    # Verify local disk reflects the new active model
    active_local = registry.get_active_model("bygait_light")
    assert active_local is not None
    assert active_local.model_version == "v2.0.0"

    # Verify outbox has a pending event
    pending = registry.outbox.list_pending()
    assert len(pending) == 1
    assert pending[0].model_version == "v2.0.0"
    assert pending[0].operation == "PROMOTE"


def test_scenario_b_local_write_fails_safely(temp_registry, monkeypatch):
    """SCENARIO B: Local atomic write fails.

    Expected: Local active model remains previous version, operation raises, Firebase untouched.
    """
    registry, _ = temp_registry
    _register_and_validate_candidate(registry, "v2.0.0")

    initial_active = registry.get_active_model("bygait_light")
    assert initial_active.model_version == "v1.0.0"

    # Force _save_registry to fail
    monkeypatch.setattr(registry, "_save_registry", lambda data: False)

    with pytest.raises(OSError, match="Failed atomic local write"):
        registry.promote_version("v2.0.0", model_type="bygait_light")

    # Verify active model is still v1.0.0
    still_active = registry.get_active_model("bygait_light")
    assert still_active.model_version == "v1.0.0"

    # Outbox must have zero pending events
    assert len(registry.outbox.list_pending()) == 0


def test_scenario_c_crash_recovery_resumes_outbox(temp_registry):
    """SCENARIO C: Process crashes after local commit but before cloud sync.

    Expected: New registry instance reloads outbox from disk and reconciles.
    """
    registry, _ = temp_registry
    _register_and_validate_candidate(registry, "v2.0.0")

    # Simulate cloud failure on first attempt
    registry._sync_outbox_event = MagicMock(return_value=False)
    registry.promote_version("v2.0.0", model_type="bygait_light")

    reg_file = registry.registry_file
    outbox_file = registry.outbox.outbox_file

    # Simulate restart by instantiating new ModelRegistry with the same files
    new_registry = ModelRegistry(registry_file=str(reg_file), outbox_file=str(outbox_file))

    # Check pending events survived restart
    pending = new_registry.outbox.list_pending()
    assert len(pending) == 1
    assert pending[0].model_version == "v2.0.0"

    # Mock successful cloud sync on recovery
    mock_firestore = MagicMock()
    mock_batch = MagicMock()
    mock_firestore.batch.return_value = mock_batch
    mock_col = MagicMock()
    mock_doc = MagicMock()
    mock_doc.get.return_value = MagicMock(exists=False)
    mock_col.document.return_value = mock_doc
    mock_firestore.collection.return_value = mock_col

    result = new_registry.reconcile_with_firebase(firestore_client=mock_firestore)
    assert result["status"] == "COMPLETED"
    assert result["reconciled_events"] >= 1

    # Outbox event is now marked SYNCHRONIZED
    assert len(new_registry.outbox.list_pending()) == 0


def test_scenario_d_idempotent_retry(temp_registry):
    """SCENARIO D: Multiple retries of the same event do not create duplicates or corrupt state."""
    registry, _ = temp_registry
    _register_and_validate_candidate(registry, "v2.0.0")

    event = ModelSyncEvent(
        event_id="sync_promote_bygait_light_v2.0.0_100",
        model_version="v2.0.0",
        model_type="bygait_light",
        desired_status="ACTIVE",
        operation="PROMOTE",
        registry_revision=2,
    )
    registry.outbox.enqueue(event)
    registry.outbox.enqueue(event)  # duplicate enqueue

    all_events = registry.outbox.list_all()
    assert len(all_events) == 1  # Deduplicated by event_id

    # Test failure counting with exponential backoff
    registry.outbox.mark_failed_attempt(event.event_id, "Network timeout 1", max_retries=3)
    ev1 = registry.outbox.get_event(event.event_id)
    assert ev1.attempt_count == 1
    assert ev1.status == "RETRYING"
    assert ev1.next_retry_at > ev1.last_attempt_at

    registry.outbox.mark_failed_attempt(event.event_id, "Network timeout 2", max_retries=3)
    registry.outbox.mark_failed_attempt(event.event_id, "Network timeout 3", max_retries=3)

    ev_exhausted = registry.outbox.get_event(event.event_id)
    assert ev_exhausted.attempt_count == 3
    assert ev_exhausted.status == "RECONCILIATION_REQUIRED"


def test_scenario_e_concurrent_promotions_protected_by_lock(temp_registry):
    """SCENARIO E: Concurrent promotion attempts are serialized by RLock."""
    registry, _ = temp_registry
    _register_and_validate_candidate(registry, "v2.0.0")
    _register_and_validate_candidate(registry, "v3.0.0")

    registry._sync_outbox_event = MagicMock(return_value=True)

    errors = []

    def promote_worker(v):
        try:
            registry.promote_version(v, model_type="bygait_light")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(target=promote_worker, args=("v2.0.0",))
    t2 = threading.Thread(target=promote_worker, args=("v3.0.0",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0

    # Active model must be a valid consistent version (either v2.0.0 or v3.0.0)
    active = registry.get_active_model("bygait_light")
    assert active.model_version in ("v2.0.0", "v3.0.0")

    # File on disk must be valid JSON and readable
    data = registry._load_registry()
    assert len(data["models"]) >= 3


def test_scenario_f_successive_promotions_converge(temp_registry):
    """SCENARIO F: Promotion v2 followed quickly by promotion v3. Local authority v3 wins."""
    registry, _ = temp_registry
    _register_and_validate_candidate(registry, "v2.0.0")
    _register_and_validate_candidate(registry, "v3.0.0")

    registry._sync_outbox_event = MagicMock(return_value=True)

    p2 = registry.promote_version("v2.0.0", model_type="bygait_light")
    assert p2.model_version == "v2.0.0"
    assert p2.previous_production_version == "v1.0.0"

    p3 = registry.promote_version("v3.0.0", model_type="bygait_light")
    assert p3.model_version == "v3.0.0"
    assert p3.previous_production_version == "v2.0.0"

    # Active model is v3.0.0
    active = registry.get_active_model("bygait_light")
    assert active.model_version == "v3.0.0"


def test_scenario_g_rollback_during_pending_sync(temp_registry):
    """SCENARIO G: Rollback during pending synchronization."""
    registry, _ = temp_registry
    _register_and_validate_candidate(registry, "v2.0.0")

    # Promote v2.0.0 with simulated cloud sync deferral
    registry._sync_outbox_event = MagicMock(return_value=False)
    registry.promote_version("v2.0.0", model_type="bygait_light")
    assert registry.get_active_model("bygait_light").model_version == "v2.0.0"

    # Now rollback to v1.0.0
    restored = registry.rollback("bygait_light", reason="Regression detected in live tests")
    assert restored.model_version == "v1.0.0"
    assert restored.deployment_status == ModelDeploymentStatus.ACTIVE

    v2_rec = registry.get_model("v2.0.0", "bygait_light")
    assert v2_rec.deployment_status == ModelDeploymentStatus.ROLLED_BACK

    # Reconcile with mock firestore
    mock_firestore = MagicMock()
    mock_batch = MagicMock()
    mock_firestore.batch.return_value = mock_batch
    mock_col = MagicMock()
    mock_doc = MagicMock()
    mock_doc.get.return_value = MagicMock(exists=False)
    mock_col.document.return_value = mock_doc
    mock_firestore.collection.return_value = mock_col

    registry.reconcile_with_firebase(firestore_client=mock_firestore)
    # The active model in local registry remains v1.0.0
    assert registry.get_active_model("bygait_light").model_version == "v1.0.0"


def test_unvalidated_candidate_promotion_rejected(temp_registry):
    """Cannot promote an unvalidated candidate."""
    registry, _ = temp_registry
    candidate = registry.register_candidate(
        model_version="v_unvalidated",
        model_type="bygait_light",
        architecture="ByGaitLight-CNN-256D",
        embedding_dim=256,
        artifact_path="",
        metadata={"allow_missing_artifact": True},
    )
    assert candidate.deployment_status == ModelDeploymentStatus.CANDIDATE

    with pytest.raises(RuntimeError, match="Cannot promote unvalidated model"):
        registry.promote_version("v_unvalidated", model_type="bygait_light")


def test_optimistic_concurrency_stale_writer_rejected(temp_registry):
    """Stale writer with smaller registry_revision is deferred for reconciliation."""
    registry, _ = temp_registry
    _register_and_validate_candidate(registry, "v2.0.0")

    mock_firestore = MagicMock()
    mock_col = MagicMock()
    mock_pointer_doc = MagicMock()
    mock_pointer_doc.exists = True
    mock_pointer_doc.to_dict.return_value = {"registry_revision": 99, "active_version": "v9.0.0"}

    mock_pointer_ref = MagicMock()
    mock_pointer_ref.get.return_value = mock_pointer_doc
    mock_col.document.return_value = mock_pointer_ref
    mock_firestore.collection.return_value = mock_col

    event = ModelSyncEvent(
        event_id="sync_test_stale",
        model_version="v2.0.0",
        model_type="bygait_light",
        desired_status="ACTIVE",
        operation="PROMOTE",
        registry_revision=2,
    )
    registry.outbox.enqueue(event)

    synced = registry._sync_outbox_event(event, registry._load_registry(), firestore_client=mock_firestore)
    assert synced is False

    updated_event = registry.outbox.get_event("sync_test_stale")
    assert updated_event.status == "RECONCILIATION_REQUIRED"
