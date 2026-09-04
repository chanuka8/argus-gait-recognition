import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from storage.firebase_embedding_store import (
    FirebaseEmbeddingDocument,
    FirebaseEmbeddingStore,
    PersistenceErrorCategory,
    validate_service_account_file,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure clean environment without ambient service account paths."""
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setattr(
        FirebaseEmbeddingStore,
        "_resolve_credential_path",
        lambda self: None,
    )


def _make_dummy_vector(dim: int = 256) -> list[float]:
    vec = [0.1] * dim
    # Normalize
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def test_01_credential_path_missing_defaults_to_offline_mode(tmp_path: Path, clean_env):
    """INVARIANT 1: Missing credential path enters offline mode without crashing."""
    offline_file = tmp_path / "offline_store.json"
    store = FirebaseEmbeddingStore(mode="auto", offline_store_path=str(offline_file))

    assert store.mode == "offline"
    assert store.credential_status == "MISSING"
    assert store.firestore_status == "UNINITIALIZED"
    healthy, info = store.check_connection_health()
    assert healthy is True
    assert info["mode"] == "offline"
    assert info["credential"] == "MISSING"


def test_02_credential_path_nonexistent_file_defaults_to_offline(tmp_path: Path, monkeypatch):
    """INVARIANT 2: Path pointing to nonexistent file falls back to offline mode."""
    nonexistent = tmp_path / "does_not_exist.json"
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_PATH", str(nonexistent))

    offline_file = tmp_path / "offline_store.json"
    store = FirebaseEmbeddingStore(mode="auto", offline_store_path=str(offline_file))

    assert store.mode == "offline"
    assert store.credential_status == "INVALID"
    healthy, info = store.check_connection_health()
    assert healthy is True
    assert info["credential"] == "INVALID"


def test_03_invalid_json_credential_defaults_to_offline(tmp_path: Path, monkeypatch):
    """INVARIANT 3: Malformed JSON file falls back to offline mode without crashing."""
    bad_json = tmp_path / "corrupt_service_account.json"
    bad_json.write_text("NOT_VALID_JSON{foo:bar", encoding="utf-8")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_PATH", str(bad_json))

    offline_file = tmp_path / "offline_store.json"
    store = FirebaseEmbeddingStore(mode="auto", offline_store_path=str(offline_file))

    assert store.mode == "offline"
    assert store.credential_status == "INVALID"


def test_04_json_missing_required_fields_defaults_to_offline(tmp_path: Path, monkeypatch):
    """INVARIANT 4: JSON missing essential service_account fields falls back to offline."""
    # Test 4a: Missing type
    incomplete_1 = tmp_path / "incomplete_1.json"
    incomplete_1.write_text(json.dumps({"project_id": "argus-17702"}), encoding="utf-8")
    is_valid, reason, _ = validate_service_account_file(incomplete_1)
    assert is_valid is False
    assert "INVALID_TYPE" in reason

    # Test 4b: Missing private_key
    incomplete_2 = tmp_path / "incomplete_2.json"
    incomplete_2.write_text(
        json.dumps({
            "type": "service_account",
            "project_id": "argus-17702",
            "client_email": "argus-sa@argus-17702.iam.gserviceaccount.com",
        }),
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_service_account_file(incomplete_2)
    assert is_valid is False
    assert "MISSING_OR_MALFORMED_PRIVATE_KEY" in reason

    # Test 4c: Missing client_email
    incomplete_3 = tmp_path / "incomplete_3.json"
    incomplete_3.write_text(
        json.dumps({
            "type": "service_account",
            "project_id": "argus-17702",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASC...\n-----END PRIVATE KEY-----\n",
        }),
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_service_account_file(incomplete_3)
    assert is_valid is False
    assert "MISSING_OR_MALFORMED_CLIENT_EMAIL" in reason

    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_PATH", str(incomplete_2))
    offline_file = tmp_path / "offline_store.json"
    store = FirebaseEmbeddingStore(mode="auto", offline_store_path=str(offline_file))
    assert store.mode == "offline"
    assert store.credential_status == "INVALID"


def test_05_valid_service_account_structure_initializes_firebase(tmp_path: Path, monkeypatch):
    """INVARIANT 5: Valid service-account structure initializes Firebase Admin SDK via mock."""
    valid_sa = tmp_path / "valid_service_account.json"
    valid_sa.write_text(
        json.dumps({
            "type": "service_account",
            "project_id": "argus-17702",
            "private_key_id": "mock_key_id_123",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC6...\n-----END PRIVATE KEY-----\n",
            "client_email": "argus-sa@argus-17702.iam.gserviceaccount.com",
            "client_id": "123456789012345678901",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_PATH", str(valid_sa))

    mock_firestore_client = MagicMock()
    mock_storage_bucket = MagicMock()

    with patch("firebase_admin.credentials.Certificate"), \
         patch("firebase_admin.initialize_app"), \
         patch("firebase_admin.firestore.client", return_value=mock_firestore_client), \
         patch("firebase_admin.storage.bucket", return_value=mock_storage_bucket):

        offline_file = tmp_path / "offline_store.json"
        store = FirebaseEmbeddingStore(mode="auto", offline_store_path=str(offline_file))

        assert store.mode == "live"
        assert store.credential_status == "FOUND"
        assert store.firestore_status == "CONNECTED"
        assert store.project_id == "argus-17702"

        healthy, info = store.check_connection_health()
        assert healthy is True
        assert info["mode"] == "live"
        assert info["firestore"] == "CONNECTED"


def test_06_live_firestore_persistence_behavior(tmp_path: Path, monkeypatch):
    """INVARIANT 6: Live mode persists to Firestore collection."""
    mock_firestore_client = MagicMock()
    mock_doc_ref = MagicMock()
    mock_firestore_client.collection.return_value.document.return_value = mock_doc_ref

    store = FirebaseEmbeddingStore(mode="offline", offline_store_path=str(tmp_path / "offline.json"))
    store.mode = "live"
    store._firestore_client = mock_firestore_client

    doc = FirebaseEmbeddingDocument(
        embedding_id="emb_gait_subj01_1700000000_abcdef",
        person_id="subj01",
        modality="gait",
        embedding_dim=256,
        vector=_make_dummy_vector(256),
    )

    result = store.persist_embedding(doc)
    assert result.success is True
    assert result.embedding_id == doc.embedding_id
    mock_firestore_client.collection.assert_called_with("biometric_embeddings")
    mock_doc_ref.set.assert_called_once()


def test_07_offline_persistence_behavior(tmp_path: Path):
    """INVARIANT 7: Offline mode persists cleanly to local JSON file."""
    offline_file = tmp_path / "offline_store.json"
    store = FirebaseEmbeddingStore(mode="offline", offline_store_path=str(offline_file))

    doc = FirebaseEmbeddingDocument(
        embedding_id="emb_gait_subj02_1700000000_123456",
        person_id="subj02",
        modality="gait",
        embedding_dim=256,
        vector=_make_dummy_vector(256),
    )

    result = store.persist_embedding(doc)
    assert result.success is True
    assert offline_file.exists()

    # Verify retrieval
    results = store.get_embeddings_by_person("subj02")
    assert len(results) == 1
    assert results[0].embedding_id == doc.embedding_id


def test_08_retry_queue_behavior_on_temporary_firestore_failure(tmp_path: Path):
    """INVARIANT 8: Temporary Firestore failure queues document in retry queue without crashing."""
    mock_firestore_client = MagicMock()
    mock_firestore_client.collection.return_value.document.return_value.set.side_effect = RuntimeError(
        "Network connection timeout to Firestore"
    )

    store = FirebaseEmbeddingStore(mode="offline", offline_store_path=str(tmp_path / "offline.json"))
    store.mode = "live"
    store._firestore_client = mock_firestore_client

    doc = FirebaseEmbeddingDocument(
        embedding_id="emb_gait_subj03_1700000000_retry01",
        person_id="subj03",
        modality="gait",
        embedding_dim=256,
        vector=_make_dummy_vector(256),
    )

    result = store.persist_embedding(doc)
    assert result.success is False
    assert result.retry_queued is True
    assert result.error_category == PersistenceErrorCategory.NETWORK_TIMEOUT
    assert store.get_retry_queue_size() == 1

    # Now simulate restored Firestore and process retry queue
    mock_firestore_client.collection.return_value.document.return_value.set.side_effect = None
    retry_results = store.process_retry_queue()
    assert len(retry_results) == 1
    assert retry_results[0].success is True
    assert store.get_retry_queue_size() == 0


def test_09_firebase_recovery_rebuilds_local_database(tmp_path: Path):
    """INVARIANT 9: Disaster recovery rebuild creates person dictionary from Firestore embeddings."""
    mock_firestore_client = MagicMock()

    doc_dict = {
        "embedding_id": "emb_gait_subj04_1700000000_recov1",
        "person_id": "subj04",
        "modality": "gait",
        "embedding_dim": 256,
        "vector": _make_dummy_vector(256),
        "model_version": "v1.0.0",
        "embedding_version": 1,
        "observation_date": "2026-09-03",
        "created_at": 1700000000.0,
        "quality_score": 0.95,
        "status": "ACTIVE",
        "source_session_id": "sess_123",
        "identity_type": "LIVE_OPERATIONAL",
        "operational_state": "VERIFIED",
        "training_eligibility": "ELIGIBLE",
        "case_id": "case_999",
    }
    mock_doc_snap = MagicMock()
    mock_doc_snap.to_dict.return_value = doc_dict

    mock_firestore_client.collection.return_value.stream.return_value = [mock_doc_snap]

    store = FirebaseEmbeddingStore(mode="offline", offline_store_path=str(tmp_path / "offline.json"))
    store.mode = "live"
    store._firestore_client = mock_firestore_client

    recovered = store.rebuild_local_from_firebase()
    assert "subj04" in recovered
    assert len(recovered["subj04"]["gait_embeddings"]) == 1
    assert recovered["subj04"]["gait_embeddings"][0]["embedding_id"] == "emb_gait_subj04_1700000000_recov1"


def test_10_no_secret_or_private_key_appears_in_logs_or_diagnostics(tmp_path: Path, caplog, monkeypatch):
    """INVARIANT 10: Private keys and secrets are NEVER logged or exposed in health check."""
    raw_secret_key = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC6_SECRET_PORTION_XYZ123\n-----END PRIVATE KEY-----\n"

    valid_sa = tmp_path / "secret_service_account.json"
    valid_sa.write_text(
        json.dumps({
            "type": "service_account",
            "project_id": "argus-17702",
            "private_key_id": "super_secret_key_id",
            "private_key": raw_secret_key,
            "client_email": "argus-sa@argus-17702.iam.gserviceaccount.com",
            "client_id": "123456789012345678901",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_PATH", str(valid_sa))

    with caplog.at_level(logging.DEBUG):
        store = FirebaseEmbeddingStore(mode="auto", offline_store_path=str(tmp_path / "offline.json"))
        _healthy, info = store.check_connection_health()

        # Check diagnostics
        info_str = json.dumps(info)
        assert "SECRET_PORTION" not in info_str
        assert "PRIVATE KEY" not in info_str
        assert "super_secret_key_id" not in info_str

        # Check log output
        log_text = caplog.text
        assert "SECRET_PORTION" not in log_text
        assert "super_secret_key_id" not in log_text
        assert "BEGIN PRIVATE KEY" not in log_text
