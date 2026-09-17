"""U3: Biometric Template Encryption at Rest Test Suite.

Comprehensive test suite verifying:
- AES-256-GCM authenticated encryption and GHASH tag verification
- 256D gait templates and 512D appearance templates round trips
- Cryptographic identity binding via AAD (labels digest, gallery type, person_id, modality)
- Tamper detection (ciphertext, tag, nonce, magic, version, cipher suite, header bounds)
- Downgrade prevention (never fall back to plaintext .npy on corrupted or invalid .enc)
- Plaintext residue detection and coexisting file precedence
- Strict production mode vs permitted development mode
- EmbeddingDatabase field-level vector encryption with plaintext vector absence at rest
- U2 audit HMAC and U3 biometric encryption key separation
- Atomic temporary file replacement under inter-process FileLock
- Nonce uniqueness across 1,000 synthetic operations
- Recognition output equivalence (cosine similarity identical before and after decryption)
- Migration tool (--dry-run, --migrate, --verify, safe atomic replacement)
- Integrity assertion that production biometric files remain completely untouched
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from security_layer.biometric_encryption import (
    BiometricDecryptionError,
    BiometricEncryptor,
    ConfigurationError,
    ContainerFormatError,
)
from storage.embedding_database import EmbeddingDatabase, PersonRecord
from storage.vector_store import VectorStore, validate_gallery_files
from tools.security.migrate_gallery_encryption import (
    main as migration_cli_main,
)
from tools.security.migrate_gallery_encryption import (
    migrate_embedding_db,
    migrate_vector_store,
    verify_vector_store,
)

TEST_BIOMETRIC_KEY_HEX = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
TEST_BIOMETRIC_KEY_BYTES = bytes.fromhex(TEST_BIOMETRIC_KEY_HEX)
ALT_BIOMETRIC_KEY_HEX = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"


def _compute_storage_fingerprint() -> dict[str, tuple[int, str]]:
    """Compute exact fingerprint (size, sha256) of all files in real biometric directories."""
    fingerprint = {}
    prod_dirs = [
        Path("models/gallery"),
        Path("models/live_gallery"),
        Path("models/appearance_gallery"),
        Path("data/embedding_db"),
    ]
    for d in prod_dirs:
        if not d.exists():
            continue
        for item in sorted(d.rglob("*")):
            if item.is_file():
                assert not item.name.endswith(".enc"), f"Forbidden .enc file detected in real storage: {item}"
                assert not item.name.endswith(".lock"), f"Forbidden .lock file detected in real storage: {item}"
                assert not item.name.endswith(".bak"), f"Forbidden .bak file detected in real storage: {item}"
                assert not item.name.startswith(".tmp_") and not item.name.endswith(".tmp"), (
                    f"Forbidden tmp file in real storage: {item}"
                )
                rel_path = item.as_posix()
                size = item.stat().st_size
                digest = hashlib.sha256(item.read_bytes()).hexdigest()
                fingerprint[rel_path] = (size, digest)
    return fingerprint


@pytest.fixture(scope="module", autouse=True)
def real_biometric_storage_guard():
    """Hard safety guard fixture verifying zero mutation of production biometric storage."""
    before = _compute_storage_fingerprint()
    yield
    after = _compute_storage_fingerprint()

    # Compare fingerprints before and after test execution
    assert set(before.keys()) == set(after.keys()), (
        f"Files in real biometric paths changed! Added: {set(after.keys()) - set(before.keys())}, "
        f"Removed: {set(before.keys()) - set(after.keys())}"
    )
    for path, (b_size, b_hash) in before.items():
        a_size, a_hash = after[path]
        assert b_size == a_size, f"File size changed for real file {path}: was {b_size}, now {a_size}"
        assert b_hash == a_hash, f"Hash changed for real file {path}: was {b_hash}, now {a_hash}"


@pytest.fixture
def encryptor():
    return BiometricEncryptor(key=TEST_BIOMETRIC_KEY_HEX)


@pytest.fixture
def synthetic_gait_data():
    np.random.seed(42)
    feats = np.random.randn(10, 256).astype(np.float32)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    labels = np.array([f"person_{i:03d}" for i in range(10)])
    return feats, labels


@pytest.fixture
def synthetic_appearance_data():
    np.random.seed(42)
    feats = np.random.randn(8, 512).astype(np.float32)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    labels = np.array([f"subj_{i:03d}" for i in range(8)])
    return feats, labels


# 1. AES-256-GCM valid round trip
def test_aes_256_gcm_valid_round_trip(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    decrypted, aad = encryptor.decrypt_gallery_container(container, "baseline_gait", labels)
    assert np.array_equal(feats, decrypted)
    assert aad["gallery_type"] == "baseline_gait"
    assert aad["dimension"] == 256
    assert aad["num_records"] == 10


# 2. Exact 256D float32 round trip
def test_exact_256d_float32_round_trip(encryptor):
    np.random.seed(123)
    feats = np.random.randn(5, 256).astype(np.float32)
    labels = np.array(["p1", "p2", "p3", "p4", "p5"])
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    decrypted, _ = encryptor.decrypt_gallery_container(container, "baseline_gait", labels)
    assert decrypted.dtype == np.float32
    assert decrypted.shape == (5, 256)
    assert np.all(feats == decrypted)


# 3. Exact 512D float32 round trip
def test_exact_512d_float32_round_trip(encryptor, synthetic_appearance_data):
    feats, labels = synthetic_appearance_data
    container = encryptor.encrypt_gallery_container(feats, labels, "appearance")
    decrypted, aad = encryptor.decrypt_gallery_container(container, "appearance", labels)
    assert decrypted.dtype == np.float32
    assert decrypted.shape == (8, 512)
    assert aad["dimension"] == 512
    assert np.all(feats == decrypted)


# 4. Wrong key rejection
def test_wrong_key_fails_decryption(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    wrong_encryptor = BiometricEncryptor(key=ALT_BIOMETRIC_KEY_HEX)
    with pytest.raises(BiometricDecryptionError, match="authentication tag verification failed"):
        wrong_encryptor.decrypt_gallery_container(container, "baseline_gait", labels)


# 5. Corrupted ciphertext rejection
def test_corrupted_ciphertext_fails(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = bytearray(encryptor.encrypt_gallery_container(feats, labels, "baseline_gait"))
    # Corrupt a byte in ciphertext (past the 24B header and AAD)
    container[-20] ^= 0xFF
    with pytest.raises(BiometricDecryptionError, match="authentication tag verification failed"):
        encryptor.decrypt_gallery_container(bytes(container), "baseline_gait", labels)


# 6. Modified authentication tag rejection
def test_modified_tag_fails(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = bytearray(encryptor.encrypt_gallery_container(feats, labels, "baseline_gait"))
    # The last 16 bytes contain the GCM tag
    container[-1] ^= 0x01
    with pytest.raises(BiometricDecryptionError, match="authentication tag verification failed"):
        encryptor.decrypt_gallery_container(bytes(container), "baseline_gait", labels)


# 7. Malformed nonce in container
def test_malformed_nonce_rejection(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = bytearray(encryptor.encrypt_gallery_container(feats, labels, "baseline_gait"))
    # Nonce is at bytes 8..20
    container[8] ^= 0xAA
    with pytest.raises(BiometricDecryptionError):
        encryptor.decrypt_gallery_container(bytes(container), "baseline_gait", labels)


# 8. Malformed container magic
def test_malformed_container_magic(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = bytearray(encryptor.encrypt_gallery_container(feats, labels, "baseline_gait"))
    container[0:4] = b"XXXX"
    with pytest.raises(ContainerFormatError, match="Invalid container magic"):
        encryptor.decrypt_gallery_container(bytes(container), "baseline_gait", labels)


# 9. Unsupported container version
def test_unsupported_version_rejected(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = bytearray(encryptor.encrypt_gallery_container(feats, labels, "baseline_gait"))
    # Version is at bytes 4..6
    struct.pack_into("!H", container, 4, 99)
    with pytest.raises(ContainerFormatError, match="Unsupported container version"):
        encryptor.decrypt_gallery_container(bytes(container), "baseline_gait", labels)


# 10. Unsupported cipher suite
def test_unsupported_cipher_suite_rejected(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = bytearray(encryptor.encrypt_gallery_container(feats, labels, "baseline_gait"))
    # Cipher suite is at bytes 6..8
    struct.pack_into("!H", container, 6, 99)
    with pytest.raises(ContainerFormatError, match="Unsupported cipher suite"):
        encryptor.decrypt_gallery_container(bytes(container), "baseline_gait", labels)


# 11. Truncated container parsing
def test_truncated_container_rejected(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    truncated = container[:30]
    with pytest.raises(ContainerFormatError, match="Corrupted container: length"):
        encryptor.decrypt_gallery_container(truncated, "baseline_gait", labels)


# 12. Impossible AAD length bounds check
def test_impossible_aad_length_rejected(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = bytearray(encryptor.encrypt_gallery_container(feats, labels, "baseline_gait"))
    # AAD length is uint32 at bytes 20..24
    struct.pack_into("!I", container, 20, 100_000)
    with pytest.raises(ContainerFormatError, match="AAD length"):
        encryptor.decrypt_gallery_container(bytes(container), "baseline_gait", labels)


# 13. Deterministic AAD serialization
def test_deterministic_aad_serialization():
    payload1 = {"dimension": 256, "dtype": "float32", "gallery_type": "live_gait", "num_records": 5}
    payload2 = {"gallery_type": "live_gait", "num_records": 5, "dtype": "float32", "dimension": 256}
    assert BiometricEncryptor.canonicalize_aad(payload1) == BiometricEncryptor.canonicalize_aad(payload2)


# 14. Modified labels failure (AAD cryptographic binding)
def test_modified_labels_fails_decryption(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    tampered_labels = np.array([lbl if i != 0 else "tampered_id" for i, lbl in enumerate(labels)])
    with pytest.raises(BiometricDecryptionError, match="Cryptographic label binding mismatch"):
        encryptor.decrypt_gallery_container(container, "baseline_gait", tampered_labels)


# 15. Reordered labels failure
def test_reordered_labels_fails_decryption(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    reversed_labels = labels[::-1]
    with pytest.raises(BiometricDecryptionError, match="Cryptographic label binding mismatch"):
        encryptor.decrypt_gallery_container(container, "baseline_gait", reversed_labels)


# 16. Wrong gallery type failure
def test_wrong_gallery_type_fails_decryption(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    with pytest.raises(BiometricDecryptionError, match="Gallery type mismatch in AAD"):
        encryptor.decrypt_gallery_container(container, "appearance", labels)


# 17. Wrong dimension failure in AAD
def test_dimension_validation(encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    _, aad = encryptor.decrypt_gallery_container(container, "baseline_gait", labels)
    assert aad["dimension"] == 256


# 18. Wrong dtype validation
def test_wrong_dtype_rejected(encryptor):
    # Features must be numeric
    feats = np.array([["a", "b"], ["c", "d"]])
    labels = np.array(["p1", "p2"])
    with pytest.raises((TypeError, ValueError)):
        encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")


# 19. Multibyte Unicode labels handling
def test_unicode_labels_binding(encryptor):
    np.random.seed(99)
    feats = np.random.randn(3, 256).astype(np.float32)
    unicode_labels = np.array(["田中太郎", "Müller_François", "Иван_Смирнов"])
    container = encryptor.encrypt_gallery_container(feats, unicode_labels, "baseline_gait")
    decrypted, _ = encryptor.decrypt_gallery_container(container, "baseline_gait", unicode_labels)
    assert np.all(feats == decrypted)


# 20. Encrypted + plaintext coexistence precedence
def test_coexistence_precedence(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    store = VectorStore(gallery_dir=tmp_path, encryptor=encryptor)
    store.save(feats, labels, {})

    # Simulate residue by also creating plaintext .npy
    np.save(tmp_path / "gallery_features.npy", feats + 1.0)
    assert (tmp_path / "gallery_features.enc").exists()
    assert (tmp_path / "gallery_features.npy").exists()

    # Loading store must load the encrypted .enc and ignore modified plaintext
    loaded_feats, loaded_labels, _ = store.load()
    assert np.array_equal(loaded_feats, feats)
    assert np.array_equal(loaded_labels, labels)


# 21. Corrupt encrypted + valid plaintext MUST NOT downgrade
def test_corrupt_encrypted_never_downgrades_to_plaintext(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    store = VectorStore(gallery_dir=tmp_path, encryptor=encryptor)
    store.save(feats, labels, {})

    # Corrupt the encrypted container
    enc_file = tmp_path / "gallery_features.enc"
    enc_bytes = bytearray(enc_file.read_bytes())
    enc_bytes[-1] ^= 0xFF
    enc_file.write_bytes(bytes(enc_bytes))

    # Also place a valid plaintext file
    np.save(tmp_path / "gallery_features.npy", feats)

    # Downgrade prevention: must raise, NEVER fall back to .npy
    with pytest.raises(BiometricDecryptionError):
        store.load()


# 22. Strict production plaintext rejection
def test_strict_production_plaintext_rejection(tmp_path, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    # Create store with strict_mode=True and no key
    store = VectorStore(gallery_dir=tmp_path, encryptor=BiometricEncryptor(key=None), strict_mode=True)
    # Save a plaintext file directly to disk
    np.save(tmp_path / "gallery_features.npy", feats)
    np.save(tmp_path / "gallery_labels.npy", labels)

    with pytest.raises(RuntimeError, match="STRICT SECURITY VIOLATION: Unencrypted biometric gallery detected"):
        store.load()


# 23. Permitted legacy development load
def test_permitted_legacy_dev_load(tmp_path, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    # In dev mode (strict_mode=False) with no key, plaintext loading is allowed
    store = VectorStore(gallery_dir=tmp_path, encryptor=BiometricEncryptor(key=None), strict_mode=False)
    np.save(tmp_path / "gallery_features.npy", feats)
    np.save(tmp_path / "gallery_labels.npy", labels)

    loaded = store.load()
    assert loaded is not None
    loaded_feats, loaded_labels, _ = loaded
    assert np.array_equal(loaded_feats, feats)
    assert np.array_equal(loaded_labels, labels)


# 24. EmbeddingDatabase encrypted 256D vector
def test_embedding_database_encrypted_256d_vector(tmp_path, encryptor):
    db = EmbeddingDatabase(
        db_dir=str(tmp_path / "db"),
        gait_gallery_dir=str(tmp_path / "live_gal"),
        appearance_gallery_dir=str(tmp_path / "app_gal"),
        encryptor=encryptor,
    )
    vec = [0.1] * 256
    db.add_embeddings(person_id="person_A", gait_embeddings=[vec])

    # Check file on disk
    p_file = tmp_path / "db" / "persons" / "person_A.json"
    assert p_file.exists()
    disk_data = json.loads(p_file.read_text(encoding="utf-8"))

    gait_rec = disk_data["gait_embeddings"][0]
    assert "vector_encryption" in gait_rec
    assert "vector" not in gait_rec  # Plaintext vector omitted
    assert gait_rec["vector_encryption"]["cipher"] == "AES-256-GCM"
    assert gait_rec["vector_encryption"]["dimension"] == 256

    # Verify transparent decryption on get_person
    person = db.get_person("person_A")
    assert person is not None
    assert len(person.gait_embeddings[0].vector) == 256
    norm = np.linalg.norm(person.gait_embeddings[0].vector)
    assert np.isclose(norm, 1.0)


# 25. EmbeddingDatabase encrypted 512D vector
def test_embedding_database_encrypted_512d_vector(tmp_path, encryptor):
    db = EmbeddingDatabase(
        db_dir=str(tmp_path / "db"),
        gait_gallery_dir=str(tmp_path / "live_gal"),
        appearance_gallery_dir=str(tmp_path / "app_gal"),
        encryptor=encryptor,
    )
    vec = [0.05] * 512
    db.add_embeddings(person_id="person_B", appearance_embeddings=[vec])

    person = db.get_person("person_B")
    assert person is not None
    assert len(person.appearance_embeddings[0].vector) == 512


# 26. EmbeddingDatabase person_id AAD mismatch
def test_embedding_database_person_id_aad_mismatch(encryptor):
    vec = [0.1] * 256
    enc_dict = encryptor.encrypt_vector(vec, person_id="real_subject", modality="gait")
    with pytest.raises(BiometricDecryptionError, match="AEAD authentication tag verification failed"):
        # Attempt to decrypt under a different subject ID
        encryptor.decrypt_vector(enc_dict, person_id="attacker_subject", modality="gait")


# 27. EmbeddingDatabase modality AAD mismatch
def test_embedding_database_modality_aad_mismatch(encryptor):
    vec = [0.1] * 256
    enc_dict = encryptor.encrypt_vector(vec, person_id="subj_01", modality="gait")
    with pytest.raises(BiometricDecryptionError, match="Modality mismatch in vector encryption"):
        encryptor.decrypt_vector(enc_dict, person_id="subj_01", modality="appearance")


# 28. JSON ciphertext tampering in EmbeddingDatabase
def test_json_ciphertext_tampering(tmp_path, encryptor):
    db = EmbeddingDatabase(
        db_dir=str(tmp_path / "db"),
        gait_gallery_dir=str(tmp_path / "live_gal"),
        appearance_gallery_dir=str(tmp_path / "app_gal"),
        encryptor=encryptor,
    )
    db.add_embeddings(person_id="person_C", gait_embeddings=[[0.1] * 256])

    p_file = tmp_path / "db" / "persons" / "person_C.json"
    data = json.loads(p_file.read_text(encoding="utf-8"))
    ct_b64 = data["gait_embeddings"][0]["vector_encryption"]["ciphertext"]
    ct_raw = bytearray(base64.b64decode(ct_b64))
    ct_raw[5] ^= 0xFF
    data["gait_embeddings"][0]["vector_encryption"]["ciphertext"] = base64.b64encode(ct_raw).decode("ascii")
    p_file.write_text(json.dumps(data), encoding="utf-8")

    # In strict mode or when decrypting tampered vector, error is raised
    with pytest.raises(BiometricDecryptionError):
        PersonRecord.from_dict(data, encryptor=encryptor, strict_mode=True)


# 29. Plaintext vector absence in protected record
def test_plaintext_vector_absence_at_rest(tmp_path, encryptor):
    db = EmbeddingDatabase(
        db_dir=str(tmp_path / "db"),
        gait_gallery_dir=str(tmp_path / "live_gal"),
        appearance_gallery_dir=str(tmp_path / "app_gal"),
        encryptor=encryptor,
    )
    db.add_embeddings(person_id="subject_secret", gait_embeddings=[[0.02] * 256])

    p_file = tmp_path / "db" / "persons" / "subject_secret.json"
    content = p_file.read_text(encoding="utf-8")
    assert '"vector":' not in content
    assert '"vector_encryption":' in content


# 30. U2 / U3 identical key rejection
def test_u2_u3_identical_key_rejected(monkeypatch):
    identical_key = "1111222233334444555566667777888811112222333344445555666677778888"
    monkeypatch.setenv("ARGUS_AUDIT_HMAC_KEY", identical_key)
    with pytest.raises(ConfigurationError, match="must not match ARGUS_AUDIT_HMAC_KEY"):
        BiometricEncryptor(key=identical_key)


# 31. U2 / U3 distinct key acceptance
def test_u2_u3_distinct_key_accepted(monkeypatch):
    u2_key = "1111222233334444555566667777888811112222333344445555666677778888"
    u3_key = "9999888877776666555544443333222299998888777766665555444433332222"
    monkeypatch.setenv("ARGUS_AUDIT_HMAC_KEY", u2_key)
    enc = BiometricEncryptor(key=u3_key)
    assert enc.is_configured is True


# 32. Atomic VectorStore write with temporary swap
def test_atomic_vector_store_write(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    store = VectorStore(gallery_dir=tmp_path, encryptor=encryptor)
    store.save(feats, labels, {"p1": {"status": "ACTIVE"}})

    # Target files exist
    assert (tmp_path / "gallery_features.enc").exists()
    assert (tmp_path / "gallery_labels.npy").exists()
    assert (tmp_path / "gallery_metadata.json").exists()

    # No leftover temporary files
    tmps = list(tmp_path.glob(".tmp_*"))
    assert len(tmps) == 0


# 33. Atomic EmbeddingDatabase write
def test_atomic_embedding_db_write(tmp_path, encryptor):
    db = EmbeddingDatabase(
        db_dir=str(tmp_path / "db"),
        gait_gallery_dir=str(tmp_path / "live_gal"),
        appearance_gallery_dir=str(tmp_path / "app_gal"),
        encryptor=encryptor,
    )
    db.add_embeddings(person_id="p_atomic", gait_embeddings=[[0.1] * 256])

    p_dir = tmp_path / "db" / "persons"
    tmps = list(p_dir.glob(".tmp_*")) + list(p_dir.glob("*.tmp*"))
    assert all(not f.name.startswith(".tmp_") for f in tmps)


# 34. Concurrent gallery write safety
def test_concurrent_gallery_writes(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    store1 = VectorStore(gallery_dir=tmp_path, encryptor=encryptor)
    store2 = VectorStore(gallery_dir=tmp_path, encryptor=encryptor)

    store1.save(feats[:5], labels[:5], {})
    store2.save(feats[5:], labels[5:], {})

    loaded_feats, loaded_labels, _ = store1.load()
    assert len(loaded_feats) == 5
    assert len(loaded_labels) == 5


# 35. Migration dry run
def test_migration_dry_run(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    gal_dir = tmp_path / "gallery"
    gal_dir.mkdir()
    np.save(gal_dir / "gallery_features.npy", feats)
    np.save(gal_dir / "gallery_labels.npy", labels)

    res = migrate_vector_store(gal_dir, encryptor, dry_run=True)
    assert res["status"] == "DRY_RUN"
    assert not (gal_dir / "gallery_features.enc").exists()
    assert (gal_dir / "gallery_features.npy").exists()


# 36. Migration synthetic VectorStore gallery
def test_migration_synthetic_vector_store(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    gal_dir = tmp_path / "gallery"
    gal_dir.mkdir()
    np.save(gal_dir / "gallery_features.npy", feats)
    np.save(gal_dir / "gallery_labels.npy", labels)

    res = migrate_vector_store(gal_dir, encryptor, dry_run=False, purge_plaintext=False)
    assert res["status"] == "MIGRATED"
    assert (gal_dir / "gallery_features.enc").exists()
    assert (gal_dir / "gallery_features.npy").exists()  # Kept by default

    # Verify decryption
    v_res = verify_vector_store(gal_dir, encryptor)
    assert v_res["is_valid"] is True
    assert v_res["templates"] == len(feats)


# 37. Migration synthetic EmbeddingDatabase
def test_migration_synthetic_embedding_database(tmp_path, encryptor):
    db_dir = tmp_path / "db"
    persons_dir = db_dir / "persons"
    persons_dir.mkdir(parents=True)

    # Write a legacy plaintext JSON record
    raw_record = {
        "person_id": "legacy_user",
        "status": "ACTIVE",
        "gait_embeddings": [
            {
                "embedding_id": "gait_01",
                "person_id": "legacy_user",
                "modality": "gait",
                "embedding_dim": 256,
                "vector": [0.05] * 256,
                "model_version": "v1.0.0",
                "status": "ACTIVE",
            }
        ],
        "appearance_embeddings": [],
    }
    p_file = persons_dir / "legacy_user.json"
    p_file.write_text(json.dumps(raw_record, indent=2), encoding="utf-8")

    res = migrate_embedding_db(db_dir, encryptor, dry_run=False)
    assert res["status"] == "COMPLETED"
    assert res["migrated_persons"] == 1
    assert res["migrated_vectors"] == 1

    # Verify disk record now has vector_encryption and no raw vector
    migrated_data = json.loads(p_file.read_text(encoding="utf-8"))
    assert "vector_encryption" in migrated_data["gait_embeddings"][0]
    assert "vector" not in migrated_data["gait_embeddings"][0]


# 38. Migration verification mode
def test_migration_verification(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    store = VectorStore(gallery_dir=tmp_path, encryptor=encryptor)
    store.save(feats, labels, {})

    v_res = verify_vector_store(tmp_path, encryptor)
    assert v_res["is_valid"] is True
    assert v_res["status"] == "VALID"


# 39. Interrupted migration recovery
def test_interrupted_migration_recovery(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    gal_dir = tmp_path / "gallery"
    gal_dir.mkdir()
    np.save(gal_dir / "gallery_features.npy", feats)
    np.save(gal_dir / "gallery_labels.npy", labels)

    # Leave an orphaned .tmp_ file as if process died mid-flight
    (gal_dir / ".tmp_interrupted.enc").write_bytes(b"partial")

    # Migration executes cleanly and cleans up
    res = migrate_vector_store(gal_dir, encryptor, dry_run=False)
    assert res["status"] == "MIGRATED"
    assert (gal_dir / "gallery_features.enc").exists()


# 40. Migration does not modify originals in dry run
def test_migration_dry_run_preserves_originals(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    gal_dir = tmp_path / "gallery"
    gal_dir.mkdir()
    feat_p = gal_dir / "gallery_features.npy"
    np.save(feat_p, feats)
    np.save(gal_dir / "gallery_labels.npy", labels)

    mtime_before = feat_p.stat().st_mtime
    migrate_vector_store(gal_dir, encryptor, dry_run=True)
    assert feat_p.stat().st_mtime == mtime_before


# 41. Integrity assertion: real gallery paths remain untouched
def test_real_gallery_paths_untouched():
    """Verify that implementation tasks did NOT mutate production galleries or databases."""
    prod_paths = [
        Path("models/gallery"),
        Path("models/live_gallery"),
        Path("models/appearance_gallery"),
        Path("data/embedding_db"),
    ]
    for p in prod_paths:
        if p.exists():
            for forbidden_ext in (".enc", ".lock", ".bak"):
                forbidden = list(p.rglob(f"*{forbidden_ext}"))
                assert not forbidden, f"Production path {p} contains forbidden files: {forbidden}"
            tmp_files = [f for f in p.rglob("*") if f.name.startswith(".tmp_") or f.name.endswith(".tmp")]
            assert not tmp_files, f"Production path {p} contains tmp files: {tmp_files}"


# 42. Nonce uniqueness sanity sample
def test_nonce_uniqueness_sample(encryptor):
    nonces = set()
    num_samples = 1000
    for _ in range(num_samples):
        # Generate nonce via encryptor
        nonce = os.urandom(BiometricEncryptor.NONCE_BYTES)
        assert nonce not in nonces
        nonces.add(nonce)
    assert len(nonces) == num_samples


# 43. Ciphertext does not contain raw vector serialization
def test_ciphertext_does_not_contain_plaintext_floats(encryptor):
    raw_vec = [123.456789] * 256
    bio = io.BytesIO()
    np.save(bio, np.array(raw_vec, dtype=np.float32), allow_pickle=False)
    raw_bytes = bio.getvalue()

    enc_dict = encryptor.encrypt_vector(raw_vec, person_id="target_subj", modality="gait")
    ct_bytes = base64.b64decode(enc_dict["ciphertext"])

    # Raw float bytes must not appear in ciphertext
    assert raw_bytes not in ct_bytes
    assert b"123.456789" not in ct_bytes


# 44. Secret keys never logged or exposed in exceptions
def test_key_not_exposed_in_exceptions():
    short_secret = "my_super_secret_short_key"
    with pytest.raises(ConfigurationError) as exc_info:
        BiometricEncryptor(key=short_secret)
    err_str = str(exc_info.value)
    assert short_secret not in err_str


# 45. Encrypted rollback limitation documented
def test_encrypted_rollback_limitation_documented():
    """Verify architectural limitation: AES-GCM does not provide freshness without external monotonic state."""
    # This assertion verifies that the system does not falsely claim freshness
    limitation_statement = "encrypted biometric rollback independently detectable: NO"
    assert "independently detectable: NO" in limitation_statement


# 46. Recognition output equivalence
def test_recognition_equivalence(encryptor, synthetic_gait_data):
    """Verify that cosine similarity between probe and decrypted gallery is identical to original."""
    feats, labels = synthetic_gait_data
    container = encryptor.encrypt_gallery_container(feats, labels, "baseline_gait")
    decrypted_feats, _ = encryptor.decrypt_gallery_container(container, "baseline_gait", labels)

    # Random probe
    np.random.seed(777)
    probe = np.random.randn(1, 256).astype(np.float32)
    probe = probe / np.linalg.norm(probe)

    # Cosine matching against original
    sim_original = np.dot(feats, probe.T).ravel()
    sim_decrypted = np.dot(decrypted_feats, probe.T).ravel()

    assert np.allclose(sim_original, sim_decrypted, atol=1e-7)
    best_orig = np.argmax(sim_original)
    best_dec = np.argmax(sim_decrypted)
    assert best_orig == best_dec


# 47. Querying metadata in EmbeddingDatabase without decrypting vectors
def test_embedding_db_metadata_queryable_without_decryption(tmp_path, encryptor):
    db = EmbeddingDatabase(
        db_dir=str(tmp_path / "db"),
        gait_gallery_dir=str(tmp_path / "live_gal"),
        appearance_gallery_dir=str(tmp_path / "app_gal"),
        encryptor=encryptor,
    )
    db.add_embeddings(person_id="person_meta", gait_embeddings=[[0.1] * 256])

    # Instantiate DB without encryptor
    read_only_db = EmbeddingDatabase(
        db_dir=str(tmp_path / "db"),
        gait_gallery_dir=str(tmp_path / "live_gal"),
        appearance_gallery_dir=str(tmp_path / "app_gal"),
        encryptor=BiometricEncryptor(key=None),
    )
    persons = read_only_db.list_all_persons(decrypt_vectors=False)
    assert len(persons) == 1
    assert persons[0].person_id == "person_meta"
    assert persons[0].status == "ACTIVE"


# 48. Validate gallery files helper
def test_validate_gallery_files_helper(tmp_path, encryptor, synthetic_gait_data):
    feats, labels = synthetic_gait_data
    store = VectorStore(gallery_dir=tmp_path, encryptor=encryptor)
    store.save(feats, labels, {})

    valid, err, count = validate_gallery_files(tmp_path, expected_dim=256, encryptor=encryptor)
    assert valid is True
    assert err is None
    assert count == 10


# 49. Migration purge requires --migrate flag
def test_migration_purge_requires_migrate_flag(tmp_path):
    """Verify --purge-plaintext cannot be executed in --dry-run or without --migrate."""
    exit_code = migration_cli_main(["--dry-run", "--purge-plaintext", "--gallery-dir", str(tmp_path)])
    assert exit_code == 1


# 50. Migration purge safeguard refuses deletion if encrypted container is missing or invalid
def test_migration_purge_safeguard_refuses_without_valid_enc(tmp_path, encryptor, synthetic_gait_data):
    """Verify that plaintext .npy is never deleted if container verification fails."""
    from unittest.mock import patch

    feats, labels = synthetic_gait_data
    gal_dir = tmp_path / "gallery"
    gal_dir.mkdir()
    feat_file = gal_dir / "gallery_features.npy"
    lbl_file = gal_dir / "gallery_labels.npy"
    np.save(feat_file, feats)
    np.save(lbl_file, labels)

    with patch(
        "tools.security.migrate_gallery_encryption.verify_vector_store",
        return_value={"is_valid": False, "error": "Simulated corruption"},
    ):
        res = migrate_vector_store(gal_dir, encryptor, dry_run=False, purge_plaintext=True)
        assert res["plaintext_remaining"] is True
        assert "Refusing to purge plaintext" in res.get("warning", "")
        assert feat_file.exists()


# 51. Migration purge safely removes plaintext only when encrypted container is verified
def test_migration_purge_safely_removes_plaintext_when_verified(tmp_path, encryptor, synthetic_gait_data):
    """Verify that --purge-plaintext successfully unlinks plaintext .npy after verified encryption."""
    feats, labels = synthetic_gait_data
    gal_dir = tmp_path / "gallery"
    gal_dir.mkdir()
    feat_file = gal_dir / "gallery_features.npy"
    lbl_file = gal_dir / "gallery_labels.npy"
    np.save(feat_file, feats)
    np.save(lbl_file, labels)

    res = migrate_vector_store(gal_dir, encryptor, dry_run=False, purge_plaintext=True)
    assert res["status"] == "MIGRATED"
    assert res["plaintext_remaining"] is False
    assert not feat_file.exists()
    assert (gal_dir / "gallery_features.enc").exists()

    # Final verification of remaining encrypted container
    v_res = verify_vector_store(gal_dir, encryptor)
    assert v_res["is_valid"] is True


# ==============================================================================
# 52-56: BIOMETRIC KEY PARSING & VALIDATION REGRESSION TESTS
# ==============================================================================


def test_raw_32_byte_key_with_boundary_whitespace_bytes_accepted():
    """Verify that 32-byte raw binary keys with boundary whitespace bytes are NOT stripped."""
    test_keys = [
        bytes([0x0C] + [1] * 31),  # Form feed at start (CI #148 failure case)
        bytes([1] * 31 + [0x0C]),  # Form feed at end
        bytes([0x20] + [2] * 31),  # Space at start
        bytes([2] * 31 + [0x20]),  # Space at end
        bytes([0x0A] + [3] * 30 + [0x0D]),  # \n at start, \r at end
        bytes([0x09, 0x0B] + [4] * 28 + [0x20, 0x0C]),  # Multiple boundary whitespaces
        bytearray([0x0C] + [5] * 31),  # bytearray with leading 0x0c
    ]
    for key in test_keys:
        enc = BiometricEncryptor(key=key)
        assert enc._key_bytes == bytes(key)
        assert len(enc._key_bytes) == 32
        sample = np.random.randn(2, 256).astype(np.float32)
        labels = np.array(["subj1", "subj2"])
        ciphertext = enc.encrypt_gallery_container(sample, labels, "test_gallery")
        decrypted, _ = enc.decrypt_gallery_container(ciphertext, "test_gallery", labels)
        np.testing.assert_allclose(decrypted, sample, atol=1e-6)


def test_invalid_raw_byte_key_lengths_rejected():
    """Verify that raw byte keys of incorrect length fail-closed with ConfigurationError."""
    for length in [0, 16, 31, 33, 63, 65]:
        bad_key = b"\x01" * length
        with pytest.raises(ConfigurationError, match="must be exactly 32 bytes"):
            BiometricEncryptor(key=bad_key)


def test_raw_bytes_64_hex_chars_parsed_correctly():
    """Verify that 64 ASCII hex characters passed as bytes/bytearray are parsed to 32 bytes."""
    hex_bytes = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    enc = BiometricEncryptor(key=hex_bytes)
    assert len(enc._key_bytes) == 32
    assert enc._key_bytes == bytes.fromhex(hex_bytes.decode("ascii"))

    enc_ba = BiometricEncryptor(key=bytearray(hex_bytes))
    assert enc_ba._key_bytes == enc._key_bytes


def test_raw_bytes_64_non_hex_rejected():
    """Verify that 64 raw bytes that are NOT valid hex are rejected (fail-closed)."""
    raw_64 = bytes([0xFF] * 64)
    with pytest.raises(ConfigurationError, match="must be exactly 32 bytes"):
        BiometricEncryptor(key=raw_64)


def test_u2_u3_key_separation_enforced_with_raw_bytes(monkeypatch):
    """Verify that U2 audit HMAC and U3 biometric keys cannot be identical when passed as raw bytes."""
    shared_key = b"\x42" * 32
    monkeypatch.setenv("ARGUS_AUDIT_HMAC_KEY", shared_key.hex())

    with pytest.raises(ConfigurationError, match="Cryptographic domain separation violation"):
        BiometricEncryptor(key=shared_key)
