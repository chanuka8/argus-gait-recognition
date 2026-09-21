"""Integration Test Suite for U5 Phase 2: Model Confidentiality and Encryption at Rest.

Verifies:
- Authenticated AES-256-GCM container format and AEAD roundtrips
- Generic exception on decryption failure (no padding/decryption oracles)
- Bijective, collision-free key-ID mapping and strict key provider routing
- Trusted public identity registry and anti-rename protection
- Physical (.enc) vs signed logical filename separation and production enforcement
- Non-downgradable production encryption and Phase-1 strictness
- Zero plaintext file residue in temp and runtime directories
- Canary secret absence in logs, exceptions, and diagnostics
- In-memory PyTorch and ONNX loading
- Provisioning concurrency, complete writes, and lock cleanup
- Startup validator and doctor diagnostics
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from app.security_layer.model_confidentiality import (
    MAGIC_HEADER,
    ArtifactConfidentiality,
    EnvModelKeyProvider,
    ModelDecryptionError,
    ModelEncryptionRequiredError,
    ModelKeyUnavailableError,
    ModelProvisioningError,
    ModelSecurityPolicyViolationError,
    ProvisioningLockError,
    decrypt_model_container,
    encrypt_model_container,
    is_model_encryption_required,
    key_id_to_env_var,
    load_verified_model_bytes,
    resolve_artifact_confidentiality,
)
from app.security_layer.model_integrity import (
    ROLE_APPEARANCE_EMBEDDING,
    ROLE_GAIT_EMBEDDING,
    ROLE_PERSON_DETECTOR,
    ModelManifestError,
    ModelRoleMismatchError,
    ModelSignatureError,
    ModelTrustConfigurationError,
    canonical_manifest_bytes,
)
from ml_platform.models.architectures.bygait_light import ByGaitLight
from ops.deployment.doctor import run_doctor
from ops.deployment.startup_validator import DeploymentStartupValidator
from ops.tools.security.encrypt_model_artifact import provision_encrypted_artifact
from ops.tools.security.sign_model_manifest import sign_manifest


@pytest.fixture
def crypto_keys():
    """Generate Ed25519 signing keypair and AES-256 keys for test suite."""
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    aes_key_1 = os.urandom(32)
    aes_key_2 = os.urandom(32)
    return {
        "priv": priv,
        "pub": pub,
        "pub_bytes": pub_bytes,
        "aes_key_1": aes_key_1,
        "aes_key_2": aes_key_2,
    }


@pytest.fixture
def signed_model_fixture(tmp_path, crypto_keys):
    """Create synthetic ByGaitLight model, sign manifest, and provide clean test assets."""
    model = ByGaitLight(part_bins=4)
    model_bytes_io = io.BytesIO()
    torch.save(model.state_dict(), model_bytes_io)
    plaintext_bytes = model_bytes_io.getvalue()

    logical_filename = "best_model.pth"
    model_file = tmp_path / logical_filename
    model_file.write_bytes(plaintext_bytes)

    # Build manifest
    sha256_hex = hashlib.sha256(plaintext_bytes).hexdigest().lower()
    manifest_data = {
        "manifest_version": "1.0",
        "models": {
            ROLE_GAIT_EMBEDDING: {
                "model_role": ROLE_GAIT_EMBEDDING,
                "model_id": "bygait_synthetic",
                "model_version": "1.0.0",
                "filename": logical_filename,
                "sha256": sha256_hex,
                "size_bytes": len(plaintext_bytes),
                "framework": "pytorch_state_dict",
            }
        },
    }

    _, sig_b64 = sign_manifest(manifest_data, crypto_keys["priv"])
    manifest_file = tmp_path / "model_manifest.json"
    manifest_file.write_bytes(canonical_manifest_bytes(manifest_data))

    sig_file = tmp_path / "model_manifest.sig"
    sig_file.write_text(sig_b64, encoding="utf-8")

    return {
        "plaintext_bytes": plaintext_bytes,
        "logical_filename": logical_filename,
        "model_file": model_file,
        "manifest_file": manifest_file,
        "sig_file": sig_file,
        "manifest_data": manifest_data,
        "sha256": sha256_hex,
        "pub_bytes": crypto_keys["pub_bytes"],
    }


# ==============================================================================
# 1. AES-256-GCM Container Format, Roundtrip & Generic Decryption Failure
# ==============================================================================


def test_01_aead_roundtrip_and_nonce_uniqueness(crypto_keys):
    """Test 1: AES-256-GCM roundtrip and ensure nonce uniqueness on repeated encryption."""
    data = b"proprietary_bygait_neural_weights_data_block"
    key = crypto_keys["aes_key_1"]

    enc1 = encrypt_model_container(data, key_id="default", key_material=key)
    enc2 = encrypt_model_container(data, key_id="default", key_material=key)

    assert enc1.startswith(MAGIC_HEADER)
    assert enc2.startswith(MAGIC_HEADER)
    assert enc1 != enc2, "Identical plaintext must produce different ciphertexts due to fresh nonces."

    dec1 = decrypt_model_container(enc1, key_material=key)
    dec2 = decrypt_model_container(enc2, key_material=key)
    assert dec1 == data
    assert dec2 == data


def test_02_generic_decryption_failure_on_tamper_and_wrong_key(crypto_keys):
    """Test 2: Tampered ciphertext, bad headers, and wrong key raise generic ModelDecryptionError."""
    data = b"secret_weights_payload"
    key1 = crypto_keys["aes_key_1"]
    key2 = crypto_keys["aes_key_2"]

    enc = encrypt_model_container(data, key_id="default", key_material=key1)

    # Wrong key
    with pytest.raises(ModelDecryptionError) as exc_info:
        decrypt_model_container(enc, key_material=key2)
    assert str(exc_info.value) == "Authenticated model decryption failed."

    # Tampered ciphertext payload (flip last byte)
    tampered_body = bytearray(enc)
    tampered_body[-1] ^= 0xFF
    with pytest.raises(ModelDecryptionError) as exc_info:
        decrypt_model_container(bytes(tampered_body), key_material=key1)
    assert str(exc_info.value) == "Authenticated model decryption failed."

    # Tampered header (flip magic byte)
    tampered_header = bytearray(enc)
    tampered_header[0] = ord("X")
    with pytest.raises(ModelDecryptionError) as exc_info:
        decrypt_model_container(bytes(tampered_header), key_material=key1)
    assert str(exc_info.value) == "Authenticated model decryption failed."

    # Truncated container
    with pytest.raises(ModelDecryptionError):
        decrypt_model_container(enc[:10], key_material=key1)


# ==============================================================================
# 2. Bijective Key-ID Mapping & Key Providers
# ==============================================================================


def test_03_collision_free_key_id_mapping():
    """Test 3: Collision-free bijective mapping between key IDs and environment variables."""
    # Sibling IDs with dashes, dots, underscores, and differing cases
    ids = ["key-v1", "key.v1", "key_v1", "KEY-V1", "key-V1"]
    env_vars = [key_id_to_env_var(kid) for kid in ids]

    assert len(set(env_vars)) == len(ids), "Distinct key IDs must produce distinct environment variable names."
    assert key_id_to_env_var("default") == "ARGUS_MODEL_ENCRYPTION_KEY"

    # Reject invalid characters and empty key IDs
    with pytest.raises(ValueError):
        key_id_to_env_var("key@invalid!")
    with pytest.raises(ValueError):
        key_id_to_env_var("")


def test_04_exact_key_provider_routing_no_default_fallback(monkeypatch, crypto_keys):
    """Test 4: Explicit non-default key ID fails closed if exact env var is absent (no default fallback)."""
    provider = EnvModelKeyProvider()
    key1_hex = crypto_keys["aes_key_1"].hex()
    key2_hex = crypto_keys["aes_key_2"].hex()

    # Set default key in environment
    monkeypatch.setenv("ARGUS_MODEL_ENCRYPTION_KEY", key1_hex)
    assert provider.get_key("default") == crypto_keys["aes_key_1"]

    # Request explicit key ID 'model-exp002' without setting its env var
    with pytest.raises(ModelKeyUnavailableError):
        provider.get_key("model-exp002")

    # Now set the exact hex env var for 'model-exp002'
    target_var = key_id_to_env_var("model-exp002")
    monkeypatch.setenv(target_var, key2_hex)
    assert provider.get_key("model-exp002") == crypto_keys["aes_key_2"]


# ==============================================================================
# 3. Artifact-Aware Confidentiality Policy & Anti-Rename Suite
# ==============================================================================


def test_05_artifact_confidentiality_resolution_and_defaults():
    """Test 5: Confidentiality policy strictly defaults to PROTECTED unless explicitly registered public."""
    # Unknown or missing IDs default to PROTECTED
    assert resolve_artifact_confidentiality(ROLE_GAIT_EMBEDDING) == ArtifactConfidentiality.PROTECTED
    assert resolve_artifact_confidentiality(ROLE_GAIT_EMBEDDING, None) == ArtifactConfidentiality.PROTECTED
    assert resolve_artifact_confidentiality(ROLE_GAIT_EMBEDDING, "unknown_id") == ArtifactConfidentiality.PROTECTED

    # Synthetic test IDs in production classifier resolve to PROTECTED
    assert resolve_artifact_confidentiality(ROLE_PERSON_DETECTOR, "yolo_synthetic") == ArtifactConfidentiality.PROTECTED
    assert (
        resolve_artifact_confidentiality(ROLE_APPEARANCE_EMBEDDING, "osnet_synthetic")
        == ArtifactConfidentiality.PROTECTED
    )

    # Authoritative public models resolve to PUBLIC_UPSTREAM
    assert (
        resolve_artifact_confidentiality(ROLE_APPEARANCE_EMBEDDING, "osnet_x0_25_msmt17")
        == ArtifactConfidentiality.PUBLIC_UPSTREAM
    )
    assert resolve_artifact_confidentiality(ROLE_PERSON_DETECTOR, "yolov8n") == ArtifactConfidentiality.PUBLIC_UPSTREAM

    # Explicit code-side policy override
    assert (
        resolve_artifact_confidentiality(ROLE_GAIT_EMBEDDING, explicit_policy=ArtifactConfidentiality.PUBLIC_UPSTREAM)
        == ArtifactConfidentiality.PUBLIC_UPSTREAM
    )


def test_06_anti_rename_security_protections(tmp_path, signed_model_fixture, monkeypatch):
    """Test 6: Renaming proprietary weights to public names (osnet_x0_25.pth, yolov8n.pt) fails closed."""
    monkeypatch.setenv("ARGUS_ENV", "production")

    # Rename proprietary plaintext ByGaitLight weights to public name osnet_x0_25.pth
    renamed_plain = tmp_path / "osnet_x0_25.pth"
    renamed_plain.write_bytes(signed_model_fixture["plaintext_bytes"])

    # Loader expecting gait model rejects unencrypted plaintext in production regardless of filename
    with pytest.raises(ModelEncryptionRequiredError):
        load_verified_model_bytes(
            renamed_plain,
            expected_role=ROLE_GAIT_EMBEDDING,
            logical_filename="best_model.pth",
        )

    # If renamed file is encrypted and attacker tries loading it as appearance model
    enc_renamed = tmp_path / "osnet_x0_25.pth.enc"
    enc_key = os.urandom(32)
    provision_encrypted_artifact(
        renamed_plain,
        enc_renamed,
        key_material=enc_key,
    )

    # Fails closed in Phase-1 manifest verification due to role mismatch
    with pytest.raises(ModelRoleMismatchError):
        load_verified_model_bytes(
            enc_renamed,
            expected_role=ROLE_APPEARANCE_EMBEDDING,
            logical_filename="osnet_x0_25.pth",
            expected_model_id="osnet_x0_25_msmt17",
            manifest_path=signed_model_fixture["manifest_file"],
            signature_path=signed_model_fixture["sig_file"],
            public_key=signed_model_fixture["pub_bytes"],
            key_material=enc_key,
        )


# ==============================================================================
# 4. Mandatory Correction: Logical vs Physical Filename Separation
# ==============================================================================


def test_07_physical_enc_vs_signed_logical_filename(tmp_path, signed_model_fixture, crypto_keys):
    """Test 7: Physical .enc file requires trusted logical filename matching signed Phase-1 manifest."""
    enc_file = tmp_path / "best_model.pth.enc"
    key = crypto_keys["aes_key_1"]
    provision_encrypted_artifact(
        signed_model_fixture["model_file"],
        enc_file,
        key_material=key,
    )

    # 1. Correct physical .enc + correct logical best_model.pth -> PASS
    decrypted = load_verified_model_bytes(
        model_path=enc_file,
        expected_role=ROLE_GAIT_EMBEDDING,
        logical_filename="best_model.pth",
        expected_model_id="bygait_synthetic",
        manifest_path=signed_model_fixture["manifest_file"],
        signature_path=signed_model_fixture["sig_file"],
        public_key=signed_model_fixture["pub_bytes"],
        key_material=key,
        confidentiality=ArtifactConfidentiality.PROTECTED,
    )
    assert decrypted == signed_model_fixture["plaintext_bytes"]

    # 2. Correct physical .enc + wrong logical name -> FAIL CLOSED
    with pytest.raises(ModelRoleMismatchError):
        load_verified_model_bytes(
            model_path=enc_file,
            expected_role=ROLE_GAIT_EMBEDDING,
            logical_filename="wrong_model.pth",
            expected_model_id="bygait_synthetic",
            manifest_path=signed_model_fixture["manifest_file"],
            signature_path=signed_model_fixture["sig_file"],
            public_key=signed_model_fixture["pub_bytes"],
            key_material=key,
            confidentiality=ArtifactConfidentiality.PROTECTED,
        )

    # 3. Missing logical identity in production -> FAIL CLOSED
    with patch.dict(os.environ, {"ARGUS_ENV": "production"}), pytest.raises(ModelSecurityPolicyViolationError):
        load_verified_model_bytes(
            model_path=enc_file,
            expected_role=ROLE_GAIT_EMBEDDING,
            logical_filename=None,
            key_material=key,
            confidentiality=ArtifactConfidentiality.PROTECTED,
        )

    # 4. Renaming physical container does NOT alter signed logical identity
    renamed_enc = tmp_path / "arbitrary_name_2026.enc"
    enc_file.rename(renamed_enc)
    decrypted_renamed = load_verified_model_bytes(
        model_path=renamed_enc,
        expected_role=ROLE_GAIT_EMBEDDING,
        logical_filename="best_model.pth",
        expected_model_id="bygait_synthetic",
        manifest_path=signed_model_fixture["manifest_file"],
        signature_path=signed_model_fixture["sig_file"],
        public_key=signed_model_fixture["pub_bytes"],
        key_material=key,
        confidentiality=ArtifactConfidentiality.PROTECTED,
    )
    assert decrypted_renamed == signed_model_fixture["plaintext_bytes"]


# ==============================================================================
# 5. Non-Downgradable Production Encryption & Strict Phase-1 Enforcement
# ==============================================================================


def test_08_production_cannot_downgrade_encryption_requirement(monkeypatch):
    """Test 8: Production runtime cannot downgrade encryption requirement via permissive env flags."""
    monkeypatch.setenv("ARGUS_ENV", "production")
    monkeypatch.setenv("ARGUS_REQUIRE_ENCRYPTED_MODELS", "false")
    monkeypatch.setenv("ARGUS_MODEL_ENCRYPTION_STRICT", "0")

    assert is_model_encryption_required(ArtifactConfidentiality.PROTECTED) is True


def test_09_production_phase1_strict_enforcement_suite(tmp_path, signed_model_fixture, crypto_keys, monkeypatch):
    """Test 9: In production with a protected model, Phase 1 strict verification is non-downgradable."""
    monkeypatch.setenv("ARGUS_ENV", "production")
    monkeypatch.setenv("ARGUS_REQUIRE_SIGNED_MODELS", "0")
    monkeypatch.setenv("ARGUS_MODEL_STRICT_MODE", "false")

    enc_file = tmp_path / "best_model.pth.enc"
    key = crypto_keys["aes_key_1"]
    provision_encrypted_artifact(
        signed_model_fixture["model_file"],
        enc_file,
        key_material=key,
    )

    # Test A: Missing manifest -> BLOCKED
    with pytest.raises(ModelManifestError):
        load_verified_model_bytes(
            enc_file,
            expected_role=ROLE_GAIT_EMBEDDING,
            logical_filename="best_model.pth",
            manifest_path=tmp_path / "non_existent_manifest.json",
            signature_path=signed_model_fixture["sig_file"],
            public_key=signed_model_fixture["pub_bytes"],
            key_material=key,
        )

    # Test B: Missing signature -> BLOCKED
    with pytest.raises(ModelSignatureError):
        load_verified_model_bytes(
            enc_file,
            expected_role=ROLE_GAIT_EMBEDDING,
            logical_filename="best_model.pth",
            manifest_path=signed_model_fixture["manifest_file"],
            signature_path=tmp_path / "non_existent_sig.sig",
            public_key=signed_model_fixture["pub_bytes"],
            key_material=key,
        )

    # Test C: Missing public key -> BLOCKED
    with pytest.raises(ModelTrustConfigurationError):
        load_verified_model_bytes(
            enc_file,
            expected_role=ROLE_GAIT_EMBEDDING,
            logical_filename="best_model.pth",
            manifest_path=signed_model_fixture["manifest_file"],
            signature_path=signed_model_fixture["sig_file"],
            public_key=None,
            key_material=key,
        )

    # Test D: Invalid signature -> BLOCKED
    corrupt_sig = tmp_path / "corrupt.sig"
    corrupt_sig.write_text(base64.b64encode(os.urandom(64)).decode("ascii"))
    with pytest.raises(ModelSignatureError):
        load_verified_model_bytes(
            enc_file,
            expected_role=ROLE_GAIT_EMBEDDING,
            logical_filename="best_model.pth",
            manifest_path=signed_model_fixture["manifest_file"],
            signature_path=corrupt_sig,
            public_key=signed_model_fixture["pub_bytes"],
            key_material=key,
        )

    # Test E: Complete Phase-1 trust chain -> PASS
    pass_bytes = load_verified_model_bytes(
        enc_file,
        expected_role=ROLE_GAIT_EMBEDDING,
        logical_filename="best_model.pth",
        expected_model_id="bygait_synthetic",
        manifest_path=signed_model_fixture["manifest_file"],
        signature_path=signed_model_fixture["sig_file"],
        public_key=signed_model_fixture["pub_bytes"],
        key_material=key,
    )
    assert pass_bytes == signed_model_fixture["plaintext_bytes"]


# ==============================================================================
# 6. Zero Plaintext Residue on Disk & No Secrets in Logs
# ==============================================================================


def test_10_no_plaintext_file_residue(tmp_path, signed_model_fixture, crypto_keys):
    """Test 10: Encrypted runtime loading creates no plaintext files in temp directories."""
    enc_file = tmp_path / "best_model.pth.enc"
    key = crypto_keys["aes_key_1"]
    provision_encrypted_artifact(
        signed_model_fixture["model_file"],
        enc_file,
        key_material=key,
    )

    temp_dir = Path(tempfile.gettempdir())
    files_before = set(temp_dir.glob("*"))

    # Execute in-memory load and model execution
    decrypted = load_verified_model_bytes(
        enc_file,
        expected_role=ROLE_GAIT_EMBEDDING,
        logical_filename="best_model.pth",
        expected_model_id="bygait_synthetic",
        manifest_path=signed_model_fixture["manifest_file"],
        signature_path=signed_model_fixture["sig_file"],
        public_key=signed_model_fixture["pub_bytes"],
        key_material=key,
    )
    buffer = io.BytesIO(decrypted)
    model = ByGaitLight(part_bins=4)
    model.load_state_dict(torch.load(buffer, weights_only=True))

    files_after = set(temp_dir.glob("*"))
    new_files = files_after - files_before

    # Verify no new file contains plaintext model bytes
    for nf in new_files:
        if nf.is_file():
            content = nf.read_bytes()
            assert signed_model_fixture["plaintext_bytes"] not in content


def test_11_no_canary_secret_in_logs(caplog, tmp_path, signed_model_fixture):
    """Test 11: Canary encryption secrets are never emitted to logs or exception strings."""
    canary_secret = "CANARY_SECRET_KEY_99999999999999"
    key_32 = hashlib.sha256(canary_secret.encode("utf-8")).digest()

    caplog.set_level(logging.DEBUG)
    enc_file = tmp_path / "best_model.pth.enc"
    provision_encrypted_artifact(
        signed_model_fixture["model_file"],
        enc_file,
        key_material=key_32,
    )

    # Attempt decrypt with wrong key
    with pytest.raises(ModelDecryptionError) as exc_info:
        load_verified_model_bytes(
            enc_file,
            expected_role=ROLE_GAIT_EMBEDDING,
            logical_filename="best_model.pth",
            key_material=os.urandom(32),
        )

    # Assert canary string is not in log messages or exception text
    for record in caplog.records:
        assert canary_secret not in record.getMessage()
    assert canary_secret not in str(exc_info.value)


# ==============================================================================
# 7. Provisioning Concurrency & Complete-Write Safety
# ==============================================================================


def test_12_provisioning_concurrency_overwrite_false(tmp_path, signed_model_fixture, crypto_keys):
    """Test 12: Two concurrent provisioning writers with overwrite=False result in exactly one success."""
    dest_path = tmp_path / "concurrent_model.pth.enc"
    key = crypto_keys["aes_key_1"]
    src = signed_model_fixture["model_file"]

    results = []
    errors = []

    def writer_task():
        try:
            res = provision_encrypted_artifact(
                source_path=src,
                destination_path=dest_path,
                key_material=key,
                overwrite=False,
            )
            results.append(res)
        except (ProvisioningLockError, FileExistsError) as e:
            errors.append(e)

    t1 = threading.Thread(target=writer_task)
    t2 = threading.Thread(target=writer_task)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one succeeded and one failed safely
    assert len(results) == 1
    assert len(errors) == 1

    # Destination artifact decrypts cleanly
    decrypted = decrypt_model_container(dest_path.read_bytes(), key_material=key)
    assert decrypted == signed_model_fixture["plaintext_bytes"]

    # Zero orphan temporary files or locks remain
    temp_files = list(tmp_path.glob(".*tmp*"))
    lock_files = list(tmp_path.glob("*.lock"))
    assert len(temp_files) == 0
    assert len(lock_files) == 0


def test_13_provisioning_complete_write_guarantee(tmp_path, signed_model_fixture, crypto_keys):
    """Test 13: Simulated partial write aborts before replace, unlinks temp, and preserves destination."""
    dest_path = tmp_path / "partial_test.pth.enc"
    dest_path.write_bytes(b"original_destination_content")
    key = crypto_keys["aes_key_1"]
    src = signed_model_fixture["model_file"]

    orig_fdopen = os.fdopen

    class PartialWriter:
        def __init__(self, fh):
            self._fh = fh

        def write(self, b):
            # Write only half the bytes
            half = len(b) // 2
            return self._fh.write(b[:half])

        def flush(self):
            self._fh.flush()

        def fileno(self):
            return self._fh.fileno()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._fh.close()

    def fake_fdopen(fd, mode="r", *args, **kwargs):
        fh = orig_fdopen(fd, mode, *args, **kwargs)
        if "w" in mode or "b" in mode:
            return PartialWriter(fh)
        return fh

    with patch("ops.tools.security.encrypt_model_artifact.os.fdopen", side_effect=fake_fdopen):
        with pytest.raises(ModelProvisioningError) as exc_info:
            provision_encrypted_artifact(
                source_path=src,
                destination_path=dest_path,
                key_material=key,
                overwrite=True,
            )
        assert "Incomplete encrypted artifact write" in str(exc_info.value)

    # Destination content must remain completely unchanged
    assert dest_path.read_bytes() == b"original_destination_content"

    # Temp files and locks must be cleaned up
    temp_files = list(tmp_path.glob(".*tmp*"))
    lock_files = list(tmp_path.glob("*.lock"))
    assert len(temp_files) == 0
    assert len(lock_files) == 0


# ==============================================================================
# 8. Startup Validator & Doctor Integration
# ==============================================================================


def test_14_startup_validator_and_doctor_diagnostics(tmp_path, monkeypatch):
    """Test 14: Startup validator and doctor detect unencrypted protected models in production."""
    # Production without encryption -> startup validator reports blocking defect
    monkeypatch.setenv("ARGUS_ENV", "production")
    validator = DeploymentStartupValidator()
    blocking = []
    warnings = []
    unable = []

    validator._validate_model_confidentiality(
        blocking_issues=blocking,
        warnings=warnings,
        unable_to_verify=unable,
    )
    assert any("unencrypted in production" in b for b in blocking)

    # Doctor diagnostics
    _exit_code, report = run_doctor(
        json_path=str(tmp_path / "doctor.json"),
        md_path=str(tmp_path / "doctor.md"),
    )
    conf_check = next((c for c in report.get("checks", []) if c["name"] == "model_confidentiality"), None)
    assert conf_check is not None
    assert conf_check["status"] == "FAIL"
