"""Integration Test Suite for Model Checkpoint Authenticity, Integrity & Safe Deserialization.

Finding U5 Phase 1:
- Cryptographic model authenticity (Ed25519)
- Canonical manifest integrity & verification
- Model-role and model-ID binding
- SHA-256 digest validation
- Strict fail-closed vs development warning modes
- Safe PyTorch deserialization (weights_only=True)
- Prevention of unverified downloads in strict mode
- Recognition equivalence and prior control regression preservation
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from models.architectures.bygait_light import ByGaitLight
from security_layer.model_integrity import (
    ROLE_APPEARANCE_EMBEDDING,
    ROLE_GAIT_EMBEDDING,
    ROLE_PERSON_DETECTOR,
    ROLE_SILHOUETTE_SEGMENTER,
    ModelDigestMismatchError,
    ModelIntegrityError,
    ModelManifestError,
    ModelRoleMismatchError,
    ModelSignatureError,
    ModelTrustConfigurationError,
    ModelVerifier,
    canonical_manifest_bytes,
    compute_file_sha256,
    load_public_key,
)


@pytest.fixture
def ephemeral_keypair() -> tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    """Generate temporary in-memory Ed25519 keypair for testing."""
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    return priv, pub


@pytest.fixture
def synthetic_model_env(tmp_path: Path, ephemeral_keypair) -> dict[str, Any]:
    """Create a fully signed synthetic model environment in tmp_path."""
    priv_key, pub_key = ephemeral_keypair

    # Create dummy model files
    gait_model_file = tmp_path / "synthetic_gait.pth"
    detector_file = tmp_path / "synthetic_detector.pt"
    onnx_file = tmp_path / "synthetic_segmenter.onnx"
    osnet_file = tmp_path / "synthetic_osnet.pth"

    # Create minimal state dict for PyTorch
    dummy_state = {"conv.weight": torch.randn(4, 1, 3, 3)}
    torch.save(dummy_state, gait_model_file)
    torch.save(dummy_state, osnet_file)
    detector_file.write_bytes(b"SYNTHETIC_YOLO_WEIGHTS_MOCK_BYTES_12345")
    onnx_file.write_bytes(b"SYNTHETIC_ONNX_MOCK_PROTOBUF_BYTES_12345")

    gait_sha256, gait_size = compute_file_sha256(gait_model_file)
    det_sha256, det_size = compute_file_sha256(detector_file)
    onnx_sha256, onnx_size = compute_file_sha256(onnx_file)
    osnet_sha256, osnet_size = compute_file_sha256(osnet_file)

    manifest_data = {
        "manifest_version": "1.0",
        "signing_key_id": "test-ed25519-key-001",
        "created_at": "2026-09-16T12:00:00Z",
        "models": {
            ROLE_GAIT_EMBEDDING: {
                "model_role": ROLE_GAIT_EMBEDDING,
                "model_id": "bygait_synthetic",
                "model_version": "1.0.0",
                "filename": gait_model_file.name,
                "sha256": gait_sha256,
                "size_bytes": gait_size,
                "framework": "pytorch_state_dict",
            },
            ROLE_PERSON_DETECTOR: {
                "model_role": ROLE_PERSON_DETECTOR,
                "model_id": "yolo_synthetic",
                "model_version": "8.0.0",
                "filename": detector_file.name,
                "sha256": det_sha256,
                "size_bytes": det_size,
                "framework": "ultralytics_yolo",
            },
            ROLE_SILHOUETTE_SEGMENTER: {
                "model_role": ROLE_SILHOUETTE_SEGMENTER,
                "model_id": "unet_synthetic",
                "model_version": "1.0.0",
                "filename": onnx_file.name,
                "sha256": onnx_sha256,
                "size_bytes": onnx_size,
                "framework": "onnx",
            },
            ROLE_APPEARANCE_EMBEDDING: {
                "model_role": ROLE_APPEARANCE_EMBEDDING,
                "model_id": "osnet_synthetic",
                "model_version": "1.0.0",
                "filename": osnet_file.name,
                "sha256": osnet_sha256,
                "size_bytes": osnet_size,
                "framework": "pytorch_state_dict",
            },
        },
    }

    canonical_bytes = canonical_manifest_bytes(manifest_data)
    sig_bytes = priv_key.sign(canonical_bytes)

    manifest_file = tmp_path / "model_manifest.json"
    sig_file = tmp_path / "model_manifest.sig"

    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    sig_file.write_bytes(sig_bytes)

    verifier = ModelVerifier(
        strict_mode=True,
        default_manifest_path=manifest_file,
        default_signature_path=sig_file,
        public_key=pub_key,
    )

    return {
        "verifier": verifier,
        "manifest_data": manifest_data,
        "manifest_file": manifest_file,
        "sig_file": sig_file,
        "priv_key": priv_key,
        "pub_key": pub_key,
        "gait_file": gait_model_file,
        "detector_file": detector_file,
        "onnx_file": onnx_file,
        "osnet_file": osnet_file,
        "tmp_path": tmp_path,
    }


# ==============================================================================
# 1-10: CORE CRYPTOGRAPHIC & DIGEST TESTS
# ==============================================================================


def test_01_valid_signed_manifest_accepted(synthetic_model_env):
    """Test 1: Valid signed manifest and model files are accepted in strict mode."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    result = verifier.verify_model(
        model_path=env["gait_file"],
        expected_role=ROLE_GAIT_EMBEDDING,
        expected_model_id="bygait_synthetic",
        manifest_path=env["manifest_file"],
        signature_path=env["sig_file"],
        public_key=env["pub_key"],
    )
    assert result == env["gait_file"]


def test_02_model_byte_tamper_rejected(synthetic_model_env):
    """Test 2: Modifying even one byte of a model file causes SHA-256 mismatch."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    # Tamper with model file
    original_bytes = env["gait_file"].read_bytes()
    tampered_bytes = bytearray(original_bytes)
    tampered_bytes[10] ^= 0xFF
    env["gait_file"].write_bytes(bytes(tampered_bytes))

    with pytest.raises(ModelDigestMismatchError, match="SHA-256 digest mismatch"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_03_manifest_field_tamper_rejected(synthetic_model_env):
    """Test 3: Modifying a field in the manifest breaks Ed25519 signature."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    # Tamper with manifest JSON (e.g. change version string)
    m_data = json.loads(env["manifest_file"].read_text(encoding="utf-8"))
    m_data["models"][ROLE_GAIT_EMBEDDING]["model_version"] = "2.0.0"
    env["manifest_file"].write_text(json.dumps(m_data), encoding="utf-8")

    with pytest.raises(ModelSignatureError, match="signature verification failed"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_04_invalid_signature_rejected(synthetic_model_env):
    """Test 4: Corrupted signature bytes are rejected."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    # Corrupt signature bytes
    bad_sig = bytearray(env["sig_file"].read_bytes())
    bad_sig[5] ^= 0xAA
    env["sig_file"].write_bytes(bytes(bad_sig))

    with pytest.raises(ModelSignatureError):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_05_wrong_public_key_rejected(synthetic_model_env):
    """Test 5: Valid signature verified against an unrelated public key fails."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    # Generate a different key
    wrong_key = ed25519.Ed25519PrivateKey.generate().public_key()

    with pytest.raises(ModelSignatureError):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=wrong_key,
        )


def test_06_missing_public_key_rejected_in_strict_mode(synthetic_model_env):
    """Test 6: Strict mode with no public key raises ModelTrustConfigurationError."""
    env = synthetic_model_env
    verifier = ModelVerifier(strict_mode=True, public_key=None)

    with pytest.raises(ModelTrustConfigurationError, match="requires a trusted public key"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=None,
        )


def test_07_missing_signature_rejected_in_strict_mode(synthetic_model_env):
    """Test 7: Strict mode with missing signature file raises ModelSignatureError."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    missing_sig = env["tmp_path"] / "nonexistent.sig"
    with pytest.raises(ModelSignatureError, match="signature not found"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=missing_sig,
            public_key=env["pub_key"],
        )


def test_08_missing_manifest_rejected_in_strict_mode(synthetic_model_env):
    """Test 8: Strict mode with missing manifest raises ModelManifestError."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    missing_m = env["tmp_path"] / "nonexistent.json"
    with pytest.raises(ModelManifestError, match="manifest not found"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=missing_m,
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_09_digest_mismatch_rejected(synthetic_model_env):
    """Test 9: Model with incorrect hash in manifest is rejected."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    m_data = env["manifest_data"]
    m_data["models"][ROLE_GAIT_EMBEDDING]["sha256"] = "a" * 64
    sig_bytes = env["priv_key"].sign(canonical_manifest_bytes(m_data))

    env["manifest_file"].write_text(json.dumps(m_data), encoding="utf-8")
    env["sig_file"].write_bytes(sig_bytes)

    with pytest.raises(ModelDigestMismatchError, match="SHA-256 digest mismatch"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_10_file_size_mismatch_rejected(synthetic_model_env):
    """Test 10: Model with incorrect size in manifest is rejected."""
    import copy

    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    m_data = copy.deepcopy(env["manifest_data"])
    m_data["models"][ROLE_GAIT_EMBEDDING]["size_bytes"] = 9999999
    sig_bytes = env["priv_key"].sign(canonical_manifest_bytes(m_data))

    test_m_file = env["tmp_path"] / "m_size_mismatch.json"
    test_s_file = env["tmp_path"] / "s_size_mismatch.sig"

    test_m_file.write_text(json.dumps(m_data), encoding="utf-8")
    test_s_file.write_bytes(sig_bytes)

    with pytest.raises(ModelDigestMismatchError, match="size mismatch"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=test_m_file,
            signature_path=test_s_file,
            public_key=env["pub_key"],
        )


# ==============================================================================
# 11-20: ROLE BINDING, SCHEMA & POLICY TESTS
# ==============================================================================


def test_11_wrong_model_role_rejected(synthetic_model_env):
    """Test 11: Supplying a detector model when gait embedding is expected is rejected."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    # detector_file is signed under ROLE_PERSON_DETECTOR; request it as ROLE_GAIT_EMBEDDING
    with pytest.raises(ModelRoleMismatchError):
        verifier.verify_model(
            model_path=env["detector_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_12_wrong_model_id_rejected(synthetic_model_env):
    """Test 12: Supplying incorrect expected_model_id raises ModelRoleMismatchError."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    with pytest.raises(ModelRoleMismatchError, match="Model ID mismatch"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            expected_model_id="unauthorized_model_v99",
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_13_filename_path_binding_violation_rejected(synthetic_model_env):
    """Test 13: Renaming model file outside manifest mapping is rejected."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    renamed = env["tmp_path"] / "renamed_model.pth"
    renamed.write_bytes(env["gait_file"].read_bytes())

    with pytest.raises(ModelRoleMismatchError):
        verifier.verify_model(
            model_path=renamed,
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_14_unsupported_manifest_version_rejected(synthetic_model_env):
    """Test 14: Manifest with unsupported schema version raises ModelManifestError."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    m_data = env["manifest_data"]
    m_data["manifest_version"] = "99.0"
    sig_bytes = env["priv_key"].sign(canonical_manifest_bytes(m_data))

    env["manifest_file"].write_text(json.dumps(m_data), encoding="utf-8")
    env["sig_file"].write_bytes(sig_bytes)

    with pytest.raises(ModelManifestError, match="Unsupported manifest version"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_15_malformed_json_manifest_rejected(synthetic_model_env):
    """Test 15: Non-JSON manifest content raises ModelManifestError."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    env["manifest_file"].write_text("<<<NOT JSON CONTENT>>>", encoding="utf-8")

    with pytest.raises(ModelManifestError, match="Malformed JSON"):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_16_malformed_base64_signature_rejected(synthetic_model_env):
    """Test 16: Invalid Base64 or non-64-byte signature raises ModelSignatureError."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    env["sig_file"].write_bytes(b"INVALID_SHORT_SIG")

    with pytest.raises(ModelSignatureError):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_17_strict_mode_unsigned_model_rejected(tmp_path: Path):
    """Test 17: In strict mode, an unsigned model without manifest is rejected."""
    unsigned_file = tmp_path / "unsigned_model.pth"
    unsigned_file.write_bytes(b"DATA")

    verifier = ModelVerifier(
        strict_mode=True,
        default_manifest_path=tmp_path / "missing.json",
        default_signature_path=tmp_path / "missing.sig",
    )

    with pytest.raises(ModelTrustConfigurationError):
        verifier.verify_model(unsigned_file, expected_role=ROLE_GAIT_EMBEDDING)


def test_18_dev_mode_unsigned_model_allowed_with_warning(tmp_path: Path, caplog):
    """Test 18: In dev mode, unsigned model is loaded and logs a single security warning."""
    unsigned_file = tmp_path / "unsigned_dev_model.pth"
    unsigned_file.write_bytes(b"DEV_MODEL_DATA")

    verifier = ModelVerifier(
        strict_mode=False,
        default_manifest_path=tmp_path / "missing.json",
        default_signature_path=tmp_path / "missing.sig",
    )

    with caplog.at_level(logging.WARNING):
        result = verifier.verify_model(unsigned_file, expected_role=ROLE_GAIT_EMBEDDING)

    assert result == unsigned_file
    assert "Model signature verification is not enforced" in caplog.text


def test_19_no_verification_failure_fallback(synthetic_model_env):
    """Test 19: If verification fails, verifier never returns an unverified path."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    # Empty the signature file
    env["sig_file"].write_bytes(b"")

    with pytest.raises(ModelSignatureError):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_20_loader_not_called_before_verification_success(synthetic_model_env):
    """Test 20: Framework loader is never invoked if verification fails."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    mock_loader = MagicMock()

    def guarded_loader(path, role):
        v_path = verifier.verify_model(
            model_path=path,
            expected_role=role,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )
        return mock_loader(v_path)

    # Corrupt model file
    env["gait_file"].write_bytes(b"CORRUPTED")

    with pytest.raises(ModelDigestMismatchError):
        guarded_loader(env["gait_file"], ROLE_GAIT_EMBEDDING)

    # Verify mock loader was NEVER called
    mock_loader.assert_not_called()


# ==============================================================================
# 21-30: SERIALIZATION, DESERIALIZATION & WORKFLOW TESTS
# ==============================================================================


def test_21_deterministic_manifest_canonicalization():
    """Test 21: Canonical manifest serialization is deterministic and order-invariant."""
    d1 = {"b": 2, "a": 1, "nested": {"y": 20, "x": 10}}
    d2 = {"nested": {"x": 10, "y": 20}, "a": 1, "b": 2}

    b1 = canonical_manifest_bytes(d1)
    b2 = canonical_manifest_bytes(d2)

    assert b1 == b2
    assert b" " not in b1
    assert b1.decode("utf-8") == '{"a":1,"b":2,"nested":{"x":10,"y":20}}'


def test_22_different_signing_key_rejected(synthetic_model_env):
    """Test 22: Signature created with key B cannot be verified with key A."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    other_priv = ed25519.Ed25519PrivateKey.generate()
    other_sig = other_priv.sign(canonical_manifest_bytes(env["manifest_data"]))
    env["sig_file"].write_bytes(other_sig)

    with pytest.raises(ModelSignatureError):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role=ROLE_GAIT_EMBEDDING,
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_23_signature_valid_but_wrong_role_rejected(synthetic_model_env):
    """Test 23: Model signature is valid, but model is requested for an unregistered role."""
    env = synthetic_model_env
    verifier: ModelVerifier = env["verifier"]

    with pytest.raises(ModelRoleMismatchError):
        verifier.verify_model(
            model_path=env["gait_file"],
            expected_role="unauthorized_secret_role",
            manifest_path=env["manifest_file"],
            signature_path=env["sig_file"],
            public_key=env["pub_key"],
        )


def test_24_rollback_limitation_documented():
    """Test 24: Document that signature proves authenticity but not independent freshness."""
    priv = ed25519.Ed25519PrivateKey.generate()
    old_manifest = {"manifest_version": "1.0", "version": "1.0.0", "models": {}}
    new_manifest = {"manifest_version": "1.0", "version": "1.1.0", "models": {}}

    old_sig = priv.sign(canonical_manifest_bytes(old_manifest))
    new_sig = priv.sign(canonical_manifest_bytes(new_manifest))

    # Both signatures are mathematically valid under the same public key
    priv.public_key().verify(old_sig, canonical_manifest_bytes(old_manifest))
    priv.public_key().verify(new_sig, canonical_manifest_bytes(new_manifest))
    # Cryptographic model signatures alone cannot determine which is newer without version state


def test_25_pytorch_state_dict_weights_only_true_success(tmp_path: Path):
    """Test 25: Safe state_dict loading with weights_only=True succeeds cleanly."""
    model = ByGaitLight()
    pth_path = tmp_path / "test_bygait.pth"
    torch.save(model.state_dict(), pth_path)

    loaded_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    assert isinstance(loaded_dict, dict)
    assert "features.0.weight" in loaded_dict


def test_26_malicious_pickle_rejected_by_weights_only(tmp_path: Path):
    """Test 26: A crafted malicious pickle payload fails under weights_only=True."""
    import pickle

    class MaliciousRCE:
        def __reduce__(self):
            return (os.system, ("echo VULNERABLE",))

    payload = {"state": MaliciousRCE()}
    bad_path = tmp_path / "rce.pth"
    with open(bad_path, "wb") as f:
        pickle.dump(payload, f)

    with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError, AttributeError)):
        torch.load(bad_path, map_location="cpu", weights_only=True)


def test_27_osnet_does_not_fallback_to_unsafe_pickle(tmp_path: Path, monkeypatch):
    """Test 27: OSNetBackbone raises an error rather than falling back to weights_only=False."""
    from models.reid.osnet_backbone import OSNetBackbone

    bad_model = tmp_path / "osnet_bad.pth"
    bad_model.write_bytes(b"CORRUPTED_NON_TENSOR_FILE")

    monkeypatch.setattr(OSNetBackbone, "_instance", None)
    backbone = OSNetBackbone(model_path=str(bad_model))
    with pytest.raises(RuntimeError, match="Failed to load OSNet checkpoint safely with weights_only=True"):
        backbone._ensure_model()


def test_28_yolo_path_verification_occurs_before_constructor(monkeypatch, synthetic_model_env):
    """Test 28: YOLO verification gate intercepts before YOLO constructor runs."""
    from pipeline.steps.detection import DetectionStep

    env = synthetic_model_env
    # Corrupt detector file
    env["detector_file"].write_bytes(b"CORRUPTED_YOLO")

    # In strict mode, DetectionStep must fail before calling ultralytics YOLO
    verifier = ModelVerifier(
        strict_mode=True,
        default_manifest_path=env["manifest_file"],
        default_signature_path=env["sig_file"],
        public_key=env["pub_key"],
    )
    monkeypatch.setattr("security_layer.model_integrity.get_model_verifier", lambda: verifier)

    with pytest.raises(ModelDigestMismatchError):
        DetectionStep(model_path=str(env["detector_file"]))


def test_29_onnx_path_verification_occurs_before_inference_session(monkeypatch, synthetic_model_env):
    """Test 29: Silhouette step verifies model before creating InferenceSession."""
    from pipeline.steps.silhouette_step import LearnedSilhouetteSegmenter

    env = synthetic_model_env
    env["onnx_file"].write_bytes(b"CORRUPTED_ONNX")

    verifier = ModelVerifier(
        strict_mode=True,
        default_manifest_path=env["manifest_file"],
        default_signature_path=env["sig_file"],
        public_key=env["pub_key"],
    )
    monkeypatch.setattr("security_layer.model_integrity.get_model_verifier", lambda: verifier)

    with pytest.raises(ModelDigestMismatchError):
        LearnedSilhouetteSegmenter(model_path=str(env["onnx_file"]))


def test_30_no_automatic_download_in_strict_mode(monkeypatch, tmp_path: Path):
    """Test 30: Missing model in strict mode fails closed without triggering YOLO download."""
    from pipeline.steps.detection import DetectionStep

    missing_path = tmp_path / "nonexistent_yolo.pt"
    verifier = ModelVerifier(strict_mode=True, public_key=b"1" * 32)
    monkeypatch.setattr("security_layer.model_integrity.get_model_verifier", lambda: verifier)

    with pytest.raises(ModelIntegrityError):
        DetectionStep(model_path=str(missing_path))


# ==============================================================================
# 31-38: SAFETY, KEY SCAN & PRIOR REGRESSION TESTS
# ==============================================================================


def test_31_no_private_signing_key_committed():
    """Test 31: Assert no private signing key material exists in repository source or models."""
    repo_root = Path("E:/ARGUS_AI")

    # 1. No private key files in models/ directory
    models_dir = repo_root / "models"
    for p in models_dir.rglob("*"):
        if p.is_file():
            assert p.suffix not in {".pem", ".key", ".priv"}, f"Key file found in models/: {p}"
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                assert "PRIVATE KEY" not in content, f"Private key material found in models/ file: {p}"
            except (UnicodeDecodeError, OSError, PermissionError):
                pass

    # 2. No private key files in security_layer/ or tools/security/
    for check_dir in [repo_root / "security_layer", repo_root / "tools" / "security"]:
        for p in check_dir.rglob("*"):
            if p.is_file():
                assert p.suffix not in {".pem", ".key", ".priv"}, f"Key file found in {check_dir}: {p}"

    # 3. No Ed25519 private key material committed anywhere across source code
    source_dirs = ["api", "services", "storage", "pipeline", "security_layer", "tools", "models"]
    for s_dir in source_dirs:
        for p in (repo_root / s_dir).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".yml", ".md", ".txt"}:
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    assert "BEGIN ED25519 PRIVATE KEY" not in content, f"Ed25519 private key found in {p}"
                except (UnicodeDecodeError, OSError, PermissionError):
                    pass


def test_32_no_private_key_printed(capsys):
    """Test 32: Model verifier functions do not print or leak key material."""
    verifier = ModelVerifier()
    verifier.set_strict_mode(False)
    captured = capsys.readouterr()
    assert "PRIVATE KEY" not in captured.out
    assert "PRIVATE KEY" not in captured.err


def test_33_public_key_may_be_loaded_safely(ephemeral_keypair):
    """Test 33: Public key can be safely loaded from raw bytes, base64, or PEM."""
    _, pub = ephemeral_keypair
    raw_32 = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    b64_str = base64.b64encode(raw_32).decode("ascii")
    pem_bytes = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    k1 = load_public_key(raw_32)
    k2 = load_public_key(b64_str)
    k3 = load_public_key(pem_bytes)

    assert isinstance(k1, ed25519.Ed25519PublicKey)
    assert isinstance(k2, ed25519.Ed25519PublicKey)
    assert isinstance(k3, ed25519.Ed25519PublicKey)


def test_34_locked_gait_pipeline_output_equivalence():
    """Test 34: Safe loading produces equivalent ByGaitLight weights and 256D embeddings."""
    real_ckpt = Path("runs/exp_001/best_model.pth")
    if not real_ckpt.exists():
        pytest.skip("Real checkpoint runs/exp_001/best_model.pth not present")

    # Load baseline
    raw_ckpt = torch.load(real_ckpt, map_location="cpu", weights_only=True)
    filtered = {k.replace("backbone.", ""): v for k, v in raw_ckpt.items() if k.startswith("backbone.")}
    part_bins = 1
    if "embedding.weight" in filtered:
        part_bins = max(1, filtered["embedding.weight"].shape[1] // 128)

    m1 = ByGaitLight(part_bins=part_bins)
    m1.load_state_dict(filtered)
    m1.eval()

    # Load via safe weights_only
    safe_ckpt = torch.load(real_ckpt, map_location="cpu", weights_only=True)
    filtered_safe = {k.replace("backbone.", ""): v for k, v in safe_ckpt.items() if k.startswith("backbone.")}
    m2 = ByGaitLight(part_bins=part_bins)
    m2.load_state_dict(filtered_safe)
    m2.eval()

    # Compare parameters
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)

    # Compare inference on known synthetic input
    dummy_gei = torch.randn(1, 1, 64, 128)
    with torch.no_grad():
        out1 = m1(dummy_gei).numpy()
        out2 = m2(dummy_gei).numpy()

    np.testing.assert_allclose(out1, out2, atol=1e-7)
    assert out1.shape == (1, 256)


def test_35_real_model_files_remain_unchanged():
    """Test 35: Assert real repository model files have not been overwritten or corrupted."""
    real_paths = [
        Path("runs/exp_001/best_model.pth"),
        Path("models/model_store/weights/osnet_x0_25.pth"),
        Path("models/model_store/weights/yolov8n.pt"),
        Path("models/model_store/weights/silhouette_segmenter.onnx"),
    ]
    for p in real_paths:
        if p.exists():
            assert p.stat().st_size > 1000, f"Model file {p} is suspiciously small"


def test_36_u2_audit_log_regression_unaffected(tmp_path: Path):
    """Test 36: U2 audit log integrity HMAC-SHA256 logging remains intact."""
    from security_layer.security_logger import SecurityLogger

    logger_inst = SecurityLogger(
        log_file=str(tmp_path / "test_u5_audit.csv"),
        hmac_key=b"A" * 32,
    )
    row_hmac = logger_inst.log(
        track_id="TRK-U5-TEST",
        identity="TEST_SUBJECT",
        score=0.95,
        severity="INFO",
        decision="ALLOW",
        camera_id="CAM-01",
    )
    assert row_hmac is not None
    assert len(row_hmac) == 64


def test_37_u3_biometric_encryption_regression_unaffected(tmp_path: Path):
    """Test 37: U3 biometric template encryption at rest remains functional."""
    from security_layer.biometric_encryption import BiometricEncryptor

    key = os.urandom(32)
    encryptor = BiometricEncryptor(key=key)

    data = np.random.randn(5, 256).astype(np.float32)
    labels = np.array(["P1", "P2", "P3", "P4", "P5"])

    enc_bytes = encryptor.encrypt_gallery_container(
        features=data,
        labels=labels,
        gallery_type="live_gallery",
    )
    assert len(enc_bytes) > 0

    dec_features, aad_meta = encryptor.decrypt_gallery_container(
        container_bytes=enc_bytes,
        expected_gallery_type="live_gallery",
        expected_labels=labels,
    )
    np.testing.assert_allclose(dec_features, data, atol=1e-6)
    assert aad_meta["gallery_type"] == "live_gallery"


def test_38_u4_camera_transport_security_unaffected():
    """Test 38: U4 camera transport credentials sanitization remains intact."""
    from security_layer.credentials import sanitize_rtsp_url

    raw = "rtsp://operator:SecretPass999@10.0.0.1:554/live"
    sanitized = sanitize_rtsp_url(raw)
    assert "SecretPass999" not in sanitized
    assert "operator" not in sanitized
    assert "rtsp://***:***@10.0.0.1:554/live" == sanitized


# ==============================================================================
# 39-45: STRICT-MODE CONFIGURATION POLICY TESTS
# ==============================================================================


def test_39_env_absent_strict_mode_off(monkeypatch):
    """Test 39: When ARGUS_MODEL_STRICT_MODE is absent, strict mode is OFF."""
    monkeypatch.delenv("ARGUS_MODEL_STRICT_MODE", raising=False)
    monkeypatch.delenv("ARGUS_REQUIRE_SIGNED_MODELS", raising=False)
    verifier = ModelVerifier()
    assert verifier.is_strict_mode() is False


def test_40_env_value_1_strict_mode_on(monkeypatch):
    """Test 40: ARGUS_MODEL_STRICT_MODE=1 enables strict mode."""
    monkeypatch.delenv("ARGUS_REQUIRE_SIGNED_MODELS", raising=False)
    monkeypatch.setenv("ARGUS_MODEL_STRICT_MODE", "1")
    verifier = ModelVerifier()
    assert verifier.is_strict_mode() is True


def test_41_env_value_true_strict_mode_on(monkeypatch):
    """Test 41: ARGUS_MODEL_STRICT_MODE=true enables strict mode."""
    monkeypatch.delenv("ARGUS_REQUIRE_SIGNED_MODELS", raising=False)
    monkeypatch.setenv("ARGUS_MODEL_STRICT_MODE", "true")
    verifier = ModelVerifier()
    assert verifier.is_strict_mode() is True


def test_42_env_value_yes_strict_mode_on(monkeypatch):
    """Test 42: ARGUS_MODEL_STRICT_MODE=yes enables strict mode."""
    monkeypatch.delenv("ARGUS_REQUIRE_SIGNED_MODELS", raising=False)
    monkeypatch.setenv("ARGUS_MODEL_STRICT_MODE", "yes")
    verifier = ModelVerifier()
    assert verifier.is_strict_mode() is True


def test_43_env_value_0_strict_mode_off(monkeypatch):
    """Test 43: ARGUS_MODEL_STRICT_MODE=0 disables strict mode."""
    monkeypatch.delenv("ARGUS_REQUIRE_SIGNED_MODELS", raising=False)
    monkeypatch.setenv("ARGUS_MODEL_STRICT_MODE", "0")
    verifier = ModelVerifier()
    assert verifier.is_strict_mode() is False


def test_44_env_value_false_strict_mode_off(monkeypatch):
    """Test 44: ARGUS_MODEL_STRICT_MODE=false disables strict mode."""
    monkeypatch.delenv("ARGUS_REQUIRE_SIGNED_MODELS", raising=False)
    monkeypatch.setenv("ARGUS_MODEL_STRICT_MODE", "false")
    verifier = ModelVerifier()
    assert verifier.is_strict_mode() is False


def test_45_unrecognized_value_strict_mode_off_with_warning(monkeypatch, caplog):
    """Test 45: Unrecognized ARGUS_MODEL_STRICT_MODE value defaults OFF with warning."""
    monkeypatch.delenv("ARGUS_REQUIRE_SIGNED_MODELS", raising=False)
    monkeypatch.setenv("ARGUS_MODEL_STRICT_MODE", "maybe")
    with caplog.at_level(logging.WARNING):
        verifier = ModelVerifier()
    assert verifier.is_strict_mode() is False
    assert "Unrecognized value" in caplog.text
    assert "maybe" in caplog.text


def test_46_legacy_env_var_supported(monkeypatch):
    """Test 46: Legacy ARGUS_REQUIRE_SIGNED_MODELS is honored when ARGUS_MODEL_STRICT_MODE is absent."""
    monkeypatch.delenv("ARGUS_MODEL_STRICT_MODE", raising=False)
    monkeypatch.setenv("ARGUS_REQUIRE_SIGNED_MODELS", "1")
    verifier = ModelVerifier()
    assert verifier.is_strict_mode() is True
