"""Offline Model Release Signing Tool.

Finding U5 — Model Checkpoint / AI Model Asset Protection.
This offline release tool creates deterministic model manifests and signs them
using an externally provisioned Ed25519 private key.

SECURITY CONSTRAINTS:
- NEVER auto-generates production private keys.
- NEVER commits or logs private key material.
- Verifier / runtime nodes must never have access to the private signing key.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.security_layer.model_integrity import (
    canonical_manifest_bytes,
    compute_file_sha256,
    normalize_role,
)


def load_private_key(key_material: str | bytes) -> ed25519.Ed25519PrivateKey:
    """Load an Ed25519 private key from raw 32 bytes or PEM format."""
    if isinstance(key_material, str):
        key_str = key_material.strip()
        if len(key_str) < 512 and Path(key_str).is_file():
            key_material = Path(key_str).read_bytes()
        elif key_str.startswith("-----BEGIN"):
            key_material = key_str.encode("utf-8")
        elif len(key_str) == 64 and all(c in "0123456789abcdefABCDEF" for c in key_str):
            key_material = bytes.fromhex(key_str)
        else:
            try:
                key_material = base64.b64decode(key_str)
            except Exception as e:
                raise ValueError(f"Failed to decode private key string: {e}") from e

    raw_bytes = bytes(key_material).strip()
    if raw_bytes.startswith(b"-----BEGIN"):
        loaded = serialization.load_pem_private_key(raw_bytes, password=None)
        if not isinstance(loaded, ed25519.Ed25519PrivateKey):
            raise ValueError(f"Expected Ed25519 private key, got {type(loaded).__name__}")
        return loaded

    if len(raw_bytes) == 32:
        return ed25519.Ed25519PrivateKey.from_private_bytes(raw_bytes)

    try:
        decoded = base64.b64decode(raw_bytes)
        if len(decoded) == 32:
            return ed25519.Ed25519PrivateKey.from_private_bytes(decoded)
    except (binascii.Error, ValueError, TypeError):
        pass

    raise ValueError("Invalid private key material; expected 32 raw bytes or PEM-encoded Ed25519 private key")


def build_manifest_entry(
    model_path: str | Path,
    role: str,
    model_id: str,
    model_version: str = "1.0.0",
    framework: str = "pytorch_state_dict",
) -> dict[str, Any]:
    """Calculate hash and size for a model file and build manifest entry."""
    p = Path(model_path)
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {p}")

    sha256_hex, size_bytes = compute_file_sha256(p)
    canonical_role = normalize_role(role)

    return {
        "model_role": canonical_role,
        "model_id": model_id,
        "model_version": model_version,
        "filename": p.name,
        "sha256": sha256_hex,
        "size_bytes": size_bytes,
        "framework": framework,
    }


def sign_manifest(
    manifest_data: dict[str, Any],
    private_key: ed25519.Ed25519PrivateKey,
) -> tuple[bytes, str]:
    """Sign canonical manifest bytes with Ed25519 private key.

    Returns:
        (raw_signature_bytes, base64_signature_string)
    """
    canonical_bytes = canonical_manifest_bytes(manifest_data)
    sig_bytes = private_key.sign(canonical_bytes)
    sig_b64 = base64.b64encode(sig_bytes).decode("ascii")
    return sig_bytes, sig_b64


def main() -> int:
    parser = argparse.ArgumentParser(description="Sign an ARGUS AI model manifest using an Ed25519 private key")
    parser.add_argument(
        "--manifest-out",
        default="ml_platform/models/model_manifest.json",
        help="Target output path for canonical manifest JSON",
    )
    parser.add_argument(
        "--sig-out",
        default="ml_platform/models/model_manifest.sig",
        help="Target output path for manifest digital signature",
    )
    parser.add_argument(
        "--private-key-file",
        help="Path to externally provisioned Ed25519 private key (PEM or raw 32-bytes)",
    )
    parser.add_argument(
        "--signing-key-id",
        default="argus-ed25519-rel-2026-v1",
        help="Identifier of the signing key",
    )
    parser.add_argument(
        "--add-model",
        action="append",
        nargs=5,
        metavar=("PATH", "ROLE", "MODEL_ID", "VERSION", "FRAMEWORK"),
        help="Add a model entry: path, role, id, version, framework",
    )

    args = parser.parse_args()

    # Resolve private key
    key_material = os.getenv("ARGUS_RELEASE_SIGNING_PRIVATE_KEY")
    if args.private_key_file:
        key_path = Path(args.private_key_file)
        if not key_path.is_file():
            print(f"[ERROR] Private key file not found: {key_path}", file=sys.stderr)
            return 1
        key_material = key_path.read_bytes()

    if not key_material:
        print(
            "[ERROR] An Ed25519 private key must be provided via --private-key-file "
            "or ARGUS_RELEASE_SIGNING_PRIVATE_KEY environment variable.",
            file=sys.stderr,
        )
        return 1

    try:
        private_key = load_private_key(key_material)
    except (ValueError, TypeError, OSError) as e:
        print(f"[ERROR] Failed to load Ed25519 private key: {e}", file=sys.stderr)
        return 1

    models_dict: dict[str, Any] = {}
    if args.add_model:
        for m_path, m_role, m_id, m_ver, m_fw in args.add_model:
            try:
                entry = build_manifest_entry(m_path, m_role, m_id, m_ver, m_fw)
                models_dict[normalize_role(m_role)] = entry
                print(f"[INFO] Added model: {m_path} (role={m_role}, sha256={entry['sha256'][:12]}...)")
            except (FileNotFoundError, ValueError, KeyError, OSError) as e:
                print(f"[ERROR] Failed to process model {m_path}: {e}", file=sys.stderr)
                return 1

    manifest_data: dict[str, Any] = {
        "manifest_version": "1.0",
        "signing_key_id": args.signing_key_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": models_dict,
    }

    # Sign canonical manifest
    sig_bytes, _sig_b64 = sign_manifest(manifest_data, private_key)

    out_manifest = Path(args.manifest_out)
    out_sig = Path(args.sig_out)

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    out_sig.write_bytes(sig_bytes)

    print(f"[SUCCESS] Signed manifest written to {out_manifest}")
    print(f"[SUCCESS] Ed25519 signature written to {out_sig} ({len(sig_bytes)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
